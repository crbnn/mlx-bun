import {captureSpeculativeOptions, speculativePrefixNamespace} from "../generation/speculative/cache-identity";
import type { MlxTokenGraph } from "../models/graph";
import type { GenerateOptions } from "../generation/index";
import type { FillSession, Proposal } from "../generation/fill/session";
import type { DraftProvider, DraftRowGroup, DraftPrefillGroup, DraftSelection } from "../generation/speculative/source";
import type { DraftRowCheckpoint } from "../contracts/mlx/draft-checkpoint";
import { makeSampler, makeStepSampler, readStepExtras, type NumberStepSampler, type Sampler } from "../sampling/index";
import { draftSamplerOptions } from "../generation/speculative/draft-policy";
import { COMPOSITION_DEFAULTS } from "../contracts/portable/composition";
import { makeSubsetDraftSampler, type DraftVocabularyHead, type SubsetDraftSampler } from "../generation/speculative/draft-vocab";
import { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import { clearCache } from "@mlx-bun/mlx/ffi";
import { applyStateChanges, cleanupFailure, disposeResources } from "../runtime/resources";
import { cloneKvCaches } from "../state/persistence";
import type { Cache } from "../contracts/mlx/cache";
import { targetCacheLayout } from "../state/layout";
import { MlxStateRows } from "../state/rows";
import { leaseCacheStates } from "../state/leases";
import { bindSpeculativeTargetModel, type MlxSpeculativeTargetBinding } from "../generation/speculative/bindings/binding";
import { bindLegacyDraftTarget } from "../generation/speculative/bindings/draft-target";
import { bindRowCacheRollback, bindRowStepRollback } from "../state/rollback";
import { disposeAttachments } from "../state/checkpoint";
import { type CheckpointAttachment } from "../contracts/mlx/checkpoint";
import { advanceSpeculativeOutputs } from "../generation/speculative/round";
import { AdaptiveDraftGate } from "../generation/speculative/adaptive-draft-gate";
import { MlxPrefillRows, type MlxPrefillState } from "./prefill-rows";
import { prefillPhase } from "../contracts/portable/prefill";
import type { MlxForwardWork } from "../contracts/mlx/forward-work";
import type { MlxGroupMethodHost, MlxGroupMethodRequest, MlxGroupPreparation, MlxGroupedMethod, Row } from "./batch-types";

interface RequestState {
  fill?: FillSession;
  proposal?: { value: Proposal; emitted: number };
  sampling: NumberStepSampler;
  draftSampling: Sampler;
  /** Coupled draw over a draft vocabulary; null when not reproducible there. */
  draftSubset: SubsetDraftSampler | null;
  retain?: () => void;
  namespace: string;
  prefixLength: number;
  /** Generated inputs retained by committed target/draft rounds. The pending
   * correction is excluded until a later round actually processes it. */
  processed?: number[];
}

/** Binding owns graph/layout selection. The executor receives only the method
 * key and lifecycle; sampling, checkpoints and numerical state remain ports.
 * Undefined when the target cannot capture the hidden layers the provider taps. */
export function bindSpeculativeGroupRequests(model: MlxTokenGraph, provider: Pick<DraftProvider, "id" | "grouped">, depth: number,
  adaptiveDepth = false) {
  if (adaptiveDepth && !model.verifyRoundCosts)
    throw new Error("adaptive draft depth needs a graph that declares its measured verify costs (verifyRoundCosts)");
  const binding = bindSpeculativeTargetModel(model);
  // The target's ports, viewed only for a provider that asks about them.
  let view: ReturnType<typeof bindLegacyDraftTarget> | undefined;
  const target = () => view ??= bindLegacyDraftTarget(model, []);
  // Resolved once for this target into a snapshot the binding owns: the checked
  // list cannot change afterwards, and prefill and decode rows must tap exactly it.
  const taps = Object.freeze([...(provider.grouped.targetTapLayers?.(target()) ?? [])]);
  if (!binding.supportsTapLayers(taps)) return undefined;
  // The selection's composition facts, resolved once for this target: argmax
  // drafts (the composition's `greedyDraft`), else the vocabulary the provider
  // declares for it.
  const greedy = (model.composition ?? COMPOSITION_DEFAULTS).greedyDraft;
  const listed = greedy ? null : provider.grouped.draftVocabulary?.(target()) ?? null;
  return (input: GenerateOptions): MlxGroupMethodRequest => {
    const options = captureSpeculativeOptions(input);
    // The request's selection, resolved once here: the listed vocabulary only
    // when its draft sampler reproduces there. Rows sharing a group share it, so
    // the draft rows bind one selection operation when they open.
    const selection: GroupSelection = greedy ? { kind: "greedy" }
      : listed && makeSubsetDraftSampler(draftSamplerOptions(options, false)) ? { kind: "listed", ids: listed } : { kind: "sampled" };
    return {
      // The KV scheme is the composition's, shared by every request.
      key: JSON.stringify(["speculative", provider.id, depth, adaptiveDepth, selection.kind]),
      data: options,
      open: host => new SpeculativeGroup(host, model, provider, binding, depth, taps, adaptiveDepth, selection),
    };
  };
}

/** A step without drafts whose samples are still unread: the GPU runs it while the step before it is delivered.
 *  Its input is kept (its window resolved, its taps committed to the drafter); `tokens` [B] are the samples the next
 *  step reads as its input, owned, and `rows` the rows it ran, in order. */
interface AheadStep {
  readonly tokens: MlxArray;
  readonly rows: readonly Row[];
}

/** A probe drafted without a verify round: proposals [B, depth] from each row's input to a step without drafts,
 *  compared with the target's samples as the following steps deliver them. `seen` counts each row's samples compared
 *  so far; a row leaves it at its first mismatch or after `depth` matches, when the gate observes it. */
interface ShadowProbe {
  tokens: MlxArray | null;
  proposals: Map<Row, readonly number[]> | null;
  readonly rows: readonly Row[];
  readonly depth: number;
  readonly seen: Map<Row, number>;
}

/** How one delivered sample left its row: still decoding, or ended. */
type StepOutcome = { readonly kind: "continue" } | { readonly kind: "stop" | "length" } |
  { readonly kind: "cancelled"; readonly reason: unknown } | { readonly kind: "failed"; readonly error: unknown };

/** The draft selection one group's requests share, before the group binds its rows' draws. */
type GroupSelection = { readonly kind: "sampled" } | { readonly kind: "greedy" } |
  { readonly kind: "listed"; readonly ids: readonly number[] };

/** Rows must tap exactly the target layers resolved for their provider at binding time. */
function declaredTaps<Rows extends { readonly tapLayers: readonly number[]; dispose(): void }>(
  provider: Pick<DraftProvider, "id">, declared: readonly number[], rows: Rows): Rows {
  if (rows.tapLayers.length !== declared.length || rows.tapLayers.some((layer, index) => layer !== declared[index])) {
    rows.dispose();
    throw new Error(`draft provider ${provider.id} taps [${rows.tapLayers}] but declared [${declared}]`);
  }
  return rows;
}

class SpeculativeGroup implements MlxGroupedMethod {
  #target: MlxStateRows<ReturnType<typeof targetCacheLayout>> | null = null;
  #draft: DraftRowGroup | null = null;
  readonly #requests = new Map<Row, RequestState>();
  #steps = 0;
  /** The step without drafts in flight, when the last round was one. */
  #ahead: AheadStep | null = null;
  /** The probe the steps without drafts are comparing, if any. */
  #shadow: ShadowProbe | null = null;
  /** Adaptive draft depth: each round's depth comes from the decayed
   *  acceptance rate and the graph's measured verify costs. */
  readonly #gate: AdaptiveDraftGate | null;

  constructor(readonly host: MlxGroupMethodHost, readonly model: MlxTokenGraph,
    readonly provider: Pick<DraftProvider, "id" | "grouped">, readonly binding: MlxSpeculativeTargetBinding, readonly depth: number,
    readonly taps: readonly number[], adaptiveDepth: boolean, readonly selection: GroupSelection) {
    this.#gate = adaptiveDepth ? new AdaptiveDraftGate(model.verifyRoundCosts!, { maxDraftTokens: depth }) : null;
  }

  get runningTokens(): number {
    const rows = this.host.rows;
    if (!rows.length) return 0;
    // Target verification consumes the pending token and the draft candidates.
    const echo = Math.max(...rows.map(row => {
      const proposal = this.#requests.get(row)?.proposal;
      return proposal?.emitted ? proposal.value.ids.length - proposal.emitted : 0;
    }));
    return rows.length * (1 + (echo || Math.min(this.depth, Math.max(...rows.map(row => row.req.maxTokens - row.generated)))));
  }

  prepare(first: Row): MlxGroupPreparation {
    const method = this;
    const maintain = this.host.kvMaintenance;
    interface Preparation extends MlxPrefillState {
      request: RequestState;
      end: number;
      chunkSize: number;
      pendingPrompt: boolean;
      boundary: number;
      transferred: boolean;
      closed: boolean;
    }
    interface Ready {
      row: Row;
      request: RequestState;
      caches: Cache[];
      draft: DraftRowCheckpoint;
      retain?: () => void;
      transferred: boolean;
    }
    let prefix: DraftPrefillGroup | undefined;
    let context: MlxArray | null = null;
    const ready: Ready[] = [];
    const closeReady = (state: Ready) => {
      const caches = state.caches.splice(0), attachments = [state.draft.attachment];
      const release = state.retain; state.retain = undefined;
      disposeResources([...caches, { dispose: () => disposeAttachments(attachments) },
        { dispose: () => release?.() }, ...(!state.transferred ? [state.request.sampling] : [])]);
    };
    const capture = (state: Preparation, snapshot: () => Cache[], index: number) => {
      let caches: Cache[] = [], attachments: CheckpointAttachment[] = [];
      try {
        caches = snapshot(); attachments.push(prefix!.capture(index).attachment);
        method.host.cache.publish(state.row.req, state.request.namespace, state.row.req.promptIds.slice(0, state.pos),
          { caches, attachments });
        caches = []; attachments = [];
      } finally { disposeResources([...caches, { dispose: () => disposeAttachments(attachments) }]); }
    };
    const target = new MlxPrefillRows<Preparation>({
      maintain, codecs: method.host.cache.codecs,
      open(row) {
        const options = row.req.method!.data as GenerateOptions;
        row.spec = { drafted: 0, accepted: 0, targetCalls: 0, rejected: 0, rounds: 0,
          acceptanceLengths: [], tokensPerForward: 0, forwardsSaved: 0, draftedByPos: [], acceptedByPos: [] };
        // Draft policy: the request sampler (mlx-lm parity) or, opt-in, argmax.
        // The verifier accepts a draft only when the TARGET's own sample equals
        // it, so the output distribution never depends on how drafts are chosen;
        // the argmax of the head's estimate maximizes the expected match.
        const draftOptions = draftSamplerOptions(options, method.selection.kind === "greedy");
        const sampler = makeSampler(draftOptions);
        const request: RequestState = { draftSampling: sampler, draftSubset: makeSubsetDraftSampler(draftOptions),
          namespace: "", prefixLength: 0,
          fill: method.provider.grouped.supportsExternalTokens ? options.fill : undefined,
          processed: method.host.cache.keeps(row.req) ? [] : undefined,
          sampling: makeStepSampler(options, { tokenRepresentation: "number", grammarWait: "before-sample",
            historyUpdate: "after-sample", initialHistory: row.req.promptIds, acceptGrammar: true,
            eosTokenIds: row.req.eosTokenIds, captureSelectedLogprob: options.logprobs === true,
            captureTopLogprobs: options.topLogprobs }) };
        if (request.fill) row.fill = request.fill.stats;
        let caches: Cache[] = [], retained: (() => void) | undefined;
        let appended = false;
        const previous = prefix?.rowCount ?? 0;
        try {
          if (!prefix) {
            // The draft prefill binds its target view once per preparation, over
            // fresh caches of the graph that bind and are released.
            const probe = method.binding.makeCache();
            try {
              prefix = declaredTaps(method.provider, method.taps,
                method.provider.grouped.openPrefill({ target: bindLegacyDraftTarget(method.model, probe), checkpoints: [] }));
            } finally { disposeResources(probe); }
          }
          const prompt = row.req.promptIds;
          const pendingPrompt = prefix.prefillMode !== "full" && method.binding.prefillTailSplit && prompt.length > 1;
          request.prefixLength = prompt.length - Number(pendingPrompt);
          request.namespace = speculativePrefixNamespace(prefix.namespace, row.cacheNamespace ?? "", options);
          // The request's state from the cache service: a reused prefix with its
          // draft state, else fresh state (paged when the composition pages).
          const acquired = method.host.cache.acquire(row.req, request.namespace);
          caches = acquired.caches; retained = acquired.retain;
          if (acquired.reusedTokens) {
            try {
              prefix.append([{ processedTokens: acquired.reusedTokens, attachment: acquired.attachments![0]! }]);
              appended = true;
              // Target membership later releases backing leases after its own
              // copies resolve. Settle companion copies under the same lease.
              if (retained) prefix.materialize();
              row.cachedTokens = acquired.reusedTokens;
            } finally { disposeAttachments(acquired.attachments); }
          } else { prefix.append([null]); appended = true; }
          // The graph's composed caches, or a stored prefix composed again by
          // the cache service: either converts inside its own appends.
          return { row, request, solo: caches, retain: retained, pos: row.cachedTokens,
            end: request.prefixLength, pendingPrompt, transferred: false, closed: false,
            chunkSize: options.prefillChunkSize ?? row.req.prefillChunkSize ?? method.host.prefillChunkSize,
            boundary: method.host.cache.keeps(row.req) ? Math.min(row.req.snapshotAt ?? prompt.length, prompt.length - 1) : -1 };
        } catch (error) {
          return cleanupFailure(error, () => disposeResources([
            { dispose: () => { if (appended) prefix!.filterRows(Array.from({ length: previous }, (_, i) => i)); } },
            ...caches, request.sampling, { dispose: () => retained?.() },
          ]));
        }
      },
      plan(state) {
        const limit = Math.min(state.end, state.boundary > state.pos ? state.boundary : Infinity);
        const whole = state.pos + state.chunkSize <= limit;
        const end = whole ? state.pos + state.chunkSize : limit;
        return { start: state.pos, end, kind: end === state.end ? "final" : "drain",
          snapshot: end === state.boundary && end > state.row.cachedTokens, batchYield: true, maintain: true,
          phase: prefillPhase(state.pos, end, whole) };
      },
      async forward(ids, caches, _states, step, work) {
        const taps = prefix!.tapLayers;
        const output = await method.binding.prefill(ids, caches, step, taps.length ? taps : undefined, work);
        context = output.ctxML;
        return output.hidden;
      },
      afterForward(ids, _caches, _states, hidden) {
        // Tapped providers consume their captured layers; assistant drafting
        // consumes the true target hidden. Other providers ignore context.
        const captured = context; context = null;
        try {
          const pending = prefix!.prefill(ids, captured ?? hidden);
          if (pending) return pending.finally(() => captured?.dispose());
        } catch (error) { captured?.dispose(); throw error; }
        captured?.dispose();
      },
      project(hidden, _caches, completed, step) {
        if (completed.every(state => state.pendingPrompt)) return null;
        using last = hidden.slice([0, hidden.shape[1]! - 1, 0], [hidden.shape[0]!, hidden.shape[1]!, hidden.shape[2]!]);
        return method.binding.projectLogits(last, step.sequences);
      },
      checkpoint: capture,
      async complete(state, logits, index) {
        const row = state.row;
        row.spec!.targetCalls++;
        if (state.pendingPrompt) { row.current = row.req.promptIds[state.end]!; row.generated = row.sampled = 0; }
        else {
          using scores = ops.reshape(logits!, [1, logits!.shape.at(-1)!]);
          const sampled = await state.request.sampling.sample(scores, 0);
          row.current = sampled.token; row.generated = row.sampled = 1;
          if (row.req.eosTokenIds.includes(row.current)) {
            if (method.host.cache.keeps(row.req)) capture(state, () => cloneKvCaches(state.solo, method.host.cache.codecs), index);
            disposeResources(state.solo.splice(0)); state.retain?.(); state.retain = undefined;
            method.host.finish(row, "stop"); return;
          }
          const more = await method.host.publish(row, row.current, readStepExtras(sampled.extras));
          state.request.fill?.observe(row.current);
          if (more === false || row.req.grammar?.isTerminated || row.generated >= row.req.maxTokens) {
            if (method.host.cache.keeps(row.req) && !row.req.signal?.aborted) capture(state, () => cloneKvCaches(state.solo, method.host.cache.codecs), index);
            disposeResources(state.solo.splice(0)); state.retain?.(); state.retain = undefined;
            method.host.finish(row, more === false || row.req.grammar?.isTerminated ? "stop" : "length"); return;
          }
        }
        method.#proposeEcho(row, state.request);
        ready.push({ row, request: state.request, caches: state.solo, retain: state.retain,
          draft: prefix!.capture(index), transferred: false });
        state.transferred = true;
      },
      reject(row, error) { row.reject(error); },
      close(state) {
        if (state.closed) return;
        state.closed = true;
        if (!state.transferred) state.request.sampling.dispose();
      },
      filterRows(keep) { prefix!.filterRows(keep); },
      dispose() {
        const held = [prefix, context].filter(value => value != null);
        prefix = undefined; context = null; disposeResources(held);
      },
    });
    const preparation: MlxGroupPreparation = {
      get rows() { return [...target.rows, ...ready.map(state => state.row)]; },
      get canAdmit() { return target.canAdmit; },
      get tokenWeight() { return target.tokenWeight + ready.reduce((sum, state) => sum + state.row.promptTokens - state.row.cachedTokens, 0); },
      admit: row => target.admit(row),
      advance: async work => {
        // Publication returns to scheduling before prepared state joins active
        // decode. This preserves the first-output flush boundary at every B.
        if (ready.length) {
          while (ready.length) {
            const state = ready[0]!;
            try {
              if (state.row.req.signal?.aborted) {
                state.row.reject(state.row.req.signal.reason);
                continue;
              }
              maintain?.prepareBatch?.(state.caches);
              this.#target ??= new MlxStateRows(state.caches.map(targetCacheLayout));
              this.#draft ??= declaredTaps(this.provider, this.taps, this.provider.grouped.open({ target: bindLegacyDraftTarget(this.model, this.#target.caches),
                checkpoints: [], sampling: { sample: (lp, steps) => this.#sampleDraftRows(lp, steps), selection: this.#draftSelection() },
                constraints: { propose: async (row, maxTokens) => this.host.rows[row]!.req.grammar?.proposeTokens(maxTokens) ?? [] } }));
              applyStateChanges([() => this.#target!.prepareAppend(state.caches),
                () => this.#draft!.prepareAppend([state.draft]), () => ({ commit: () => {
                  state.request.retain = state.retain; state.retain = undefined;
                  this.#requests.set(state.row, state.request); state.transferred = true;
                  this.host.join(state.row);
                  // New text: what the gate observed of the rows before says little about it.
                  this.#gate?.reset();
                }, dispose() {} })]);
            } finally { ready.shift(); closeReady(state); }
          }
          return target.rows.length === 0;
        }
        const done = await target.advance(work);
        return done && ready.length === 0;
      },
      dispose() {
        const held = ready.splice(0);
        disposeResources([target, ...held.map(state => ({ dispose: () => closeReady(state) }))]);
      },
    };
    try { target.admit(first); return preparation; }
    catch (error) { return cleanupFailure(error, () => preparation.dispose()); }
  }

  /** The selection this group's requests resolved when they were bound. Under
   *  `listed` every row's draft sampler reproduces on the subset (the request
   *  binding keyed its group on that), so each row draws its own coupled token. */
  #draftSelection(): DraftSelection {
    const selection = this.selection;
    if (selection.kind !== "listed") return selection;
    return { kind: "listed", ids: selection.ids, sample: (logits: MlxArray, head: DraftVocabularyHead, steps: readonly number[]): MlxArray => {
      const rows = this.host.rows.map(row => this.#requests.get(row)!);
      const tokens: MlxArray[] = [];
      try {
        for (let row = 0; row < rows.length; row++) {
          using scores = logits.slice([row, 0], [row + 1, logits.shape[1]!]);
          tokens.push(rows[row]!.draftSubset!(scores, head, steps[row]!));
        }
        return tokens.length === 1 ? tokens.pop()! : ops.concatAxis(tokens, 0);
      } finally { disposeResources(tokens); }
    } };
  }

  #sampleDraftRows(logprobs: MlxArray, steps: readonly number[]): MlxArray {
    const tokens: MlxArray[] = [];
    try {
      for (let row = 0; row < this.host.rows.length; row++) {
        using scores = logprobs.slice([row, 0], [row + 1, logprobs.shape[1]!]);
        tokens.push(this.#requests.get(this.host.rows[row]!)!.draftSampling(scores, steps[row]!));
      }
      return ops.concatAxis(tokens, 0);
    } finally { disposeResources(tokens); }
  }

  async advance(work?: MlxForwardWork): Promise<void> {
    const active = [...this.host.rows];
    const live = active.flatMap((row, index) => row.req.signal?.aborted ? [] : [index]);
    if (live.length !== active.length) {
      for (const row of active) if (row.req.signal?.aborted) row.reject(row.req.signal.reason);
      this.host.filterRows(live);
    }
    // A row that joined since the last step has no sample in flight: deliver the ones in flight first.
    if (this.#ahead && !sameRows(this.#ahead.rows, this.host.rows)) await this.#flushAhead();
    let rows = [...this.host.rows];
    if (!rows.length) return;
    const cap = Math.min(this.depth, Math.max(...rows.map(row => row.req.maxTokens - row.generated)));
    // A round without drafts verifies the pending token alone (one plain decode step) and still advances the
    // drafter's context through its commit; rows that sample on the device pipeline such rounds as plain decode does.
    let depth = this.#gate && cap > 0 ? this.#gate.next(cap, rows.length) : cap;
    // A probe from a drafter that can propose without committing is compared with the samples instead of verified.
    const shadow = depth > 0 && this.#gate?.probing && this.#draft!.preview && !this.#shadow ? cap : 0;
    if ((depth === 0 || shadow) && this.#pipelinesZero(rows)) return this.#zeroStep(rows, work, shadow);
    this.#settleShadow();
    if (this.#ahead) {
      await this.#flushAhead();
      rows = [...this.host.rows];
      if (!rows.length) return;
      depth = Math.min(depth, Math.max(...rows.map(row => row.req.maxTokens - row.generated)));
    }
    // The ordinary MTP round already supplies samples that can establish a
    // copied prefix. Do not verify a wide span before any sample matches it.
    const proposals = rows.map(row => {
      const proposal = this.#requests.get(row)!.proposal;
      return proposal?.emitted ? proposal.value.ids.slice(proposal.emitted) : [];
    });
    const echoDepth = Math.max(...proposals.map(ids => ids.length));
    let externalTokens: MlxArray | undefined;
    const rollback = bindRowCacheRollback(this.#target!.caches, rows.length);
    const fills = echoDepth ? rows.flatMap((row, index) => {
      const request = this.#requests.get(row)!;
      return proposals[index]!.length ? [request.fill!] : [];
    }) : [];
    const halted = new Set<Row>();
    try {
      const completed = await advanceSpeculativeOutputs(rows.map(row => ({ pending: row.current,
        step: row.sampled, remaining: row.req.maxTokens - row.generated, eosTokenIds: row.req.eosTokenIds,
        sampling: this.#requests.get(row)!.sampling, grammarDone: () => row.req.grammar?.isTerminated === true,
        output: { commit: async (ids, metadata) => {
          row.generated++;
          const more = await this.host.publish(row, ids[0]!, metadata?.[0]);
          if (more === false) { halted.add(row); return false; }
        } },
      })), echoDepth || depth, echoDepth ? {
        draft: () => proposals,
        commit: (accepted, context) => this.#draft!.consume!(externalTokens!, context, accepted.map(n => n + 1)),
      } : this.#draft!, {
        materialize: () => {
          const state = leaseCacheStates(this.#target!.caches);
          try { ops.evalAll([...state.borrow()]); }
          finally { state.close(); }
        },
        transaction: !echoDepth ? rollback : { ...rollback,
          begin(depth) {
            const start = performance.now(); rollback.begin(depth);
            const elapsed = performance.now() - start;
            for (const fill of fills) fill.noteVerifyEvent(elapsed);
          },
          resolve(accepted) {
            const start = performance.now(); rollback.resolve(accepted);
            const elapsed = performance.now() - start;
            for (const fill of fills) fill.stats.checkpointMs += elapsed;
          },
        },
        forward: async (ids, window) => {
          if (echoDepth) externalTokens = ops.copyOf(ids);
          const taps = this.#draft!.tapLayers;
          const result = await this.binding.verify(ids, this.#target!.caches, window, taps.length ? taps : undefined, work);
          const context = result.ctxML ?? result.hidden;
          try { return { logits: this.binding.projectLogits(result.hidden, window.sequences * window.positions), context }; }
          catch (error) { context.dispose(); throw error; }
          finally { if (result.ctxML) result.hidden.dispose(); }
        },
      });
      const keep: number[] = [];
      for (let index = 0; index < rows.length; index++) {
        const row = rows[index]!, output = completed.outputs[index]!, round = completed.rounds[index]!;
        const request = this.#requests.get(row)!;
        if (request.fill) {
          for (const token of round.acceptance.emitted.slice(0, output.generated)) {
            const proposal = request.proposal;
            if (proposal && token === proposal.value.ids[proposal.emitted]) {
              if (++proposal.emitted === proposal.value.ids.length) this.#settleEcho(request);
            } else {
              this.#settleEcho(request); request.fill.observe(token);
            }
          }
          if (output.kind === "continue") this.#proposeEcho(row, request);
          else this.#settleEcho(request);
        }
        if (output.kind !== "failed" && output.kind !== "cancelled")
          request.processed?.push(row.current, ...round.drafts.slice(0, output.accepted));
        const stats = row.spec!;
        stats.drafted += echoDepth ? 0 : round.drafts.length;
        stats.accepted += echoDepth ? 0 : round.acceptance.accepted;
        stats.rejected = stats.drafted - stats.accepted;
        stats.targetCalls++;
        if (!echoDepth) {
          this.#gate?.observe(round.drafts.length, round.acceptance.accepted);
          stats.rounds!++;
          stats.acceptanceLengths!.push(round.acceptance.accepted);
          for (let p = 0; p < round.drafts.length; p++) stats.draftedByPos![p] = (stats.draftedByPos![p] ?? 0) + 1;
          for (let p = 0; p < round.acceptance.accepted; p++) stats.acceptedByPos![p] = (stats.acceptedByPos![p] ?? 0) + 1;
        }
        row.sampled += round.acceptance.emitted.length + Number(round.acceptance.sawEos);
        // A consumer may stop inside a verified burst before its terminal EOS.
        if (output.kind === "stop" && !halted.has(row) && round.acceptance.sawEos &&
          output.generated === round.acceptance.emitted.length && row.generated < row.req.maxTokens)
          row.generated++;
        const forwards = stats.targetCalls - 1;
        stats.tokensPerForward = row.generated / forwards;
        stats.forwardsSaved = Math.max(0, row.generated - 1 - forwards);
        if (completed.phaseMs) {
          // Diagnostic phase attribution, summed per request. A shared round
          // costs every row the same wall time; it is not divided by B.
          const phase = stats.phaseMs ??= { draft: 0, verify: 0, sample: 0, commit: 0, rounds: 0 };
          phase.draft += completed.phaseMs.draft; phase.verify += completed.phaseMs.verify;
          phase.sample += completed.phaseMs.sample; phase.commit += completed.phaseMs.commit;
          phase.rounds += completed.phaseMs.rounds;
          for (const field of ["verifyOps", "draftOps"] as const) {
            const counts = completed.phaseMs[field];
            if (!counts) continue;
            const total = phase[field] ??= {};
            for (const [name, n] of Object.entries(counts)) total[name] = (total[name] ?? 0) + n;
          }
          if (completed.phaseMs.layers) {
            const layers = phase.layers ??= {};
            for (const [name, ms] of Object.entries(completed.phaseMs.layers)) layers[name] = (layers[name] ?? 0) + ms;
          }
        }
        if (output.kind === "continue") { row.current = output.pending; keep.push(index); }
        else if (output.kind === "failed") row.reject(output.error);
        else if (output.kind === "cancelled") row.reject(row.req.signal?.reason ?? new Error(output.reason));
        else {
          if (!row.req.signal?.aborted) this.#publishCheckpoint(row, index, request);
          this.host.finish(row, output.kind);
        }
      }
      if (keep.length !== rows.length) this.host.filterRows(keep);
      if (++this.#steps % 256 === 0) clearCache();
    } finally { externalTokens?.dispose(); }
  }

  /** Whether rounds without drafts pipeline for these rows: each draws its sample on the device without reading it
   *  (no grammar or captured probabilities; processor history takes the draw unread) and none echoes proposals. */
  #pipelinesZero(rows: readonly Row[]): boolean {
    return rows.every(row => {
      const request = this.#requests.get(row)!;
      return !request.fill && request.sampling.unread !== undefined;
    });
  }

  /** One sample per row of a step's logits [B, 1, V], left on the device [B]: the draws `sample` would make. */
  #sampleStep(rows: readonly Row[], logits: MlxArray): MlxArray {
    const samplers = rows.map(row => this.#requests.get(row)!.sampling);
    const V = logits.shape[2]!, B = rows.length;
    const independent = samplers[0]!.independent;
    if (independent && samplers.every(sampling => sampling.independent === independent)) {
      using flat = ops.reshape(logits, [B, V]);
      return independent.sample(flat);
    }
    const tokens: MlxArray[] = [];
    try {
      for (const [index, sampling] of samplers.entries()) {
        using slice = logits.slice([index, 0, 0], [index + 1, 1, V]);
        using scores = ops.reshape(slice, [1, V]);
        const step = rows[index]!.sampled;
        tokens.push(sampling.unread!.sample(scores, step));
      }
      return tokens.length === 1 ? tokens.pop()! : ops.concatAxis(tokens, 0);
    } finally { disposeResources(tokens); }
  }

  /** One round without drafts, pipelined as plain decode is: this step's window (one position per row, fed by the
   *  previous step's samples on the device) is built and submitted before those samples are read and delivered, so
   *  the GPU runs it meanwhile. Until delivered, its input may still end its row: rows that continue keep it and
   *  commit its taps to the drafter, rows that end drop it (`bindRowStepRollback`), so a finished row's state covers
   *  exactly the tokens it processed, as after an unpipelined round. A graph or state failure throws and discards
   *  the group. */
  async #zeroStep(rows: readonly Row[], work: MlxForwardWork | undefined, shadow = 0): Promise<void> {
    // A round for every row, which delivers the sample in flight; the forward runs unless every row has drawn its last.
    const forward = !this.#ahead || rows.some(row => row.sampled < row.req.maxTokens);
    for (const row of rows) {
      const stats = row.spec!;
      stats.rounds!++; stats.acceptanceLengths!.push(0);
      if (forward) stats.targetCalls++;
      stats.tokensPerForward = row.generated / (stats.targetCalls - 1);
      stats.forwardsSaved = Math.max(0, row.generated - stats.targetCalls);
    }
    if (!forward) return this.#flushAhead();
    const previous = this.#ahead;
    this.#ahead = null;
    const B = rows.length, caches = this.#target!.caches;
    let context: MlxArray | null = null, tokens: MlxArray | null = null;
    try {
      const window = previous ? bindRowStepRollback(caches, B) : null;
      {
        using ids = previous ? ops.reshape(previous.tokens, [B, 1]) : ops.fromInt32(rows.map(row => row.current), [B, 1]);
        window?.begin(1);
        const taps = this.#draft!.tapLayers;
        const result = await this.binding.decode(ids, caches, taps.length ? taps : undefined, work);
        context = result.ctxML ?? result.hidden;
        try {
          using logits = this.binding.projectLogits(result.hidden, B);
          tokens = this.#sampleStep(rows, logits);
        } finally { if (result.ctxML) result.hidden.dispose(); }
      }
      ops.asyncEvalAll([tokens]);
      const steps = rows.map(row => row.sampled++);
      // The previous step's samples, delivered while the GPU runs this step: this step's inputs.
      let inputs = rows.map(row => row.current), outcomes: StepOutcome[] = rows.map(() => ({ kind: "continue" }));
      if (previous) {
        inputs = previous.tokens.toIntTokens();
        outcomes = [];
        for (const [index, row] of rows.entries()) outcomes.push(await this.#deliverSample(row, inputs[index]!));
        window!.resolve(outcomes.map(outcome => Number(outcome.kind === "continue")));
      }
      const keep = this.#retire(rows, outcomes);
      if (keep.length !== B) this.host.filterRows(keep);
      if (keep.length) {
        if (keep.length !== B) {
          using index = ops.fromInt32(keep, [keep.length]);
          const keptContext = ops.takeAxis(context, index, 0), keptTokens = ops.takeAxis(tokens, index, 0);
          context.dispose(); tokens.dispose();
          context = keptContext; tokens = keptTokens;
        }
        // A probe drafts from the kept inputs before their taps join the drafter's context, as a round would.
        if (shadow) this.#previewShadow(keep.map(index => rows[index]!), keep.map(index => inputs[index]!),
          shadow, keep.map(index => steps[index]!));
        // The drafter takes the kept inputs as a round's pending tokens, then their taps.
        await this.#draft!.draft(keep.map(index => inputs[index]!), 0, keep.map(index => steps[index]!));
        await this.#draft!.commit(keep.map(() => 0), context);
        for (const index of keep) {
          const row = rows[index]!;
          this.#requests.get(row)!.processed?.push(inputs[index]!);
          row.current = inputs[index]!;
        }
        this.#ahead = { tokens, rows: keep.map(index => rows[index]!) };
        tokens = null;
      }
      if (++this.#steps % 256 === 0) clearCache();
    } finally { disposeResources([previous?.tokens, context, tokens].filter(value => value != null)); }
  }

  /** Read and deliver the step in flight, leaving its rows' samples pending: the rounds that follow draft from them. */
  async #flushAhead(): Promise<void> {
    const ahead = this.#ahead;
    if (!ahead) return;
    this.#ahead = null;
    let read: number[];
    try { read = ahead.tokens.toIntTokens(); } finally { ahead.tokens.dispose(); }
    // Rows that joined after the step follow its rows; they have nothing in flight.
    const rows = [...this.host.rows], outcomes: StepOutcome[] = [];
    for (const [index, row] of rows.entries())
      outcomes.push(index < ahead.rows.length ? await this.#deliverSample(row, read[index]!) : { kind: "continue" });
    for (const [index, row] of ahead.rows.entries()) if (outcomes[index]!.kind === "continue") row.current = read[index]!;
    const keep = this.#retire(rows, outcomes);
    if (keep.length !== rows.length) this.host.filterRows(keep);
  }

  /** Deliver one verified sample of a step without drafts, as a round delivers its window: an EOS ends the row
   *  unpublished, a consumer may stop it, and its budget ends it. */
  /** Start a probe: each row's proposals from `pending`, left on the device and submitted behind the step in flight. */
  #previewShadow(rows: readonly Row[], pending: readonly number[], depth: number, steps: readonly number[]): void {
    const tokens = this.#draft!.preview!(pending, depth, steps);
    if (!tokens) return;
    ops.asyncEvalAll([tokens]);
    this.#shadow = { tokens, proposals: null, rows, depth: tokens.shape[1]!, seen: new Map(rows.map(row => [row, 0])) };
  }

  /** Compare one delivered sample with its row's probe; the gate observes the row at its first mismatch or after
   *  every proposal matched. */
  #compareShadow(row: Row, token: number): void {
    const shadow = this.#shadow, seen = shadow?.seen.get(row);
    if (!shadow || seen === undefined) return;
    if (!shadow.proposals) {
      const read = shadow.tokens!.toIntTokens();
      shadow.tokens!.dispose(); shadow.tokens = null;
      shadow.proposals = new Map(shadow.rows.map((probed, index) => [probed, read.slice(index * shadow.depth, (index + 1) * shadow.depth)]));
    }
    const matched = shadow.proposals.get(row)![seen] === token;
    if (matched && seen + 1 < shadow.depth) { shadow.seen.set(row, seen + 1); return; }
    this.#gate?.observe(shadow.depth, matched ? shadow.depth : seen);
    shadow.seen.delete(row);
    if (!shadow.seen.size) this.#settleShadow();
  }

  /** End the probe: rows still matching are observed over the samples they compared. */
  #settleShadow(): void {
    const shadow = this.#shadow;
    if (!shadow) return;
    this.#shadow = null;
    shadow.tokens?.dispose();
    for (const seen of shadow.seen.values()) if (seen > 0) this.#gate?.observe(seen, seen);
  }

  async #deliverSample(row: Row, token: number): Promise<StepOutcome> {
    if (row.req.signal?.aborted) return { kind: "cancelled", reason: row.req.signal.reason };
    this.#compareShadow(row, token);
    if (row.req.eosTokenIds.includes(token)) {
      if (row.generated < row.req.maxTokens) row.generated++;
      return { kind: "stop" };
    }
    try {
      row.generated++;
      if (await this.host.publish(row, token) === false) return { kind: "stop" };
    } catch (error) { return { kind: "failed", error }; }
    return row.generated >= row.req.maxTokens ? { kind: "length" } : { kind: "continue" };
  }

  /** Retire the rows a step ended (a stopped or finished row publishes its state first) and return the indices
   *  of the rest. The caller filters the group to them. */
  #retire(rows: readonly Row[], outcomes: readonly StepOutcome[]): number[] {
    const keep: number[] = [];
    for (const [index, row] of rows.entries()) {
      const outcome = outcomes[index]!;
      if (outcome.kind === "continue") { keep.push(index); continue; }
      if (outcome.kind === "failed") row.reject(outcome.error);
      else if (outcome.kind === "cancelled") row.reject(outcome.reason);
      else {
        if (!row.req.signal?.aborted) this.#publishCheckpoint(row, index, this.#requests.get(row)!);
        this.host.finish(row, outcome.kind);
      }
    }
    return keep;
  }

  #proposeEcho(row: Row, request: RequestState): void {
    if (request.proposal || !request.fill) return;
    const value = request.fill.propose(row.req.maxTokens - row.generated, "verify");
    if (value) request.proposal = { value, emitted: 0 };
  }

  #settleEcho(request: RequestState): void {
    if (!request.proposal) return;
    request.fill!.commit(request.proposal.value, request.proposal.emitted);
    request.proposal = undefined;
  }

  /** The method aligns token coverage and captures its state. The cache owns
   * RAM retention and eventual SSD persistence; no storage work is awaited. */
  #publishCheckpoint(row: Row, index: number, request: RequestState): void {
    if (!this.host.cache.keeps(row.req) || !request.processed) return;
    let target: Cache[] = [], attachments: CheckpointAttachment[] = [];
    try {
      target = this.#target!.extractRow(index);
      attachments.push(this.#draft!.capture(index).attachment);
      this.host.cache.publish(row.req, request.namespace,
        [...row.req.promptIds.slice(0, request.prefixLength), ...request.processed], { caches: target, attachments });
      target = []; attachments = [];
    } finally { disposeResources([...target, { dispose: () => disposeAttachments(attachments) }]); }
  }

  filterRows(keep: readonly number[], _discard: boolean): void {
    this.#target?.filterRows(keep); this.#draft?.filterRows(keep);
    const ahead = this.#ahead;
    if (ahead) {
      // The step in flight ran a prefix of the rows: rows that joined after it follow.
      const kept = keep.filter(index => index < ahead.rows.length);
      if (!kept.length) { ahead.tokens.dispose(); this.#ahead = null; }
      else if (kept.length !== ahead.rows.length) {
        using index = ops.fromInt32(kept, [kept.length]);
        const tokens = ops.takeAxis(ahead.tokens, index, 0);
        ahead.tokens.dispose();
        this.#ahead = { tokens, rows: kept.map(row => ahead.rows[row]!) };
      }
    }
    const retained = new Set(keep.map(index => this.host.rows[index]!));
    const releases: { dispose(): void }[] = [];
    for (const [row, request] of this.#requests) if (!retained.has(row)) {
      this.#settleEcho(request);
      this.#requests.delete(row);
      releases.push(request.sampling, { dispose: () => request.retain?.() });
    }
    disposeResources(releases);
  }

  dispose(): void {
    this.#ahead?.tokens.dispose();
    this.#ahead = null;
    this.#shadow?.tokens?.dispose();
    this.#shadow = null;
    for (const request of this.#requests.values()) this.#settleEcho(request);
    const resources = [...(this.#target ? [this.#target] : []), ...(this.#draft ? [this.#draft] : []),
      ...[...this.#requests.values()].flatMap(request => [request.sampling, { dispose: () => request.retain?.() }])];
    this.#target = null; this.#draft = null; this.#requests.clear();
    disposeResources(resources);
  }
}

/** Whether a step in flight ran exactly the group's current rows. */
function sameRows(ran: readonly Row[], rows: readonly Row[]): boolean {
  return ran.length === rows.length && ran.every((row, index) => row === rows[index]);
}
