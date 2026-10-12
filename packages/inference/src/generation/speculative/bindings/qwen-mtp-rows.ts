import type { MlxArray } from "@mlx-bun/mlx/array";
import { materializeCopy } from "@mlx-bun/mlx/materialize";
import * as ops from "@mlx-bun/mlx/ops";
import type { PreparedStateChange } from "../../../contracts/portable/resources";
import { normalizedArgmax } from "../../../kernels/sampling/normalized-argmax";
import type { RecurrentMtpModule } from "../../../contracts/mlx/drafter";
import { widthClass, type MlxWidthTable } from "../../../contracts/mlx/graph";
import { byWidth } from "../../../models/graph";
import { applyStateChanges,cleanupFailure,disposeResources } from "../../../runtime/resources";
import { toLogprobs } from "../../../sampling/index";
import { BatchedKVCache } from "../../../state/batched-kv";
import { KVCache } from "../../../state/kv";
import { DraftVocabularyHead } from "../draft-vocab";
import type { DraftRowSampling,RecurrentMtpTarget } from "../source";

/** Sampling owns each request's RNG policy. Scores are [B,V] log-probabilities;
 * the returned owned [B] token array stays on device through the draft chain. */
export type MtpRowSampling = DraftRowSampling;

/** One draft step's token selection: the module output [B, 1, H] (borrowed) and
 *  each row's step in, owned full-vocabulary ids [B] out. */
type MtpSelect = (output: MlxArray, steps: readonly number[]) => MlxArray;

/** The selection the draft group bound, per width class of the rows it drafts
 *  for: the listed vocabulary's projection and each row's coupled draw over it,
 *  the full head and its fused normalized argmax, or the full head and each
 *  row's sampler over its log-probabilities. Resolved once when the rows open;
 *  `owned` receives the listed projection, which the rows dispose. */
function bindMtpSelection(target: RecurrentMtpTarget, sampling: MtpRowSampling,
  owned: { vocabulary: DraftVocabularyHead | null }): MlxWidthTable<MtpSelect> {
  const selection = sampling.selection;
  if (selection.kind === "listed") {
    // Frequency-ranked draft vocabulary: project onto the listed rows only.
    // The target still verifies over the full vocabulary.
    if (!target.vocabularyHead) throw new Error("a listed draft vocabulary needs the target's quantized vocabulary head");
    const head = owned.vocabulary = new DraftVocabularyHead(target.vocabularyHead, selection.ids);
    return byWidth(() => (output, steps) => {
      using subsetLogits = head.project(output);
      using subsetFlat = ops.reshape(subsetLogits, [steps.length, head.size]);
      return selection.sample(subsetFlat, head, steps);
    });
  }
  // Greedy drafting: one fused normalized-argmax over [B,V]; no logsumexp
  // pass, no per-row top-p/top-k sort, no categorical draw.
  if (selection.kind === "greedy") return byWidth(width => (output, steps) => {
    using logits = target.logits[width](output);
    using flat = ops.reshape(logits, [steps.length, logits.shape.at(-1)!]);
    return normalizedArgmax(flat);
  });
  return byWidth(width => (output, steps) => {
    using logits = target.logits[width](output);
    using flat = ops.reshape(logits, [steps.length, logits.shape.at(-1)!]);
    using logprobs = toLogprobs(flat);
    return sampling.sample(logprobs, steps);
  });
}

/** Prefill membership drafts nothing, so its rows bind no selection. */
const prefillOnly: MlxWidthTable<MtpSelect> = byWidth(() => () => {
  throw new Error("Qwen MTP prefill rows do not draft");
});

/** The same draft graph and state lifecycle for one or several requests.
 * Admission, target verification, output publication and persistence belong to
 * the caller. Rows remain fixed within a draft/commit round. */
export class QwenMtpRows {
  #cache = new BatchedKVCache();
  #hidden: MlxArray | null = null;
  #prefilled: boolean[] = [];
  #drafts: number[][] = [];
  #depth = 0;
  /** The listed vocabulary's row subset of the target's projection, when the
   *  bound selection drafts over it. */
  readonly #owned: { vocabulary: DraftVocabularyHead | null } = { vocabulary: null };
  /** The draft token selection, bound once from the group's sampling. */
  readonly #select: MlxWidthTable<MtpSelect>;

  constructor(
    readonly target: RecurrentMtpTarget,
    readonly module: RecurrentMtpModule,
    sampling: MtpRowSampling | null,
    states: readonly (MtpRowState | null)[],
  ) {
    this.#select = sampling ? bindMtpSelection(target, sampling, this.#owned) : prefillOnly;
    try { this.append(states); }
    catch (error) { cleanupFailure(error, () => this.dispose()); }
  }

  get rowCount(): number { return this.#cache.rowOffsets.length; }

  /** Add committed state at a method boundary. Inputs remain caller-owned. */
  append(states: readonly (MtpRowState | null)[]): void {
    if (!states.length) return;
    applyStateChanges([() => this.prepareAppend(states)]);
  }

  prepareAppend(states: readonly (MtpRowState | null)[]): PreparedStateChange {
    if (!states.length) return { commit() {}, dispose() {} };
    const prefilled = [...this.#prefilled, ...states.map(state => state !== null)];
    const example = this.#hidden ?? states.find(state => state !== null)?.hidden;
    let hidden: MlxArray | null = null;
    let cache: BatchedKVCache | null = new BatchedKVCache();
    const held: { dispose(): void }[] = [];
    try {
      if (example) {
        const zero = (rows: number) => {
          const array = ops.zeros([rows, 1, this.target.hiddenSize], example.dtype);
          held.push(array); return array;
        };
        hidden = ops.concatAxis([
          ...(this.rowCount ? [this.#hidden ?? zero(this.rowCount)] : []),
          ...states.map(state => state?.hidden ?? zero(1)),
        ], 0);
      }
      cache.mergeRows([...(this.rowCount ? [this.#cache] : []), ...states.map(state => {
        if (state) return state.cache;
        const empty = new KVCache(); held.push(empty); return empty;
      })]);
    } catch (error) { return cleanupFailure(error, () => disposeResources([cache!, ...(hidden ? [hidden] : [])])); }
    finally { disposeResources(held); }
    return {
      commit: () => {
        const previousCache = this.#cache, previousHidden = this.#hidden;
        this.#cache = cache!; this.#hidden = hidden;
        this.#prefilled = prefilled;
        cache = previousCache; hidden = previousHidden;
      },
      dispose: () => {
        const held = [cache, hidden].filter(value => value !== null);
        cache = null; hidden = null;
        disposeResources(held);
      },
    };
  }

  /** Retire/reorder committed requests without involving their sampler policy. */
  filterRows(keep: readonly number[]): void {
    // A drained group keeps its bound selection for the requests that rejoin it.
    if (!keep.length) { this.#releaseRows(); return; }
    using indices = ops.fromInt32([...keep], [keep.length]);
    const hidden = this.#hidden ? ops.takeAxis(this.#hidden, indices, 0) : null;
    try { this.#cache.filterRows(keep); }
    catch (error) { hidden?.dispose(); throw error; }
    this.#hidden?.dispose(); this.#hidden = hidden;
    this.#prefilled = keep.map(row => this.#prefilled[row]!);
  }

  /** Build companion KV from borrowed new target tokens/taps at the same B.
   * The first token bridges to each restored row's preceding true hidden.
   * Cold rows discard that bridge. All later rows pair token[p+1] with h[p].
   * Only KV and the final true hidden are evaluated; unused attention/MLP
   * outputs remain outside the evaluated prefill graph. */
  prefill(tokens: MlxArray, context: MlxArray): void {
    const [B, length] = tokens.shape as [number, number];
    const H = context.shape[2]!;
    if (this.#prefilled.some(Boolean)) {
      using first = tokens.slice([0, 0], [B, 1]);
      using embeds = this.target.embed(first);
      const mixed = this.#prefilled.some(value => !value);
      if (mixed) this.#cache.specRoundBegin();
      this.module.decode[widthClass(B)](embeds, this.#hidden!, this.#cache).dispose();
      if (mixed) this.#cache.specRoundRollback(this.#prefilled.map(Number));
    }
    for (let pos = 0; pos + 1 < length; pos += 2048) {
      const end = Math.min(pos + 2048, length - 1);
      using ids = tokens.slice([0, pos + 1], [B, end + 1]);
      using embeds = this.target.embed(ids);
      using hidden = context.slice([0, pos, 0], [B, end, H]);
      // A single position per row appends through the one-position read.
      (end - pos === 1 ? this.module.decode[widthClass(B)] : this.module.window)(embeds, hidden, this.#cache).dispose();
      ops.evalAll(this.#cache.state());
    }
    using tail = context.slice([0, length - 1, 0], [B, length, H]);
    const hidden = materializeCopy(tail);
    this.#hidden?.dispose(); this.#hidden = hidden;
    ops.evalAll([hidden, ...this.#cache.state()]);
    this.#prefilled.fill(true);
  }

  materialize(): void {
    const state = [...this.#cache.state(), ...(this.#hidden ? [this.#hidden] : [])];
    if (state.length) ops.evalAll(state);
  }

  /** Align the companion to verified external continuations through the same
   * true-hidden prefill graph. Different rows retain different prefix lengths. */
  consume(tokens: MlxArray, context: MlxArray, lengths: readonly number[]): void {
    this.#cache.specRoundBegin();
    this.prefill(tokens, context);
    this.#cache.specRoundRollback([...lengths]);
    using positions = ops.fromInt32(lengths.map(length => length - 1), [lengths.length, 1, 1]);
    const hidden = ops.takeAlongAxis(context, positions, 1);
    this.#hidden!.dispose(); this.#hidden = hidden;
  }

  /** Nonnegative depth, with every request's first pending token already known.
   * Build the complete device dependency chain before reading proposal IDs. */
  draft(pending: readonly number[], depth: number, steps: readonly number[]): number[][] {
    if (depth === 0) return this.#chain(pending, depth, steps) as number[][];
    using packed = this.#chain(pending, depth, steps) as MlxArray;
    return this.#resolve(packed.toIntTokens(), pending.length, depth, true);
  }

  /** The same chain with the proposal IDs left on the device ([B, depth]), so the
   *  caller can build its verify graph before anything is read back. `resolve`
   *  takes the row-major host copy of `tokens`. Null at depth 0. */
  draftDevice(pending: readonly number[], depth: number, steps: readonly number[]):
    { tokens: MlxArray; resolve(read: readonly number[]): number[][] } | null {
    if (depth === 0) return null;
    const B = pending.length;
    using packed = this.#chain(pending, depth, steps) as MlxArray; // [depth * B], position-major
    using grid = ops.reshape(packed, [depth, B]);
    using rows = ops.transposeAxes(grid, [1, 0]);
    return { tokens: ops.contiguous(rows), resolve: read => this.#resolve(read, B, depth, false) };
  }

  #resolve(read: readonly number[], B: number, depth: number, positionMajor: boolean): number[][] {
    this.#drafts = Array.from({ length: B }, (_, row) =>
      Array.from({ length: depth }, (_, position) => read[positionMajor ? position * B + row : row * depth + position]!));
    return this.#drafts;
  }

  /** Build the draft chain. Returns the packed device tokens [depth * B]
   *  (position-major), or the empty proposals at depth 0. */
  #chain(pending: readonly number[], depth: number, steps: readonly number[]): MlxArray | number[][] {
    const B = pending.length;
    const tokens: MlxArray[] = [];
    let ids: MlxArray | null = ops.fromInt32([...pending], [B, 1]);
    let chained: MlxArray | null = null;
    this.#cache.specRoundBegin();
    try {
      // A zero-proposal round still processes the pending target token. Keep
      // its companion KV aligned so a later round can resume speculation.
      if (depth === 0) {
        using embeds = this.target.embed(ids!);
        this.module.decode[widthClass(B)](embeds, this.#hidden!, this.#cache).dispose();
        this.#drafts = Array.from({ length: B }, () => []);
        this.#depth = 0;
        this.#hidden!.dispose(); this.#hidden = null;
        return this.#drafts;
      }
      const width = widthClass(B), select = this.#select[width];
      for (let position = 0; position < depth; position++) {
        using embeds = this.target.embed(ids!);
        const output = this.module.decode[width](embeds, chained ?? this.#hidden!, this.#cache);
        chained?.dispose(); chained = output;
        const token = select(output, steps.map(step => step + position));
        tokens.push(token);
        ids!.dispose(); ids = ops.reshape(token, [B, 1]);
      }
      const packed = ops.concatAxis(tokens, 0);
      this.#depth = depth;
      this.#hidden!.dispose(); this.#hidden = null;
      return packed;
    } finally {
      ids?.dispose(); chained?.dispose();
      for (const token of tokens) token.dispose();
    }
  }

  /** Resolve draft storage using each request's accepted prefix. The verify
   * context is borrowed [B,depth+1,H], before the target's final norm. */
  commit(accepted: readonly number[], context: MlxArray): void {
    const B = accepted.length, depth = this.#depth;
    const appended = Math.max(1, depth);
    const kept = accepted.map(count => Math.min(count + 1, appended));
    if (kept.every(count => count === appended)) this.#cache.specRoundCommit();
    else this.#cache.specRoundRollback(kept);

    // Full acceptance needs the last accepted token's missing draft row,
    // using the TRUE target hidden at its preceding position. Other requests
    // keep their current coverage; their temporary append is never retained.
    const extra = accepted.map(count => Number(depth > 0 && count === depth));
    if (extra.some(Boolean)) {
      using ids = ops.fromInt32(this.#drafts.map(row => row[depth - 1]!), [B, 1]);
      using embeds = this.target.embed(ids);
      using hidden = context.slice([0, depth - 1, 0], [B, depth, context.shape[2]!]);
      this.#cache.specRoundBegin();
      this.module.decode[widthClass(B)](embeds, hidden, this.#cache).dispose();
      this.#cache.specRoundRollback(extra);
    }
    using positions = ops.fromInt32([...accepted], [B, 1, 1]);
    this.#hidden = ops.takeAlongAxis(context, positions, 1);
    this.#drafts = []; this.#depth = 0;
  }

  /** Only between committed rounds. Returned state is independent of the
   * group's storage and uses the existing single-request cache representation. */
  extractRow(row: number): MtpRowState {
    const cache = this.#cache.extractRow(row);
    try {
      using view = this.#hidden!.slice([row, 0, 0], [row + 1, 1, this.#hidden!.shape[2]!]);
      return { cache, hidden: materializeCopy(view) };
    } catch (error) { cache.dispose(); throw error; }
  }

  /** Release every row's state; the bound selection stays. */
  #releaseRows(): void {
    this.#cache.dispose(); this.#hidden?.dispose(); this.#hidden = null;
    this.#drafts = []; this.#depth = 0; this.#prefilled = [];
  }

  dispose(): void {
    this.#owned.vocabulary?.dispose(); this.#owned.vocabulary = null;
    this.#releaseRows();
  }
}

import { MtpRowState } from "../../../state/speculative/qwen-mtp-state";
export { type MtpRowState } from "../../../state/speculative/qwen-mtp-state";
