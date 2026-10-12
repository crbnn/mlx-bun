// Concrete model graph for Qwen3.5 hybrid models (e.g. Qwen3.6-27B-OptiQ-4bit).
// Port target: mlx_lm.models.qwen3_5 (+ qwen3_next Attention/MLP/RMSNormGated,
// gated_delta recurrence). The architecture is a 64-layer stack where every
// `fullAttentionInterval`-th layer is standard softmax attention and the rest
// are gated-DeltaNet linear-attention layers. Weights carry a
// `language_model.` prefix. The blocks live in `blocks.ts`; this graph composes
// them and hands every block the phase its caller names.
//
// Parity bars: bf16 KV → bit-exact vs mlx-lm; mixed-precision KV → bit-exact vs
// mlx-optiq (the 16 full-attention layers quantized per kv_config.json, via the
// shared maybeQuantizeKv path).

import type { ModelConfig } from "../../artifacts/config";
import type { Weights } from "../../artifacts/weights";

import { mapTokenGroups } from "../../input/token-groups";
import type { MlxArray } from "@mlx-bun/mlx/array";

import * as ops from "@mlx-bun/mlx/ops";
import { qwenAppendChunkSize } from "../../layers/qwen-append";
import { disposing } from "../../layers/helpers";
import { composedFullAttentionCache } from "../../state/bf16-first-kv";
import type { Composition } from "../../contracts/portable/composition";
import { AttentionMasks } from "../../state/attention-read";
import { denseTurboQuantKernels } from "../../state/turboquant-attention";
import { LoraState } from "../../layers/lora";
import { QuantizedEmbedding } from "../../layers/quantized-embedding";
import { QuantizedLinear } from "../../layers/quantized-linear";
import { RMSNorm } from "../../layers/normalization";
import { type Cache } from "../../contracts/mlx/cache";
import { qwen35WeightsView } from "./checkpoint";
import type { GraphCapabilities } from "../../contracts/portable/graph";
import type {
  MlxCapturingForward, MlxDeclaredGraph, MlxHeadForward, MlxKnownSpans, MlxLayerCapture, MlxMixedForward, MlxPhaseForward, MlxPrefillChunk,
  MlxTokenPhases, MlxVerifyForward, MlxWidthClass,
  MlxWidthTable,
} from "../../contracts/mlx/graph";
import type { TargetView } from "../../contracts/mlx/draft-target";
import type { MediaEncoders, MlxPromptInput, Vision } from "../../contracts/mlx/media";
import { declareGraph } from "../capabilities";
import { atEveryWidth, byWidth, type ReadPhase } from "../graph";
import { bindQwenMediaInput, qwenDraftTarget, qwenMediaEncoders } from "./media-input";
import { SSMCache } from "../../state/ssm";
import { buildMropePositions, mropeInvFreq } from "../../layers/qwen-mrope";
import { type MropeRequestState } from "../../contracts/mlx/positions";
import { Qwen3Layer } from "./blocks";

const PREFIX = "language_model";

/** A prompt window above four rows evaluates each layer and its recurrent state before the next (the bounded
 *  workspace a retained prefix leaves room for); four rows or fewer stay one lazy graph. */
const boundedWindow = (width: MlxWidthClass) => width === "fiveToEight" || width === "wider";

export class Qwen35Model implements MlxDeclaredGraph, MlxTokenPhases {
  readonly config: ModelConfig;
  readonly weightsBytes: number;
  /** Base path for LoRA target keys (weights carry the language_model prefix). */
  readonly prefixBase = "language_model.model";
  readonly loraState = new LoraState();
  /** Layers whose attention reads plain keys and values: none; it attends the storage its caches hold. */
  readonly requiredDenseKvLayers: readonly number[] = Object.freeze([]);
  /** Prepared image and video embeddings with request-owned mRoPE positions,
   * layer taps for drafts. Its caches convert delayed affine rows and every
   * method is qualified over them. Speculation over affine KV is the
   * composition's `speculativeAffineKv`; immediate 4-bit KV is qualified even
   * without a grouped method. Its few full-attention layers materialize their
   * scores at long context, so prefill chunks shrink to bound that workspace. */
  get graphCapabilities(): GraphCapabilities {
    return declareGraph({
      media: { input: "embeddings+positions", video: true }, hiddenLayerTaps: true,
      kv: { delayedAffine: "all" }, prefill: { boundedWorkspace: true },
      speculation: { affineKv: this.composition.speculativeAffineKv, immediateAffine4: true },
    });
  }
  bindMediaInput(input: Vision): MlxPromptInput { return bindQwenMediaInput(this, input.embeddings, input.mrope!); }
  mediaEncoders(modelDir: string): Promise<MediaEncoders> { return qwenMediaEncoders(this, modelDir); }
  draftTarget(_caches: Cache[]): TargetView { return qwenDraftTarget(this); }
  readonly embed: QuantizedEmbedding;
  readonly layers: Qwen3Layer[];
  readonly finalNorm: RMSNorm;
  /** null when tied: the output head reuses embed_tokens (embed.asLinear). */
  readonly lmHead: QuantizedLinear | null;
  readonly tied: boolean;
  readonly faIdx: number;
  /** The served configuration this graph was built for: its KV scheme (what
   *  `makeCache` composes) and its switches. */
  readonly composition: Composition;
  /** The composition's `packedMixedMlp`: the groups of a mixed forward share
   *  one MLP call. */
  readonly #packMixedMlp: boolean;

  constructor(weights: Weights, config: ModelConfig, composition: Composition) {
    // Checkpoint-generation normalization BEFORE the first tensor() call: the
    // 5.8-family export names the trunk `model.language_model.*` with a
    // top-level `lm_head`, ships `mtp.*` in-repo, and stores RMSNorm gains as
    // γ−1 with HF-layout conv1d. The view maps all of that onto the names and
    // values this graph reads (mlx-lm's post-sanitize space). No-op — nothing
    // installed at all — for artifacts already in that space.
    const view = qwen35WeightsView(weights);
    if (view) weights.setView(view);
    this.config = config;
    this.composition = composition;
    this.tied = config.text.tieWordEmbeddings;
    this.weightsBytes = [...weights.shards.files.values()]
      .reduce((a, f) => a + f.mmap.size, 0);
    this.embed = QuantizedEmbedding.load(weights, `${PREFIX}.model.embed_tokens`, config);
    this.layers = Array.from(
      { length: config.text.numHiddenLayers },
      (_, i) => new Qwen3Layer(weights, config, i),
    );
    this.finalNorm = new RMSNorm(weights.tensor(`${PREFIX}.model.norm.weight`), config.text.rmsNormEps);
    // Tied models (e.g. Qwen3.5-4B) ship no lm_head; the reference uses
    // embed_tokens.as_linear (mlx-lm qwen3_5 TextModel.__call__).
    this.lmHead = this.tied ? null : QuantizedLinear.load(weights, `${PREFIX}.lm_head`, config);
    this.faIdx = config.text.fullAttentionInterval - 1;
    this.#packMixedMlp = composition.packedMixedMlp;
    this.knownSpans = this.#qualifiesKnownSpans() ? Object.freeze({
      affineKvBits: Object.freeze([4, 8]), turboQuantFormats: Object.freeze([Object.freeze({ kBits: 8, vBits: 3 })]),
      maxChunkSize: (state: readonly Cache[]) => qwenAppendChunkSize(state[0]!.offset),
    }) : undefined;
    const known: ReadPhase = this.knownSpans ? "committed" : "window";
    this.appendKnown = byWidth(width => (ids, state) =>
      this.forwardLayers(this.embed.encode(ids), state, known, undefined, null, width, false));
  }

  loraTargets(): Map<string, QuantizedLinear> {
    const out = new Map<string, QuantizedLinear>();
    for (let i = 0; i < this.layers.length; i++) {
      const l = this.layers[i]!;
      const p = `${PREFIX}.model.layers.${i}`;
      if (l.selfAttn) {
        out.set(`${p}.self_attn.q_proj`, l.selfAttn.qProj);
        out.set(`${p}.self_attn.k_proj`, l.selfAttn.kProj);
        out.set(`${p}.self_attn.v_proj`, l.selfAttn.vProj);
        out.set(`${p}.self_attn.o_proj`, l.selfAttn.oProj);
      } else if (l.linearAttn) {
        out.set(`${p}.linear_attn.in_proj_qkv`, l.linearAttn.inProjQkv);
        out.set(`${p}.linear_attn.in_proj_z`, l.linearAttn.inProjZ);
        out.set(`${p}.linear_attn.in_proj_b`, l.linearAttn.inProjB);
        out.set(`${p}.linear_attn.in_proj_a`, l.linearAttn.inProjA);
        out.set(`${p}.linear_attn.out_proj`, l.linearAttn.outProj);
      }
      // Trellis-coded MLP tensors are not LoRA targets (no adapter seam yet).
      if (l.mlp.kind === "affine") {
        out.set(`${p}.mlp.gate_proj`, l.mlp.gate);
        out.set(`${p}.mlp.up_proj`, l.mlp.up);
        out.set(`${p}.mlp.down_proj`, l.mlp.down);
      }
    }
    return out;
  }

  /** One cache per layer, the full-attention caches sharing one mask memo:
   * the cache the composition's KV scheme names for each attention layer
   * (`composedFullAttentionCache`). Their TurboQuant read is dense
   * (`denseTurboQuantKernels`): this graph read TurboQuant storage decoded,
   * through `updateAndFetch` and the fused SDPA, before its blocks read through
   * the cache, so the TurboQuant lego keeps its served numbers. */
  makeCache(): Cache[] {
    const masks = new AttentionMasks(), kv = this.composition.kv;
    return this.layers.map((l, layer) => (l.isLinear ? new SSMCache() : composedFullAttentionCache(kv, layer, masks, denseTurboQuantKernels)));
  }

  /** The 27B composition on the M4 Pro qualifies committed known-token spans:
   * its append keeps the native affine arithmetic while Trellis shares up to
   * four rows. Read from the composition and the quant table at construction;
   * every other composition takes known tokens as a window, one position per
   * fill step. */
  #qualifiesKnownSpans(): boolean {
    const t = this.config.text;
    if (t.hiddenSize !== 5120 || t.intermediateSize !== 17408 || this.layers.length !== 64 ||
        t.headDim !== 256 || t.numAttentionHeads !== 24 || t.numKeyValueHeads !== 4 ||
        this.composition.device !== "applegpu_g16s")
      return false;
    const qualified = (linear: QuantizedLinear) => {
      const { mode, bits, groupSize } = linear.spec;
      return mode === "affine" && [2, 3, 4, 6, 8].includes(bits) && [32, 64, 128].includes(groupSize);
    };
    return this.layers.every(layer => {
      const a = layer.linearAttn, s = layer.selfAttn;
      const attention = a ? [a.inProjQkv, a.inProjZ, a.inProjB, a.inProjA, a.outProj]
        : [s!.qProj, s!.kProj, s!.vProj, s!.oProj];
      const mlp = layer.mlp.kind === "affine" ? [layer.mlp.gate, layer.mlp.up, layer.mlp.down] : [];
      return attention.every(qualified) && mlp.every(qualified);
    });
  }

  /** The scoring and training forward (`MlxSequenceForward`): a window on the
   * MLP kernels of `width`, above four rows each layer evaluated before the next. */
  sequenceHidden(ids: MlxArray, cache: Cache[], width: MlxWidthClass): MlxArray {
    return this.forwardLayers(this.embed.encode(ids), cache, "window", undefined, null, width, boundedWindow(width));
  }

  /** One whole chunk of a prompt; rows are sequences times the chunk size, and
   * each layer's output and recurrent state is evaluated before the next. The
   * GPU is this graph's one unit. */
  readonly prefillChunk: MlxPrefillChunk = {
    gpu: (ids, state, capture) => this.#captured(this.forwardLayers(this.embed.encode(ids), state, "window", undefined,
      capture, "wider", true), capture),
  };

  /** A prompt's positions that are not a whole chunk, on the MLP kernels of
   * their width; above four rows each layer is evaluated before the next. */
  readonly prefillTail: MlxWidthTable<MlxCapturingForward> = byWidth(width => (ids, state, capture) =>
    this.#captured(this.forwardLayers(this.embed.encode(ids), state, "window", undefined, capture, width,
      boundedWindow(width)), capture));

  /** One new token per sequence: `ids` [sequences, 1], so rows are sequences. */
  readonly decode: MlxWidthTable<MlxCapturingForward> = byWidth(width => (ids, state, capture) =>
    this.#captured(this.forwardLayers(this.embed.encode(ids), state, "decode", undefined, capture, width, false), capture));

  /** A speculative window: each sequence's last accepted token and its drafted
   * tokens, right-padded by the caller to the round's depth, `ids`
   * [sequences, depth + 1], so rows are sequences times (depth + 1). The graph
   * appends all depth + 1 positions; dropping the rejected suffix is the
   * caller's speculative transaction, begun before this call. */
  readonly verify: MlxWidthTable<MlxVerifyForward> = byWidth(width => (ids, state, capture) =>
    this.#captured(this.forwardLayers(this.embed.encode(ids), state, "window", undefined, capture, width, false), capture));

  /** Committed known-token spans, when the composition qualifies them (`#qualifiesKnownSpans`). */
  readonly knownSpans: MlxKnownSpans | undefined;
  /** Known tokens: the committed read under `knownSpans`, the window read otherwise. */
  readonly appendKnown: MlxWidthTable<MlxPhaseForward>;

  /** The one output head at every width. */
  readonly logits = atEveryWidth<MlxHeadForward>(hidden => this.logitsFromHidden(hidden));

  /** Hands `capture` the post-final-norm output after its layers, and returns it. */
  #captured(hidden: MlxArray, capture: MlxLayerCapture | null): MlxArray {
    capture?.(this.layers.length, hidden);
    return hidden;
  }

  /** The vision prefill phase: a prompt's spliced embeddings (borrowed) at request positions. */
  prefillEmbeddingsAtPositions(embeds: MlxArray, cache: Cache[], positions: MlxArray, width: MlxWidthClass): MlxArray {
    return this.forwardLayers(ops.contiguous(embeds), cache, "window", positions, null, width, boundedWindow(width));
  }

  /** A prompt's token ids after a reused prefix, at request positions. */
  prefillAtPositions(ids: MlxArray, cache: Cache[], positions: MlxArray, width: MlxWidthClass): MlxArray {
    return this.forwardLayers(this.embed.encode(ids), cache, "window", positions, null, width, boundedWindow(width));
  }

  /** One new token per row at request positions, including different positions for each row. */
  decodeAtPositions(ids: MlxArray, cache: Cache[], positions: MlxArray, width: MlxWidthClass): MlxArray {
    return this.forwardLayers(this.embed.encode(ids), cache, "decode", positions, null, width, false);
  }

  /** Mixed rows in one forward: each group's attention and DeltaNet read its
   * own caches in its own phase (a verify group reads a window). The MLP rows
   * of the groups that may share it run as one call at `width`, the class of
   * the groups' rows, when the composition packs mixed MLP; a verify group
   * keeps its own geometry (`mapTokenGroups`). A prompt group above four rows
   * evaluates each layer and the groups' recurrent state before the next, as
   * the prompt tail does. */
  readonly mixed: MlxMixedForward = (groups, width) => {
    const hidden: MlxArray[] = [];
    const results: MlxArray[] = [];
    const bounded = groups.some(group => group.phase === "window" && boundedWindow(group.width));
    try {
      for (const group of groups) hidden.push(this.embed.encode(group.ids));
      for (const [i, layer] of this.layers.entries()) {
        const mids: MlxArray[] = [];
        try {
          for (const [row, group] of groups.entries())
            mids.push(layer.forwardAttn(hidden[row]!, group.state[i]!, group.phase === "decode" ? "decode" : "window"));
          const outputs = mapTokenGroups(groups, mids, (h, w) => layer.forwardMlp(h, "window", w), width,
            this.#packMixedMlp);
          for (const row of groups.keys()) { hidden[row]!.dispose(); hidden[row] = outputs[row]!; }
          // Materialize every layer's recurrent tail as well as the residual.
          // It is not a dependency of h and otherwise pins prefill buffers.
          if (bounded) {
            const caches = groups.map(group => group.state[i]!);
            const state = caches.map(cache => cache.state());
            try { ops.evalAll([...hidden, ...state.flat()]); }
            finally { for (const [row, cache] of caches.entries())
              if (cache.stateNeedsDispose) for (const value of state[row]!) value.dispose(); }
          }
          for (const [row, group] of groups.entries()) group.capture?.(i, hidden[row]!);
        } finally { for (const mid of mids) mid.dispose(); }
      }
      for (const [row, group] of groups.entries()) {
        const out = this.finalNorm.forward(hidden[row]!); results.push(out);
        group.capture?.(this.layers.length, out);
      }
      return results;
    } catch (error) { for (const result of results) result.dispose(); throw error; }
    finally { for (const h of hidden) h.dispose(); }
  };

  /** Active vision mRoPE request state (serial lane; set by the generation
   *  gateway around a vision request's run, null for text-only — which keeps
   *  the bit-exact fast-rope path). The explicit serial executor scopes this value.
   *  Shared media forwards pass request-owned positions directly. */
  mrope: MropeRequestState | null = null;
  #mropeInvFreq: MlxArray | null = null;

  /** The vision prefill phase without request positions: spliced input embeddings [1, L, H] (image features
   *  overwriting image-token rows — the caller builds them). Caller keeps
   *  ownership of `embeds`. bidir/ids/multimodal are the Gemma-shaped extras
   *  and are unused: Qwen3.5 vision attends fully causally. */
  prefillEmbeddings(
    embeds: MlxArray, cache: Cache[], _bidir: MlxArray | null,
    _ids: MlxArray | null, _multimodal: MlxArray | null, width: MlxWidthClass,
  ): MlxArray {
    const h = ops.contiguous(embeds); // fresh handle — the layer loop consumes it
    return this.forwardLayers(h, cache, "window", undefined, null, width, boundedWindow(width));
  }

  /** The decoder over `h0` (consumed), every block reading its cache for
   *  `phase` and every MLP on the kernels of `width`, the class of `h0`'s rows
   *  the calling operation names. `bounded` evaluates each layer's output and
   *  recurrent state before the next. `capture` receives each layer's residual
   *  output after the layer runs; the caller hands it the final hidden. */
  protected forwardLayers(h0: MlxArray, cache: Cache[], phase: ReadPhase,
    positions: MlxArray | undefined, capture: MlxLayerCapture | null, width: MlxWidthClass, bounded: boolean): MlxArray {
    const L = h0.shape[1]!;
    // Vision requests: one interleaved-mRoPE cos/sin table per forward,
    // shared by all 12 full-attention layers (positions are layer-invariant).
    let mropeFwd: ReturnType<typeof buildMropePositions> | null = null;
    if (positions || this.mrope) {
      const t = this.config.text;
      const ropeDims = Math.trunc(t.headDim * t.partialRotaryFactor);
      const base = t.ropeParameters.full_attention?.ropeTheta ?? 10000;
      this.#mropeInvFreq ??= mropeInvFreq(ropeDims, base);
      mropeFwd = positions ? { posIds: positions, invFreq: this.#mropeInvFreq, rotaryDims: ropeDims }
        : buildMropePositions(this.mrope!, cache[this.faIdx]!.offset, L, this.#mropeInvFreq, ropeDims);
    }
    let h: MlxArray | null = h0;
    const prof = (globalThis as Record<string, unknown>).__deltaProf as
      Record<string, number>
      | undefined;
    try {
      for (let i = 0; i < this.layers.length; i++) {
        const tl = prof ? performance.now() : 0;
        const next = this.layers[i]!.forward(h, cache[i]!, phase, mropeFwd, width);
        h.dispose();
        h = next;
        if (bounded) {
          // Affine prefills also need bounded evaluation: a retained prefix
          // history can leave too little room for the entire deferred graph.
          // Materialize the cache outputs too. In particular, the tiny
          // copied conv tail is not a dependency of `h`; leaving it lazy
          // pins the whole prefill conv buffer until the chunk ends.
          const state = cache[i]!.state();
          try { ops.evalAll([h, ...state]); }
          finally { if (cache[i]!.stateNeedsDispose) for (const a of state) a.dispose(); }
        }
        if (prof) {
          ops.evalAll([h]);
          const key = this.layers[i]!.isLinear ? "layerLin" : "layerFull";
          prof[key] = (prof[key] ?? 0) + performance.now() - tl;
        }
        capture?.(i, h);
      }
      const out = disposing(h, this.finalNorm.forward(h));
      h = null; // consumed — the finally must not double-free
      return out;
    } finally {
      if (mropeFwd && !positions) mropeFwd.posIds.dispose();
      // A mid-loop layer throw must not strand the in-flight [1,L,H]
      // residual (2026-08-18 review).
      h?.dispose();
    }
  }

  logitsFromHidden(h: MlxArray): MlxArray {
    return this.tied ? this.embed.asLinear(h) : this.lmHead!.forward(h);
  }
}
