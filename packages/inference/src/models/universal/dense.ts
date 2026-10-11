// UniversalDense — the Tier-0 config-driven dense llama-family module
// (`02d723a:docs/design/generic-model-support.md` §4.1). One module, shaped exactly
// like qwen3.ts, with every arch delta selected by the UniversalArgs
// descriptor (archs.ts). Monolith path only: no compiled
// decode. Each attention block reads its cache for the phase its caller names
// (`ReadPhase`) with the read the descriptor composes at construction: the
// cache's own read, or the score-softcapped read over plain KV
// (`state/softcap-attention.ts`). The bar is L1 bit-exactness
// vs mlx-lm on this machine's GPU (`02d723a:tests/parity/universal-parity.test.ts`).
//
// Porting discipline: each branch transcribes its mlx-lm source op-for-op
// (q/k-norm before vs after reshape, rope order, scale points, softcap,
// mask kinds). The descriptor selects branches; it never approximates.

import type { ModelConfig } from "../../artifacts/config";
import type { Weights } from "../../artifacts/weights";
import { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import { argmaxLastPosition, logitSoftcap } from "../../kernels/logits";
import { disposing } from "../../layers/helpers";
import { composedFullAttentionCache, composedSlidingAttentionCache } from "../../state/bf16-first-kv";
import type { Composition } from "../../contracts/portable/composition";
import { AttentionMasks } from "../../state/attention-read";
import { softcapAttentionReads } from "../../state/softcap-attention";
import { denseTurboQuantKernels } from "../../state/turboquant-attention";
import { LoraState } from "../../layers/lora";
import type { GraphCapabilities } from "../../contracts/portable/graph";
import type {
  MlxDeclaredGraph, MlxHeadForward, MlxPhaseForward, MlxPrefillChunk, MlxTokenPhases, MlxVerifyForward,
} from "../../contracts/mlx/graph";
import { PLAIN_KV_VERIFICATION, declareGraph } from "../capabilities";
import { atEveryWidth, deprecatedPhase, type ReadPhase } from "../graph";
import type { QuantizedLinear } from "../../layers/quantized-linear";
import { TrellisLinear } from "../../layers/trellis-linear";
import { RMSNorm } from "../../layers/normalization";
import { type Cache } from "../../contracts/mlx/cache";
import { genericArgsFor, universalCacheWindows, universalMaskRecipe, type UniversalArgs } from "./archs";
import { compiledSwiglu } from "../../layers/swiglu";
import { initializeRope, NoRope, phi3SuRope, UniversalRope } from "../../layers/rope";
import { loadEmbedding, loadLayerNorm, loadLinear, loadRmsNorm, type AnyEmbedding, type AnyLinear, type AnyNorm } from "../../layers/loading";
import { WeightAudit } from "../../artifacts/weight-audit";
import { cleanupFailure } from "../../runtime/resources";
import { cacheRead, type AttentionReadStrategy } from "../qwen/blocks";

/** The score-softcapped read as a block's read strategy: decode reads
 * `appendDecode`, a window `appendWindow`; this graph has no committed append
 * (no `createAppend`), so a committed span reads as a window. */
function softcapRead(softcap: number, window: number | null, arrayMasks: boolean): AttentionReadStrategy {
  const reads = softcapAttentionReads(softcap, window, arrayMasks);
  const byPhase: Readonly<Record<ReadPhase, typeof reads.appendDecode>> = {
    decode: reads.appendDecode, window: reads.appendWindow, committed: reads.appendWindow,
  };
  return Object.freeze({ read: (cache: Cache, phase: ReadPhase, k: MlxArray, v: MlxArray) => byPhase[phase](cache, k, v) });
}

/** The linear `loadLinear` loads for `path` when the checkpoint stores it
 * affine-quantized (a `.scales` tensor and no Trellis entry in the quant
 * table): a `QuantizedLinear`, the only flavor adapters mount on. Read from the
 * artifact at load, as `loadLinear` reads it. */
function adapterTarget(weights: Weights, config: ModelConfig, path: string, linear: AnyLinear): QuantizedLinear | null {
  return !TrellisLinear.isTrellis(config, path) && weights.has(`${path}.scales`) ? linear as QuantizedLinear : null;
}

/** Rope for one attention module, per the descriptor (llama's
 *  initialize_rope, the plain-nn.RoPE archs, or phi3's Su/linear branch). */
function buildRope(a: UniversalArgs): UniversalRope {
  const ropeDims = Math.floor(a.headDim * a.partialRotaryFactor);
  if (a.rope === "phi3") {
    const sc = a.ropeScaling;
    if (sc && ["longrope", "su"].includes(String(sc.type))) {
      return phi3SuRope(
        ropeDims,
        a.ropeTheta,
        a.maxPositionEmbeddings ?? 131072,
        a.originalMaxPositionEmbeddings,
        (sc.long_factor as number[] | number | undefined) ?? 1.0,
      );
    }
    let scale = 1.0;
    if (sc && String(sc.type) === "linear") scale = 1 / (sc.factor as number);
    return new UniversalRope(ropeDims, a.ropeTraditional, a.ropeTheta, scale, null, 1, false);
  }
  if (a.rope === "plain")
    return new UniversalRope(ropeDims, a.ropeTraditional, a.ropeTheta, 1.0, null, 1, false);
  // initialize_rope path (llama, qwen2, qwen3, olmo2, granite, smollm3)
  return initializeRope(a.headDim, a.ropeTheta, a.ropeTraditional, a.ropeScaling, a.maxPositionEmbeddings);
}

class UniversalAttention {
  readonly qProj: AnyLinear | null;
  readonly kProj: AnyLinear | null;
  readonly vProj: AnyLinear | null;
  readonly qkvProj: AnyLinear | null;
  readonly oProj: AnyLinear;
  readonly qNorm: RMSNorm | null;
  readonly kNorm: RMSNorm | null;
  /** Bound at construction: the score softcap composes the manual attention
   * over plain K/V arrays as this block's read; without it, the block reads
   * through the cache's own read, which also attends encoded KV. */
  readonly softcap: number | null;
  /** The projections adapters mount on (affine-quantized), read at load. */
  readonly adapters: { readonly q: QuantizedLinear | null; readonly k: QuantizedLinear | null;
    readonly v: QuantizedLinear | null; readonly qkv: QuantizedLinear | null; readonly o: QuantizedLinear | null };

  constructor(
    weights: Weights,
    config: ModelConfig,
    prefix: string,
    readonly a: UniversalArgs,
    readonly rope: UniversalRope,
    audit: WeightAudit,
    /** How the block reads its cache, composed by the graph from the descriptor. */
    readonly reads: AttentionReadStrategy,
  ) {
    this.softcap = a.attnLogitSoftcap;
    const load = (name: string) => loadLinear(weights, `${prefix}.${name}`, config, audit);
    const target = (name: string, linear: AnyLinear | null) =>
      linear && adapterTarget(weights, config, `${prefix}.${name}`, linear);
    if (a.fusedQkv) {
      this.qkvProj = load("qkv_proj");
      this.qProj = this.kProj = this.vProj = null;
    } else {
      this.qProj = load("q_proj");
      this.kProj = load("k_proj");
      this.vProj = load("v_proj");
      this.qkvProj = null;
    }
    this.oProj = load("o_proj");
    this.adapters = Object.freeze({ q: target("q_proj", this.qProj), k: target("k_proj", this.kProj),
      v: target("v_proj", this.vProj), qkv: target("qkv_proj", this.qkvProj), o: target("o_proj", this.oProj) });
    if (a.qkNorm !== "none") {
      // qwen3 ("head"): RMSNorm(head_dim); olmo2 ("full"): RMSNorm(width) —
      // both live at q_norm/k_norm and use rms_norm_eps.
      this.qNorm = loadRmsNorm(weights, `${prefix}.q_norm`, a.normEps, false, audit);
      this.kNorm = loadRmsNorm(weights, `${prefix}.k_norm`, a.normEps, false, audit);
    } else {
      this.qNorm = this.kNorm = null;
    }
  }

  /** `x` [B, L, hidden] (borrowed) to an owned [B, L, hidden], appending to
   *  `cache` and attending through the read `phase` names. */
  forward(x: MlxArray, cache: Cache, phase: ReadPhase): MlxArray {
    const a = this.a;
    const [B, L] = x.shape as [number, number, number];

    let q: MlxArray, k: MlxArray, v: MlxArray;
    if (this.qkvProj) {
      // phi3: fused projection, split as ACTIVATIONS (mx.split) like mlx-lm.
      const qkv = this.qkvProj.forward(x);
      const queryPos = a.numHeads * a.headDim;
      const parts = ops.split(qkv, [queryPos, queryPos + a.numKvHeads * a.headDim], -1);
      qkv.dispose();
      [q, k, v] = parts as [MlxArray, MlxArray, MlxArray];
    } else {
      q = this.qProj!.forward(x);
      k = this.kProj!.forward(x);
      v = this.vProj!.forward(x);
    }

    if (a.qkNorm === "full") {
      // olmo2: norm the FLAT projections before any reshape.
      q = disposing(q, this.qNorm!.forward(q));
      k = disposing(k, this.kNorm!.forward(k));
    }

    q = disposing(q, ops.reshape(q, [B, L, a.numHeads, -1]));
    k = disposing(k, ops.reshape(k, [B, L, a.numKvHeads, -1]));
    v = disposing(v, ops.reshape(v, [B, L, a.numKvHeads, -1]));

    if (a.qkNorm === "head") {
      // qwen3: per-head norm over head_dim, BEFORE transpose + rope.
      q = disposing(q, this.qNorm!.forward(q));
      k = disposing(k, this.kNorm!.forward(k));
    }

    q = disposing(q, ops.transposeAxes(q, [0, 2, 1, 3]));
    k = disposing(k, ops.transposeAxes(k, [0, 2, 1, 3]));
    v = disposing(v, ops.transposeAxes(v, [0, 2, 1, 3]));

    // Batched decode: the scheduler's mask wrapper exposes each row's REAL
    // position as ropeOffsetArr (rows have different prompt lengths under
    // left-padding; the scalar offset would mis-position every padded row by
    // its pad amount — the 2026-07-03 uneven-batch bug). Serial lane: scalar
    // offset, byte-identical to before (the S1b.1 pattern, cf. qwen3_5.ts).
    const offArr = (cache as { ropeOffsetArr?: MlxArray }).ropeOffsetArr;
    q = disposing(q, offArr
      ? this.rope.applyDynamic(q, offArr)
      : this.rope.apply(q, cache.offset));
    k = disposing(k, offArr
      ? this.rope.applyDynamic(k, offArr)
      : this.rope.apply(k, cache.offset));

    // The read holds its mask: a failed release keeps the attention output
    // from leaking, and an attention failure stays the primary error.
    let attn: MlxArray;
    try {
      const read = this.reads.read(cache, phase, k, v);
      try { attn = read.attend(q, a.attnScale); }
      catch (error) { return cleanupFailure(error, () => read.dispose()); }
      try { read.dispose(); }
      catch (error) { return cleanupFailure(error, () => attn.dispose()); }
    } finally { k.dispose(); v.dispose(); q.dispose(); }

    const attnT = ops.transposeAxes(attn, [0, 2, 1, 3]);
    attn.dispose();
    const merged = ops.reshape(attnT, [B, L, -1]);
    attnT.dispose();
    const out = this.oProj.forward(merged);
    merged.dispose();
    return out;
  }
}

/** swiglu(gate, up) = nn.silu(gate) * up as ONE call — the whole activation,
 *  not just silu. Defaults to the `@mx.compile` version (compiledSwiglu) — the
 *  same libmlx kernel mlx-lm dispatches (L1-faithful), bit-identical to the old
 *  uncompiled composition but one dispatch instead of three. Set to null to force
 *  the uncompiled path (the slower same-parity opt-in). */
export type SwigluFn = (gate: MlxArray, up: MlxArray) => MlxArray;

export class UniversalMLP {
  readonly gate: AnyLinear | null;
  readonly up: AnyLinear | null;
  readonly gateUp: AnyLinear | null;
  readonly down: AnyLinear;
  /** Fused swiglu (silu(gate)*up), default = the oracle's `@mx.compile` closure.
   *  Only consumed for the swiglu / fused_swiglu kinds; geglu/geglu_approx/gelu_mlp
   *  are unaffected (the oracle does NOT @mx.compile those — gemma/gemma2/starcoder2
   *  use a plain nn.gelu composition). Set to null for the uncompiled composition. */
  swigluFn: SwigluFn | null = compiledSwiglu;
  /** The projections adapters mount on (affine-quantized), read at load. */
  readonly adapters: { readonly gate: QuantizedLinear | null; readonly up: QuantizedLinear | null;
    readonly gateUp: QuantizedLinear | null; readonly down: QuantizedLinear | null };

  constructor(
    weights: Weights,
    config: ModelConfig,
    prefix: string,
    readonly kind: UniversalArgs["mlp"],
    audit: WeightAudit,
  ) {
    const load = (name: string) => loadLinear(weights, `${prefix}.${name}`, config, audit);
    const target = (name: string, linear: AnyLinear) => adapterTarget(weights, config, `${prefix}.${name}`, linear);
    if (kind === "fused_swiglu") {
      this.gateUp = load("gate_up_proj");
      this.gate = this.up = null;
      this.down = load("down_proj");
      this.adapters = Object.freeze({ gate: null, up: null, gateUp: target("gate_up_proj", this.gateUp), down: target("down_proj", this.down) });
    } else if (kind === "gelu_mlp") {
      // starcoder2: c_fc → gelu(precise) → c_proj
      this.gate = load("c_fc");
      this.up = this.gateUp = null;
      this.down = load("c_proj");
      this.adapters = Object.freeze({ gate: target("c_fc", this.gate), up: null, gateUp: null, down: target("c_proj", this.down) });
    } else {
      this.gate = load("gate_proj");
      this.up = load("up_proj");
      this.gateUp = null;
      this.down = load("down_proj");
      this.adapters = Object.freeze({ gate: target("gate_proj", this.gate), up: target("up_proj", this.up), gateUp: null,
        down: target("down_proj", this.down) });
    }
  }

  forward(x: MlxArray): MlxArray {
    if (this.kind === "gelu_mlp") {
      const h = this.gate!.forward(x);
      const g = ops.geluPrecise(h);
      h.dispose();
      const out = this.down.forward(g);
      g.dispose();
      return out;
    }

    let gate: MlxArray;
    let up: MlxArray;
    if (this.kind === "fused_swiglu") {
      const gu = this.gateUp!.forward(x);
      const half = Math.floor(gu.shape[gu.shape.length - 1]! / 2);
      const parts = ops.split(gu, [half], -1); // mx.split(x, 2, axis=-1)
      gu.dispose();
      [gate, up] = parts as [MlxArray, MlxArray];
    } else {
      gate = this.gate!.forward(x);
      up = this.up!.forward(x);
    }

    // swiglu / fused_swiglu run silu(gate)*up as ONE fused call (mlx-lm's
    // `@mx.compile swiglu`), matching the oracle kernel set exactly. This is the
    // default (swigluFn = compiledSwiglu); null forces the uncompiled composition.
    if (this.swigluFn && (this.kind === "swiglu" || this.kind === "fused_swiglu")) {
      const hidden = this.swigluFn(gate, up);
      gate.dispose();
      up.dispose();
      const out = this.down.forward(hidden);
      hidden.dispose();
      return out;
    }

    let act: MlxArray;
    if (this.kind === "geglu") {
      act = ops.geluPrecise(gate); // gemma-1: nn.gelu (erf)
    } else if (this.kind === "geglu_approx") {
      act = ops.geluApprox(gate); // gemma2: nn.gelu_approx
    } else {
      // swiglu — the L1-verified uncompiled composition (minicpm5/qwen3):
      const sig = ops.sigmoid(gate);
      act = ops.mul(gate, sig);
      sig.dispose();
    }
    gate.dispose();
    const hidden = ops.mul(act, up);
    act.dispose();
    up.dispose();
    const out = this.down.forward(hidden);
    hidden.dispose();
    return out;
  }
}

export class UniversalLayer {
  readonly attn: UniversalAttention;
  readonly mlp: UniversalMLP;
  /** pre/gemma2/glm4: input_layernorm. post (olmo2): unused (null). */
  readonly inputNorm: AnyNorm | null;
  /** pre: post_attention_layernorm (pre-MLP). post: post_attention (on
   *  attn out). gemma2: post_attention (on attn out). glm4:
   *  post_attention (pre-MLP). */
  readonly postAttnNorm: AnyNorm;
  /** gemma2: pre_feedforward. glm4: post_self_attn (on attn out). */
  readonly extraNorm1: AnyNorm | null;
  /** gemma2: post_feedforward. glm4: post_mlp. post: post_feedforward. */
  readonly extraNorm2: AnyNorm | null;

  constructor(
    weights: Weights,
    config: ModelConfig,
    prefix: string,
    readonly a: UniversalArgs,
    rope: UniversalRope,
    audit: WeightAudit,
    reads: AttentionReadStrategy,
  ) {
    this.attn = new UniversalAttention(weights, config, `${prefix}.self_attn`, a, rope, audit, reads);
    this.mlp = new UniversalMLP(weights, config, `${prefix}.mlp`, a.mlp, audit);
    const norm = (name: string): AnyNorm =>
      a.norm === "layernorm"
        ? loadLayerNorm(weights, `${prefix}.${name}`, a.normEps, audit)
        : loadRmsNorm(weights, `${prefix}.${name}`, a.normEps, a.norm === "rmsnorm_plus_one", audit);

    this.postAttnNorm = norm("post_attention_layernorm");
    if (a.block === "post") {
      this.inputNorm = null;
      this.extraNorm1 = null;
      this.extraNorm2 = norm("post_feedforward_layernorm");
    } else if (a.block === "gemma2") {
      this.inputNorm = norm("input_layernorm");
      this.extraNorm1 = norm("pre_feedforward_layernorm");
      this.extraNorm2 = norm("post_feedforward_layernorm");
    } else if (a.block === "glm4") {
      this.inputNorm = norm("input_layernorm");
      this.extraNorm1 = norm("post_self_attn_layernorm");
      this.extraNorm2 = norm("post_mlp_layernorm");
    } else {
      this.inputNorm = norm("input_layernorm");
      this.extraNorm1 = null;
      this.extraNorm2 = null;
    }
  }

  forward(x: MlxArray, cache: Cache, phase: ReadPhase): MlxArray {
    const a = this.a;
    if (a.block === "post") {
      // olmo2: h = x + norm(attn(x)); out = h + norm(mlp(h))
      const r1 = this.attn.forward(x, cache, phase);
      const n1 = disposing(r1, this.postAttnNorm.forward(r1));
      const h = ops.add(x, n1);
      n1.dispose();
      const r2 = this.mlp.forward(h);
      const n2 = disposing(r2, this.extraNorm2!.forward(r2));
      const out = ops.add(h, n2);
      h.dispose();
      n2.dispose();
      return out;
    }
    if (a.block === "gemma2") {
      // h = x + postAttn(attn(inputNorm(x))); out = h + postFfn(mlp(preFfn(h)))
      const xn = this.inputNorm!.forward(x);
      const r1 = this.attn.forward(xn, cache, phase);
      xn.dispose();
      const n1 = disposing(r1, this.postAttnNorm.forward(r1));
      const h = ops.add(x, n1);
      n1.dispose();
      const fn = this.extraNorm1!.forward(h);
      const r2 = this.mlp.forward(fn);
      fn.dispose();
      const n2 = disposing(r2, this.extraNorm2!.forward(r2));
      const out = ops.add(h, n2);
      h.dispose();
      n2.dispose();
      return out;
    }
    if (a.block === "glm4") {
      // x1 = x + postSelfAttn(attn(inputNorm(x)));
      // out = postMlp(mlp(postAttn(x1))) + x1
      const xn = this.inputNorm!.forward(x);
      const r1 = this.attn.forward(xn, cache, phase);
      xn.dispose();
      const n1 = disposing(r1, this.extraNorm1!.forward(r1));
      const x1 = ops.add(x, n1);
      n1.dispose();
      const pn = this.postAttnNorm.forward(x1);
      const r2 = this.mlp.forward(pn);
      pn.dispose();
      const n2 = disposing(r2, this.extraNorm2!.forward(r2));
      const out = ops.add(n2, x1);
      n2.dispose();
      x1.dispose();
      return out;
    }
    // pre (llama family): h = x + r·mult; out = h + r·mult
    const xn = this.inputNorm!.forward(x);
    let r1 = this.attn.forward(xn, cache, phase);
    xn.dispose();
    if (a.residualMultiplier !== null)
      r1 = disposing(r1, ops.mulScalar(r1, a.residualMultiplier));
    const h = ops.add(x, r1);
    r1.dispose();
    const hn = this.postAttnNorm.forward(h);
    let r2 = this.mlp.forward(hn);
    hn.dispose();
    if (a.residualMultiplier !== null)
      r2 = disposing(r2, ops.mulScalar(r2, a.residualMultiplier));
    const out = ops.add(h, r2);
    h.dispose();
    r2.dispose();
    return out;
  }
}

export class UniversalDenseModel implements MlxDeclaredGraph, MlxTokenPhases {
  readonly config: ModelConfig;
  readonly args: UniversalArgs;
  /** The served configuration this graph was built for: its KV scheme is what `makeCache` composes. */
  readonly composition: Composition;
  readonly weightsBytes: number;
  readonly prefixBase = "model";
  readonly loraState = new LoraState();
  readonly embed: AnyEmbedding;
  readonly layers: UniversalLayer[];
  readonly finalNorm: AnyNorm;
  readonly lmHead: AnyLinear | null;
  readonly sharedRope: UniversalRope;
  readonly ropes: UniversalRope[];
  /** Whether every attention layer, as bound, reads encoded (affine) KV views. */
  readonly encodedKvAttention: boolean;
  /** Layers whose bound attention reads plain keys and values: the manual softcap layers. */
  readonly requiredDenseKvLayers: readonly number[];

  /** Softcapped attention reads plain K/V in every layer it softcaps, so encoded
   * storage decodes on read and verification qualifies plain-KV rounds only.
   * Otherwise attention attends the storage its caches hold: delayed affine rows
   * convert per layer and serve ordinary decoding only. Before the attention
   * facts are bound, neither is declared. */
  get graphCapabilities(): GraphCapabilities {
    const denseReads = this.encodedKvAttention === false;
    return declareGraph({
      kv: { denseReads, delayedAffine: this.encodedKvAttention === true ? "ordinary" : "none" },
      ...(denseReads ? { speculation: PLAIN_KV_VERIFICATION } : {}),
    });
  }
  // Cache layout and each layer's read come from the same descriptor facts,
  // bound at construction; later edits to `args` cannot make them disagree.
  readonly #cacheWindows: readonly (number | null)[];

  constructor(weights: Weights, config: ModelConfig, composition: Composition, args?: UniversalArgs) {
    const a = args ?? genericArgsFor(config);
    if (!a)
      throw new Error(`UniversalDenseModel: no descriptor for model_type ${config.modelType}`);
    this.config = config;
    this.args = a;
    this.composition = composition;
    this.#cacheWindows = universalCacheWindows(a);
    // Each layer's read: the cache's own read, or the softcapped read over plain
    // KV with the layer group's window (gemma2: create_attention_mask(...,
    // return_array=True) materializes the causal mask). The cache's own read
    // builds the mask the cache answers for its query count, so array masks
    // without a score softcap have no read here.
    const recipe = universalMaskRecipe(a);
    const softcap = a.attnLogitSoftcap;
    if (softcap === null && recipe.array)
      throw new Error(`UniversalDenseModel: ${a.modelType} asks for array masks without a score softcap; ` +
        "the composed attention reads take the cache's own mask (state/softcap-attention.ts reads array masks)");
    const reads = (layer: number): AttentionReadStrategy => softcap === null ? cacheRead
      : softcapRead(softcap, recipe.slidingLayers[layer] ? recipe.window : null, recipe.array);
    this.weightsBytes = [...weights.shards.files.values()]
      .reduce((acc, f) => acc + f.mmap.size, 0);

    const audit = new WeightAudit();
    this.embed = loadEmbedding(weights, "model.embed_tokens", config, audit);
    // One rope instance shared by every rope-bearing layer (identical
    // params ⇒ identical table); NoPE layers (smollm3) get the no-op.
    this.sharedRope = buildRope(a);
    const noRope = new NoRope();
    this.ropes = Array.from({ length: a.numHiddenLayers }, (_, i) =>
      a.noRopeLayers && a.noRopeLayers[i] === 0 ? noRope : this.sharedRope,
    );
    this.layers = Array.from(
      { length: a.numHiddenLayers },
      (_, i) => new UniversalLayer(weights, config, `model.layers.${i}`, a, this.ropes[i]!, audit, reads(i)),
    );
    this.encodedKvAttention = this.layers.every(layer => layer.attn.softcap === null);
    this.requiredDenseKvLayers = Object.freeze(this.layers.flatMap((layer, index) => layer.attn.softcap === null ? [] : [index]));
    this.finalNorm = a.norm === "layernorm"
      ? loadLayerNorm(weights, "model.norm", a.normEps, audit)
      : loadRmsNorm(weights, "model.norm", a.normEps, a.norm === "rmsnorm_plus_one", audit);
    this.lmHead = a.tieWordEmbeddings
      ? null
      : loadLinear(weights, "lm_head", config, audit);

    // Load-time weight audit: unconsumed tensors (outside the arch's
    // sanitize drops) are a LOAD ERROR, never a silently-wrong model.
    audit.finish(weights, a.dropWeightPatterns);
  }

  /** LoRA hot-swap targets — quantized linears only (dense checkpoints
   *  serve without adapter support at Tier-0), read at load. */
  loraTargets(): Map<string, QuantizedLinear> {
    const out = new Map<string, QuantizedLinear>();
    const put = (path: string, lin: QuantizedLinear | null) => {
      if (lin) out.set(path, lin);
    };
    for (let i = 0; i < this.layers.length; i++) {
      const { attn, mlp } = this.layers[i]!;
      const p = `model.layers.${i}`;
      put(`${p}.self_attn.q_proj`, attn.adapters.q);
      put(`${p}.self_attn.k_proj`, attn.adapters.k);
      put(`${p}.self_attn.v_proj`, attn.adapters.v);
      put(`${p}.self_attn.qkv_proj`, attn.adapters.qkv);
      put(`${p}.self_attn.o_proj`, attn.adapters.o);
      put(`${p}.mlp.gate_proj`, mlp.adapters.gate);
      put(`${p}.mlp.up_proj`, mlp.adapters.up);
      put(`${p}.mlp.gate_up_proj`, mlp.adapters.gateUp);
      put(`${p}.mlp.down_proj`, mlp.adapters.down);
    }
    return out;
  }

  /** One cache per layer (a ring on a sliding layer with a window), sharing
   * one mask memo. The full-attention caches' TurboQuant read is dense
   * (`denseTurboQuantKernels`): this graph read TurboQuant storage decoded,
   * through `updateAndFetch`, before its blocks read through the cache (the
   * softcapped read still fetches it that way). */
  makeCache(): Cache[] {
    const masks = new AttentionMasks();
    const kv = this.composition.kv;
    return this.#cacheWindows.map((window, layer) => window ? composedSlidingAttentionCache(kv, layer, window, masks)
      : composedFullAttentionCache(kv, layer, masks, denseTurboQuantKernels));
  }

  /** @deprecated E2 calls the named phases. */
  forwardHidden(ids: MlxArray, cache: Cache[]): MlxArray {
    return this.forwardLayers(this.#embed(ids), cache, deprecatedPhase(ids));
  }

  /** One whole chunk of a prompt; rows are sequences times the chunk size. The GPU is this graph's one unit. */
  readonly prefillChunk: MlxPrefillChunk = {
    gpu: (ids, cache) => this.forwardLayers(this.#embed(ids), cache, "window"),
  };

  /** The positions left after a prompt's whole chunks: the same semantics as
   * `prefillChunk`, named apart because a composition may run it elsewhere. */
  prefillTail(ids: MlxArray, cache: Cache[]): MlxArray {
    return this.forwardLayers(this.#embed(ids), cache, "window");
  }

  /** The one output head at every width. */
  readonly logits = atEveryWidth<MlxHeadForward>(hidden => this.logitsFromHidden(hidden));

  /** One new token per sequence: `ids` [sequences, 1]. */
  readonly decode = atEveryWidth<MlxPhaseForward>((ids, cache) =>
    this.forwardLayers(this.#embed(ids), cache, "decode"));

  /** A speculative window of depth + 1 positions per sequence. This graph
   * declares no hidden-layer taps, so `capture` is always null. */
  readonly verify = atEveryWidth<MlxVerifyForward>((ids, cache) =>
    this.forwardLayers(this.#embed(ids), cache, "window"));

  forwardEmbeddings(_embeds: MlxArray, _cache: Cache[], _bidir: MlxArray | null): MlxArray {
    throw new Error("universal-dense input-embedding path is not supported");
  }

  /** Token embeddings, scaled where the descriptor scales them (owned). */
  #embed(ids: MlxArray): MlxArray {
    const h = this.embed.encode(ids);
    return this.args.embedMultiplier !== null ? disposing(h, ops.mulScalar(h, this.args.embedMultiplier)) : h;
  }

  /** The decoder over `h0` (consumed), every block reading its cache for
   *  `phase`. This call owns h0 and each intermediate hidden state, and
   *  releases them on failure as on success; an execution failure stays the
   *  primary error. */
  protected forwardLayers(h0: MlxArray, cache: Cache[], phase: ReadPhase): MlxArray {
    let cur = h0;
    try {
      for (let i = 0; i < this.layers.length; i++) {
        const next = this.layers[i]!.forward(cur, cache[i]!, phase);
        const done = cur;
        cur = next;
        done.dispose();
      }
    } catch (error) {
      return cleanupFailure(error, () => cur.dispose());
    }
    let out: MlxArray;
    try { out = this.finalNorm.forward(cur); }
    catch (error) { return cleanupFailure(error, () => cur.dispose()); }
    try { cur.dispose(); }
    catch (error) { return cleanupFailure(error, () => out.dispose()); }
    return out;
  }

  logitsFromHidden(h: MlxArray): MlxArray {
    let logits = this.lmHead ? this.lmHead.forward(h) : this.embed.asLinear(h);
    if (this.args.finalLogitSoftcap !== null)
      logits = disposing(logits, logitSoftcap(logits, this.args.finalLogitSoftcap));
    if (this.args.logitsDivisor !== null) {
      const d = ops.scalarLike(this.args.logitsDivisor, logits);
      logits = disposing(logits, ops.div(logits, d));
      d.dispose();
    }
    return logits;
  }

  forward(tokens: number[] | MlxArray, cache: Cache[]): MlxArray {
    const ids = Array.isArray(tokens)
      ? ops.fromInt32(tokens, [1, tokens.length])
      : tokens;
    const h = this.forwardHidden(ids, cache);
    if (Array.isArray(tokens)) ids.dispose();
    const logits = this.logitsFromHidden(h);
    h.dispose();
    return logits;
  }

  generate(promptTokens: number[], maxTokens: number, eosIds: number[] = []): number[] {
    const cache = this.makeCache();
    const out: number[] = [];
    try {
      let tokens = promptTokens;
      for (let step = 0; step < maxTokens; step++) {
        const logits = this.forward(tokens, cache);
        const next = argmaxLastPosition(logits);
        logits.dispose();
        if (eosIds.includes(next)) break;
        out.push(next);
        tokens = [next];
      }
    } finally {
      for (const c of cache) c.dispose();
    }
    return out;
  }
}
