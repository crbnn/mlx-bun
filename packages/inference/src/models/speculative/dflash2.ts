// DFlash 2 drafter for Qwen3.8-27B (incoai/Qwen3.8-27B-DFlash2, Apache-2.0).
// Port of z-lab/dflash@07ebd93 `dflash/model_mlx.py` (DFlash2DraftModel),
// cross-checked against Splash `runtime/model/DFlashDraft.cpp`, SGLang PR
// #35371 and vLLM PR #52816. Five Qwen3-style layers draft one block
// [anchor, MASK×(G−1)] in a single pass:
//   context  c = hidden_norm(fc(concat(target layers 5,19,33,47,61)))
//            — each layer projects c with its own k/v (k_norm, RoPE at the
//            row's absolute position) into a K/V prefix the block attends to.
//   layer    n = rms(x); D = kernel_projection(n) → [stage, tap, group]
//            x += Conv₁(Attn(Conv₀(n)));  m = rms(x); D' from m
//            x += Conv₁(MLP(Conv₀(m)))
//            Conv(y)[t,c] = Σ_τ (base[τ,c] + D[t,τ,c/16])·y[t−τ,c], y[−1] = 0
//   output   rows 1..G−1 → norm → the TARGET's lm_head (the drafter has no
//            embedding or head of its own), or the head's rows of the draft
//            vocabulary a target's composition declares
//            (`DraftProjection.vocabulary`): the walk then selects among them.
//   selector top-16 per position; greedy left-to-right walk scoring
//            logit + Σ_r pred[prev,r]·z[r]·succ[cand,r], z = hidden_projection(h).
// Attention is bidirectional within the block. The 2048-row sliding window
// over context is not applied: context longer than 2047 rows attends to all
// of it (verification keeps outputs exact; acceptance can drop).
//
// Target basis: a TurboQuant R1 fold stores the residual stream as h·R1,
// R1 = diag(s)·M with M = MLX's hadamard_transform/√n (x ↦ x·M), and folds the
// final-norm gain γ into the head. The drafter was trained on the unrotated
// model and loads in that basis; the target declares its own
// (`TargetView.residualBasis`) and the drafter applies it once, when a draft
// group binds: taps reach fc as tap·R1ᵀ (folded into fc as fc·diag(s)·M),
// embedding rows are rotated back as (x·Mᵀ)⊙s, and draft hiddens enter the
// folded head as ((v/γ)⊙s)·M. For n = 5120 = 20·256, M is orthogonal but NOT
// symmetric, so x·Mᵀ needs the dense transpose, not a second transform.
//
// Kernels: each 4-bit projection runs, per width class of its rows, the kernel
// the target's composition declares for its device (`TargetView.draftProjections`)
// or, without a declaration, the drafter's own rule: five to eight rows (the
// fiveToEight class, one block of 8) on the simdgroup matrix unit, every other
// width on MLX's quantized matmul. The rule's kernels are built at load (fc when
// the first draft group binds) and the declared ones when it binds. The context
// projection (fc, then each layer's k and v, over the B × L rows a prefill or a
// commit appends) and the draft block (B anchors × G columns) are width tables;
// the caller indexes them by the rows it projects or drafts.

import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { Weights } from "../../artifacts/weights";
import type { DraftProjection } from "../../contracts/mlx/draft-projection";
import type { DraftProjectionKernel, ResidualBasis } from "../../contracts/mlx/draft-target";
import type {
  ContextKV, Dflash2ContextAttention, Dflash2DraftBlock, Dflash2DrafterModel, DrafterContextKV, DrafterContextProjection,
} from "../../contracts/mlx/drafter";
import { widthClass, type MlxWidthClass, type MlxWidthTable } from "../../contracts/mlx/graph";
import { atEveryWidth, byWidth } from "../graph";
import { isDflash2Architecture } from "./dflash2-artifact";
import { QuantizedLinear } from "../../layers/quantized-linear";
import { Affine4MmaLinear } from "../../layers/affine-verify-linear";

export interface Dflash2Config {
  hidden: number;
  layers: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  eps: number;
  ropeTheta: number;
  blockSize: number;
  maskTokenId: number;
  convGroupSize: number;
  topK: number;
  targetLayerIds: number[];
  numTargetLayers: number;
}

/** One projection on one kernel. */
interface Projection { forward(x: MlxArray): MlxArray }

/** A BF16 projection over its stored transpose. */
class Bf16Projection implements Projection {
  constructor(readonly t: MlxArray) {}
  forward(x: MlxArray): MlxArray { return ops.matmul(x, this.t); }
}

/** A projection's kernel at each width class of its rows. Built at load on the drafter's own rule (fc when
 *  the target binds): a BF16 projection one matmul at every class; a 4-bit one the simdgroup matrix unit at
 *  five to eight rows where the weight's shape takes it (N a multiple of 8, K of 256) and MLX's quantized matmul
 *  at every other class. A target that declares its composition's kernels rebuilds `at` over `affine4` when it
 *  binds. */
interface Weight {
  at: MlxWidthTable<Projection>;
  /** The 4-bit, group-64 weight a declared kernel builds over; null for any other width. */
  readonly affine4: { readonly w: MlxArray; readonly scales: MlxArray; readonly biases: MlxArray } | null;
  dispose(): void;
}

function bf16Weight(projection: Bf16Projection): Weight {
  return { at: atEveryWidth<Projection>(projection), affine4: null, dispose: () => projection.t.dispose() };
}
function quantizedWeight(source: MlxArray, bits: number): Weight {
  const q = ops.quantize(source, 64, bits);
  ops.evalAll([q.packed, q.scales, q.biases]);
  const linear = new QuantizedLinear(q.packed, q.scales, q.biases, { bits, groupSize: 64, mode: "affine" });
  const mma = bits === 4 && q.scales.shape[0]! % 8 === 0 && (q.scales.shape[1]! * 64) % 256 === 0 ? new Affine4MmaLinear(linear) : null;
  return {
    at: byWidth<Projection>(width => width === "fiveToEight" ? mma ?? linear : linear),
    affine4: bits === 4 ? { w: q.packed, scales: q.scales, biases: q.biases } : null,
    dispose() { linear.w.dispose(); linear.scales.dispose(); linear.biases!.dispose(); },
  };
}

/** R1 sign lane 0 of the rotation fold (packages/quantize/src/rotate.ts signVector). */
function r1Signs(seed: number, n: number): Float32Array {
  let state = (seed ^ Math.imul(1, 0x9e3779b9)) >>> 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
    z ^= z >>> 15;
    out[i] = (z & 1) === 0 ? 1 : -1;
  }
  return out;
}

interface Layer {
  inNorm: MlxArray; postNorm: MlxArray;
  q: Weight; k: Weight; v: Weight; o: Weight; qNorm: MlxArray; kNorm: MlxArray;
  gate: Weight; up: Weight; down: Weight;
  attnBase: MlxArray; attnProj: Bf16Projection; mlpBase: MlxArray; mlpProj: Bf16Projection;
}

function readConfig(raw: Record<string, any>): Dflash2Config {
  if (!isDflash2Architecture(raw.architectures))
    throw new Error(`Dflash2Drafter: expected architectures[0]==="DFlash2DraftModel", got ${JSON.stringify(raw.architectures)}`);
  const d = raw.dflash_config ?? {};
  if (d.conv_kernel_size !== 2) throw new Error(`Dflash2Drafter: conv_kernel_size ${d.conv_kernel_size} not implemented (two-tap only)`);
  if ((raw.rope_parameters?.rope_type ?? "default") !== "default") throw new Error("Dflash2Drafter: only default RoPE is implemented");
  return {
    hidden: raw.hidden_size, layers: raw.num_hidden_layers, heads: raw.num_attention_heads,
    kvHeads: raw.num_key_value_heads, headDim: raw.head_dim, eps: raw.rms_norm_eps,
    ropeTheta: raw.rope_parameters?.rope_theta ?? raw.rope_theta,
    blockSize: d.block_size, maskTokenId: d.mask_token_id, convGroupSize: d.conv_group_size,
    topK: d.selector_top_k, targetLayerIds: d.target_layer_ids, numTargetLayers: raw.num_target_layers,
  };
}

export class Dflash2Drafter implements Dflash2DrafterModel {
  readonly cfg: Dflash2Config;
  readonly tapLayers: number[];
  /** Draft tokens per block (block size minus the anchor). */
  readonly gamma: number;
  readonly #bits: number;
  #w: Weights | null;
  /** fc in the trained basis, owned until the target binds and fc is built from it. */
  #fcSource: MlxArray | null;
  #fc: Weight | null = null;
  #hiddenNorm: MlxArray;
  #norm: MlxArray;
  #selProj: Bf16Projection;
  #pred: MlxArray;
  #succ: MlxArray;
  #layers: Layer[] = [];
  /** The target declarations bound by the first draft group: undefined until then. */
  #bound: { basis: ResidualBasis | null; projections: MlxWidthTable<DraftProjectionKernel> | null } | undefined = undefined;
  /** R1's signs, the folded gain and dense Mᵀ [H, H] (bf16; every entry is ±1/√H) of a bound fold. */
  #rotation: { signs: MlxArray; finalGain: MlxArray; hadamardT: MlxArray } | null = null;
  /** Target-basis rows → drafter basis; identity until a fold binds. */
  #fromTarget: (x: MlxArray) => MlxArray = x => ops.contiguous(x);
  /** Drafter's normed hidden → input of the target's head; identity until a fold binds. */
  #toTargetHead: (v: MlxArray) => MlxArray = v => ops.contiguous(v);

  private constructor(w: Weights, cfg: Dflash2Config, bits: number) {
    this.#w = w;
    this.cfg = cfg;
    this.#bits = bits;
    this.tapLayers = cfg.targetLayerIds;
    this.gamma = cfg.blockSize - 1;
    const T = (name: string) => w.tensor(name);
    const bf16 = (name: string) => new Bf16Projection(ops.transposeAxes(T(name), [1, 0]));
    // Splash quantizes projections only; convs, norms and codebooks stay BF16.
    const weight = (name: string): Weight => {
      if (!bits) return bf16Weight(bf16(name));
      using source = T(name);
      return quantizedWeight(source, bits);
    };
    this.#fcSource = T("fc.weight");
    this.#hiddenNorm = T("hidden_norm.weight");
    this.#norm = T("norm.weight");
    this.#selProj = bf16("candidate_selector.hidden_projection.weight");
    this.#pred = T("candidate_selector.predecessor_codebook");
    this.#succ = T("candidate_selector.successor_codebook");
    for (let i = 0; i < cfg.layers; i++) {
      const p = `layers.${i}`;
      this.#layers.push({
        inNorm: T(`${p}.input_layernorm.weight`), postNorm: T(`${p}.post_attention_layernorm.weight`),
        q: weight(`${p}.self_attn.q_proj.weight`), k: weight(`${p}.self_attn.k_proj.weight`),
        v: weight(`${p}.self_attn.v_proj.weight`), o: weight(`${p}.self_attn.o_proj.weight`),
        qNorm: T(`${p}.self_attn.q_norm.weight`), kNorm: T(`${p}.self_attn.k_norm.weight`),
        gate: weight(`${p}.mlp.gate_proj.weight`), up: weight(`${p}.mlp.up_proj.weight`),
        down: weight(`${p}.mlp.down_proj.weight`),
        attnBase: this.#baseKernel(T(`${p}.attention_conv.base_kernel`)),
        attnProj: bf16(`${p}.attention_conv.kernel_projection.weight`),
        mlpBase: this.#baseKernel(T(`${p}.mlp_conv.base_kernel`)),
        mlpProj: bf16(`${p}.mlp_conv.kernel_projection.weight`),
      });
    }
    if (bits) this.#releaseCheckpoint();
  }

  /** Quantized projections own their storage; copy the remaining BF16 tensors
   *  (fc's source among them) out of the checkpoint map so its 3.85 GB mapping
   *  can be released. */
  #releaseCheckpoint(): void {
    const own = (a: MlxArray): MlxArray => {
      const copy = a.dtype === Dtype.float32 ? ops.contiguous(a) : ops.mulScalar(a, 1);
      copy.eval(); a.dispose(); return copy;
    };
    const ownBf16 = (m: Bf16Projection): Bf16Projection => {
      const t = ops.contiguous(m.t); // materializes the transposed view
      t.eval(); m.t.dispose(); return new Bf16Projection(t);
    };
    this.#fcSource = own(this.#fcSource!);
    this.#hiddenNorm = own(this.#hiddenNorm); this.#norm = own(this.#norm);
    this.#pred = own(this.#pred); this.#succ = own(this.#succ); this.#selProj = ownBf16(this.#selProj);
    for (const l of this.#layers) {
      l.inNorm = own(l.inNorm); l.postNorm = own(l.postNorm); l.qNorm = own(l.qNorm); l.kNorm = own(l.kNorm);
      l.attnBase = own(l.attnBase); l.mlpBase = own(l.mlpBase);
      l.attnProj = ownBf16(l.attnProj); l.mlpProj = ownBf16(l.mlpProj);
    }
    const w = this.#w!;
    w.dispose();
    for (const file of w.shards.files.values()) file.mmap.unmap();
    this.#w = null;
  }

  /** fc acting on target-basis taps: per tap block, fc_k·diag(s)·M (f32 fold). Consumes `source`. */
  #foldedFc(source: MlxArray, signs: Float32Array, bits: number): Weight {
    const H = this.cfg.hidden, m = this.cfg.targetLayerIds.length;
    using s = MlxArray.fromFloat32(signs, [H]);
    using f = source.astype(Dtype.float32);
    source.dispose();
    using blocks = ops.reshape(f, [H, m, H]);
    using signed = ops.mul(blocks, s);
    using rotated = ops.hadamardTransform(signed, 1 / Math.sqrt(H));
    using flat = ops.reshape(rotated, [H, m * H]);
    using folded = flat.astype(Dtype.bfloat16);
    if (!bits) {
      const t = ops.transposeAxes(folded, [1, 0]);
      t.eval();
      return bf16Weight(new Bf16Projection(t));
    }
    return quantizedWeight(folded, bits);
  }

  /** `dir`: the drafter checkpoint, loaded in the basis it was trained on.
   *  `bits`: the projections' quantized width; 0 keeps BF16 projections. */
  static async load(dir: string, opts: { bits: number }): Promise<Dflash2Drafter> {
    const raw = (await Bun.file(`${dir}/config.json`).json()) as Record<string, any>;
    const cfg = readConfig(raw);
    const w = await Weights.open(dir);
    try { return new Dflash2Drafter(w, cfg, opts.bits); }
    catch (error) { w.dispose(); throw error; }
  }

  /** Build fc for the target's basis and, for a fold, the rotations at the
   *  embedding and the head; then, where the target declares its composition's
   *  kernels, every 4-bit projection's kernel per width class from them. The
   *  first call applies; later calls (every draft group binds) must pass the
   *  same declarations. */
  bindTarget(basis: ResidualBasis | null, projections: MlxWidthTable<DraftProjectionKernel> | null): void {
    if (this.#bound !== undefined) {
      if (this.#bound.basis !== basis) throw new Error("this DFlash 2 drafter is bound to a target with a different residual basis");
      if (this.#bound.projections !== projections) throw new Error("this DFlash 2 drafter is bound to a target with different projection kernels");
      return;
    }
    this.#bindBasis(basis);
    if (projections) for (const weight of [this.#fc!, ...this.#layers.flatMap(l => [l.q, l.k, l.v, l.o, l.gate, l.up, l.down])]) {
      if (!weight.affine4) continue;
      const { w, scales, biases } = weight.affine4;
      weight.at = byWidth<Projection>(width => ({ forward: projections[width](w, scales, biases) }));
    }
    this.#bound = { basis, projections };
  }

  #bindBasis(basis: ResidualBasis | null): void {
    const source = this.#fcSource!;
    this.#fcSource = null;
    if (!basis) {
      if (this.#bits) {
        using unrotated = source;
        this.#fc = quantizedWeight(unrotated, this.#bits);
      } else {
        this.#fc = bf16Weight(new Bf16Projection(ops.transposeAxes(source, [1, 0])));
        source.dispose();
      }
      return;
    }
    const H = this.cfg.hidden;
    const signs = r1Signs(basis.r1Seed, H);
    this.#fc = this.#foldedFc(source, signs, this.#bits);
    const eye = new Float32Array(H * H);
    for (let i = 0; i < H; i++) eye[i * H + i] = 1;
    using identity = MlxArray.fromFloat32(eye, [H, H]);
    using dense = ops.hadamardTransform(identity, 1 / Math.sqrt(H)); // rows e_i·M
    using transposed = ops.transposeAxes(dense, [1, 0]);
    using contiguous = ops.contiguous(transposed);
    const hadamardT = contiguous.astype(Dtype.bfloat16);
    hadamardT.eval();
    const rotation = this.#rotation = {
      signs: MlxArray.fromFloat32(signs, [H]), finalGain: MlxArray.fromFloat32(basis.finalGain, [H]), hadamardT,
    };
    // x·R1ᵀ = (x·Mᵀ)⊙s over the last axis (target basis → drafter basis).
    this.#fromTarget = x => {
      using h = ops.matmul(x.dtype === Dtype.bfloat16 ? x : x.astype(Dtype.bfloat16), rotation.hadamardT);
      using back = ops.mul(h, rotation.signs);
      return back.astype(x.dtype);
    };
    // (v/γ)·R1 (drafter's normed hidden → input of the target's folded head).
    this.#toTargetHead = v => {
      const n = v.shape[v.shape.length - 1]!;
      using f = v.astype(Dtype.float32);
      using unscaled = ops.div(f, rotation.finalGain);
      using signed = ops.mul(unscaled, rotation.signs);
      using h = ops.hadamardTransform(signed, 1 / Math.sqrt(n));
      return h.astype(v.dtype);
    };
  }

  /** [stage, tap, H] base kernel as f32 [1, 1, stage, tap, H] for broadcasting. */
  #baseKernel(source: MlxArray): MlxArray {
    using f = source.astype(Dtype.float32);
    source.dispose();
    return ops.reshape(f, [1, 1, 2, 2, this.cfg.hidden]);
  }

  #rms(x: MlxArray, weight: MlxArray): MlxArray { return ops.rmsNorm(x, weight, this.cfg.eps); }

  /** Per-row dynamic coefficients [B,G,stage,tap,H] (f32) for one conv site. */
  #coefficients(n: MlxArray, proj: Bf16Projection, base: MlxArray): MlxArray {
    const B = n.shape[0]!, G = n.shape[1]!, H = this.cfg.hidden, groups = H / this.cfg.convGroupSize;
    using d = proj.forward(n);
    using d32 = d.astype(Dtype.float32);
    using grouped = ops.reshape(d32, [B, G, 2, 2, groups, 1]);
    using wide = ops.broadcastTo(grouped, [B, G, 2, 2, groups, this.cfg.convGroupSize]);
    using perChannel = ops.reshape(wide, [B, G, 2, 2, H]);
    return ops.add(perChannel, base);
  }

  /** Two-tap conv of y [B,G,H] with stage `stage` of coef; f32 result. */
  #conv(y: MlxArray, coef: MlxArray, stage: number): MlxArray {
    const B = y.shape[0]!, G = y.shape[1]!, H = this.cfg.hidden;
    using c0 = coef.slice([0, 0, stage, 0, 0], [B, G, stage + 1, 1, H]);
    using c1 = coef.slice([0, 0, stage, 1, 0], [B, G, stage + 1, 2, H]);
    using t0 = ops.reshape(c0, [B, G, H]);
    using t1 = ops.reshape(c1, [B, G, H]);
    using y32 = y.astype(Dtype.float32);
    using head = ops.zeros([B, 1, H], Dtype.float32);
    using body = y32.slice([0, 0, 0], [B, G - 1, H]);
    using prev = ops.concatAxis([head, body], 1);
    using a = ops.mul(t0, y32);
    using b = ops.mul(t1, prev);
    return ops.add(a, b);
  }

  #rope(x: MlxArray, position: number | MlxArray): MlxArray {
    return typeof position === "number"
      ? ops.rope(x, this.cfg.headDim, this.cfg.ropeTheta, position, null)
      : ops.ropeDynamic(x, this.cfg.headDim, this.cfg.ropeTheta, position, null);
  }

  /** Target taps [B,L,m·H] (target basis; fc carries the bound fold) → context rows [B,L,H], per width class of
   *  the B·L rows: fc on its kernel for the class (see `Weight`). */
  readonly projectContext: MlxWidthTable<DrafterContextProjection> = byWidth(width => taps => {
    using projected = this.#fc!.at[width].forward(taps);
    return this.#rms(projected, this.#hiddenNorm);
  });

  /** Context rows [B,L,H] → per-layer K/V [B,kv,L,D] at absolute positions from `position`, per width class of
   *  the B·L rows: each layer's k and v on the kernel `projectContext` runs for that class. */
  readonly projectContextKVRows: MlxWidthTable<DrafterContextKV> = byWidth(width =>
    (context, position) => this.#contextKV(context, position, width));

  #contextKV(context: MlxArray, position: number | MlxArray, width: MlxWidthClass): ContextKV[] {
    const B = context.shape[0]!, L = context.shape[1]!, { kvHeads, headDim } = this.cfg;
    return this.#layers.map(layer => {
      using kFlat = layer.k.at[width].forward(context);
      using k4 = ops.reshape(kFlat, [B, L, kvHeads, headDim]);
      using kn = this.#rms(k4, layer.kNorm);
      using kT = ops.transposeAxes(kn, [0, 2, 1, 3]);
      using vFlat = layer.v.at[width].forward(context);
      using v4 = ops.reshape(vFlat, [B, L, kvHeads, headDim]);
      return { k: this.#rope(kT, position), v: ops.transposeAxes(v4, [0, 2, 1, 3]) };
    });
  }

  /** The columns of a block drafting `depth` tokens: the anchor and up to `depth` (≤ gamma) mask positions. */
  blockColumns(depth: number): number { return Math.min(this.cfg.blockSize, Math.max(1, depth) + 1); }

  /** The draft block per width class of its rows (anchors × columns): every projection on its kernel for the
   *  class (see `Weight`). */
  readonly draft: MlxWidthTable<Dflash2DraftBlock> = byWidth(width =>
    (context, projection, anchors, position, columns) => this.#block(context, projection, anchors, position, columns, width));

  /** One block per anchor over `columns` positions, every projection on its kernel for `width`: the proposals left on the
   *  device, [B, columns - 1] uint32, row-major. */
  #block(context: Dflash2ContextAttention, projection: DraftProjection, anchors: readonly number[],
    position: number | MlxArray, columns: number, width: MlxWidthClass): MlxArray {
    const B = anchors.length, G = columns;
    const { hidden: H, heads, kvHeads, headDim } = this.cfg;
    using ids = ops.fromInt32(anchors.flatMap(anchor => [anchor, ...Array(G - 1).fill(this.cfg.maskTokenId)]), [B, G]);
    using embedded = projection.embed.encode(ids);
    let x = this.#fromTarget(embedded);
    try {
      for (const [li, layer] of this.#layers.entries()) {
        {
          using n = this.#rms(x, layer.inNorm);
          using coef = this.#coefficients(n, layer.attnProj, layer.attnBase);
          using a32 = this.#conv(n, coef, 0);
          using a = a32.astype(x.dtype);
          using qFlat = layer.q.at[width].forward(a);
          using q4 = ops.reshape(qFlat, [B, G, heads, headDim]);
          using qn = this.#rms(q4, layer.qNorm);
          using qT = ops.transposeAxes(qn, [0, 2, 1, 3]);
          using q = this.#rope(qT, position);
          using kFlat = layer.k.at[width].forward(a);
          using k4 = ops.reshape(kFlat, [B, G, kvHeads, headDim]);
          using kn = this.#rms(k4, layer.kNorm);
          using kT = ops.transposeAxes(kn, [0, 2, 1, 3]);
          using k = this.#rope(kT, position);
          using vFlat = layer.v.at[width].forward(a);
          using v4 = ops.reshape(vFlat, [B, G, kvHeads, headDim]);
          using v = ops.transposeAxes(v4, [0, 2, 1, 3]);
          using attn = context.attend(li, q, k, v, 1 / Math.sqrt(headDim));
          using attnT = ops.transposeAxes(attn, [0, 2, 1, 3]);
          using attnFlat = ops.reshape(attnT, [B, G, heads * headDim]);
          using out = layer.o.at[width].forward(attnFlat);
          using finished = this.#conv(out, coef, 1);
          using x32 = x.astype(Dtype.float32);
          using sum = ops.add(x32, finished);
          x.dispose();
          x = sum.astype(embedded.dtype);
        }
        {
          using m = this.#rms(x, layer.postNorm);
          using coef = this.#coefficients(m, layer.mlpProj, layer.mlpBase);
          using inner32 = this.#conv(m, coef, 0);
          using inner = inner32.astype(x.dtype);
          using g = layer.gate.at[width].forward(inner);
          using u = layer.up.at[width].forward(inner);
          using act = ops.silu(g);
          using prod = ops.mul(act, u);
          using mlp = layer.down.at[width].forward(prod);
          using finished = this.#conv(mlp, coef, 1);
          using x32 = x.astype(Dtype.float32);
          using sum = ops.add(x32, finished);
          x.dispose();
          x = sum.astype(embedded.dtype);
        }
      }
      using rows = x.slice([0, 1, 0], [B, G, H]);
      using h = this.#rms(rows, this.#norm);
      using headInput = this.#toTargetHead(h);
      // A target that declares its draft vocabulary is projected onto those rows only (its candidates their ids).
      const listed = projection.vocabulary;
      using logits = (listed?.logits ?? projection.logits)[widthClass(B * (G - 1))](headInput);
      using z = this.#selProj.forward(h);
      using columns = this.#select(logits, z, anchors, G - 1, listed?.ids ?? null); // [steps, B]
      using rowsMajor = ops.transposeAxes(columns, [1, 0]);
      return ops.contiguous(rowsMajor);
    } finally { x.dispose(); }
  }

  /** Greedy candidate-path walk (fp32 scores, as vLLM requires). Every position's top-K candidates and their
   *  scores come from one partition over all positions (each row partitions alone, as one position's would);
   *  only the path term walks position by position. `ids` (uint32 [N]) maps the columns of logits over a draft
   *  vocabulary to token ids; null: the columns are the token ids. */
  #select(logits: MlxArray, z: MlxArray, anchors: readonly number[], steps: number, ids: MlxArray | null): MlxArray {
    const B = anchors.length, R = z.shape[2]!, K = this.cfg.topK;
    const tokens: MlxArray[] = [];
    using scoresAll = logits.astype(Dtype.float32);
    using negatedAll = ops.mulScalar(scoresAll, -1);
    using orderAll = ops.argpartitionAxis(negatedAll, K - 1, -1);
    using localAll = orderAll.slice([0, 0, 0], [B, steps, K]);
    using unaryAll = ops.takeAlongAxis(scoresAll, localAll, -1); // [B, steps, K]
    using listedAll = ids ? ops.takeAxis(ids, localAll, 0) : null; // their token ids over a draft vocabulary
    const candidateAll = listedAll ?? localAll;
    using anchorIds = ops.fromInt32([...anchors], [B]);
    let previous = anchorIds.astype(Dtype.uint32);
    try {
      for (let p = 0; p < steps; p++) {
        using localRow = candidateAll.slice([0, p, 0], [B, p + 1, K]);
        using local = ops.reshape(localRow, [B, K]);
        using unaryRow = unaryAll.slice([0, p, 0], [B, p + 1, K]);
        using unary = ops.reshape(unaryRow, [B, K]);
        const candidates = local;
        using zRow = z.slice([0, p, 0], [B, p + 1, R]);
        using zFlat = ops.reshape(zRow, [B, R]);
        using z32 = zFlat.astype(Dtype.float32);
        using predRows = ops.takeAxis(this.#pred, previous, 0);
        using pred32 = predRows.astype(Dtype.float32);
        using weighted = ops.mul(pred32, z32);
        using column = ops.reshape(weighted, [B, R, 1]);
        using candidateFlat = ops.reshape(candidates, [B * K]);
        using succRows = ops.takeAxis(this.#succ, candidateFlat, 0);
        using succ32 = succRows.astype(Dtype.float32);
        using succ3 = ops.reshape(succ32, [B, K, R]);
        using edge3 = ops.matmul(succ3, column);
        using edge = ops.reshape(edge3, [B, K]);
        using total = ops.add(unary, edge);
        using best = ops.argmaxAxis(total, -1);
        using bestCol = ops.reshape(best, [B, 1]);
        using chosen = ops.takeAlongAxis(candidates, bestCol, -1);
        const token = ops.reshape(chosen, [1, B]);
        tokens.push(token);
        previous.dispose();
        using flatToken = ops.reshape(token, [B]);
        previous = flatToken.astype(Dtype.uint32);
      }
      return ops.concatAxis(tokens, 0);
    } finally {
      previous.dispose();
      for (const token of tokens) token.dispose();
    }
  }

  dispose(): void {
    this.#fc?.dispose(); this.#fcSource?.dispose(); this.#selProj.t.dispose();
    this.#hiddenNorm.dispose(); this.#norm.dispose(); this.#pred.dispose(); this.#succ.dispose();
    for (const l of this.#layers) {
      l.inNorm.dispose(); l.postNorm.dispose(); l.qNorm.dispose(); l.kNorm.dispose();
      for (const m of [l.q, l.k, l.v, l.o, l.gate, l.up, l.down]) m.dispose();
      l.attnProj.t.dispose(); l.mlpProj.t.dispose();
      l.attnBase.dispose(); l.mlpBase.dispose();
    }
    if (this.#rotation) { this.#rotation.signs.dispose(); this.#rotation.finalGain.dispose(); this.#rotation.hadamardT.dispose(); }
    this.#w?.dispose();
  }
}
