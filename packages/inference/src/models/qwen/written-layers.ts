import type { MlxArray } from "@mlx-bun/mlx/array";
import type { Weights } from "../../artifacts/weights";
import type { ModelConfig } from "../../artifacts/config";
import { QuantizedLinear } from "../../layers/quantized-linear";
import { TrellisLinear } from "../../layers/trellis-linear";
import type { TrellisRepresentations } from "../../kernels/trellis/index";
import type { RowProjection } from "../../layers/trellis-prompt";
import { QwenGatedDeltaNet, Qwen3TextAttention, Qwen3MediaAttention, QwenDeltaLayer, QwenAttentionLayer } from "./blocks";
import { AneAffineSplit, type AneMlpSplit } from "../../layers/ane-prefill-split";
import { compiledSwiglu } from "../../layers/swiglu";
import { qwen35WeightsView } from "./checkpoint";

/** Checkpoint naming only; it never selects a layer or numerical operation. */
export function canonicalQwenWeights(weights: Weights): Weights {
  const view = qwen35WeightsView(weights);
  if (view) weights.setView(view);
  return weights;
}

/** A weight attachment port. The graph names every constructor at its binding;
 * these helpers only put the named projections on the named block. */
export type AffineProjectionClass = new (linear: QuantizedLinear) => RowProjection;
export type GateUpClass = new (gate: TrellisLinear, up: TrellisLinear) => RowProjection;
export type TrellisProjectionClass = new (linear: TrellisLinear) => RowProjection;
export type GateDownClass = new (gateUp: RowProjection, down: RowProjection) => RowProjection;
export type UnfusedClass = new (gate: RowProjection, up: RowProjection, down: RowProjection) => RowProjection;
export type FusedClass = new (gate: TrellisLinear, up: TrellisLinear, down: RowProjection) => RowProjection;

/** Program/buffer ownership shared by written ANE bindings. Shape lookup only
 * shares resources; the graph supplies the operation and row capacity. */
export class AneWeightShares {
  readonly affine = new Map<string, AneAffineSplit>();
  mlp: AneMlpSplit | undefined;
  readonly owned: { dispose(): void }[] = [];
  projection(linear: QuantizedLinear, Split: typeof AneAffineSplit, fraction: number, rows: number): AneAffineSplit {
    const shape = `${linear.inFeatures}:${linear.outFeatures}`;
    const projection = Split.build(linear, fraction, rows, this.affine.get(shape));
    this.affine.set(shape, projection);
    this.owned.push(projection);
    return projection;
  }
  [Symbol.dispose](): void { for (const resource of this.owned.splice(0)) resource.dispose(); }
}

export interface AneMlpClass {
  build(gate: TrellisLinear, up: TrellisLinear, down: TrellisLinear,
    fraction: number, rows: number, share?: AneMlpSplit): AneMlpSplit;
}
export class AneBoundMlp {
  readonly split: AneMlpSplit;
  constructor(trellis: TrellisWeights, Split: AneMlpClass, fraction: number, rows: number, shares: AneWeightShares) {
    this.split = Split.build(trellis.gate, trellis.up, trellis.down, fraction, rows, shares.mlp);
    shares.mlp = this.split;
    shares.owned.push(this.split);
  }
  forward(x: MlxArray): MlxArray { return this.split.forward(x, compiledSwiglu); }
}

export class TrellisWeights {
  readonly gate: TrellisLinear;
  readonly up: TrellisLinear;
  readonly down: TrellisLinear;
  constructor(weights: Weights, config: ModelConfig, prefix: string, representations: TrellisRepresentations) {
    this.gate = TrellisLinear.load(weights, `${prefix}.mlp.gate_proj`, config, false, representations);
    this.up = TrellisLinear.load(weights, `${prefix}.mlp.up_proj`, config, false, representations);
    this.down = TrellisLinear.load(weights, `${prefix}.mlp.down_proj`, config, true, representations);
  }
}

/** A gate/up operation already includes the activation; its down projection is
 * another operation the graph names. Both borrow their input. */
export class GateDownMLP {
  constructor(readonly gateUp: RowProjection, readonly down: RowProjection) {}
  forward(x: MlxArray): MlxArray {
    using hidden = this.gateUp.forward(x);
    return this.down.forward(hidden);
  }
}

export class DeltaWeights {
  readonly trellis: TrellisWeights;
  readonly inProjQkv: QuantizedLinear;
  readonly inProjZ: QuantizedLinear;
  readonly inProjA: QuantizedLinear;
  readonly inProjB: QuantizedLinear;
  readonly outProj: QuantizedLinear;
  constructor(weights: Weights, config: ModelConfig, readonly prefix: string, representations: TrellisRepresentations) {
    this.trellis = new TrellisWeights(weights, config, prefix, representations);
    const block = `${prefix}.linear_attn`;
    this.inProjQkv = QuantizedLinear.load(weights, `${block}.in_proj_qkv`, config);
    this.inProjZ = QuantizedLinear.load(weights, `${block}.in_proj_z`, config);
    this.inProjA = QuantizedLinear.load(weights, `${block}.in_proj_a`, config);
    this.inProjB = QuantizedLinear.load(weights, `${block}.in_proj_b`, config);
    this.outProj = QuantizedLinear.load(weights, `${block}.out_proj`, config);
  }
  stage(weights: Weights, config: ModelConfig, Layer: typeof QwenDeltaLayer,
    Mlp: GateDownClass, GateUp: GateUpClass, Down: TrellisProjectionClass,
    Qkv: AffineProjectionClass, Z: AffineProjectionClass, A: AffineProjectionClass,
    B: AffineProjectionClass, Out: AffineProjectionClass, qkScale: { q: MlxArray; k: MlxArray }): QwenDeltaLayer {
    return new Layer(weights, config, this.prefix, this.bind(weights, config, Qkv, Z, A, B, Out, qkScale),
      new Mlp(new GateUp(this.trellis.gate, this.trellis.up), new Down(this.trellis.down)));
  }
  prompt(weights: Weights, config: ModelConfig, Layer: typeof QwenDeltaLayer,
    Mlp: UnfusedClass, Gate: TrellisProjectionClass, Down: TrellisProjectionClass,
    Plain: AffineProjectionClass, qkScale: { q: MlxArray; k: MlxArray }): QwenDeltaLayer {
    return new Layer(weights, config, this.prefix, this.bind(weights, config, Plain, Plain, Plain, Plain, Plain, qkScale),
      new Mlp(new Gate(this.trellis.gate), new Gate(this.trellis.up), new Down(this.trellis.down)));
  }
  fused(weights: Weights, config: ModelConfig, Layer: typeof QwenDeltaLayer,
    Mlp: FusedClass, Down: TrellisProjectionClass, Plain: AffineProjectionClass,
    qkScale: { q: MlxArray; k: MlxArray }): QwenDeltaLayer {
    return new Layer(weights, config, this.prefix, this.bind(weights, config, Plain, Plain, Plain, Plain, Plain, qkScale),
      new Mlp(this.trellis.gate, this.trellis.up, new Down(this.trellis.down)));
  }
  withMlp(weights: Weights, config: ModelConfig, Layer: typeof QwenDeltaLayer,
    mlp: RowProjection, Projection: AffineProjectionClass, qkScale: { q: MlxArray; k: MlxArray }): QwenDeltaLayer {
    return new Layer(weights, config, this.prefix, this.bind(weights, config, Projection, Projection, Projection, Projection, Projection, qkScale), mlp);
  }
  aneStage(weights: Weights, config: ModelConfig, Layer: typeof QwenDeltaLayer,
    Mlp: typeof AneBoundMlp, Split: AneMlpClass, Affine: typeof AneAffineSplit,
    Plain: AffineProjectionClass, fraction: number, rows: number, shares: AneWeightShares,
    qkScale: { q: MlxArray; k: MlxArray }): QwenDeltaLayer {
    const mlp = new Mlp(this.trellis, Split, fraction, rows, shares);
    return new Layer(weights, config, this.prefix, this.ane(weights, config, Plain, Affine, fraction, rows, shares, qkScale), mlp);
  }
  bind(weights: Weights, config: ModelConfig,
    Qkv: AffineProjectionClass, Z: AffineProjectionClass, A: AffineProjectionClass,
    B: AffineProjectionClass, Out: AffineProjectionClass,
    qkScale: { q: MlxArray; k: MlxArray }): QwenGatedDeltaNet<RowProjection> {
    const block = new QwenGatedDeltaNet(weights, config, `${this.prefix}.linear_attn`, {
      inProjQkv: new Qkv(this.inProjQkv), inProjZ: new Z(this.inProjZ),
      inProjA: new A(this.inProjA), inProjB: new B(this.inProjB), outProj: new Out(this.outProj),
    });
    block.qkScale = qkScale;
    return block;
  }
  ane(weights: Weights, config: ModelConfig, Plain: AffineProjectionClass,
    Split: typeof AneAffineSplit, fraction: number, rows: number, shares: AneWeightShares,
    qkScale: { q: MlxArray; k: MlxArray }): QwenGatedDeltaNet<RowProjection> {
    const block = new QwenGatedDeltaNet(weights, config, `${this.prefix}.linear_attn`, {
      inProjQkv: new Plain(this.inProjQkv), inProjZ: shares.projection(this.inProjZ, Split, fraction, rows),
      inProjA: new Plain(this.inProjA), inProjB: new Plain(this.inProjB),
      outProj: shares.projection(this.outProj, Split, fraction, rows),
    });
    block.qkScale = qkScale;
    return block;
  }
  targets(): readonly (readonly [string, QuantizedLinear])[] {
    const prefix = `${this.prefix}.linear_attn`;
    return [[`${prefix}.in_proj_qkv`, this.inProjQkv], [`${prefix}.in_proj_z`, this.inProjZ],
      [`${prefix}.in_proj_a`, this.inProjA], [`${prefix}.in_proj_b`, this.inProjB], [`${prefix}.out_proj`, this.outProj]];
  }
}

export class AttentionWeights {
  readonly trellis: TrellisWeights;
  readonly qProj: QuantizedLinear;
  readonly kProj: QuantizedLinear;
  readonly vProj: QuantizedLinear;
  readonly oProj: QuantizedLinear;
  constructor(weights: Weights, config: ModelConfig, readonly prefix: string, representations: TrellisRepresentations) {
    this.trellis = new TrellisWeights(weights, config, prefix, representations);
    const block = `${prefix}.self_attn`;
    this.qProj = QuantizedLinear.load(weights, `${block}.q_proj`, config);
    this.kProj = QuantizedLinear.load(weights, `${block}.k_proj`, config);
    this.vProj = QuantizedLinear.load(weights, `${block}.v_proj`, config);
    this.oProj = QuantizedLinear.load(weights, `${block}.o_proj`, config);
  }
  stage(weights: Weights, config: ModelConfig, Layer: typeof QwenAttentionLayer,
    Mlp: GateDownClass, GateUp: GateUpClass, Down: TrellisProjectionClass,
    Q: AffineProjectionClass, K: AffineProjectionClass, V: AffineProjectionClass, O: AffineProjectionClass): QwenAttentionLayer {
    return new Layer(weights, config, this.prefix, this.text(weights, config, Q, K, V, O),
      new Mlp(new GateUp(this.trellis.gate, this.trellis.up), new Down(this.trellis.down)));
  }
  prompt(weights: Weights, config: ModelConfig, Layer: typeof QwenAttentionLayer,
    Mlp: UnfusedClass, Gate: TrellisProjectionClass, Down: TrellisProjectionClass,
    Plain: AffineProjectionClass): QwenAttentionLayer {
    return new Layer(weights, config, this.prefix, this.text(weights, config, Plain, Plain, Plain, Plain),
      new Mlp(new Gate(this.trellis.gate), new Gate(this.trellis.up), new Down(this.trellis.down)));
  }
  fused(weights: Weights, config: ModelConfig, Layer: typeof QwenAttentionLayer,
    Mlp: FusedClass, Down: TrellisProjectionClass, Plain: AffineProjectionClass): QwenAttentionLayer {
    return new Layer(weights, config, this.prefix, this.text(weights, config, Plain, Plain, Plain, Plain),
      new Mlp(this.trellis.gate, this.trellis.up, new Down(this.trellis.down)));
  }
  withMlp(weights: Weights, config: ModelConfig, Layer: typeof QwenAttentionLayer,
    mlp: RowProjection, Projection: AffineProjectionClass): QwenAttentionLayer {
    return new Layer(weights, config, this.prefix, this.text(weights, config, Projection, Projection, Projection, Projection), mlp);
  }
  mediaLayer(weights: Weights, config: ModelConfig, Layer: typeof QwenAttentionLayer,
    mlp: RowProjection, Projection: AffineProjectionClass): QwenAttentionLayer {
    return new Layer(weights, config, this.prefix, this.media(weights, config, Projection, Projection, Projection, Projection), mlp);
  }
  aneStage(weights: Weights, config: ModelConfig, Layer: typeof QwenAttentionLayer,
    Mlp: typeof AneBoundMlp, Split: AneMlpClass, Affine: typeof AneAffineSplit,
    Plain: AffineProjectionClass, fraction: number, rows: number, shares: AneWeightShares): QwenAttentionLayer {
    const mlp = new Mlp(this.trellis, Split, fraction, rows, shares);
    return new Layer(weights, config, this.prefix, this.ane(weights, config, Plain, Affine, fraction, rows, shares), mlp);
  }
  text(weights: Weights, config: ModelConfig, Q: AffineProjectionClass, K: AffineProjectionClass,
    V: AffineProjectionClass, O: AffineProjectionClass): Qwen3TextAttention<RowProjection> {
    return new Qwen3TextAttention(weights, config, `${this.prefix}.self_attn`, {
      qProj: new Q(this.qProj), kProj: new K(this.kProj), vProj: new V(this.vProj), oProj: new O(this.oProj),
    });
  }
  media(weights: Weights, config: ModelConfig, Q: AffineProjectionClass, K: AffineProjectionClass,
    V: AffineProjectionClass, O: AffineProjectionClass): Qwen3MediaAttention<RowProjection> {
    return new Qwen3MediaAttention(weights, config, `${this.prefix}.self_attn`, {
      qProj: new Q(this.qProj), kProj: new K(this.kProj), vProj: new V(this.vProj), oProj: new O(this.oProj),
    });
  }
  ane(weights: Weights, config: ModelConfig, Plain: AffineProjectionClass,
    Split: typeof AneAffineSplit, fraction: number, rows: number, shares: AneWeightShares): Qwen3TextAttention<RowProjection> {
    return new Qwen3TextAttention(weights, config, `${this.prefix}.self_attn`, {
      qProj: new Plain(this.qProj), kProj: new Plain(this.kProj), vProj: new Plain(this.vProj),
      oProj: shares.projection(this.oProj, Split, fraction, rows),
    });
  }
  targets(): readonly (readonly [string, QuantizedLinear])[] {
    const prefix = `${this.prefix}.self_attn`;
    return [[`${prefix}.q_proj`, this.qProj], [`${prefix}.k_proj`, this.kProj],
      [`${prefix}.v_proj`, this.vProj], [`${prefix}.o_proj`, this.oProj]];
  }
}
