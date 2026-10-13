import type { MlxArray } from "@mlx-bun/mlx/array";
import { gateUpReduceRows } from "../kernels/trellis/gate-up";
import { gateUpCodebookRow } from "../kernels/trellis/gate-up-codebook";
import { fusedGateUpSwigluMixed, type MixedGateUpTail } from "../kernels/trellis/mixed-gate-up";
import type { TrellisLinear } from "./trellis-linear";

// Fused gate/up/SwiGLU layers over a Qwen MLP's axis-1 Trellis gate and up
// projections on the reduce kernels (one SIMD group per output row, the decoded
// weight times the scale per weight; kernels/trellis/index.ts). Each layer runs
// one kernel for one row form and takes everything that kernel varies on at
// construction. The owning graph picks the layer for the rows of its width
// class and the weights' bit widths.

/** `sameWidth` null: either. */
function sameAxis1(name: string, gate: TrellisLinear, up: TrellisLinear, sameWidth: boolean | null): void {
  const a = gate.geometry, b = up.geometry;
  if (a.axis !== 1 || b.axis !== 1 || a.rows !== b.rows || a.cols !== b.cols ||
      a.T !== b.T || a.L !== b.L || (sameWidth !== null && (a.k === b.k) !== sameWidth))
    throw new Error(`${name}: gate/up need packed axis-1 codes of one geometry${sameWidth === null ? ""
      : `, ${sameWidth ? "the same" : "different"} widths`}`);
  if (gate.variant !== up.variant) throw new Error(`${name}: gate and up decode with different variants`);
}

/** One activation row; gate and up at the same or different widths, decoding through a threadgroup codebook
 *  (kernels/trellis/gate-up-codebook.ts) with fusedGateUpSwiglu's (same widths) and ReduceMixedGateUp's (different
 *  widths) output bits. `mixedTail` is the SwiGLU arithmetic where the widths differ (ReduceMixedGateUp's `tail`);
 *  same widths take the float32 sigmoid. */
export class CodebookGateUpRow {
  constructor(readonly gate: TrellisLinear, readonly up: TrellisLinear, readonly mixedTail: MixedGateUpTail) {
    sameAxis1("CodebookGateUpRow", gate, up, null);
    if (gate.geometry.L > 12) throw new Error("CodebookGateUpRow: the codebook holds at most 2^12 states");
  }

  /** `x` [.., in] (borrowed, one row) to an owned [.., rows]. */
  forward(x: MlxArray): MlxArray { return gateUpCodebookRow(x, this.gate, this.up, this.mixedTail); }
}

/** 2..4 activation rows sharing each decoded weight; gate and up at the same width. */
export class ReduceGateUpRows {
  readonly #variant: number;

  constructor(readonly gate: TrellisLinear, readonly up: TrellisLinear) {
    sameAxis1("ReduceGateUpRows", gate, up, true);
    this.#variant = gate.variant;
  }

  /** `x` [.., in] (borrowed, 2..4 rows) to an owned [.., rows]. */
  forward(x: MlxArray): MlxArray { return gateUpReduceRows(x, this.gate, this.up, this.#variant); }
}

/** 1..4 activation rows; gate and up at different widths. `tail` is the SwiGLU arithmetic the kernel reproduces
 *  (`"split"`: MLX's compiled swiglu over the two bf16 projections, bit for bit; `"fused"`: the float32 sigmoid of the
 *  same-width kernel). */
export class ReduceMixedGateUp {
  readonly #variant: number;

  constructor(readonly gate: TrellisLinear, readonly up: TrellisLinear, readonly tail: MixedGateUpTail) {
    sameAxis1("ReduceMixedGateUp", gate, up, false);
    this.#variant = gate.variant;
  }

  /** `x` [.., in] (borrowed, 1..4 rows) to an owned [.., rows]. */
  forward(x: MlxArray): MlxArray { return fusedGateUpSwigluMixed(x, this.gate, this.up, this.#variant, this.tail); }
}
