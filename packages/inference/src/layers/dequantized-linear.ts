import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { QuantizedLinear } from "./quantized-linear";

// A prompt piece's affine projection over a loaded QuantizedLinear (borrowed
// weights, no additive bias). From DEQUANTIZED_MIN_ROWS rows the weight is
// dequantized to bf16 for the call and multiplied by MLX's dense GEMM: the same
// bf16 weights MLX's quantized matmul builds in its tiles and the same float
// accumulation, so the output is that matmul's bit for bit, from the GEMM's
// larger tiles (measured on the M1 Max at 3 and 4 bits, group 64, 640 to 2048
// rows, every Qwen3.8 projection whose outputs are whole 64-wide tiles:
// bit-identical, 1.27x to 1.38x faster, 7.8 against 5.7 TFLOPS at 2048 rows).
// Fewer rows, and the other projections (DeltaNet's in_proj_a and in_proj_b, 48
// outputs), take MLX's quantized matmul itself, where the two differ in kernel
// choice and bits. The identity is MLX's kernel choice at those shapes, not a
// property of every shape (at 512 inputs the two round differently): a graph
// selects this layer for the artifact it was measured on. The dense weight is a transient of one call (outputs
// x inputs x 2 bytes). Applies no mounted adapters: a graph builds it only where
// the composition mounts none.

/** The fewest rows the dense GEMM takes: below this MLX's quantized matmul selects other kernels (no longer
 *  bit-identical to the GEMM at every projection shape) and the dequantization costs more than it saves. */
export const DEQUANTIZED_MIN_ROWS = 640;

export class DequantizedLinear {
  readonly #spec: ops.QuantSpec;
  /** The fewest rows this projection takes the dense GEMM at: DEQUANTIZED_MIN_ROWS, or never when its outputs are
   *  not whole 64-wide tiles. */
  readonly #minRows: number;

  constructor(readonly linear: QuantizedLinear) {
    if (!linear.biases || linear.bias || linear.w.ndim !== 2 || linear.spec.mode !== "affine")
      throw new Error("DequantizedLinear: unsupported projection (affine, two-dimensional, no additive bias)");
    this.#spec = linear.spec;
    this.#minRows = linear.outFeatures % 64 === 0 ? DEQUANTIZED_MIN_ROWS : Infinity;
  }

  /** `x` [.., inputs] (borrowed) to an owned [.., outputs], lazy. */
  forward(x: MlxArray): MlxArray {
    const { w, scales, biases } = this.linear;
    if (x.size / x.shape.at(-1)! < this.#minRows) return ops.quantizedMatmul(x, w, scales, biases, this.#spec, true);
    using dense = ops.dequantize(w, scales, biases, this.#spec);
    using weightT = ops.transposeAxes(dense, [1, 0]);
    return ops.matmul(x, weightT);
  }
}
