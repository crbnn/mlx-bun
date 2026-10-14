// The prompt-piece projection (layers/dequantized-linear) on synthetic 3- and 4-bit group-64 affine weights at the
// Qwen3.8 projection shapes against the QuantizedLinear it wraps: below a projection's fewest rows (DEQUANTIZED_MIN_ROWS,
// or above MLX's split-K GEMM range: 513 for k_proj and v_proj's 1024 outputs), and at every row count for outputs that
// are not whole 64-wide tiles (in_proj_a and in_proj_b), it is MLX's quantized matmul itself; from there the
// dequantized bf16 weight and MLX's dense GEMM, which on the M1 Max (where the layer was measured and is selected)
// return the quantized matmul bit for bit, and elsewhere agree to bf16 precision.
import { expect, test } from "bun:test";
import { Dtype, MlxArray, ops } from "@mlx-bun/mlx";
import { deviceArchitecture } from "@mlx-bun/mlx/ffi";
import { DEQUANTIZED_MIN_ROWS, DequantizedLinear, dequantizedMinRows } from "../../src/layers/dequantized-linear";
import { QuantizedLinear } from "../../src/layers/quantized-linear";

function rows(m: number, width: number): MlxArray {
  using raw = MlxArray.fromFloat32(Float32Array.from({ length: m * width }, (_, i) => Math.sin(i * 0.613 + m) * 0.5), [1, m, width]);
  return raw.astype(Dtype.bfloat16);
}
function affine(out: number, inp: number, bits: number): QuantizedLinear {
  using w32 = MlxArray.fromFloat32(Float32Array.from({ length: out * inp }, (_, i) => Math.sin(i * 0.137 + bits) * 0.05), [out, inp]);
  using wb = w32.astype(Dtype.bfloat16);
  const q = ops.quantize(wb, 64, bits);
  return new QuantizedLinear(q.packed, q.scales, q.biases, { bits, groupSize: 64, mode: "affine" });
}
/** Elements that differ, and the largest difference relative to the largest reference magnitude. */
function compare(actual: MlxArray, expected: MlxArray): { differ: number; rel: number } {
  expect(actual.shape).toEqual(expected.shape);
  using a32 = actual.astype(Dtype.float32), b32 = expected.astype(Dtype.float32);
  const a = a32.toFloat32(), b = b32.toFloat32();
  let differ = 0, err = 0, mag = 0;
  for (let i = 0; i < b.length; i++) {
    if (a[i] !== b[i]) differ++;
    err = Math.max(err, Math.abs(a[i]! - b[i]!)); mag = Math.max(mag, Math.abs(b[i]!));
  }
  return { differ, rel: err / mag };
}

test("the fewest rows: DEQUANTIZED_MIN_ROWS, above the split-K GEMM's range at narrow outputs, never off 64-wide tiles", () => {
  expect([5120, 6144, 10240, 12288].map(dequantizedMinRows)).toEqual(Array(4).fill(DEQUANTIZED_MIN_ROWS));
  expect(dequantizedMinRows(1024)).toBe(513);
  expect(dequantizedMinRows(48)).toBe(Infinity);
});

test("the quantized matmul's output below the threshold, and from it the dense GEMM's, which is the same on the M1 Max", () => {
  const measured = deviceArchitecture() === "applegpu_g13s";
  // in_proj_qkv, in_proj_z, out_proj and o_proj, q_proj, k_proj and v_proj, in_proj_a and in_proj_b.
  const shapes: readonly (readonly [number, number, number])[] =
    [[3, 5120, 10240], [3, 5120, 6144], [4, 6144, 5120], [4, 5120, 12288], [4, 5120, 1024], [4, 5120, 48]];
  for (const [bits, inputs, outputs] of shapes) {
    const linear = affine(outputs, inputs, bits), layer = new DequantizedLinear(linear);
    for (const m of [9, DEQUANTIZED_MIN_ROWS - 1, DEQUANTIZED_MIN_ROWS, 257, 512, 513, 1023, 2048]) {
      using x = rows(m, inputs);
      using expected = linear.forward(x), actual = layer.forward(x);
      const { differ, rel } = compare(actual, expected);
      const label = `${bits}-bit ${inputs}→${outputs}, ${m} rows`;
      if (m < dequantizedMinRows(outputs) || measured) expect(differ, label).toBe(0);
      else expect(rel, label).toBeLessThan(1e-2);
    }
  }
}, 120_000);

test("a projection it does not serve is refused", () => {
  const linear = affine(256, 512, 4);
  expect(() => new DequantizedLinear(new QuantizedLinear(linear.w, linear.scales, null, linear.spec))).toThrow(/unsupported/);
  using bias = ops.zeros([256], Dtype.bfloat16);
  expect(() => new DequantizedLinear(new QuantizedLinear(linear.w, linear.scales, linear.biases, linear.spec, bias))).toThrow(/unsupported/);
});
