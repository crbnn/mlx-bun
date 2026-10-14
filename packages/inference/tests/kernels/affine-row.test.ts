// The one-row affine kernel (3- and 4-bit, group 64) against MLX's quantized_matmul at one row over the same
// weights: it keeps MLX's qmv arithmetic, so the outputs are equal bit for bit, for activations of several
// magnitudes (and zeros), at every K and N the kernel takes.
import { expect, test } from "bun:test";
import { Dtype, MlxArray, ops } from "@mlx-bun/mlx";
import { affineRow, affineRowEligible } from "../../src/kernels/quantization/affine-row";

function sameBits(actual: MlxArray, expected: MlxArray): number {
  expect(actual.shape).toEqual(expected.shape);
  expect(actual.dtype).toBe(expected.dtype);
  const a = new Uint16Array(actual.rawBytesView().slice().buffer), b = new Uint16Array(expected.rawBytesView().slice().buffer);
  let differ = 0;
  for (let i = 0; i < b.length; i++) if (a[i] !== b[i]) differ++;
  return differ;
}

test("3- and 4-bit one-row affine kernel equals quantized_matmul bit for bit", () => {
  for (const [K, N] of [[512, 64], [1536, 128], [5120, 192]] as const) {
    using w32 = MlxArray.fromFloat32(Float32Array.from({ length: N * K }, (_, i) => Math.sin(i * 0.137) * 0.05 + Math.cos(i * 0.0071) * 0.02), [N, K]);
    using wb = w32.astype(Dtype.bfloat16);
    for (const bits of [3, 4] as const) {
      const q = ops.quantize(wb, 64, bits), spec = { bits, groupSize: 64, mode: "affine" };
      try {
        expect(affineRowEligible(1, q.packed, q.scales, spec)).toBe(true);
        expect(affineRowEligible(2, q.packed, q.scales, spec)).toBe(false);
        for (const [label, scale, zeroEvery] of [["unit", 1 / 64, 0], ["wide", 37.5, 0], ["small", 1e-3, 0], ["sparse", 1 / 64, 3]] as const) {
          using x32 = MlxArray.fromFloat32(Float32Array.from({ length: K }, (_, i) =>
            zeroEvery && i % zeroEvery !== 0 ? 0 : ((i * 73 + 19) % 257 - 128) * scale * (1 + (i % 7) / 3)), [1, 1, K]);
          using x = x32.astype(Dtype.bfloat16);
          using expected = ops.quantizedMatmul(x, q.packed, q.scales, q.biases, spec, true);
          using actual = affineRow(x, q.packed, q.scales, q.biases!, bits);
          expect(sameBits(actual, expected), `${bits}-bit K${K} N${N} ${label}`).toBe(0);
        }
      } finally { q.packed.dispose(); q.scales.dispose(); q.biases?.dispose(); }
    }
  }
}, 60_000);

test("one-row affine kernel takes only what it was built for", () => {
  using w32 = MlxArray.fromFloat32(Float32Array.from({ length: 64 * 512 }, (_, i) => Math.sin(i * 0.137) * 0.05), [64, 512]);
  using wb = w32.astype(Dtype.bfloat16);
  const q8 = ops.quantize(wb, 64, 8), q32 = ops.quantize(wb, 32, 4);
  try {
    expect(affineRowEligible(1, q8.packed, q8.scales, { bits: 8, groupSize: 64, mode: "affine" })).toBe(false);
    expect(affineRowEligible(1, q32.packed, q32.scales, { bits: 4, groupSize: 32, mode: "affine" })).toBe(false);
  } finally { for (const q of [q8, q32]) { q.packed.dispose(); q.scales.dispose(); q.biases?.dispose(); } }
  using wide32 = MlxArray.fromFloat32(Float32Array.from({ length: 64 * 8192 }, (_, i) => Math.sin(i * 0.137) * 0.05), [64, 8192]);
  using wide = wide32.astype(Dtype.bfloat16);
  const q = ops.quantize(wide, 64, 3);
  try {
    // Its x_thread would not fit threadgroup memory.
    expect(affineRowEligible(1, q.packed, q.scales, { bits: 3, groupSize: 64, mode: "affine" })).toBe(false);
  } finally { q.packed.dispose(); q.scales.dispose(); q.biases?.dispose(); }
});
