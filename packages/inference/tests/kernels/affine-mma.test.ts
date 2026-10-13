// 3- and 4-bit verify-width affine kernels (simdgroup-matrix, split
// simdgroup-matrix and shared multi-row) against MLX's quantized_matmul over
// the same group-64 weights. All reorder the reduction relative to MLX, so they
// agree to bf16 precision.
import { expect, test } from "bun:test";
import { Dtype, MlxArray, ops } from "@mlx-bun/mlx";
import { affineSplitMma, affineSplitMmaEligible, affineSplitMmaPlan } from "../../src/kernels/quantization/affine-split-mma";
import { affine3Mma, affine3MmaEligible } from "../../src/kernels/quantization/affine3-mma";
import { affine3Rows, affine3RowsEligible } from "../../src/kernels/quantization/affine3-rows";
import { affine4Mma, affine4MmaEligible } from "../../src/kernels/quantization/affine4-mma";
import { affine4Rows, affine4RowsEligible } from "../../src/kernels/quantization/affine4-rows";

const KERNELS = {
  3: { mma: affine3Mma, mmaEligible: affine3MmaEligible, rows: affine3Rows, rowsEligible: affine3RowsEligible },
  4: { mma: affine4Mma, mmaEligible: affine4MmaEligible, rows: affine4Rows, rowsEligible: affine4RowsEligible },
} as const;

function relErr(actual: MlxArray, expected: MlxArray): number {
  expect(actual.shape).toEqual(expected.shape);
  const a = actual.astype(Dtype.float32).toFloat32(), b = expected.astype(Dtype.float32).toFloat32();
  let err = 0, mag = 0;
  for (let i = 0; i < b.length; i++) { err = Math.max(err, Math.abs(a[i]! - b[i]!)); mag = Math.max(mag, Math.abs(b[i]!)); }
  return err / mag;
}

test("3- and 4-bit affine MMA and multi-row kernels match quantized_matmul at verify widths", () => {
  const K = 1024, N = 128;
  using w32 = MlxArray.fromFloat32(Float32Array.from({ length: N * K }, (_, i) => Math.sin(i * 0.137) * 0.05), [N, K]);
  using wb = w32.astype(Dtype.bfloat16);
  for (const bits of [3, 4] as const) {
  const kernel = KERNELS[bits];
  const q = ops.quantize(wb, 64, bits), spec = { bits, groupSize: 64, mode: "affine" };
  try {
    for (const m of [1, 2, 3, 5, 8]) {
      using x32 = MlxArray.fromFloat32(Float32Array.from({ length: m * K }, (_, i) => ((i * 73 + 19) % 257 - 128) / 64), [m, K]);
      using x = x32.astype(Dtype.bfloat16);
      using expected = ops.quantizedMatmul(x, q.packed, q.scales, q.biases, spec, true);
      expect(kernel.mmaEligible(m, q.packed, q.scales, spec)).toBe(true);
      using mma = kernel.mma(x, q.packed, q.scales, q.biases!);
      expect(relErr(mma, expected), `mma ${bits}-bit m${m}`).toBeLessThan(1e-2);
      if (m >= 2) {
        expect(kernel.rowsEligible(m, q.packed, q.scales, spec)).toBe(true);
        using rows = kernel.rows(x, q.packed, q.scales, q.biases!);
        expect(relErr(rows, expected), `rows ${bits}-bit m${m}`).toBeLessThan(1e-2);
      }
    }
  } finally { q.packed.dispose(); q.scales.dispose(); q.biases?.dispose(); }
  }
}, 60_000);

test("3- and 4-bit split simdgroup-matrix kernel matches quantized_matmul", () => {
  // N 128 takes the narrow launch (one tile over eight K splits, more splits than K steps), N 2048 the wide one.
  for (const [K, N] of [[1024, 128], [1536, 128], [1024, 2048]] as const) {
    using w32 = MlxArray.fromFloat32(Float32Array.from({ length: N * K }, (_, i) => Math.sin(i * 0.137) * 0.05), [N, K]);
    using wb = w32.astype(Dtype.bfloat16);
    const plan = affineSplitMmaPlan(N);
    expect(plan).toEqual(N === 128 ? { tiles: 1, splits: 8 } : { tiles: 2, splits: 2 });
    for (const bits of [3, 4] as const) {
      const q = ops.quantize(wb, 64, bits), spec = { bits, groupSize: 64, mode: "affine" };
      try {
        for (const m of [1, 2, 3, 4, 5, 8]) {
          using x32 = MlxArray.fromFloat32(Float32Array.from({ length: m * K }, (_, i) => ((i * 73 + 19) % 257 - 128) / 64), [1, m, K]);
          using x = x32.astype(Dtype.bfloat16);
          using expected = ops.quantizedMatmul(x, q.packed, q.scales, q.biases, spec, true);
          expect(affineSplitMmaEligible(m, q.packed, q.scales, spec)).toBe(true);
          using split = affineSplitMma(x, q.packed, q.scales, q.biases!, bits, plan);
          expect(relErr(split, expected), `split mma ${bits}-bit K${K} N${N} m${m}`).toBeLessThan(1e-2);
        }
      } finally { q.packed.dispose(); q.scales.dispose(); q.biases?.dispose(); }
    }
  }
}, 60_000);
