// Factored-scale and simdgroup-matrix Trellis kernels on synthetic codes
// against main's packed kernels at variant 6 (f32 code·scale weights, the same
// decoded values). The new kernels apply the scale after the reduction (or to
// the activation) and the matrix kernels reorder the sum, so they agree to bf16
// precision, not bit for bit. The matrix kernels that read code values from a
// threadgroup codebook (gate/up for 1..8 rows, down for any number) give the
// 1..8-row matrix kernels' bits.
import { describe, expect, test } from "bun:test";
import { Dtype, MlxArray, ops } from "@mlx-bun/mlx";
import {
  downFactoredRow, downFactoredRows, downK3InterleavedFactoredRow, downK3InterleavedFactoredRows,
  DOWN_ONE_BLOCK, downK3InterleavedMma, downK3InterleavedMmaWide, downK3InterleavedMmaWidePass, downMma, downMmaWide,
  downMmaWidePass,
  fusedGateUpSwiglu, fusedGateUpSwigluMixed, gateUpCodebookMma, gateUpFactoredRow, gateUpFactoredRows, gateUpMma,
  gateUpMmaWide, mixedGateUpFactoredRows, trellisScatter, type TrellisGeometry,
} from "@mlx-bun/inference/kernels/trellis";

// The #311 representations apply only to variant 13; main's variant 6 reference takes none.
const noScatterRepresentation = { bits: false, genericBits: false, floatCodebook: false };

function weights(k: number, axis: 0 | 1 = 1, interleave = false, seedBase = 31) {
  const rows = 64, cols = 512, words = cols * k / 32;
  let seed = seedBase + k;
  const data = Int32Array.from({ length: rows * words }, () => seed = Math.imul(seed, 1664525) + 1013904223);
  using ints = MlxArray.fromInt32(data, interleave ? [1, rows, 48] : [rows, words]);
  const codes = ints.astype(Dtype.uint32);
  using scale32 = MlxArray.fromFloat32(Float32Array.from({ length: rows }, (_, i) => (i % 17 + 1) / 200), [rows]);
  const scales = scale32.astype(Dtype.float16);
  const geometry: TrellisGeometry = { k, L: 12, T: 256, axis, rows, cols,
    inFeatures: axis === 1 ? cols : rows, outFeatures: axis === 1 ? rows : cols,
    ...(interleave ? { blockInterleave: 2 as const } : {}) };
  return { codes, scales, geometry, [Symbol.dispose]() { codes.dispose(); scales.dispose(); } };
}

function input(m: number, k: number, batched = false) {
  using raw = MlxArray.fromFloat32(Float32Array.from({ length: m * k }, (_, i) => ((i * 73 + 19) % 257 - 128) / 31), batched ? [1, m, k] : [m, k]);
  return raw.astype(Dtype.bfloat16);
}

/** Whether rows [lo, hi) of `actual` (rows on axis `axis`) equal `expected` byte for byte. */
function sameRows(actual: MlxArray, expected: MlxArray, lo: number, hi: number, axis: 0 | 1): boolean {
  const start = actual.shape.map(() => 0), stop = [...actual.shape];
  start[axis] = lo; stop[axis] = hi;
  using part = actual.slice(start, stop);
  using same = ops.equal(part, expected);
  using all = ops.allReduce(same);
  return all.toFloat32()[0] === 1;
}

/** max |a - b| relative to max |b| (both cast to f32). */
function relErr(actual: MlxArray, expected: MlxArray): number {
  expect(actual.shape).toEqual(expected.shape);
  const a = actual.astype(Dtype.float32).toFloat32(), b = expected.astype(Dtype.float32).toFloat32();
  let err = 0, mag = 0;
  for (let i = 0; i < b.length; i++) { err = Math.max(err, Math.abs(a[i]! - b[i]!)); mag = Math.max(mag, Math.abs(b[i]!)); }
  return err / mag;
}

/** Main's packed gate/up (variant 6) over row groups of at most 4. */
function gateUpReference(x: MlxArray, gate: ReturnType<typeof weights>, up: ReturnType<typeof weights>): MlxArray {
  const m = x.shape[1]!, pieces = [] as MlxArray[];
  for (let lo = 0; lo < m; lo += 4) {
    using part = x.slice([0, lo, 0], [1, Math.min(m, lo + 4), x.shape[2]!]);
    pieces.push(gate.geometry.k === up.geometry.k ? fusedGateUpSwiglu(part, gate, up, 6, false)
      : fusedGateUpSwigluMixed(part, gate, up, 6, "split"));
  }
  try { return ops.concatAxis(pieces, 1); } finally { for (const p of pieces) p.dispose(); }
}

/** Main's packed scatter (variant 6) over row groups of at most 4. */
function downReference(x: MlxArray, w: ReturnType<typeof weights>): MlxArray {
  const m = x.shape[0]!, pieces = [] as MlxArray[];
  for (let lo = 0; lo < m; lo += 4) {
    using part = x.slice([lo, 0], [Math.min(m, lo + 4), x.shape[1]!]);
    pieces.push(trellisScatter(part, w.codes, w.scales, w.geometry, 6, false, noScatterRepresentation));
  }
  try { return ops.concatAxis(pieces, 0); } finally { for (const p of pieces) p.dispose(); }
}

describe("factored gate/up", () => {
  test("same width: one row and 2..4 shared rows", () => {
    for (const k of [2, 3, 4]) {
      using gate = weights(k), up = weights(k, 1, false, 97);
      for (const m of [1, 2, 3, 4]) {
        using x = input(m, gate.geometry.inFeatures, true);
        using expected = gateUpReference(x, gate, up);
        using actual = m === 1 ? gateUpFactoredRow(x, gate, up) : gateUpFactoredRows(x, gate, up);
        expect(relErr(actual, expected), `k${k} m${m}`).toBeLessThan(2e-2);
      }
    }
  }, 60_000);

  test("mixed widths: 1..4 rows", () => {
    for (const [kg, ku] of [[2, 3], [3, 2], [3, 4]] as const) {
      using gate = weights(kg), up = weights(ku, 1, false, 97);
      for (const m of [1, 2, 3, 4]) {
        using x = input(m, gate.geometry.inFeatures, true);
        using expected = gateUpReference(x, gate, up);
        using actual = mixedGateUpFactoredRows(x, gate, up);
        expect(relErr(actual, expected), `k${kg}/${ku} m${m}`).toBeLessThan(2e-2);
      }
    }
  }, 60_000);
});

describe("factored down projection", () => {
  test("row-major codes: one row and 2..4 rows", () => {
    for (const k of [2, 3, 4]) {
      using w = weights(k, 0);
      for (const m of [1, 2, 3, 4]) {
        using x = input(m, w.geometry.inFeatures);
        using expected = downReference(x, w);
        using actual = m === 1 ? downFactoredRow(x, w.codes, w.scales, w.geometry) : downFactoredRows(x, w.codes, w.scales, w.geometry);
        expect(relErr(actual, expected), `k${k} m${m}`).toBeLessThan(2e-2);
      }
    }
  }, 60_000);

  test("3-bit block-interleaved codes: one row and 2..4 rows", () => {
    using w = weights(3, 0, true);
    for (const m of [1, 2, 3, 4]) {
      using x = input(m, w.geometry.inFeatures);
      using expected = downReference(x, w);
      using actual = m === 1 ? downK3InterleavedFactoredRow(x, w.codes, w.scales, w.geometry)
        : downK3InterleavedFactoredRows(x, w.codes, w.scales, w.geometry);
      expect(relErr(actual, expected), `m${m}`).toBeLessThan(2e-2);
    }
  }, 60_000);
});

describe("simdgroup-matrix kernels", () => {
  test("gate/up for 1..8 rows, same and mixed widths", () => {
    for (const [kg, ku] of [[2, 2], [3, 3], [4, 4], [2, 3]] as const) {
      using gate = weights(kg), up = weights(ku, 1, false, 97);
      for (const m of [1, 2, 4, 5, 8]) {
        using x = input(m, gate.geometry.inFeatures, true);
        using expected = gateUpReference(x, gate, up);
        using actual = gateUpMma(x, gate, up);
        expect(relErr(actual, expected), `k${kg}/${ku} m${m}`).toBeLessThan(2e-2);
      }
    }
  }, 60_000);

  test("down projection for 1..8 rows, both code layouts", () => {
    for (const [k, interleave] of [[2, false], [3, false], [4, false], [3, true]] as const) {
      using w = weights(k, 0, interleave);
      for (const m of [1, 3, 4, 8]) {
        using x = input(m, w.geometry.inFeatures);
        using expected = downReference(x, w);
        using actual = (interleave ? downK3InterleavedMma : downMma)(x, w.codes, w.scales, w.geometry);
        expect(relErr(actual, expected), `k${k}${interleave ? "i" : ""} m${m}`).toBeLessThan(2e-2);
      }
    }
  }, 60_000);
});

describe("simdgroup-matrix kernels at any number of rows", () => {
  test("gate/up: the reference, and each block of eight rows the 1..8-row kernel's bit for bit", () => {
    for (const [kg, ku] of [[2, 2], [3, 3], [4, 4], [2, 3]] as const) {
      using gate = weights(kg), up = weights(ku, 1, false, 97);
      for (const m of [1, 8, 9, 16, 21, 24, 40]) {
        using x = input(m, gate.geometry.inFeatures, true);
        using expected = gateUpReference(x, gate, up);
        using actual = gateUpMmaWide(x, gate, up);
        expect(relErr(actual, expected), `k${kg}/${ku} m${m}`).toBeLessThan(2e-2);
        for (let lo = 0; lo < m; lo += 8) {
          using block = x.slice([0, lo, 0], [1, Math.min(m, lo + 8), x.shape[2]!]);
          using alone = gateUpMma(block, gate, up);
          expect(sameRows(actual, alone, lo, Math.min(m, lo + 8), 1), `k${kg}/${ku} m${m} rows ${lo}+`).toBe(true);
        }
      }
    }
  }, 60_000);

  test("down projection, both code layouts: the reference, and each block of eight rows the 1..8-row kernel's bit for bit", () => {
    for (const [k, interleave] of [[2, false], [3, false], [4, false], [3, true]] as const) {
      using w = weights(k, 0, interleave);
      for (const m of [1, 2, 3, 4, 5, 8, 9, 16, 21, 24, 40]) {
        using x = input(m, w.geometry.inFeatures);
        using expected = downReference(x, w);
        using actual = (interleave ? downK3InterleavedMmaWide : downMmaWide)(x, w.codes, w.scales, w.geometry);
        expect(relErr(actual, expected), `k${k}${interleave ? "i" : ""} m${m}`).toBeLessThan(2e-2);
        for (let lo = 0; lo < m; lo += 8) {
          using block = x.slice([lo, 0], [Math.min(m, lo + 8), x.shape[1]!]);
          using alone = (interleave ? downK3InterleavedMma : downMma)(block, w.codes, w.scales, w.geometry);
          expect(sameRows(actual, alone, lo, Math.min(m, lo + 8), 0), `k${k}${interleave ? "i" : ""} m${m} rows ${lo}+`).toBe(true);
        }
      }
    }
  }, 60_000);
});

describe("threadgroup-codebook simdgroup-matrix gate/up", () => {
  test("gate/up for 1..8 rows: the reference, and the 1..8-row matrix kernel's bits", () => {
    for (const [kg, ku] of [[2, 2], [3, 3], [4, 4], [2, 3], [4, 3]] as const) {
      using gate = weights(kg), up = weights(ku, 1, false, 97);
      for (const m of [1, 2, 3, 4, 5, 8]) {
        using x = input(m, gate.geometry.inFeatures, true);
        using expected = gateUpReference(x, gate, up);
        using actual = gateUpCodebookMma(x, gate, up);
        expect(relErr(actual, expected), `k${kg}/${ku} m${m}`).toBeLessThan(2e-2);
        using alone = gateUpMma(x, gate, up);
        expect(sameRows(actual, alone, 0, m, 1), `k${kg}/${ku} m${m} bits`).toBe(true);
      }
    }
  }, 60_000);

});

describe("threadgroup-codebook any-row down projection in a fixed pass", () => {
  test("one block per pass, and two and three for row-major codes: each block of eight rows the 1..8-row kernel's bits", () => {
    for (const [k, interleave] of [[2, false], [3, false], [4, false], [3, true]] as const) {
      using w = weights(k, 0, interleave);
      const passes = interleave ? [DOWN_ONE_BLOCK] : [DOWN_ONE_BLOCK, { S: 8, MT: 2 }, { S: 8, MT: 3 }];
      for (const pass of passes) for (const m of [1, 2, 3, 5, 8, 12]) {
        using x = input(m, w.geometry.inFeatures);
        using actual = interleave ? downK3InterleavedMmaWidePass(x, w.codes, w.scales, w.geometry, pass)
          : downMmaWidePass(x, w.codes, w.scales, w.geometry, pass);
        for (let lo = 0; lo < m; lo += 8) {
          using block = x.slice([lo, 0], [Math.min(m, lo + 8), x.shape[1]!]);
          using alone = (interleave ? downK3InterleavedMma : downMma)(block, w.codes, w.scales, w.geometry);
          expect(sameRows(actual, alone, lo, Math.min(m, lo + 8), 0), `k${k}${interleave ? "i" : ""} S${pass.S} MT${pass.MT} m${m} rows ${lo}+`).toBe(true);
        }
      }
    }
  }, 60_000);
});
