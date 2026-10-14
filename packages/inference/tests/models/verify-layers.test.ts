// Verify-width layers (layers/trellis-gate-up, layers/trellis-down,
// layers/affine-verify-linear) on small synthetic projections against the
// projections they replace: TrellisLinear (+ compiled SwiGLU) and
// QuantizedLinear. Each layer runs one kernel that reassociates the reduction,
// so they agree to bf16 precision. The codebook layers (layers/trellis-reduce-gate-up,
// layers/trellis-scatter-down) and the one-row affine layer reproduce the one-row
// paths they replace bit for bit. Each layer also refuses weights it was not built for.
// TrellisLinear itself, at prompt rows, leaves its weight expansion unbounded on request
// with the same bits.
import { describe, expect, test } from "bun:test";
import { Dtype, MlxArray, ops } from "@mlx-bun/mlx";
import { QuantizedLinear } from "../../src/layers/quantized-linear";
import { TrellisLinear, fusedGateUpSwiglu } from "../../src/layers/trellis-linear";
import { compiledSwiglu } from "../../src/layers/swiglu";
import { fusedGateUpSwigluMixed as mixedGateUpKernel } from "../../src/kernels/trellis/mixed-gate-up";
import { CodebookGateUpRow } from "../../src/layers/trellis-reduce-gate-up";
import { CodebookDownK3iRow, CodebookDownRow } from "../../src/layers/trellis-scatter-down";
import { FactoredGateUpRow, FactoredGateUpRows, FactoredMixedGateUp, MmaGateUp, MmaGateUpWide } from "../../src/layers/trellis-gate-up";
import {
  FactoredDownK3iRow, FactoredDownK3iRows, FactoredDownRow, FactoredDownRows, MmaDown, MmaDownK3i, MmaDownWide, MmaDownWideK3i,
} from "../../src/layers/trellis-down";
import {
  Affine3MmaLinear, Affine3RowsLinear, Affine4MmaLinear, Affine4RowsLinear, AffineRowLinear, AffineSplitMmaLinear,
} from "../../src/layers/affine-verify-linear";

const D = 512, R = 256;

function relErr(actual: MlxArray, expected: MlxArray): number {
  expect(actual.shape).toEqual(expected.shape);
  const a = actual.astype(Dtype.float32).toFloat32(), b = expected.astype(Dtype.float32).toFloat32();
  let err = 0, mag = 0;
  for (let i = 0; i < b.length; i++) { err = Math.max(err, Math.abs(a[i]! - b[i]!)); mag = Math.max(mag, Math.abs(b[i]!)); }
  return err / mag;
}
function rows(m: number, width: number): MlxArray {
  using raw = MlxArray.fromFloat32(Float32Array.from({ length: m * width }, (_, i) => ((i * 73 + 19) % 257 - 128) / 128), [1, m, width]);
  return raw.astype(Dtype.bfloat16);
}
function trellis(stored: number, cols: number, axis: 0 | 1, k: number, seedBase: number, interleave = false): TrellisLinear {
  let seed = seedBase;
  const words = cols * k / 32;
  const shape = interleave ? [cols / 512, stored, 48] : [stored, words];
  using ints = MlxArray.fromInt32(Int32Array.from({ length: stored * words }, () => seed = Math.imul(seed, 1664525) + 1013904223), shape);
  using scale32 = MlxArray.fromFloat32(Float32Array.from({ length: stored }, (_, i) => (i % 17 + 1) / 400), [stored]);
  return new TrellisLinear(ints.astype(Dtype.uint32), scale32.astype(Dtype.float16),
    { bits: k, groupSize: 256, mode: "trellis", trellis: { L: 12, code: "1mad", axis } });
}

describe("Trellis verify layers", () => {
  test("gate/up layers match the projections plus SwiGLU", () => {
    for (const [kg, ku] of [[3, 3], [2, 3]] as const) {
      const gate = trellis(R, D, 1, kg, 11), up = trellis(R, D, 1, ku, 23);
      for (const m of [1, 2, 4, 5, 8]) {
        using x = rows(m, D);
        using g = gate.forward(x, true), u = up.forward(x, true);
        using expected = compiledSwiglu(g, u);
        const layer = m > 4 ? new MmaGateUp(gate, up) : kg !== ku ? new FactoredMixedGateUp(gate, up)
          : m === 1 ? new FactoredGateUpRow(gate, up) : new FactoredGateUpRows(gate, up);
        using actual = layer.forward(x);
        expect(relErr(actual, expected), `k${kg}/${ku} m${m}`).toBeLessThan(2e-2);
      }
      const wide = new MmaGateUpWide(gate, up);
      for (const m of [1, 9, 16, 40]) {
        using x = rows(m, D);
        using g = gate.forward(x, true), u = up.forward(x, true);
        using expected = compiledSwiglu(g, u);
        using actual = wide.forward(x);
        expect(relErr(actual, expected), `wide k${kg}/${ku} m${m}`).toBeLessThan(2e-2);
      }
    }
  }, 60_000);

  test("down layers match the projection for both code layouts", () => {
    for (const interleave of [false, true]) {
      const down = trellis(R, D, 0, interleave ? 3 : 2, 37, interleave);
      for (const m of [1, 3, 4, 8]) {
        using x = rows(m, R);
        using expected = down.forward(x, true);
        const layer = interleave
          ? m === 1 ? new FactoredDownK3iRow(down) : m < 4 ? new FactoredDownK3iRows(down) : new MmaDownK3i(down)
          : m === 1 ? new FactoredDownRow(down) : m < 4 ? new FactoredDownRows(down) : new MmaDown(down);
        using actual = layer.forward(x);
        expect(relErr(actual, expected), `${interleave ? "k3i" : "row-major"} m${m}`).toBeLessThan(2e-2);
      }
      const wide = interleave ? new MmaDownWideK3i(down) : new MmaDownWide(down);
      for (const m of [1, 9, 16, 40]) {
        using x = rows(m, R);
        using expected = down.forward(x, true);
        using actual = wide.forward(x);
        expect(relErr(actual, expected), `wide ${interleave ? "k3i" : "row-major"} m${m}`).toBeLessThan(2e-2);
      }
    }
  }, 60_000);

  test("layers refuse weights they were not built for", () => {
    const rowMajor = trellis(R, D, 0, 3, 41), interleaved = trellis(R, D, 0, 3, 43, true);
    const gate = trellis(R, D, 1, 3, 47), up = trellis(R, D, 1, 2, 53);
    expect(() => new FactoredDownRow(interleaved)).toThrow();
    expect(() => new FactoredDownK3iRows(rowMajor)).toThrow();
    expect(() => new FactoredGateUpRows(gate, up)).toThrow();
    expect(() => new FactoredMixedGateUp(gate, gate)).toThrow();
    expect(() => new MmaDownWide(interleaved)).toThrow();
    expect(() => new MmaDownWideK3i(rowMajor)).toThrow();
    expect(() => new MmaGateUpWide(gate, rowMajor)).toThrow();
  });
});

/** Bytes equal (shape, dtype and every bit). */
function sameBits(actual: MlxArray, expected: MlxArray): boolean {
  expect(actual.shape).toEqual(expected.shape);
  expect(actual.dtype).toBe(expected.dtype);
  using a = ops.contiguous(actual), b = ops.contiguous(expected);
  const x = a.rawBytesView(), y = b.rawBytesView();
  return Buffer.from(x.buffer, x.byteOffset, x.byteLength).equals(Buffer.from(y.buffer, y.byteOffset, y.byteLength));
}

describe("Trellis codebook layers", () => {
  // The paths they replace: the reduce gate/up (fusedGateUpSwiglu, and the mixed kernel at the projections' variant)
  // and TrellisLinear's own one-row down projection.
  test("each layer computes its replaced path's bits at one row", () => {
    for (const [kg, ku] of [[3, 3], [2, 3], [4, 3]] as const) {
      const gate = trellis(R, D, 1, kg, 61), up = trellis(R, D, 1, ku, 67);
      using x = rows(1, D);
      if (kg === ku) {
        using expected = fusedGateUpSwiglu(x, gate, up);
        using actual = new CodebookGateUpRow(gate, up, "split").forward(x);
        expect(sameBits(actual, expected), `k${kg}`).toBe(true);
      } else for (const tail of ["split", "fused"] as const) {
        using expected = mixedGateUpKernel(x, gate, up, gate.variant, tail);
        using actual = new CodebookGateUpRow(gate, up, tail).forward(x);
        expect(sameBits(actual, expected), `k${kg}/${ku} ${tail}`).toBe(true);
      }
    }
    for (const [k, interleave] of [[2, false], [4, false], [3, true]] as const) {
      const down = trellis(R, D, 0, k, 71, interleave);
      using x = rows(1, R);
      using expected = down.forward(x);
      using actual = (interleave ? new CodebookDownK3iRow(down) : new CodebookDownRow(down)).forward(x);
      expect(sameBits(actual, expected), `k${k}${interleave ? " interleaved" : ""}`).toBe(true);
    }
  }, 60_000);

  test("layers refuse weights they were not built for", () => {
    const rowMajor3 = trellis(R, D, 0, 3, 73), rowMajor2 = trellis(R, D, 0, 2, 79), interleaved = trellis(R, D, 0, 3, 83, true);
    const gate = trellis(R, D, 1, 3, 89);
    expect(() => new CodebookDownRow(interleaved)).toThrow();
    expect(() => new CodebookDownRow(rowMajor3)).toThrow();
    expect(() => new CodebookDownK3iRow(rowMajor2)).toThrow();
    expect(() => new CodebookGateUpRow(gate, rowMajor2, "split")).toThrow();
  });
});

describe("affine verify layers", () => {
  test("3- and 4-bit layers match QuantizedLinear", () => {
    using w32 = MlxArray.fromFloat32(Float32Array.from({ length: R * D }, (_, i) => Math.sin(i * 0.137) * 0.05), [R, D]);
    using wb = w32.astype(Dtype.bfloat16);
    for (const bits of [3, 4] as const) {
      const q = ops.quantize(wb, 64, bits);
      const lin = new QuantizedLinear(q.packed, q.scales, q.biases, { bits, groupSize: 64, mode: "affine" });
      const rowsLayer = bits === 3 ? new Affine3RowsLinear(lin) : new Affine4RowsLinear(lin);
      const mmaLayer = bits === 3 ? new Affine3MmaLinear(lin) : new Affine4MmaLinear(lin);
      for (const m of [2, 4, 5, 8]) {
        using x = rows(m, D);
        using expected = lin.forward(x);
        using a = rowsLayer.forward(x);
        using b = mmaLayer.forward(x);
        expect(relErr(a, expected), `rows ${bits}-bit m${m}`).toBeLessThan(1e-2);
        expect(relErr(b, expected), `mma ${bits}-bit m${m}`).toBeLessThan(1e-2);
      }
      expect(() => bits === 3 ? new Affine4RowsLinear(lin) : new Affine3RowsLinear(lin)).toThrow();
      // The one-row layer takes either width, as stored, and gives QuantizedLinear's bits.
      {
        using x = rows(1, D);
        using expected = lin.forward(x);
        using actual = new AffineRowLinear(lin).forward(x);
        using same = ops.equal(actual, expected);
        using all = ops.allReduce(same);
        expect(all.toFloat32()[0], `one row ${bits}-bit`).toBe(1);
      }
      // The M1 Max layer takes either width, as stored.
      const splitLayer = new AffineSplitMmaLinear(lin);
      // Against the float32 product of the dequantized weights: from 14 rows MLX's quantized matmul here is its
      // matrix kernel, which rounds its weights to bf16.
      using scales32 = q.scales.astype(Dtype.float32), biases32 = q.biases!.astype(Dtype.float32);
      using dense32 = ops.dequantize(q.packed, scales32, biases32, { bits, groupSize: 64, mode: "affine" });
      using weightT = ops.transposeAxes(dense32, [1, 0]);
      for (const m of [1, 3, 8, 9, 16, 40]) {
        using x = rows(m, D);
        using x32 = x.astype(Dtype.float32);
        using expected = ops.matmul(x32, weightT);
        using split = splitLayer.forward(x);
        expect(relErr(split, expected), `split mma ${bits}-bit m${m}`).toBeLessThan(1e-2);
      }
    }
    // It takes no 8-bit weights.
    const q8 = ops.quantize(wb, 64, 8);
    try {
      const lin8 = new QuantizedLinear(q8.packed, q8.scales, q8.biases, { bits: 8, groupSize: 64, mode: "affine" });
      expect(() => new AffineSplitMmaLinear(lin8)).toThrow();
      expect(() => new AffineRowLinear(lin8)).toThrow();
    } finally { q8.packed.dispose(); q8.scales.dispose(); q8.biases?.dispose(); }
  }, 60_000);
});

describe("TrellisLinear at prompt rows", () => {
  test("an unbounded weight expansion gives the bounded product's bits", () => {
    using x = rows(96, D);
    // Gate/up form, row-major down, 3-bit interleaved down: each takes D inputs.
    for (const lin of [trellis(R, D, 1, 3, 7), trellis(D, R, 0, 2, 19), trellis(D, 1024, 0, 3, 23, true)]) {
      using bounded = lin.forward(x, true);
      using unbounded = lin.forward(x, true, false);
      expect(unbounded.shape).toEqual(bounded.shape);
      expect(Buffer.from(unbounded.rawBytes()).equals(Buffer.from(bounded.rawBytes()))).toBe(true);
      lin.codes.dispose(); lin.scales.dispose();
    }
  }, 60_000);
});
