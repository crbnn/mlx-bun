// Opt in after checking training is inactive: MLX_BUN_TEST_FIXED_LAYERS=1.
// Every comparison uses the original dispatcher as the bit-level control.
import { expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { TrellisLinear } from "../../src/layers/trellis-linear";
import { TrellisReducePrompt, TrellisScatterPrompt, TrellisWidePrompt, TrellisTiledPrompt, TrellisSplitKPrompt, TrellisExpandReducePrompt, TrellisExpandScatterPrompt } from "../../src/layers/trellis-prompt";
import { QuantizedLinear, QuantizedLinearPlain, QuantizedLinearRows } from "../../src/layers/quantized-linear";
import { PlainGeglu, CompiledGeglu, geglu } from "../../src/layers/geglu";
import { LoraDelta, LoraState } from "../../src/layers/lora";
import { SwitchGLU, SortedSwitchGLU, UnsortedSwitchGLU } from "../../src/models/gemma4/blocks";
import { QuantizedSwitchLinear } from "../../src/layers/quantized-switch-linear";

const numerical = test.skipIf(!process.env.MLX_BUN_TEST_FIXED_LAYERS);
const bitsEqual = (a: MlxArray, b: MlxArray) => {
  expect(a.shape).toEqual(b.shape);
  expect(a.dtype).toEqual(b.dtype);
  using ac = ops.contiguous(a), bc = ops.contiguous(b);
  expect(Buffer.compare(Buffer.from(ac.rawBytes()), Buffer.from(bc.rawBytes()))).toBe(0);
};
function input(rows: number, width: number) {
  using wide = MlxArray.fromFloat32(Float32Array.from({ length: rows * width }, (_, i) => Math.sin(i * 0.073 + 0.4) / 3), [1, rows, width]);
  return wide.astype(Dtype.bfloat16);
}
function trellis(rows: number, cols: number, axis: 0 | 1): TrellisLinear {
  const words = new Uint32Array(rows * cols * 3 / 32);
  for (let i = 0; i < words.length; i++) words[i] = Math.imul(i + 31, 0x9e3779b1) >>> 0;
  const codes = MlxArray.fromBytesCopy(new Uint8Array(words.buffer), [rows, cols * 3 / 32], Dtype.uint32);
  using wide = MlxArray.fromFloat32(Float32Array.from({ length: rows }, (_, i) => (i % 23 + 1) / 400), [rows]);
  return new TrellisLinear(codes, wide.astype(Dtype.float16), { bits: 3, groupSize: 256, mode: "trellis", trellis: { L: 12, code: "1mad", axis } });
}
function release(lin: TrellisLinear) { lin.codes.dispose(); lin.scales.dispose(); }

numerical("fixed Trellis reduce, scatter and dense expansion retain dispatcher bits and leading dimensions", () => {
  for (const axis of [0, 1] as const) {
    const lin = trellis(512, 512, axis);
    try {
      for (const rows of [1, 2, 3, 4, 9, 16, 33]) {
        using x = input(rows, 512);
        using expected = lin.forward(x, true);
        const layer = rows <= 4
          ? axis === 1 ? new TrellisReducePrompt(lin) : new TrellisScatterPrompt(lin)
          : axis === 1 ? new TrellisExpandReducePrompt(lin) : new TrellisExpandScatterPrompt(lin);
        using actual = layer.forward(x);
        bitsEqual(actual, expected);
      }
    } finally { release(lin); }
  }
}, 120_000);

numerical("fixed Trellis prompt kernels retain dispatcher bits over the served 27B geometry", () => {
  const gate = trellis(17408, 5120, 1), down = trellis(17408, 5120, 0);
  try {
    for (const rows of [5, 8, 9, 15, 16, 32]) {
      using x = input(rows, 5120);
      using expected = gate.forward(x, true);
      const layer = gate.representations.widePrefill && rows <= 15 ? new TrellisWidePrompt(gate) : new TrellisTiledPrompt(gate);
      using actual = layer.forward(x);
      bitsEqual(actual, expected);
    }
    for (const rows of [5, 6, 7, 8]) {
      using x = input(rows, 17408);
      using expected = down.forward(x, true);
      using actual = new TrellisSplitKPrompt(down).forward(x);
      bitsEqual(actual, expected);
    }
  } finally { release(gate); release(down); }
}, 120_000);

numerical("fixed affine plain and independent rows retain bias and mounted delta arithmetic", () => {
  using w = input(128, 128);
  using flat = ops.reshape(w, [128, 128]);
  const q = ops.quantize(flat, 64, 4);
  using bias = input(1, 128);
  const base = new QuantizedLinear(q.packed, q.scales, q.biases, { groupSize: 64, bits: 4, mode: "affine" }, bias);
  using a = input(128, 4), b = input(4, 128);
  using af = ops.reshape(a, [128, 4]), bf = ops.reshape(b, [4, 128]);
  const state = new LoraState();
  base.loraState = state;
  base.adapters = new Map([["adapter", { a: af, b: bf, scale: 0.13, rank: 4 }]]);
  try {
    for (const active of [[], ["adapter"]]) for (const rows of [1, 2, 4, 8]) {
      state.active = active;
      using x = input(rows, 128);
      using plainExpected = base.forward(x, false), rowExpected = base.forward(x, true);
      using plain = new QuantizedLinearPlain(base).forward(x), row = new QuantizedLinearRows(base).forward(x);
      bitsEqual(plain, plainExpected); bitsEqual(row, rowExpected);
    }
  } finally { q.packed.dispose(); q.scales.dispose(); q.biases?.dispose(); }
});

numerical("the adapter delta composes after an arbitrary projection and returns an untouched handle when inactive", () => {
  const delta = new LoraDelta(), state = new LoraState();
  using x = input(4, 128), a = input(128, 4), b = input(4, 128);
  using af = ops.reshape(a, [128, 4]), bf = ops.reshape(b, [4, 128]);
  delta.loraState = state; delta.adapters = new Map([["adapter", { a: af, b: bf, scale: 0.13, rank: 4 }]]);
  using plain = ops.mulScalar(x, 2);
  expect(delta.forward(x, plain)).toBe(plain);
  state.active = ["adapter"];
  using xa = ops.matmul(x, af), z = ops.matmul(xa, bf), scaled = ops.mulScalar(z, 0.13), cast = scaled.astype(x.dtype);
  using expected = ops.add(plain, cast);
  using actual = delta.forward(x, ops.mulScalar(x, 2));
  bitsEqual(actual, expected);
});

numerical("fixed GeGLU forms retain each old form's bits and consume both inputs", () => {
  for (const compiled of [false, true]) for (const rows of [1, 4, 32]) {
    const gate = input(rows, 128), up = input(rows, 128);
    using expected = geglu(input(rows, 128), input(rows, 128), compiled);
    using actual = (compiled ? new CompiledGeglu() : new PlainGeglu()).forward(gate, up);
    bitsEqual(actual, expected);
    expect(() => gate.shape).toThrow(); expect(() => up.shape).toThrow();
  }
});

numerical("fixed sorted and unsorted expert gathers retain the threshold dispatcher's bits", () => {
  using raw = input(4 * 128, 128), flat = ops.reshape(raw, [4, 128, 128]);
  const q = ops.quantize(flat, 64, 4);
  const linear = new QuantizedSwitchLinear(q.packed, q.scales, q.biases, { groupSize: 64, bits: 4, mode: "affine" });
  const base = Object.assign(Object.create(SwitchGLU.prototype), { gate: linear, up: linear, down: linear, compiled: true }) as SwitchGLU;
  try {
    for (const rows of [1, 8, 31, 32, 40]) {
      using x = input(rows, 128), indices = ops.fromInt32(Array.from({ length: rows * 2 }, (_, i) => i % 4), [1, rows, 2]);
      using expected = base.forward(x, indices);
      const fixed = rows * 2 >= 64 ? new SortedSwitchGLU(base, new CompiledGeglu()) : new UnsortedSwitchGLU(base, new CompiledGeglu());
      using actual = fixed.forward(x, indices);
      bitsEqual(actual, expected);
    }
  } finally { q.packed.dispose(); q.scales.dispose(); q.biases?.dispose(); }
});
