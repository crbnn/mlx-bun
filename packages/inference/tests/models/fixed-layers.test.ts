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

numerical("literal fused and unfused Trellis MLP operations retain the old width bindings", () => {
  const gate = trellis(512, 512, 1), up = trellis(512, 512, 1), down = trellis(512, 512, 0);
  const old = new TrellisMlp(gate, up, down);
  try {
    for (const rows of [1, 2, 3, 4, 8]) {
      using x = input(rows, 512);
      using expected = old.forward(x,"window",rows);
      const operation = rows <= 4 ? new TrellisFusedMLP(gate, up, new TrellisScatterPrompt(down))
        : new TrellisUnfusedMLP(new TrellisExpandReducePrompt(gate), new TrellisExpandReducePrompt(up), new TrellisExpandScatterPrompt(down));
      using actual = operation.forward(x);
      bitsEqual(actual, expected);
    }
  } finally { release(gate); release(up); release(down); }
});

numerical("named Qwen text and media attention retain outputs and KV state across prompt and decode", () => {
  using normWide = MlxArray.fromFloat32(new Float32Array(64).fill(1), [64]);
  using norm = normWide.astype(Dtype.bfloat16);
  const weights = { tensor: () => norm } as unknown as Weights;
  const config = { text: { numAttentionHeads: 2, numKeyValueHeads: 2, headDim: 64, partialRotaryFactor: 1, rmsNormEps: 1e-6,
    ropeParameters: { full_attention: { ropeTheta: 10000 } } } } as unknown as ModelConfig;
  const owned: MlxArray[] = [];
  const projection = (rows: number) => {
    using raw = input(rows, 128), flat = ops.reshape(raw, [rows, 128]);
    const q = ops.quantize(flat, 64, 4);
    owned.push(q.packed, q.scales); if (q.biases) owned.push(q.biases);
    return new QuantizedLinear(q.packed, q.scales, q.biases, { bits: 4, groupSize: 64, mode: "affine" });
  };
  const ports = { qProj: projection(256), kProj: projection(128), vProj: projection(128), oProj: projection(128) };
  const old = new Qwen3Attention(weights, config, "attention", undefined, undefined, ports);
  const fixed = { qProj: new QuantizedLinearPlain(ports.qProj), kProj: new QuantizedLinearPlain(ports.kProj),
    vProj: new QuantizedLinearPlain(ports.vProj), oProj: new QuantizedLinearPlain(ports.oProj) };
  const text = new Qwen3TextAttention(weights, config, "attention", fixed), media = new Qwen3MediaAttention(weights, config, "attention", fixed);
  using invFreq = mropeInvFreq(64, 10000);
  try {
    for (const variant of ["text", "media"] as const) {
      const controlCache = new KVCache(), candidateCache = new KVCache();
      try {
        for (const rows of [3, 1, 1]) {
          using x = input(rows, 128);
          using positions = ops.fromInt32(Array.from({ length: rows * 3 }, (_, i) => controlCache.offset + i % rows), [3, 1, rows]);
          const state = { posIds: positions, invFreq, rotaryDims: 64 };
          const phase = rows === 1 ? "decode" : "window";
          using expected = old.forward(x, controlCache, phase, variant === "media" ? state : null);
          using actual = variant === "media" ? media.forward(x, candidateCache, phase, state) : text.forward(x, candidateCache, phase);
          bitsEqual(actual, expected);
          expect(candidateCache.offset).toEqual(controlCache.offset);
          bitsEqual(candidateCache.keys!, controlCache.keys!); bitsEqual(candidateCache.values!, controlCache.values!);
        }
      } finally { controlCache.dispose(); candidateCache.dispose(); }
    }
  } finally { for (const array of owned) array.dispose(); }
});

import type { ModelConfig } from "../../src/artifacts/config";
import type { Weights } from "../../src/artifacts/weights";
import { Qwen3Attention, Qwen3TextAttention, Qwen3MediaAttention, TrellisMlp } from "../../src/models/qwen/blocks";
import { TrellisFusedMLP, TrellisUnfusedMLP } from "../../src/layers/trellis-prompt";
import { mropeInvFreq } from "../../src/layers/qwen-mrope";
import { KVCache } from "../../src/state/kv";
