import { expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import type { Weights } from "../../src/artifacts/weights";
import type { ModelConfig } from "../../src/artifacts/config";
import { DecoderLayer } from "../../src/models/gemma4/blocks";
import { GemmaLayerWeights, GemmaInputLayerWeights, GemmaMoeLayerWeights, GemmaDenseLayer, GemmaInputLayer, GemmaMoeLayer, SlidingAttention, FullAttention, FullKEqVAttention, SharedSlidingAttention, CompiledGegluMLP, FixedExperts, CompiledGeglu, SortedSwitchGLU, UnsortedSwitchGLU } from "../../src/models/gemma4/written-blocks";
import { KVCache } from "../../src/state/kv";

const numerical = test.skipIf(!process.env.MLX_BUN_TEST_FIXED_LAYERS);
function input(rows: number, width: number) {
  using wide = MlxArray.fromFloat32(Float32Array.from({ length: rows * width }, (_, i) => Math.sin(i * 0.079 + 0.3) / 3), [1, rows, width]);
  return wide.astype(Dtype.bfloat16);
}
function equal(a: MlxArray, b: MlxArray) {
  expect(a.shape).toEqual(b.shape); expect(a.dtype).toEqual(b.dtype);
  using ac = ops.contiguous(a), bc = ops.contiguous(b);
  expect(Buffer.compare(Buffer.from(ac.rawBytes()), Buffer.from(bc.rawBytes()))).toBe(0);
}
function fixture(inputs: boolean, moe: boolean, kEqV: boolean) {
  const arrays = new Map<string, MlxArray>();
  const value = (name: string): MlxArray => {
    if (arrays.has(name)) return arrays.get(name)!;
    const module = name.replace(/\.(weight|scales|biases)$/, "");
    if (name.endsWith("layer_scalar")) {
      const result = MlxArray.fromFloat32(new Float32Array([0.9]), []); arrays.set(name, result); return result;
    }
    if (/norm(?:_[12])?\.weight$/.test(name) || /\.scale$/.test(name) || /per_expert_scale$/.test(name)) {
      const width = /[qk]_norm/.test(name) ? 64 : /per_expert_scale$/.test(name) ? 4 : 128;
      using raw = MlxArray.fromFloat32(new Float32Array(width).fill(1), [width]);
      const result = raw.astype(Dtype.bfloat16); arrays.set(name, result); return result;
    }
    let rows = 128, cols = 128;
    if (module.endsWith("router.proj")) rows = 4;
    if (module.endsWith("per_layer_input_gate")) rows = 64;
    if (module.endsWith("per_layer_projection")) cols = 64;
    const experts = module.includes("switch_glu") ? 4 : 1;
    const shape = experts > 1 ? [experts, rows, cols] : [rows, cols];
    let seed = 0; for (const char of module) seed += char.charCodeAt(0);
    using raw = MlxArray.fromFloat32(Float32Array.from({ length: experts * rows * cols }, (_, i) => Math.sin(i * 0.11 + seed) * 0.03), shape);
    using bf16 = raw.astype(Dtype.bfloat16);
    const q = ops.quantize(bf16, 64, 4);
    arrays.set(`${module}.weight`, q.packed); arrays.set(`${module}.scales`, q.scales); arrays.set(`${module}.biases`, q.biases!);
    return arrays.get(name)!;
  };
  const weights = { tensor: value, has: (name: string) => !name.endsWith(".bias") } as unknown as Weights;
  const config = { text: { hiddenSize: 128, intermediateSize: 128, numAttentionHeads: 2, numKeyValueHeads: 2,
    numGlobalKeyValueHeads: 2, headDim: 64, globalHeadDim: 64, attentionKEqV: kEqV,
    hiddenSizePerLayerInput: inputs ? 64 : 0, layerTypes: ["sliding_attention", "full_attention"],
    rmsNormEps: 1e-6, slidingWindow: 64, numExperts: 4, topKExperts: 2, enableMoeBlock: moe,
    ropeParameters: { sliding_attention: { ropeType: "default", ropeTheta: 10000 }, full_attention: { ropeType: "default", ropeTheta: 10000 } } },
    quantization: { default: { groupSize: 64, bits: 4, mode: "affine" }, perLayer: new Map() } } as unknown as ModelConfig;
  return { config, weights, dispose() { for (const array of arrays.values()) array.dispose(); } };
}

numerical("written dense and per-layer-input blocks retain outputs, donor reads and KV continuation", () => {
  for (const inputs of [false, true]) for (const kEqV of [false, true]) {
    const data = fixture(inputs, false, kEqV);
    const index = kEqV ? 1 : 0, prefix = `language_model.model.layers.${index}`;
    const original = new DecoderLayer(data.weights, data.config, "language_model.model", index, true, true);
    const attention = kEqV ? new FullKEqVAttention(data.weights, data.config, `${prefix}.self_attn`) : new SlidingAttention(data.weights, data.config, `${prefix}.self_attn`);
    const parts = inputs ? new GemmaInputLayerWeights(data.weights, data.config, prefix, attention) : new GemmaLayerWeights(data.weights, data.config, prefix, attention);
    const fixed = inputs ? new GemmaInputLayer(parts as GemmaInputLayerWeights, new CompiledGegluMLP(parts.mlp)) : new GemmaDenseLayer(parts, new CompiledGegluMLP(parts.mlp));
    const a = new KVCache(), b = new KVCache();
    try {
      for (const rows of [3, 1, 1]) {
        using x = input(rows, 128), layerInput = input(rows, 64);
        const control = original.forward(x, a, rows === 1 ? "decode" : "window", null, inputs ? layerInput : null);
        const actual = fixed.forward(x, b, rows === 1 ? "decode" : "window", null, layerInput);
        try { equal(actual.h, control.h); equal(a.keys!, b.keys!); equal(a.values!, b.values!); expect(a.offset).toBe(b.offset); }
        finally { actual.h.dispose(); control.h.dispose(); actual.donor?.dispose(); control.donor?.dispose(); }
      }
    } finally { a.dispose(); b.dispose(); data.dispose(); }
  }
});

numerical("written MoE blocks retain unsorted and sorted expert arithmetic at the threshold", () => {
  const data = fixture(false, true, true), prefix = "language_model.model.layers.1";
  const original = new DecoderLayer(data.weights, data.config, "language_model.model", 1, true, true);
  const parts = new GemmaMoeLayerWeights(data.weights, data.config, prefix, new FullKEqVAttention(data.weights, data.config, `${prefix}.self_attn`));
  const unsorted = new GemmaMoeLayer(parts, new CompiledGegluMLP(parts.mlp), new FixedExperts(new UnsortedSwitchGLU(parts.experts, new CompiledGeglu())));
  const sorted = new GemmaMoeLayer(parts, new CompiledGegluMLP(parts.mlp), new FixedExperts(new SortedSwitchGLU(parts.experts, new CompiledGeglu())));
  try {
    for (const rows of [1, 7, 31, 32, 40]) {
      using x = input(rows, 128);
      const expected = original.moeFeedforward(x);
      using actual = (rows >= 32 ? sorted : unsorted).feedforward(x);
      using scaled = original.scaled(expected);
      equal(actual, scaled);
    }
  } finally { parts.router.normWeight.dispose(); original.router!.normWeight.dispose(); data.dispose(); }
});

numerical("a written shared attention layer attends the same donor without owning a cache", () => {
  const data = fixture(true, false, false), prefix = "language_model.model.layers.0.self_attn";
  const donor = new SlidingAttention(data.weights, data.config, prefix);
  const original = new Attention(data.weights, data.config, prefix, "sliding_attention", false);
  const fixed = new SharedSlidingAttention(data.weights, data.config, prefix);
  const cache = new KVCache();
  using x = input(3, 128);
  const source = donor.forward(x, cache, "window", null);
  try {
    const control = original.forward(x, null, "window", source.donor);
    const candidate = fixed.forward(x, null, "window", source.donor);
    try { equal(candidate.out, control.out); expect(candidate.donor).toBeNull(); expect(control.donor).toBeNull(); }
    finally { control.out.dispose(); candidate.out.dispose(); }
  } finally { source.out.dispose(); source.donor.dispose(); cache.dispose(); data.dispose(); }
});

import { Attention } from "../../src/models/gemma4/blocks";
