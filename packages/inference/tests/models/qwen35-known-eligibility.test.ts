import { expect, test } from "bun:test";
import { parseQuantization, type ModelConfig } from "../../src/artifacts/config";
import { qwen35KnownSpansEligible } from "../../src/models/qwen/known-span-eligibility";

function config(quant: Record<string, unknown> = {}): ModelConfig {
  return {
    text: { hiddenSize: 5120, intermediateSize: 17408, numHiddenLayers: 64,
      headDim: 256, numAttentionHeads: 24, numKeyValueHeads: 4, fullAttentionInterval: 4 },
    quantization: parseQuantization({ bits: 4, group_size: 64, mode: "affine", ...quant }),
  } as ModelConfig;
}

test("known-span artifact qualification preserves every supported affine width and group", () => {
  for (const bits of [2, 3, 4, 6, 8]) for (const group_size of [32, 64, 128])
    expect(qwen35KnownSpansEligible(config({ bits, group_size }))).toBe(true);
  for (const quant of [{ bits: 5 }, { group_size: 256 }, { mode: "mxfp4", bits: 4, group_size: 32 }])
    expect(qwen35KnownSpansEligible(config(quant))).toBe(false);
});

test("qualification checks the full geometry and every layer's actual projection roles", () => {
  for (const field of ["hiddenSize", "intermediateSize", "numHiddenLayers", "headDim", "numAttentionHeads", "numKeyValueHeads"] as const) {
    const c = config(); c.text = { ...c.text, [field]: c.text[field] + 1 };
    expect(qwen35KnownSpansEligible(c)).toBe(false);
  }
  for (const path of ["language_model.model.layers.62.linear_attn.out_proj", "language_model.model.layers.63.self_attn.o_proj",
    "language_model.model.layers.63.mlp.down_proj"])
    expect(qwen35KnownSpansEligible(config({ [path]: { bits: 5, group_size: 64, mode: "affine" } }))).toBe(false);
});

test("all-Trellis MLPs retain qualification while mixed or unquantized projections do not", () => {
  const overrides: Record<string, unknown> = {};
  for (let layer = 0; layer < 64; layer++) for (const role of ["gate_proj", "up_proj", "down_proj"])
    overrides[`language_model.model.layers.${layer}.mlp.${role}`] = {
      bits: 3, group_size: 256, mode: "trellis", trellis: { L: 12, code: "1mad", axis: role === "down_proj" ? 0 : 1 },
    };
  expect(qwen35KnownSpansEligible(config(overrides))).toBe(true);
  delete overrides["language_model.model.layers.63.mlp.up_proj"];
  expect(qwen35KnownSpansEligible(config(overrides))).toBe(false);
  expect(qwen35KnownSpansEligible(config({ "language_model.model.layers.0.linear_attn.in_proj_qkv": { mode: "none" } }))).toBe(false);
});
