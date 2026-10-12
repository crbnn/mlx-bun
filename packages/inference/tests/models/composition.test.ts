import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { deviceArchitecture } from "@mlx-bun/mlx/ffi";
import { loadModelConfig, type ModelConfig } from "../../src/artifacts/config";
import type { Weights } from "../../src/artifacts/weights";
import type { Composition } from "../../src/contracts/portable/composition";
import { resolveComposition } from "../../src/models/composition";
import { createModel, openModel } from "../../src/models/factory";
import { ModelImplementationRegistry } from "../../src/models/implementation";
import { resolveModelProfile } from "../../src/models/profile";
import { createRuntimeConfig, withRuntimeConfig } from "../../src/runtime/config";

let dir: string, config: ModelConfig;
beforeAll(async () => {
  dir = mkdtempSync(join(tmpdir(), "mlx-composition-"));
  writeFileSync(join(dir, "config.json"), JSON.stringify({ model_type: "qwen3", hidden_size: 8, num_hidden_layers: 2,
    num_attention_heads: 2, num_key_value_heads: 2, intermediate_size: 16, vocab_size: 32,
    max_position_embeddings: 64, tie_word_embeddings: true }));
  // A header-valid single shard: the stand-in implementations below never read it.
  const header = JSON.stringify({ "model.norm.weight": { dtype: "BF16", shape: [8], data_offsets: [0, 16] } }).padEnd(64, " ");
  const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(header.length));
  writeFileSync(join(dir, "model.safetensors"), Buffer.concat([length, Buffer.from(header), Buffer.alloc(16)]));
  config = await loadModelConfig(dir);
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

test("the default record: this machine, bf16, no drafter, the 2048-token chunk, no adapters, eight rows", () => {
  const composition = resolveComposition(config);
  expect(composition).toEqual({ device: deviceArchitecture(), aneBridge: true, kv: { kind: "bf16" }, pagedKv: null,
    draftDepth: 0, prefillChunkTokens: 2048, adapters: false, maxRows: 8, packedMixedMlp: true,
    vectorGreedySampling: true, compiledDecode: true, mixedPrefill: false, mixedTokenBudget: 256, prefillTailSplit: true,
    grammarJump: false });
  // The Neural Engine bridge is composed unless the caller turns it off; nothing probes the machine.
  expect(resolveComposition(config, { aneBridge: false }).aneBridge).toBe(false);
  expect(Object.isFrozen(composition)).toBe(true);
  expect(Object.isFrozen(composition.kv)).toBe(true);
});

test("the prefill chunk comes from the prefill policy's override, read through the runtime configuration, else 2048", () => {
  const chunk = (values: Record<string, string>) =>
    withRuntimeConfig(createRuntimeConfig(values), () => resolveComposition(config).prefillChunkTokens);
  expect(chunk({})).toBe(2048);
  expect(chunk({ MLX_BUN_RD_PREFILL_CHUNK: "512" })).toBe(512);
  // As the policy reads it: floored, at least one token.
  expect(chunk({ MLX_BUN_RD_PREFILL_CHUNK: "300.9" })).toBe(300);
  expect(chunk({ MLX_BUN_RD_PREFILL_CHUNK: "0" })).toBe(1);
});

test("the KV flags resolve to the scheme maintenance applies, with a delayed start only when nonzero", () => {
  const kv = (request: Parameters<typeof resolveComposition>[1] & {}, base = config) => resolveComposition(base, request).kv;
  expect(kv({ kv: { override: "off" } })).toEqual({ kind: "bf16" });
  expect(kv({ kv: { override: 4 } })).toEqual({ kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" });
  expect(kv({ kv: { override: 8, quantizedKvStart: 0 } })).toEqual({ kind: "affine", bits: 8, groupSize: 64, kernels: "unfused" });
  expect(kv({ kv: { override: 8, quantizedKvStart: 512 } })).toEqual({ kind: "affine", bits: 8, groupSize: 64, quantizedKvStart: 512, kernels: "unfused" });
  // `config` reads the artifact's kv_config.json, and stays bf16 without one.
  expect(kv({ kv: { override: "config" } })).toEqual({ kind: "bf16" });
  const layered = { ...config, kvQuant: [{ layerIdx: 0, bits: 4, groupSize: 32 }, { layerIdx: 1, bits: 8, groupSize: 64 }] };
  expect(kv({ kv: { override: "config" } }, layered)).toEqual({ kind: "affine-layers",
    layers: [{ layer: 0, bits: 4, groupSize: 32 }, { layer: 1, bits: 8, groupSize: 64 }], kernels: "tiled-causal" });
  expect(kv({ kv: { override: "config", quantizedKvStart: 100 } }, layered)).toMatchObject({ kind: "affine-layers", quantizedKvStart: 100 });
  expect(kv({ kv: { turboQuant: { kBits: 8, vBits: 3 } } })).toEqual({ kind: "turbo", kBits: 8, vBits: 3 });
  expect(kv({ kv: { turboQuant: { kBits: 4, vBits: 4 }, quantizedKvStart: 64 } })).toEqual({ kind: "turbo", kBits: 4, vBits: 4, quantizedKvStart: 64 });
  expect(() => kv({ kv: { override: 4, quantizedKvStart: -1 } })).toThrow("quantizedKvStart must be a nonnegative integer");
});

test("--fused-sdpa picks the affine read kernels: on tiled-causal, off unfused, unset by the scheme's kind", () => {
  const kernels = (request: NonNullable<Parameters<typeof resolveComposition>[1]>["kv"], base = config) => {
    const kv = resolveComposition(base, { kv: request }).kv;
    return kv.kind === "affine" || kv.kind === "affine-layers" ? kv.kernels : undefined;
  };
  const layered = { ...config, kvQuant: [{ layerIdx: 0, bits: 4, groupSize: 64 }] };
  expect([kernels({ override: 4 }), kernels({ override: 4, fusedSdpa: true }), kernels({ override: 4, fusedSdpa: false })])
    .toEqual(["unfused", "tiled-causal", "unfused"]);
  expect([kernels({ override: "config" }, layered), kernels({ override: "config", fusedSdpa: true }, layered),
    kernels({ override: "config", fusedSdpa: false }, layered)]).toEqual(["tiled-causal", "tiled-causal", "unfused"]);
  // Schemes without affine reads ignore it.
  expect(resolveComposition(config, { kv: { turboQuant: { kBits: 8, vBits: 3 }, fusedSdpa: false } }).kv)
    .toEqual({ kind: "turbo", kBits: 8, vBits: 3 });
  expect(resolveComposition(config, { kv: { fusedSdpa: true } }).kv).toEqual({ kind: "bf16" });
});

test("the graph switch is a composition field, on by default; the runtime configuration does not reach it", () => {
  const packed = (request: Parameters<typeof resolveComposition>[1] = {}, values: Record<string, string> = {}) =>
    withRuntimeConfig(createRuntimeConfig(values), () => resolveComposition(config, request).packedMixedMlp);
  expect(packed()).toBe(true);
  expect(packed({ packedMixedMlp: false })).toBe(false);
  expect(packed({}, { MLX_BUN_MIXED_PACKED_MLP: "0" })).toBe(true);
  // Speculation over a KV scheme is no composition fact: every cache lego holds the transaction.
  expect(resolveComposition(config)).not.toHaveProperty("speculativeAffineKv");
});

test("the scheduler switches are composition fields at their defaults; the runtime configuration does not reach them", () => {
  const facts = (request: Parameters<typeof resolveComposition>[1] = {}, values: Record<string, string> = {}) =>
    withRuntimeConfig(createRuntimeConfig(values), () => {
      const { compiledDecode, mixedPrefill, mixedTokenBudget, prefillTailSplit, grammarJump } = resolveComposition(config, request);
      return { compiledDecode, mixedPrefill, mixedTokenBudget, prefillTailSplit, grammarJump };
    });
  const defaults = { compiledDecode: true, mixedPrefill: false, mixedTokenBudget: 256, prefillTailSplit: true, grammarJump: false };
  expect(facts()).toEqual(defaults);
  const flipped = { compiledDecode: false, mixedPrefill: true, mixedTokenBudget: 9, prefillTailSplit: false, grammarJump: true };
  expect(facts(flipped)).toEqual(flipped);
  // The environment keys these fields replaced change nothing.
  expect(facts({}, { MLX_BUN_COMPILED_DECODE: "0", MLX_BUN_MIXED_PREFILL: "1", MLX_BUN_MIXED_TOKEN_BUDGET: "9",
    MLX_BUN_PREFILL_TAIL_SPLIT: "0", MLX_BUN_GRAMMAR_JUMP: "1" })).toEqual(defaults);
  for (const mixedTokenBudget of [0, 1.5, -2])
    expect(() => resolveComposition(config, { mixedTokenBudget })).toThrow("composition mixedTokenBudget must be a positive integer");
});

test("the loaded drafter fixes the depth, or caps it when the scheduler adapts; rows and adapters are recorded as given", () => {
  expect(resolveComposition(config, { draft: { numDraftTokens: 3 } }).draftDepth).toBe(3);
  expect(resolveComposition(config, { draft: { numDraftTokens: 7, adaptive: true } }).draftDepth).toEqual({ adaptive: true, max: 7 });
  expect(resolveComposition(config, { draft: null }).draftDepth).toBe(0);
  expect(resolveComposition(config, { maxRows: 1, adapters: true })).toMatchObject({ maxRows: 1, adapters: true });
  expect(() => resolveComposition(config, { maxRows: 0 })).toThrow("composition maxRows must be a positive integer");
  expect(() => resolveComposition(config, { draft: { numDraftTokens: 0 } })).toThrow("composition draft tokens must be a positive integer");
});

/** A registry whose qwen3 implementation reports what it was built for. */
const reporting = () => {
  const seen: Composition[] = [];
  const registry = new ModelImplementationRegistry<Weights, { composition: Composition; weights: Weights }>([{
    id: "qwen3", graph: "qwen3", loader: "safetensors", loop: "autoregressive",
    create: (weights, _config, _profile, composition) => { seen.push(composition); return { composition, weights }; },
  }]);
  return { seen, registry };
};

test("createModel hands the record to the implementation, and resolves the default record when given none", () => {
  const { seen, registry } = reporting();
  const profile = resolveModelProfile(config);
  const chosen = resolveComposition(config, { kv: { override: 4 }, maxRows: 2 });
  expect(createModel({} as Weights, config, chosen, profile, registry).composition).toBe(chosen);
  expect(createModel({} as Weights, config, undefined, profile, registry).composition).toEqual(resolveComposition(config));
  expect(seen).toHaveLength(2);
});

test("openModel, the convenience load, takes the same record and the same default", async () => {
  const { registry } = reporting();
  const chosen = resolveComposition(config, { draft: { numDraftTokens: 2 } });
  const given = await openModel(dir, { implementations: registry, composition: chosen });
  try { expect(given.composition).toBe(chosen); } finally { given.weights.dispose(); }
  const defaulted = await openModel(dir, { implementations: registry });
  try { expect(defaulted.composition).toEqual(resolveComposition(config)); } finally { defaulted.weights.dispose(); }
});

test("paging is a composition fact: bf16 and uniform 4- or 8-bit pages, the block size given or 256", () => {
  expect(resolveComposition(config).pagedKv).toBeNull();
  expect(resolveComposition(config, { pagedKv: {} }).pagedKv).toEqual({ blockSize: 256 });
  expect(resolveComposition(config, { pagedKv: { blockSize: 16 }, kv: { override: 8 } })).toMatchObject({
    pagedKv: { blockSize: 16 }, kv: { kind: "affine", bits: 8 } });
  expect(() => resolveComposition(config, { pagedKv: { blockSize: 0 } })).toThrow("composition paged KV block size must be a positive integer");
  // The paged lego has no per-layer, other-width or TurboQuant pages yet: such a composition cannot be built as named.
  const listed = { ...config, kvQuant: [{ layerIdx: 0, bits: 4, groupSize: 64 }] };
  for (const kv of [{ override: "config" as const }, { turboQuant: { kBits: 8, vBits: 3 } }, { override: 3 }]) {
    expect(resolveComposition(listed, { kv }).kv.kind).not.toBe("bf16");
    expect(() => resolveComposition(listed, { kv, pagedKv: {} })).toThrow("per-layer and TurboQuant pages are not implemented");
  }
});

test("a graph that carries no composition record composes its own caches: a KV scheme or paging for it is refused", () => {
  let released = 0;
  const registry = new ModelImplementationRegistry<Weights, { weights: Weights; dispose(): void }>([{
    id: "qwen3", graph: "qwen3", loader: "safetensors", loop: "autoregressive",
    create: weights => ({ weights, dispose() { released++; } }),
  }]);
  const profile = resolveModelProfile(config);
  expect(() => createModel({} as Weights, config, resolveComposition(config), profile, registry)).not.toThrow();
  for (const request of [{ kv: { override: 4 as const } }, { pagedKv: {} }])
    expect(() => createModel({} as Weights, config, resolveComposition(config, request), profile, registry))
      .toThrow("this graph composes its own caches");
  expect(released).toBe(2);
});
