// The loader's selection tables (E1): every resident family's graphs are data,
// selected once at load from the artifact fingerprint and the composition. No
// model loads here: configs are synthetic or the committed generator fixtures,
// compositions come from resolveComposition with the device overridden, and the
// selected entry is read, never built. Real-weight selection is
// graph-table-real-weights.test.ts.
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { loadModelConfig, type ModelConfig } from "../../src/artifacts/config";
import { configFingerprint } from "../../src/artifacts/fingerprint";
import type { Composition } from "../../src/contracts/portable/composition";
import { resolveComposition, type CompositionRequest } from "../../src/models/composition";
import { MLX_GRAPH_TABLES } from "../../src/models/factory";
import { GraphTable, type GraphEntry, type ServedComposition } from "../../src/models/implementation";
import { QWEN38_TRELLIS_M4PRO_GRAPH } from "../../src/models/qwen/qwen38-27b-trellis-m4pro";
import { QWEN38_TRELLIS_TQ_GRAPH, qwen38TrellisTqFingerprint } from "../../src/models/qwen/qwen38-27b-trellis-tq";

const TRELLIS = "cfa205c8f5af046a";
const M4_PRO = "applegpu_g16s", M1_MAX = "applegpu_g13s";

let root: string, qwen: ModelConfig;
beforeAll(async () => {
  root = mkdtempSync(join(tmpdir(), "mlx-graph-table-"));
  writeFileSync(join(root, "config.json"), JSON.stringify({ model_type: "qwen3_5_text", hidden_size: 64, num_hidden_layers: 2,
    num_attention_heads: 2, num_key_value_heads: 1, head_dim: 32, vocab_size: 64, intermediate_size: 64, rms_norm_eps: 1e-6,
    rope_theta: 1e6, max_position_embeddings: 128, tie_word_embeddings: true, eos_token_id: 1, linear_num_value_heads: 4,
    linear_num_key_heads: 2, linear_key_head_dim: 16, linear_value_head_dim: 16, linear_conv_kernel_dim: 4,
    full_attention_interval: 2, layer_types: ["linear_attention", "full_attention"] }));
  qwen = await loadModelConfig(root);
});
afterAll(() => rmSync(root, { recursive: true, force: true }));

/** The record the app resolves for `request`, as if loaded on `device`. */
const on = (device: string, config: ModelConfig, request: CompositionRequest = {}): Composition =>
  ({ ...resolveComposition(config, request), device });

describe("a graph table", () => {
  const entry = (id: string, serves: ServedComposition): GraphEntry<null, string> => ({ id, serves, create: () => id });
  const composition = (fields: Partial<Composition> = {}): Composition => ({ device: M4_PRO, aneBridge: true, kv: { kind: "bf16" },
    draftDepth: 0, prefillChunkTokens: 2048, adapters: false, maxRows: 8, speculativeAffineKv: true, packedMixedMlp: true,
    vectorGreedySampling: true, ...fields });

  test("the most constrained entry the composition meets wins; the generic entry serves the rest", () => {
    const table = new GraphTable([
      entry("device", { device: M4_PRO }),
      entry("device+kv", { device: M4_PRO, kv: [{ kind: "affine", bits: 4, groupSize: 64 }] }),
      entry("fingerprint+device+rows", { fingerprint: "f", device: M4_PRO, maxRows: 4 }),
      entry("generic", {}),
    ], config => config.modelDir);
    expect(table.generic.id).toBe("generic");
    expect(table.match("x", composition({ device: M1_MAX })).id).toBe("generic");
    expect(table.match("x", composition()).id).toBe("device");
    expect(table.match("x", composition({ kv: { kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" } })).id).toBe("device+kv");
    // A delayed start is the cache's timing, and the read kernels the cache's composition: the same scheme.
    expect(table.match("x", composition({ kv: { kind: "affine", bits: 4, groupSize: 64, quantizedKvStart: 512, kernels: "unfused" } })).id).toBe("device+kv");
    expect(table.match("x", composition({ kv: { kind: "affine", bits: 4, groupSize: 64, kernels: "tiled-causal" } })).id).toBe("device+kv");
    expect(table.match("x", composition({ kv: { kind: "affine", bits: 4, groupSize: 32, kernels: "unfused" } })).id).toBe("device");
    // maxRows bounds from above, inclusive.
    expect(table.match("f", composition({ maxRows: 4 })).id).toBe("fingerprint+device+rows");
    expect(table.match("f", composition({ maxRows: 1 })).id).toBe("fingerprint+device+rows");
    expect(table.match("f", composition({ maxRows: 5 })).id).toBe("device");
    // select computes the fingerprint from the config with the table's own function.
    expect(table.select({ modelDir: "f" } as ModelConfig, composition({ maxRows: 2 })).id).toBe("fingerprint+device+rows");
    expect(table.select({ modelDir: "g" } as ModelConfig, composition({ maxRows: 2 })).id).toBe("device");
  });

  test("every field constrains when present: ANE bridge, draft depth, chunk, adapters, per-layer and TurboQuant KV", () => {
    const layered = { kind: "affine-layers", layers: [{ layer: 1, bits: 8, groupSize: 64 }, { layer: 0, bits: 4, groupSize: 64 }] } as const;
    const table = new GraphTable([
      entry("ane", { aneBridge: false }),
      entry("depth", { draftDepth: { adaptive: true, max: 7 }, device: M1_MAX }),
      entry("chunk", { prefillChunkTokens: 512, device: M1_MAX, adapters: true }),
      entry("layers", { kv: [layered], device: "applegpu_g15s" }),
      entry("turbo", { kv: [{ kind: "turbo", kBits: 8, vBits: 3 }], device: "applegpu_g14s" }),
      entry("generic", {}),
    ]);
    expect(table.match(undefined, composition({ aneBridge: false })).id).toBe("ane");
    expect(table.match(undefined, composition({ device: M1_MAX, draftDepth: { adaptive: true, max: 7 } })).id).toBe("depth");
    expect(table.match(undefined, composition({ device: M1_MAX, draftDepth: 7 })).id).toBe("generic");
    expect(table.match(undefined, composition({ device: M1_MAX, prefillChunkTokens: 512, adapters: true })).id).toBe("chunk");
    expect(table.match(undefined, composition({ device: M1_MAX, prefillChunkTokens: 512 })).id).toBe("generic");
    // Per-layer schemes compare layer by layer, in any order.
    expect(table.match(undefined, composition({ device: "applegpu_g15s",
      kv: { kind: "affine-layers", layers: [{ layer: 0, bits: 4, groupSize: 64 }, { layer: 1, bits: 8, groupSize: 64 }], kernels: "tiled-causal" } })).id).toBe("layers");
    expect(table.match(undefined, composition({ device: "applegpu_g15s",
      kv: { kind: "affine-layers", layers: [{ layer: 0, bits: 4, groupSize: 64 }, { layer: 1, bits: 4, groupSize: 64 }], kernels: "tiled-causal" } })).id).toBe("generic");
    expect(table.match(undefined, composition({ device: "applegpu_g14s", kv: { kind: "turbo", kBits: 8, vBits: 3 } })).id).toBe("turbo");
    expect(table.match(undefined, composition({ device: "applegpu_g14s", kv: { kind: "turbo", kBits: 4, vBits: 4 } })).id).toBe("generic");
  });

  test("construction rejects tables whose selection would refuse or depend on order", () => {
    expect(() => new GraphTable([entry("a", { device: M4_PRO })])).toThrow("exactly one generic entry");
    expect(() => new GraphTable([entry("a", {}), entry("b", {})])).toThrow("exactly one generic entry");
    expect(() => new GraphTable([entry("a", { fingerprint: "f" }), entry("g", {})])).toThrow("the table computes none");
    expect(() => new GraphTable([entry("a", { device: M4_PRO }), entry("a", { device: M1_MAX }), entry("g", {})])).toThrow("duplicate entry a");
    expect(() => new GraphTable([entry("a", { kv: [] }), entry("g", {})])).toThrow("serves no KV scheme");
    // Equal counts that one composition could meet: device M4 Pro with bf16 meets both.
    expect(() => new GraphTable([entry("a", { device: M4_PRO }), entry("b", { kv: [{ kind: "bf16" }] }), entry("g", {})]))
      .toThrow("a and b constrain as many fields");
    expect(() => new GraphTable([entry("a", { device: M4_PRO, maxRows: 8 }), entry("b", { device: M4_PRO, maxRows: 16 }), entry("g", {})]))
      .toThrow("a and b constrain as many fields");
    // Disjoint on a shared field, or of different counts: allowed.
    expect(() => new GraphTable([entry("a", { device: M4_PRO }), entry("b", { device: M1_MAX }), entry("g", {})])).not.toThrow();
    expect(() => new GraphTable([entry("a", { kv: [{ kind: "bf16" }] }), entry("b", { kv: [{ kind: "affine", bits: 4, groupSize: 64 }] }),
      entry("g", {})])).not.toThrow();
    expect(() => new GraphTable([entry("a", { device: M4_PRO }), entry("b", { device: M4_PRO, maxRows: 8 }), entry("g", {})])).not.toThrow();
  });
});

describe("the Qwen 3.5 table", () => {
  const table = MLX_GRAPH_TABLES["qwen3.5"]!;
  const select = (composition: Composition, fingerprint = TRELLIS) => table.match(fingerprint, composition).id;

  test("the M4 Pro graph serves the Trellis artifact on applegpu_g16s with bf16 or affine 4/64, adapters or not, up to 8 rows", () => {
    expect(select(on(M4_PRO, qwen))).toBe(QWEN38_TRELLIS_M4PRO_GRAPH);
    expect(select(on(M4_PRO, qwen, { kv: { override: 4 } }))).toBe(QWEN38_TRELLIS_M4PRO_GRAPH);
    expect(select(on(M4_PRO, qwen, { kv: { override: 4, quantizedKvStart: 512 } }))).toBe(QWEN38_TRELLIS_M4PRO_GRAPH);
    for (const fusedSdpa of [true, false])
      expect(select(on(M4_PRO, qwen, { kv: { override: 4, fusedSdpa } }))).toBe(QWEN38_TRELLIS_M4PRO_GRAPH);
    expect(select(on(M4_PRO, qwen, { adapters: true }))).toBe(QWEN38_TRELLIS_M4PRO_GRAPH);
    expect(select(on(M4_PRO, qwen, { maxRows: 1, draft: { numDraftTokens: 3 } }))).toBe(QWEN38_TRELLIS_M4PRO_GRAPH);
    expect(select({ ...on(M4_PRO, qwen, { draft: { numDraftTokens: 7, adaptive: true } }), aneBridge: false, prefillChunkTokens: 512 }))
      .toBe(QWEN38_TRELLIS_M4PRO_GRAPH);
  });

  test("the same artifact on applegpu_g16s with another KV scheme or more rows gets the generic graph", () => {
    expect(select(on(M4_PRO, qwen, { kv: { override: 8 } }))).toBe("qwen3.5");
    const layered = { ...qwen, kvQuant: [{ layerIdx: 1, bits: 4, groupSize: 64 }] };
    expect(on(M4_PRO, layered, { kv: { override: "config" } }).kv.kind).toBe("affine-layers");
    expect(select(on(M4_PRO, layered, { kv: { override: "config" } }))).toBe("qwen3.5");
    expect(select(on(M4_PRO, qwen, { kv: { turboQuant: { kBits: 8, vBits: 3 } } }))).toBe("qwen3.5");
    expect(select(on(M4_PRO, qwen, { maxRows: 9 }))).toBe("qwen3.5");
    expect(select(on(M4_PRO, qwen, { maxRows: 16 }))).toBe("qwen3.5");
  });

  test("on applegpu_g13s the TQ graph serves the artifact whatever its KV scheme or rows; other GPUs get the generic graph", () => {
    expect(select(on(M1_MAX, qwen))).toBe(QWEN38_TRELLIS_TQ_GRAPH);
    expect(select(on(M1_MAX, qwen, { kv: { override: 8 } }))).toBe(QWEN38_TRELLIS_TQ_GRAPH);
    expect(select(on(M1_MAX, qwen, { kv: { override: 4 }, maxRows: 16 }))).toBe(QWEN38_TRELLIS_TQ_GRAPH);
    expect(select(on("applegpu_g15s", qwen))).toBe("qwen3.5");
  });

  test("another Qwen 3.5 artifact gets the generic graph on every device; the table fingerprints with the Trellis graph fingerprint", () => {
    const fingerprint = qwen38TrellisTqFingerprint(qwen);
    expect(fingerprint).not.toBe(TRELLIS);
    for (const device of [M4_PRO, M1_MAX]) {
      expect(table.select(qwen, on(device, qwen)).id).toBe("qwen3.5");
      expect(select(on(device, qwen), fingerprint)).toBe("qwen3.5");
    }
  });

  test("the published Trellis config selects by its own fingerprint (config only; skips when the snapshot is not downloaded)", async () => {
    const dir = join(homedir(), ".cache/huggingface/hub/models--mlx-bun--Qwen3.8-27B-Trellis-3.2bpw/snapshots/cbb7a58a7222a19a3d02d36540c9cab5d9aadf0d");
    if (!existsSync(join(dir, "config.json"))) return;
    const config = await loadModelConfig(dir);
    expect(qwen38TrellisTqFingerprint(config)).toBe(TRELLIS);
    expect(table.select(config, on(M4_PRO, config)).id).toBe(QWEN38_TRELLIS_M4PRO_GRAPH);
    expect(table.select(config, on(M1_MAX, config)).id).toBe(QWEN38_TRELLIS_TQ_GRAPH);
    expect(table.select(config, on(M4_PRO, config, { kv: { override: 8 } })).id).toBe("qwen3.5");
  });
});

describe("the Gemma 4 table", () => {
  const table = MLX_GRAPH_TABLES["gemma4"]!;
  const fixtures = join(import.meta.dir, "../fixtures/gemma4-graphs");
  const stems = { "gemma4-12b": "gemma-4-12B-it-OptiQ-4bit", "gemma4-e4b": "gemma-4-e4b-it-OptiQ-4bit", "gemma4-26b": "gemma-4-26B-A4B-it-OptiQ-4bit" };
  const fixture = async (repo: string) => {
    const snapshots = join(fixtures, `models--mlx-community--${repo}`, "snapshots");
    return loadModelConfig(join(snapshots, readdirSync(snapshots)[0]!));
  };

  for (const [stem, repo] of Object.entries(stems)) {
    test(`${stem}: its fingerprint with its kv_config scheme selects the generated graph; any other scheme the generic graph`, async () => {
      const config = await fixture(repo);
      const pick = (request: CompositionRequest) => table.select(config, on(M4_PRO, config, request)).id;
      expect(on(M4_PRO, config, { kv: { override: "config" } }).kv.kind).toBe("affine-layers");
      expect(pick({ kv: { override: "config" } })).toBe(stem);
      expect(pick({ kv: { override: "config", quantizedKvStart: 100 } })).toBe(stem);
      // `--fused-sdpa` names the read kernels its caches compose, not another graph.
      for (const fusedSdpa of [true, false]) expect(pick({ kv: { override: "config", fusedSdpa } })).toBe(stem);
      // Selection does not depend on the device, rows or adapters for these entries.
      expect(table.select(config, on(M1_MAX, config, { kv: { override: "config" }, maxRows: 32, adapters: true })).id).toBe(stem);
      for (const other of [{}, { kv: { override: "off" as const } }, { kv: { override: 4 } }, { kv: { override: 8 } },
        { kv: { turboQuant: { kBits: 8, vBits: 3 } } }]) expect(pick(other)).toBe("gemma4");
    });
  }

  test("a generated file's scheme on another fingerprint, or a non-generated Gemma 4 artifact, gets the generic graph", async () => {
    const e4b = await fixture(stems["gemma4-e4b"]), twelve = await fixture(stems["gemma4-12b"]);
    const e4bScheme = on(M4_PRO, e4b, { kv: { override: "config" } });
    expect(table.match(configFingerprint(twelve), e4bScheme).id).toBe("gemma4");
    expect(table.match("0000000000000000", e4bScheme).id).toBe("gemma4");
  });
});

test("every resident family's table has its generic graph, named for the family", () => {
  for (const [graph, table] of Object.entries(MLX_GRAPH_TABLES)) expect([graph, table.generic.id]).toEqual([graph, graph]);
});
