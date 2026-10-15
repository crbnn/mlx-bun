import { expect, test } from "bun:test";
import { parseQuantization, quantFor, type ModelConfig } from "../../src/artifacts/config";
import type { Qwen3Layer } from "../../src/models/qwen/blocks";
import type { Weights } from "../../src/artifacts/weights";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxRowTable } from "../../src/contracts/mlx/graph";
import { Qwen35Model, qwen35KnownSpansEligible } from "../../src/models/qwen/qwen3_5";
import { rowOperationFor } from "../../src/models/graph";

function fixture(quant: Record<string, unknown> = {}) {
  const config = {
    text: { hiddenSize: 5120, intermediateSize: 17408, numHiddenLayers: 64,
      headDim: 256, numAttentionHeads: 24, numKeyValueHeads: 4, fullAttentionInterval: 4 },
    quantization: parseQuantization({ bits: 4, group_size: 64, mode: "affine", ...quant }),
  } as ModelConfig;
  // Host-only loaded projection facts: the constructor's predicate reads the
  // actual carriers' specs, never tensors or the loader's recommendation table.
  const layers = Array.from({ length: 64 }, (_, layer) => {
    const prefix = `language_model.model.layers.${layer}`;
    const projection = (role: string) => ({ spec: quantFor(config.quantization, `${prefix}.${role}`) ?? { mode: "none" } });
    const mlp = { gate: projection("mlp.gate_proj"), up: projection("mlp.up_proj"), down: projection("mlp.down_proj") };
    return {
      linearAttn: (layer + 1) % 4 ? { inProjQkv: projection("linear_attn.in_proj_qkv"), inProjZ: projection("linear_attn.in_proj_z"),
        inProjB: projection("linear_attn.in_proj_b"), inProjA: projection("linear_attn.in_proj_a"), outProj: projection("linear_attn.out_proj") } : null,
      selfAttn: (layer + 1) % 4 ? null : { qProj: projection("self_attn.q_proj"), kProj: projection("self_attn.k_proj"),
        vProj: projection("self_attn.v_proj"), oProj: projection("self_attn.o_proj") },
      mlp: { ...mlp, kind: [mlp.gate, mlp.up, mlp.down].every(p => p.spec?.mode === "trellis") ? "trellis" : "affine" },
    } as unknown as Qwen3Layer;
  });
  return { config, layers, eligible: (device = "applegpu_g16s") => qwen35KnownSpansEligible(config, layers, device) };
}

test("generic committed-span qualification preserves every supported affine width and group", () => {
  for (const bits of [2, 3, 4, 6, 8]) for (const group_size of [32, 64, 128])
    expect(fixture({ bits, group_size }).eligible()).toBe(true);
  for (const quant of [{ bits: 5 }, { group_size: 256 }, { mode: "mxfp4", bits: 4, group_size: 32 }])
    expect(fixture(quant).eligible()).toBe(false);
});

test("construction qualification uses device, full geometry and every actual loaded projection", () => {
  for (const device of ["applegpu_g13s", "applegpu_g15s", "applegpu_g16x", "unknown-gpu"])
    expect(fixture().eligible(device)).toBe(false);
  for (const field of ["hiddenSize", "intermediateSize", "headDim", "numAttentionHeads", "numKeyValueHeads"] as const) {
    const f = fixture(); f.config.text = { ...f.config.text, [field]: f.config.text[field] + 1 };
    expect(f.eligible()).toBe(false);
  }
  const short = fixture(); short.layers.pop(); expect(short.eligible()).toBe(false);
  for (const path of ["language_model.model.layers.62.linear_attn.out_proj", "language_model.model.layers.63.self_attn.o_proj",
    "language_model.model.layers.63.mlp.down_proj"])
    expect(fixture({ [path]: { bits: 5, group_size: 64, mode: "affine" } }).eligible()).toBe(false);
  const loaded = fixture();
  // A config edit after loading cannot hide an incompatible actual carrier.
  Object.assign(loaded.layers[63]!.selfAttn!.oProj, { spec: { mode: "affine", bits: 5, groupSize: 64 } });
  expect(loaded.eligible()).toBe(false);
});

test("all-Trellis MLPs retain qualification while mixed or unquantized projections do not", () => {
  const overrides: Record<string, unknown> = {};
  for (let layer = 0; layer < 64; layer++) for (const role of ["gate_proj", "up_proj", "down_proj"])
    overrides[`language_model.model.layers.${layer}.mlp.${role}`] = {
      bits: 3, group_size: 256, mode: "trellis", trellis: { L: 12, code: "1mad", axis: role === "down_proj" ? 0 : 1 },
    };
  expect(fixture(overrides).eligible()).toBe(true);
  delete overrides["language_model.model.layers.63.mlp.up_proj"];
  expect(fixture(overrides).eligible()).toBe(false);
  expect(fixture({ "language_model.model.layers.0.linear_attn.in_proj_qkv": { mode: "none" } }).eligible()).toBe(false);
});

test("a small admission cap retains Qwen's exact narrow rows and explicit expanded default", () => {
  // Zero layers and borrowed host sentinels exercise real constructor
  // declarations without creating, evaluating or submitting native arrays.
  const payload = Object.freeze({}) as MlxArray;
  const weights = { tensorNames: [], shards: { files: new Map() }, has: () => false,
    tensor: () => payload } as unknown as Weights;
  const config = { ...fixture().config, text: { ...fixture().config.text,
    hiddenSize: 512, intermediateSize: 512, numHiddenLayers: 0, tieWordEmbeddings: true, rmsNormEps: 1e-6 } };
  for (const cap of [1, 8]) {
    const graph = new Qwen35Model(weights, config, cap, "applegpu_g16s");
    try {
      const tables: readonly MlxRowTable<unknown>[] = [graph.decode, graph.verify, graph.mediaDecode];
      for (const table of tables) {
        expect(Object.hasOwn(table, "default")).toBe(true);
        for (let rows = 1; rows <= 32; rows++) {
          expect(Object.hasOwn(table, rows)).toBe(true);
          expect(rowOperationFor(table, rows)).not.toBe(table.default!);
        }
        for (const [first, last] of [[1, 4], [5, 8], [9, 15], [16, 32]])
          expect(rowOperationFor(table, first!)).toBe(rowOperationFor(table, last!));
        expect(rowOperationFor(table, 33)).toBe(table.default!);
        expect(rowOperationFor(table, 262144)).toBe(table.default!);
        expect(table[33]).toBeUndefined();
      }
      expect(rowOperationFor(graph.logits, 1)).toBe(rowOperationFor(graph.logits, 262144));
    } finally { graph.dispose(); }
  }
});
