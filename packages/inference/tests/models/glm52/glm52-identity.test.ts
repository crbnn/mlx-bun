// GLM-5.2's graph against its frozen copy (tests/models/glm52-reference, the
// graph, MLA block, MTP row and compressed cache at the B4 base) over synthetic
// weights, at maxDiff 0: a dense layer and two MoE layers (shared and routed
// experts), DSA on and off (a full indexer layer, a layer that shares its
// selection, and selections past `index_topk`, so the token-serial DSA loops
// run), native MTP. Every phase the scheduler calls: a prompt chunk, a prompt
// tail, decodes, a verify window, known tokens, two batched rows of different
// lengths (decode and verify), the scoring forward; the hidden states, the
// logits and every stored cache tensor after each. No artifact exists on this
// machine; this is the oracle for the graph's own changes.
import { afterAll, describe, expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import type { ModelConfig } from "../../../src/artifacts/config";
import { parseGlm52Config, type Glm52Config } from "../../../src/artifacts/glm52-config";
import type { Cache, LatentAttentionCache } from "../../../src/contracts/mlx/cache";
import type { Glm52WeightSource } from "../../../src/contracts/mlx/glm52-weights";
import { widthClass } from "../../../src/contracts/mlx/width";
import { Glm52Model } from "../../../src/models/glm52/model";
import { Glm52Model as FrozenGlm52Model } from "../glm52-reference/model";
import { MLACache as FrozenMLACache } from "../glm52-reference/cache";
import { MLACache } from "../../../src/state/glm52-cache";

// ---------------------------------------------------------------------------
// Synthetic geometry and weights

const HIDDEN = 16, HEADS = 2, Q_RANK = 8, KV_RANK = 8, NOPE = 4, ROPE = 4, VALUE = 4;
const INTER = 16, MOE_INTER = 8, EXPERTS = 4, VOCAB = 48, LAYERS = 3;
const INDEX_TOPK = 6, INDEX_HEADS = 2, INDEX_DIM = 8;

function glmConfig(dsa: boolean): Glm52Config {
  return parseGlm52Config("/synthetic", {
    model_type: "glm_moe_dsa", architectures: ["GlmMoeDsaForCausalLM"],
    hidden_size: HIDDEN, num_hidden_layers: LAYERS, num_attention_heads: HEADS, num_key_value_heads: HEADS,
    q_lora_rank: Q_RANK, kv_lora_rank: KV_RANK, qk_nope_head_dim: NOPE, qk_rope_head_dim: ROPE, v_head_dim: VALUE,
    first_k_dense_replace: 1, intermediate_size: INTER, moe_intermediate_size: MOE_INTER,
    n_routed_experts: EXPERTS, num_experts_per_tok: 2, n_shared_experts: 1, n_group: 1, topk_group: 1,
    norm_topk_prob: true, routed_scaling_factor: 1.5, rms_norm_eps: 1e-5, rope_theta: 10_000, rope_interleave: true,
    vocab_size: VOCAB, max_position_embeddings: 256,
    ...(dsa ? { index_topk: INDEX_TOPK, index_n_heads: INDEX_HEADS, index_head_dim: INDEX_DIM,
      indexer_types: ["full", "shared", "full"] } : {}),
    num_nextn_predict_layers: 1, eos_token_id: [1], pad_token_id: 0,
  });
}

const values = (count: number, seed: number, scale = 0.3) => Float32Array.from({ length: count },
  (_, i) => Math.fround(Math.sin((i + 1) * (seed * 0.731 + 0.37)) * scale + Math.cos((i + 3) * 0.19 + seed) * scale * 0.25));

/** Every tensor the graph and its MTP row read, by name. */
function syntheticTensors(config: Glm52Config): Map<string, { data: Float32Array; shape: number[] }> {
  const tensors = new Map<string, { data: Float32Array; shape: number[] }>();
  let seed = 1;
  const add = (name: string, shape: number[], scale = 0.3) => {
    tensors.set(name, { data: values(shape.reduce((a, b) => a * b, 1), seed++, scale), shape });
  };
  const norm = (name: string, width: number) =>
    tensors.set(name, { data: Float32Array.from({ length: width }, (_, i) => 1 + 0.05 * Math.sin(i + seed++)), shape: [width] });
  const swiglu = (prefix: string, inter: number) => {
    add(`${prefix}.gate_proj.weight`, [inter, HIDDEN]);
    add(`${prefix}.up_proj.weight`, [inter, HIDDEN]);
    add(`${prefix}.down_proj.weight`, [HIDDEN, inter]);
  };
  add("model.embed_tokens.weight", [VOCAB, HIDDEN], 1);
  norm("model.norm.weight", HIDDEN);
  add("lm_head.weight", [VOCAB, HIDDEN]);
  for (let layer = 0; layer <= LAYERS; layer++) {
    const p = `model.layers.${layer}`;
    norm(`${p}.input_layernorm.weight`, HIDDEN);
    norm(`${p}.post_attention_layernorm.weight`, HIDDEN);
    add(`${p}.self_attn.q_a_proj.weight`, [Q_RANK, HIDDEN]);
    norm(`${p}.self_attn.q_a_layernorm.weight`, Q_RANK);
    add(`${p}.self_attn.q_b_proj.weight`, [HEADS * (NOPE + ROPE), Q_RANK]);
    add(`${p}.self_attn.kv_a_proj_with_mqa.weight`, [KV_RANK + ROPE, HIDDEN]);
    norm(`${p}.self_attn.kv_a_layernorm.weight`, KV_RANK);
    add(`${p}.self_attn.kv_b_proj.weight`, [HEADS * (NOPE + VALUE), KV_RANK]);
    add(`${p}.self_attn.o_proj.weight`, [HIDDEN, HEADS * VALUE]);
    if (layer < LAYERS && config.indexTopk > 0 && config.indexerTypes[layer] === "full") {
      add(`${p}.self_attn.indexer.wq_b.weight`, [INDEX_HEADS * INDEX_DIM, Q_RANK]);
      add(`${p}.self_attn.indexer.wk.weight`, [INDEX_DIM, HIDDEN]);
      norm(`${p}.self_attn.indexer.k_norm.weight`, INDEX_DIM);
      add(`${p}.self_attn.indexer.k_norm.bias`, [INDEX_DIM], 0.05);
      add(`${p}.self_attn.indexer.weights_proj.weight`, [INDEX_HEADS, HIDDEN]);
    }
    if (layer < config.firstKDenseReplace) swiglu(`${p}.mlp`, INTER);
    else {
      add(`${p}.mlp.gate.weight`, [EXPERTS, HIDDEN], 1);
      add(`${p}.mlp.gate.e_score_correction_bias`, [EXPERTS], 0.1);
      swiglu(`${p}.mlp.shared_experts`, MOE_INTER);
      for (let expert = 0; expert < EXPERTS; expert++) swiglu(`${p}.mlp.experts.${expert}`, MOE_INTER);
    }
  }
  const mtp = `model.layers.${LAYERS}`;
  norm(`${mtp}.enorm.weight`, HIDDEN);
  norm(`${mtp}.hnorm.weight`, HIDDEN);
  add(`${mtp}.eh_proj.weight`, [HIDDEN, 2 * HIDDEN]);
  norm(`${mtp}.shared_head.norm.weight`, HIDDEN);
  return tensors;
}

/** An in-memory f32 weight source (the structural seam GLM-5.2's fixtures use). */
class SyntheticWeights implements Glm52WeightSource {
  readonly arrays = new Map<string, MlxArray>();
  readonly weightsBytes: number;
  constructor(config: Glm52Config) {
    let bytes = 0;
    for (const [name, value] of syntheticTensors(config)) {
      this.arrays.set(name, MlxArray.fromFloat32(value.data, value.shape));
      bytes += value.data.byteLength;
    }
    this.weightsBytes = bytes;
  }
  has(name: string): boolean { return this.arrays.has(name); }
  tensor(name: string): MlxArray {
    const value = this.arrays.get(name);
    if (!value) throw new Error(`missing synthetic tensor ${name}`);
    return value;
  }
  dequantized(name: string, outputRows: number, inputColumns: number): MlxArray {
    const source = this.tensor(name);
    expect(source.shape).toEqual([outputRows, inputColumns]);
    return ops.contiguous(source);
  }
  linear(input: MlxArray, name: string, outputRows: number, inputColumns: number): MlxArray {
    const weight = this.tensor(name);
    expect(weight.shape).toEqual([outputRows, inputColumns]);
    using transpose = ops.transposeAxes(weight, [1, 0]);
    return ops.matmul(input, transpose);
  }
  embedding(ids: MlxArray, name: string, vocabSize: number, hiddenSize: number): MlxArray {
    const table = this.tensor(name);
    expect(table.shape).toEqual([vocabSize, hiddenSize]);
    using flat = ops.reshape(ids, [ids.size]);
    using rows = ops.takeAxis(table, flat, 0);
    return ops.reshape(rows, [...ids.shape, hiddenSize]);
  }
  dispose(): void {
    for (const array of this.arrays.values()) array.dispose();
    this.arrays.clear();
  }
}

// ---------------------------------------------------------------------------
// Comparison

const totals = new Map<string, { tensors: number; worst: number }>();
afterAll(() => {
  for (const [label, { tensors, worst }] of totals)
    console.log(`glm52-identity ${label}: maxDiff ${worst} over ${tensors} tensors`);
});

function host(a: MlxArray): Float32Array {
  using wide = a.astype(Dtype.float32);
  using flat = ops.contiguous(wide);
  return flat.toFloat32();
}
function expectSame(label: string, scenario: string, actual: MlxArray, expected: MlxArray): void {
  expect(actual.shape, `${label} shape`).toEqual(expected.shape);
  expect(actual.dtype, `${label} dtype`).toBe(expected.dtype);
  const x = host(actual), y = host(expected);
  let worst = 0;
  for (let i = 0; i < x.length; i++) {
    const d = Math.abs(x[i]! - y[i]!);
    if (!(d <= worst)) worst = Number.isNaN(d) ? Infinity : d;
  }
  const total = totals.get(scenario) ?? { tensors: 0, worst: 0 };
  totals.set(scenario, { tensors: total.tensors + 1, worst: Math.max(total.worst, worst) });
  expect(worst, `${label} maxDiff`).toBe(0);
}
/** Every stored tensor of both cache lists, in order. */
function expectCaches(label: string, scenario: string, actual: Cache[], expected: Cache[]): void {
  expect(actual.length).toBe(expected.length);
  for (let layer = 0; layer < actual.length; layer++) {
    const a = actual[layer]!, e = expected[layer]!;
    expect(a.offset, `${label} layer ${layer} offset`).toBe(e.offset);
    expect([...(a as LatentAttentionCache).rowOffsets], `${label} layer ${layer} rows`)
      .toEqual([...(e as LatentAttentionCache).rowOffsets]);
    const as = a.state(), es = e.state();
    expect(as.length, `${label} layer ${layer} planes`).toBe(es.length);
    for (let plane = 0; plane < as.length; plane++)
      expectSame(`${label} layer ${layer} plane ${plane}`, scenario, as[plane]!, es[plane]!);
  }
}

type Graph = Glm52Model | FrozenGlm52Model;
type Phase = "chunk" | "tail" | "decode" | "verify" | "known";
/** The phase operation for the rows of `ids` (sequences times tokens), as the scheduler indexes it. */
async function phase(graph: Graph, which: Phase, ids: MlxArray, cache: Cache[]): Promise<MlxArray> {
  const width = widthClass(ids.shape[0]! * ids.shape[1]!);
  switch (which) {
    case "chunk": return await graph.prefillChunk.gpu(ids, cache, null);
    case "tail": return await graph.prefillTail[width](ids, cache, null);
    case "decode": return await graph.decode[width](ids, cache);
    case "verify": return await graph.verify[width](ids, cache, null);
    case "known": return await graph.appendKnown[width](ids, cache);
  }
}
const tokens = (count: number, salt: number) => Array.from({ length: count }, (_, i) => 3 + (i * 7 + salt * 5) % (VOCAB - 3));

/** Run one phase on both graphs over the same ids and compare the hidden, the
 * logits and the caches. */
async function both(scenario: string, label: string, graphs: { current: Graph; frozen: Graph },
  caches: { current: Cache[]; frozen: Cache[] }, which: Phase, rows: number[][]): Promise<void> {
  using ids = ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);
  using actual = await phase(graphs.current, which, ids, caches.current);
  using expected = await phase(graphs.frozen, which, ids, caches.frozen);
  expectSame(`${scenario} ${label} hidden`, scenario, actual, expected);
  using actualLogits = graphs.current.logitsFromHidden(actual);
  using expectedLogits = graphs.frozen.logitsFromHidden(expected);
  expectSame(`${scenario} ${label} logits`, scenario, actualLogits, expectedLogits);
  expectCaches(`${scenario} ${label}`, scenario, caches.current, caches.frozen);
}

function dispose(...lists: Cache[][]): void { for (const list of lists) for (const cache of list) cache.dispose(); }

for (const dsa of [true, false]) describe(`GLM-5.2 graph identity (DSA ${dsa ? "on" : "off"})`, () => {
  const glm = glmConfig(dsa);
  const config = { modelType: "glm_moe_dsa", text: { numHiddenLayers: LAYERS }, eosTokenIds: [1] } as unknown as ModelConfig;
  const weights = new SyntheticWeights(glm);
  const capabilities = { dsa, mtpMetadata: true, mtpEnabled: true, mtpDraftTokens: 3 };
  const graphs = { current: new Glm52Model(weights, config, glm, capabilities), frozen: new FrozenGlm52Model(weights, config, glm, capabilities) };
  const scenario = `DSA ${dsa ? "on" : "off"}`;
  afterAll(() => weights.dispose());

  test("one sequence: chunk past index_topk, tail, decodes, verify, known tokens", async () => {
    const caches = { current: graphs.current.makeCache() as Cache[], frozen: graphs.frozen.makeCache() as Cache[] };
    try {
      const modes: string[] = [];
      graphs.current.setDsaSelectionObserver(selection => modes.push(selection.mode));
      await both(scenario, "chunk 12", graphs, caches, "chunk", [tokens(12, 1)]);
      await both(scenario, "tail 3", graphs, caches, "tail", [tokens(3, 2)]);
      for (let step = 0; step < 3; step++) await both(scenario, `decode ${step}`, graphs, caches, "decode", [tokens(1, 3 + step)]);
      await both(scenario, "verify 4", graphs, caches, "verify", [tokens(4, 7)]);
      await both(scenario, "known 2", graphs, caches, "known", [tokens(2, 8)]);
      await both(scenario, "known 1", graphs, caches, "known", [tokens(1, 9)]);
      await both(scenario, "decode after", graphs, caches, "decode", [tokens(1, 10)]);
      // The DSA scenario reads sparse selections past index_topk (the token-serial loops) as well as dense ones.
      if (dsa) expect([...new Set(modes)].sort()).toEqual(["dense", "sparse"]);
      else expect(modes).toEqual([]);
    } finally { graphs.current.setDsaSelectionObserver(null); dispose(caches.current, caches.frozen); }
  });

  test("a short prompt below index_topk, then decodes across it", async () => {
    const caches = { current: graphs.current.makeCache() as Cache[], frozen: graphs.frozen.makeCache() as Cache[] };
    try {
      await both(scenario, "chunk 4", graphs, caches, "chunk", [tokens(4, 11)]);
      for (let step = 0; step < 4; step++) await both(scenario, `decode ${step}`, graphs, caches, "decode", [tokens(1, 12 + step)]);
      await both(scenario, "verify 3", graphs, caches, "verify", [tokens(3, 16)]);
    } finally { dispose(caches.current, caches.frozen); }
  });

  test("two batched rows of different lengths: decode and verify", async () => {
    const serial = (graph: Graph) => graph.makeCache() as Cache[];
    const rows = { current: [serial(graphs.current), serial(graphs.current)], frozen: [serial(graphs.frozen), serial(graphs.frozen)] };
    const batched: { current: Cache[]; frozen: Cache[] } = { current: [], frozen: [] };
    try {
      for (const [row, length] of [9, 5].entries()) {
        await both(scenario, `row ${row} prompt`, graphs, { current: rows.current[row]!, frozen: rows.frozen[row]! }, "chunk",
          [tokens(length, 21 + row)]);
      }
      for (const side of ["current", "frozen"] as const) {
        for (let layer = 0; layer < LAYERS; layer++) {
          const first = rows[side][0]![layer]! as LatentAttentionCache & { makeEmptyBatch(): Cache; };
          const layout = first.makeEmptyBatch() as Cache & { mergeRows(rows: readonly Cache[]): void };
          layout.mergeRows(rows[side].map(list => list[layer]!));
          batched[side].push(layout);
        }
      }
      expectCaches(`${scenario} merged`, scenario, batched.current, batched.frozen);
      for (let step = 0; step < 2; step++)
        await both(scenario, `rows decode ${step}`, graphs, batched, "decode", [tokens(1, 30 + step), tokens(1, 40 + step)]);
      await both(scenario, "rows verify 3", graphs, batched, "verify", [tokens(3, 50), tokens(3, 51)]);
    } finally { dispose(...rows.current, ...rows.frozen, batched.current, batched.frozen); }
  });

  test("the scoring forward over the whole sequence", () => {
    const caches = { current: graphs.current.makeCache() as Cache[], frozen: graphs.frozen.makeCache() as Cache[] };
    try {
      using ids = ops.fromInt32(tokens(5, 60), [1, 5]);
      using actual = graphs.current.sequenceHidden(ids, caches.current, "fiveToEight");
      using expected = graphs.frozen.sequenceHidden(ids, caches.frozen, "fiveToEight");
      expectSame(`${scenario} scoring hidden`, scenario, actual, expected);
      expectCaches(`${scenario} scoring`, scenario, caches.current, caches.frozen);
    } finally { dispose(caches.current, caches.frozen); }
  });

  test("native MTP: the draft row's forward and head over its own compressed state", async () => {
    const current = graphs.current.nativeDraftHead(), frozen = graphs.frozen.nativeDraftHead();
    const caches = { current: graphs.current.makeCache() as Cache[], frozen: graphs.frozen.makeCache() as Cache[] };
    const geometry = { ...current.cache, role: "mtp" as const };
    const mtpCache = { current: new MLACache(geometry) as Cache, frozen: new FrozenMLACache(geometry) as Cache };
    try {
      using ids = ops.fromInt32(tokens(4, 70), [1, 4]);
      using hidden = await graphs.current.prefillChunk.gpu(ids, caches.current, null);
      using hiddenFrozen = await graphs.frozen.prefillChunk.gpu(ids, caches.frozen, null);
      expectSame(`${scenario} mtp anchor`, scenario, hidden, hiddenFrozen);
      for (let step = 0; step < 2; step++) {
        using draftIds = ops.fromInt32(tokens(step === 0 ? 4 : 1, 71 + step), [1, step === 0 ? 4 : 1]);
        using anchor = step === 0 ? ops.contiguous(hidden) : hidden.slice([0, 3, 0], [1, 4, HIDDEN]);
        // The current head names the phase; the frozen one chose it by token count.
        using out = await (step === 0 ? current.window(draftIds, anchor, mtpCache.current as LatentAttentionCache) : current.decode(draftIds, anchor, mtpCache.current as LatentAttentionCache));
        using outFrozen = await frozen.forward(draftIds, anchor, mtpCache.frozen as LatentAttentionCache);
        expectSame(`${scenario} mtp ${step} hidden`, scenario, out, outFrozen);
        using logits = current.project(out);
        using logitsFrozen = frozen.project(outFrozen);
        expectSame(`${scenario} mtp ${step} logits`, scenario, logits, logitsFrozen);
      }
      expectCaches(`${scenario} mtp`, scenario, [mtpCache.current], [mtpCache.frozen]);
    } finally { dispose(caches.current, caches.frozen, [mtpCache.current, mtpCache.frozen]); }
  });
});
