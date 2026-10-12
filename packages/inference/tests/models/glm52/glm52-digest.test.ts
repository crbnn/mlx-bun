// GLM-5.2's graph over synthetic weights, held to its recorded digest
// (tests/fixtures/synthetic-digests/glm52.json; tests/models/synthetic-digest.ts):
// a dense layer and two MoE layers (shared and routed experts), DSA on and off
// (a full indexer layer, a layer that shares its selection, and selections past
// `index_topk`, so the token-serial DSA loops run), native MTP. Every phase the
// scheduler calls: a prompt chunk, a prompt tail, decodes, a verify window, known
// tokens, two batched rows of different lengths (decode and verify), the scoring
// forward; the hidden states, the logits and every stored cache tensor after
// each. No artifact holds GLM-5.2 numbers on either machine; the digest was
// recorded from the tree whose graph equalled the B4 base's at maxDiff 0 on
// these cases, so it carries that check forward. The latent legos follow: f32
// pages against the f32 lego at maxDiff 0, the encoded forms by KL.
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
import { MLACache } from "../../../src/state/glm52-cache";
import type { KvStorageScheme } from "../../../src/contracts/portable/composition";
import { syntheticDigests, type DigestCase } from "../synthetic-digest";

// ---------------------------------------------------------------------------
// Synthetic geometry and weights

const HIDDEN = 16, HEADS = 2, Q_RANK = 8, KV_RANK = 8, NOPE = 4, ROPE = 4, VALUE = 4;
const INTER = 16, MOE_INTER = 8, EXPERTS = 4, VOCAB = 48, LAYERS = 3;
const INDEX_TOPK = 6, INDEX_HEADS = 2, INDEX_DIM = 8;

/** The latent widths: the digest scenarios' small ones, or widths every latent lego encodes (affine groups of 64, TurboQuant's widths). */
interface LatentWidths { readonly kvRank: number; readonly rope: number; readonly index: number }
const SMALL: LatentWidths = { kvRank: KV_RANK, rope: ROPE, index: INDEX_DIM };

function glmConfig(dsa: boolean, widths: LatentWidths = SMALL): Glm52Config {
  return parseGlm52Config("/synthetic", {
    model_type: "glm_moe_dsa", architectures: ["GlmMoeDsaForCausalLM"],
    hidden_size: HIDDEN, num_hidden_layers: LAYERS, num_attention_heads: HEADS, num_key_value_heads: HEADS,
    q_lora_rank: Q_RANK, kv_lora_rank: widths.kvRank, qk_nope_head_dim: NOPE, qk_rope_head_dim: widths.rope, v_head_dim: VALUE,
    first_k_dense_replace: 1, intermediate_size: INTER, moe_intermediate_size: MOE_INTER,
    n_routed_experts: EXPERTS, num_experts_per_tok: 2, n_shared_experts: 1, n_group: 1, topk_group: 1,
    norm_topk_prob: true, routed_scaling_factor: 1.5, rms_norm_eps: 1e-5, rope_theta: 10_000, rope_interleave: true,
    vocab_size: VOCAB, max_position_embeddings: 256,
    ...(dsa ? { index_topk: INDEX_TOPK, index_n_heads: INDEX_HEADS, index_head_dim: widths.index,
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
    add(`${p}.self_attn.q_b_proj.weight`, [HEADS * (NOPE + config.qkRopeHeadDim), Q_RANK]);
    add(`${p}.self_attn.kv_a_proj_with_mqa.weight`, [config.kvLoraRank + config.qkRopeHeadDim, HIDDEN]);
    norm(`${p}.self_attn.kv_a_layernorm.weight`, config.kvLoraRank);
    add(`${p}.self_attn.kv_b_proj.weight`, [HEADS * (NOPE + VALUE), config.kvLoraRank]);
    add(`${p}.self_attn.o_proj.weight`, [HIDDEN, HEADS * VALUE]);
    if (layer < LAYERS && config.indexTopk > 0 && config.indexerTypes[layer] === "full") {
      add(`${p}.self_attn.indexer.wq_b.weight`, [INDEX_HEADS * config.indexHeadDim, Q_RANK]);
      add(`${p}.self_attn.indexer.wk.weight`, [config.indexHeadDim, HIDDEN]);
      norm(`${p}.self_attn.indexer.k_norm.weight`, config.indexHeadDim);
      add(`${p}.self_attn.indexer.k_norm.bias`, [config.indexHeadDim], 0.05);
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
// Digest

const digests = syntheticDigests("glm52");

function host(a: MlxArray): Float32Array {
  using wide = a.astype(Dtype.float32);
  using flat = ops.contiguous(wide);
  return flat.toFloat32();
}
/** Same shape, dtype and bytes. */
function expectSame(label: string, actual: MlxArray, expected: MlxArray): void {
  expect(actual.shape, `${label} shape`).toEqual(expected.shape);
  expect(actual.dtype, `${label} dtype`).toBe(expected.dtype);
  using left = ops.contiguous(actual), right = ops.contiguous(expected);
  expect(Buffer.from(left.rawBytes()).equals(Buffer.from(right.rawBytes())), `${label} maxDiff 0`).toBe(true);
}

type Phase = "chunk" | "tail" | "decode" | "verify" | "known";
/** The phase operation for the rows of `ids` (sequences times tokens), as the scheduler indexes it. */
async function phase(graph: Glm52Model, which: Phase, ids: MlxArray, cache: Cache[]): Promise<MlxArray> {
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

/** Run one phase over `rows` and hand the hidden, the logits and the caches to the digest. */
async function step(digest: DigestCase, label: string, graph: Glm52Model, caches: Cache[], which: Phase,
  rows: number[][]): Promise<void> {
  using ids = ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);
  using hidden = await phase(graph, which, ids, caches);
  using logits = graph.logitsFromHidden(hidden);
  digest.tensors(`${label} hidden`, hidden);
  digest.tensors(`${label} logits`, logits);
  digest.caches(`${label} caches`, caches);
}

function dispose(...lists: Cache[][]): void { for (const list of lists) for (const cache of list) cache.dispose(); }

for (const dsa of [true, false]) describe(`GLM-5.2 graph (DSA ${dsa ? "on" : "off"})`, () => {
  const glm = glmConfig(dsa);
  const config = { modelType: "glm_moe_dsa", text: { numHiddenLayers: LAYERS }, eosTokenIds: [1] } as unknown as ModelConfig;
  const weights = new SyntheticWeights(glm);
  const capabilities = { dsa, mtpMetadata: true, mtpEnabled: true, mtpDraftTokens: 3 };
  const graph = new Glm52Model(weights, config, glm, capabilities);
  const scenario = `DSA ${dsa ? "on" : "off"}`;
  afterAll(() => weights.dispose());

  digests.test(`${scenario}: one sequence: chunk past index_topk, tail, decodes, verify, known tokens`, async digest => {
    const caches = graph.makeCache() as Cache[];
    try {
      const modes: string[] = [];
      graph.setDsaSelectionObserver(selection => modes.push(selection.mode));
      await step(digest, "chunk 12", graph, caches, "chunk", [tokens(12, 1)]);
      await step(digest, "tail 3", graph, caches, "tail", [tokens(3, 2)]);
      for (let index = 0; index < 3; index++) await step(digest, `decode ${index}`, graph, caches, "decode", [tokens(1, 3 + index)]);
      await step(digest, "verify 4", graph, caches, "verify", [tokens(4, 7)]);
      await step(digest, "known 2", graph, caches, "known", [tokens(2, 8)]);
      await step(digest, "known 1", graph, caches, "known", [tokens(1, 9)]);
      await step(digest, "decode after", graph, caches, "decode", [tokens(1, 10)]);
      // The DSA scenario reads sparse selections past index_topk (the token-serial loops) as well as dense ones.
      if (dsa) expect([...new Set(modes)].sort()).toEqual(["dense", "sparse"]);
      else expect(modes).toEqual([]);
    } finally { graph.setDsaSelectionObserver(null); dispose(caches); }
  });

  digests.test(`${scenario}: a short prompt below index_topk, then decodes across it`, async digest => {
    const caches = graph.makeCache() as Cache[];
    try {
      await step(digest, "chunk 4", graph, caches, "chunk", [tokens(4, 11)]);
      for (let index = 0; index < 4; index++) await step(digest, `decode ${index}`, graph, caches, "decode", [tokens(1, 12 + index)]);
      await step(digest, "verify 3", graph, caches, "verify", [tokens(3, 16)]);
    } finally { dispose(caches); }
  });

  digests.test(`${scenario}: two batched rows of different lengths: decode and verify`, async digest => {
    const rows = [graph.makeCache() as Cache[], graph.makeCache() as Cache[]];
    const batched: Cache[] = [];
    try {
      for (const [row, length] of [9, 5].entries())
        await step(digest, `row ${row} prompt`, graph, rows[row]!, "chunk", [tokens(length, 21 + row)]);
      for (let layer = 0; layer < LAYERS; layer++) {
        const first = rows[0]![layer]! as LatentAttentionCache & { makeEmptyBatch(): Cache; };
        const layout = first.makeEmptyBatch() as Cache & { mergeRows(rows: readonly Cache[]): void };
        layout.mergeRows(rows.map(list => list[layer]!));
        batched.push(layout);
      }
      digest.caches("merged", batched);
      for (let index = 0; index < 2; index++)
        await step(digest, `rows decode ${index}`, graph, batched, "decode", [tokens(1, 30 + index), tokens(1, 40 + index)]);
      await step(digest, "rows verify 3", graph, batched, "verify", [tokens(3, 50), tokens(3, 51)]);
    } finally { dispose(...rows, batched); }
  });

  digests.test(`${scenario}: the scoring forward over the whole sequence`, digest => {
    const caches = graph.makeCache() as Cache[];
    try {
      using ids = ops.fromInt32(tokens(5, 60), [1, 5]);
      using hidden = graph.sequenceHidden(ids, caches, "fiveToEight");
      digest.tensors("scoring hidden", hidden);
      digest.caches("scoring caches", caches);
    } finally { dispose(caches); }
  });

  digests.test(`${scenario}: native MTP: the draft row's forward and head over its own compressed state`, async digest => {
    const head = graph.nativeDraftHead();
    const caches = graph.makeCache() as Cache[];
    const mtpCache = new MLACache({ ...head.cache, role: "mtp" as const });
    try {
      using ids = ops.fromInt32(tokens(4, 70), [1, 4]);
      using hidden = await graph.prefillChunk.gpu(ids, caches, null);
      digest.tensors("mtp anchor", hidden);
      for (let index = 0; index < 2; index++) {
        using draftIds = ops.fromInt32(tokens(index === 0 ? 4 : 1, 71 + index), [1, index === 0 ? 4 : 1]);
        using anchor = index === 0 ? ops.contiguous(hidden) : hidden.slice([0, 3, 0], [1, 4, HIDDEN]);
        using out = await (index === 0 ? head.window(draftIds, anchor, mtpCache) : head.decode(draftIds, anchor, mtpCache));
        using logits = head.project(out);
        digest.tensors(`mtp ${index} hidden`, out);
        digest.tensors(`mtp ${index} logits`, logits);
      }
      digest.caches("mtp caches", [mtpCache]);
    } finally { dispose(caches, [mtpCache]); }
  });
});

// ---------------------------------------------------------------------------
// The latent legos (B4): the caches GLM-5.2's graph composes for the
// composition's KV scheme and paging. Latent widths every lego encodes (affine
// groups of 64; TurboQuant's widths). f32 pages read the f32 lego's bits, so the
// paged graph equals the graph over the f32 lego at maxDiff 0 (hidden and logits; the
// pages are a different layout, so their tensors are not compared). The encoded
// forms have no oracle: their logits are compared with the f32 graph's over the
// same teacher-forced phases, by KL, and reported.

const WIDE: LatentWidths = { kvRank: 64, rope: 64, index: 64 };

/** KL(f32 ‖ form) per position of two logit tensors [.., V], with argmax agreement. */
function kl(reference: MlxArray, form: MlxArray): { sum: number; max: number; agree: number; positions: number } {
  const x = host(reference), y = host(form), positions = x.length / VOCAB;
  let sum = 0, max = 0, agree = 0;
  for (let p = 0; p < positions; p++) {
    const a = Array.from(x.subarray(p * VOCAB, (p + 1) * VOCAB)), b = Array.from(y.subarray(p * VOCAB, (p + 1) * VOCAB));
    const lse = (r: number[]) => { const m = Math.max(...r); return m + Math.log(r.reduce((s, v) => s + Math.exp(v - m), 0)); };
    const za = lse(a), zb = lse(b);
    let d = 0;
    for (let i = 0; i < VOCAB; i++) d += Math.exp(a[i]! - za) * ((a[i]! - za) - (b[i]! - zb));
    sum += d; max = Math.max(max, d);
    agree += Number(a.indexOf(Math.max(...a)) === b.indexOf(Math.max(...b)));
  }
  return { sum, max, agree, positions };
}

for (const dsa of [true, false]) describe(`GLM-5.2 latent legos (DSA ${dsa ? "on" : "off"})`, () => {
  const glm = glmConfig(dsa, WIDE);
  const config = { modelType: "glm_moe_dsa", text: { numHiddenLayers: LAYERS }, eosTokenIds: [1] } as unknown as ModelConfig;
  const weights = new SyntheticWeights(glm);
  const capabilities = { dsa, mtpMetadata: true, mtpEnabled: true, mtpDraftTokens: 3 };
  const plain = new Glm52Model(weights, config, glm, capabilities);
  const composed = (kv: KvStorageScheme, paged: boolean) => new Glm52Model(weights, config, glm, capabilities, null, null,
    "exact", 0, { kv, pagedKv: paged ? { blockSize: 4 } : null });
  const scenario = `latent legos DSA ${dsa ? "on" : "off"}`;
  afterAll(() => weights.dispose());
  /** The phases of one sequence, then two merged rows: [phase, rows of ids]. */
  const sequence: [Phase, number[][]][] = [["chunk", [tokens(12, 1)]], ["tail", [tokens(3, 2)]],
    ["decode", [tokens(1, 3)]], ["decode", [tokens(1, 4)]], ["decode", [tokens(1, 5)]], ["verify", [tokens(4, 7)]],
    ["known", [tokens(2, 8)]], ["decode", [tokens(1, 10)]]];

  test("f32 pages: every phase, and two merged rows, equal the f32 lego at maxDiff 0", async () => {
    const paged = composed({ kind: "bf16" }, true);
    const caches = { paged: paged.makeCache(), plain: plain.makeCache() as Cache[] };
    try {
      expect(caches.paged.every(cache => cache.signature().startsWith("kv:latent-paged:"))).toBe(true);
      for (const [index, [which, rows]] of sequence.entries()) {
        using ids = ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);
        using actual = await phase(paged, which, ids, caches.paged);
        using expected = await phase(plain, which, ids, caches.plain);
        expectSame(`${scenario} paged ${which} ${index} hidden`, actual, expected);
        using a = paged.logitsFromHidden(actual), b = plain.logitsFromHidden(expected);
        expectSame(`${scenario} paged ${which} ${index} logits`, a, b);
      }
    } finally { dispose(caches.paged, caches.plain); }
    // Two rows of different lengths, merged into each storage's own layout.
    const rows = { paged: [paged.makeCache(), paged.makeCache()], plain: [plain.makeCache() as Cache[], plain.makeCache() as Cache[]] };
    const batched: { paged: Cache[]; plain: Cache[] } = { paged: [], plain: [] };
    try {
      for (const [row, length] of [9, 5].entries()) {
        using ids = ops.fromInt32(tokens(length, 21 + row), [1, length]);
        (await phase(paged, "chunk", ids, rows.paged[row]!)).dispose();
        (await phase(plain, "chunk", ids, rows.plain[row]!)).dispose();
      }
      for (const side of ["paged", "plain"] as const)
        for (let layer = 0; layer < LAYERS; layer++) {
          const layout = (rows[side][0]![layer] as Cache & { makeEmptyBatch(): Cache & { mergeRows(rows: readonly Cache[]): void } }).makeEmptyBatch();
          layout.mergeRows(rows[side].map(list => list[layer]!));
          batched[side].push(layout);
        }
      for (const [which, ids] of [["decode", [tokens(1, 30), tokens(1, 40)]], ["verify", [tokens(3, 50), tokens(3, 51)]]] as const) {
        using input = ops.fromInt32(ids.flat(), [2, ids[0]!.length]);
        using actual = await phase(paged, which, input, batched.paged);
        using expected = await phase(plain, which, input, batched.plain);
        expectSame(`${scenario} paged rows ${which} hidden`, actual, expected);
      }
    } finally { dispose(...rows.paged, ...rows.plain, batched.paged, batched.plain); }
  });

  test("encoded and paged encoded forms serve every phase, reported by KL against the f32 graph", async () => {
    const reference = composed({ kind: "bf16" }, false);
    const forms: [string, KvStorageScheme, boolean][] = [
      ["affine 4", { kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" }, false],
      ["affine 8", { kind: "affine", bits: 8, groupSize: 64, kernels: "unfused" }, false],
      ["affine 8 from 16 (converts during decode)", { kind: "affine", bits: 8, groupSize: 64, kernels: "unfused", quantizedKvStart: 16 }, false],
      ["TurboQuant 8/3", { kind: "turbo", kBits: 8, vBits: 3 }, false],
      ["paged affine 4", { kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" }, true],
      ["paged TurboQuant 8/3", { kind: "turbo", kBits: 8, vBits: 3 }, true],
    ];
    const report: Record<string, unknown> = {};
    for (const [name, kv, paged] of forms) {
      const graph = composed(kv, paged);
      expect(graph.stateCodecs, name).toBeDefined();
      const caches = { form: graph.makeCache(), reference: reference.makeCache() };
      const totals = { sum: 0, max: 0, agree: 0, positions: 0 };
      try {
        for (const [which, rows] of sequence) {
          using ids = ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);
          using x = await phase(reference, which, ids, caches.reference);
          using y = await phase(graph, which, ids, caches.form);
          using lx = reference.logitsFromHidden(x), ly = graph.logitsFromHidden(y);
          const d = kl(lx, ly);
          expect(Number.isFinite(d.sum), `${name} ${which}`).toBe(true);
          totals.sum += d.sum; totals.max = Math.max(totals.max, d.max); totals.agree += d.agree; totals.positions += d.positions;
        }
      } finally { dispose(caches.form, caches.reference); }
      report[name] = { positions: totals.positions, klMean: totals.sum / totals.positions, klMax: totals.max,
        argmaxAgree: totals.agree / totals.positions };
    }
    console.log(`glm52 latent legos (synthetic weights, DSA ${dsa ? "on" : "off"}, logits KL vs f32): ${JSON.stringify(report)}`);
  });
});
