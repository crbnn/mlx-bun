import type { TestGraph } from "../models/test-composition";
// A two-model draft provider owns its draft graph and the weights it opened:
// its dispose releases what the graph built at construction (here a llama3
// scaled rope's frequency table), then the weights. A synthetic checkpoint
// written to a temporary directory; no model, no download.
import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Dtype, MlxArray, ops } from "@mlx-bun/mlx";
import { writeShardedSafetensors, type NamedTensor } from "../../src/artifacts/safetensors-writer";
import { TwoModelProvider } from "../../src/generation/speculative/sources/two-model";
import { UniversalDenseModel } from "../../src/models/universal/dense";

const HIDDEN = 64, HEADS = 2, KV_HEADS = 1, HEAD_DIM = 32, INTER = 64, VOCAB = 64;
let dir: string;

beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "mlx-bun-two-model-dispose-"));
  const tensors: NamedTensor[] = [];
  let seed = 1;
  const values = (rows: number, cols: number) => {
    const phase = ++seed;
    using wide = MlxArray.fromFloat32(Float32Array.from({ length: rows * cols }, (_, i) => Math.sin(i * 0.17 + phase) * 0.1), [rows, cols]);
    return wide.astype(Dtype.bfloat16);
  };
  const linear = (path: string, rows: number, cols: number) => {
    using w = values(rows, cols);
    const q = ops.quantize(w, 64, 4);
    tensors.push({ name: `${path}.weight`, array: q.packed }, { name: `${path}.scales`, array: q.scales }, { name: `${path}.biases`, array: q.biases });
  };
  const norm = (path: string) => {
    using ones = MlxArray.fromFloat32(new Float32Array(HIDDEN).fill(1), [HIDDEN]);
    tensors.push({ name: `${path}.weight`, array: ones.astype(Dtype.bfloat16) });
  };
  linear("model.embed_tokens", VOCAB, HIDDEN);
  norm("model.norm");
  const p = "model.layers.0";
  norm(`${p}.input_layernorm`);
  norm(`${p}.post_attention_layernorm`);
  linear(`${p}.self_attn.q_proj`, HEADS * HEAD_DIM, HIDDEN);
  linear(`${p}.self_attn.k_proj`, KV_HEADS * HEAD_DIM, HIDDEN);
  linear(`${p}.self_attn.v_proj`, KV_HEADS * HEAD_DIM, HIDDEN);
  linear(`${p}.self_attn.o_proj`, HIDDEN, HEADS * HEAD_DIM);
  for (const name of ["gate_proj", "up_proj"]) linear(`${p}.mlp.${name}`, INTER, HIDDEN);
  linear(`${p}.mlp.down_proj`, HIDDEN, INTER);
  writeShardedSafetensors(dir, tensors);
  for (const { array } of tensors) array.dispose();
  writeFileSync(join(dir, "config.json"), JSON.stringify({
    model_type: "llama", hidden_size: HIDDEN, num_hidden_layers: 1, num_attention_heads: HEADS, num_key_value_heads: KV_HEADS,
    head_dim: HEAD_DIM, intermediate_size: INTER, vocab_size: VOCAB, rms_norm_eps: 1e-5, rope_theta: 500000,
    rope_scaling: { rope_type: "llama3", factor: 8, low_freq_factor: 1, high_freq_factor: 4, original_max_position_embeddings: 8192 },
    max_position_embeddings: 1024, tie_word_embeddings: true, eos_token_id: [1], quantization: { group_size: 64, bits: 4 },
  }));
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

test("a two-model provider's dispose releases its graph's constants, then its weights", async () => {
  const provider = await TwoModelProvider.load(dir);
  const graph = provider.model;
  expect(graph instanceof UniversalDenseModel).toBe(true);
  const freqs = (graph as TestGraph<UniversalDenseModel>).sharedRope.freqs;
  expect(freqs).not.toBeNull();
  // The provider's weights are private; the order is read off both disposals.
  const weights = (provider as unknown as { weights: { dispose(): void } }).weights;
  const order: string[] = [];
  const graphDispose = graph.dispose.bind(graph), weightsDispose = weights.dispose.bind(weights);
  graph.dispose = () => { order.push("graph"); graphDispose(); };
  weights.dispose = () => { order.push("weights"); weightsDispose(); };
  provider.dispose();
  expect(order).toEqual(["graph", "weights"]);
  expect(() => freqs!.handle).toThrow("MlxArray used after dispose");
});
