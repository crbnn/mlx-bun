// The TurboQuant head dimensions beyond the reference's four (signs.ts,
// scripts/gen-turboquant-signs.ts), gated where they sit beside the existing
// ones. The rotation is orthogonal at every generated size (its inverse
// recovers the input). The value codec's round trip at 3 bits keeps the
// Lloyd-Max distortion it has at 64 and 128. On a synthetic llama-shaped graph
// of each head dimension (two layers, four query and two KV heads), a 40-token
// prompt prefills and 32 tokens decode teacher-forced under bf16 KV and under
// the TurboQuant k8v3 lego (bf16 through its first append, encoded after): the
// per-position KL of the TurboQuant logits against bf16 for the new size sits
// within the range the existing sizes give. No model, no download.
import { afterAll, expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { parseQuantization, type ModelConfig } from "../../src/artifacts/config";
import type { Weights } from "../../src/artifacts/weights";
import type { Cache } from "../../src/contracts/mlx/cache";
import type { KvStorageScheme } from "../../src/contracts/portable/composition";
import { decodeValues, encodeValues, fwht } from "../../src/kernels/turboquant/ops";
import { TURBOQUANT_HEAD_DIMS } from "../../src/kernels/turboquant/signs";
import { UniversalDenseModel } from "../../src/models/universal/dense";
import { klPerToken } from "../../src/scoring/kl";
import { testComposition } from "../models/test-composition";

/** The reference's sizes and the ones generated beyond them. */
const EXISTING = [64, 128, 256], NEW = TURBOQUANT_HEAD_DIMS.filter(d => ![64, 128, 256, 512].includes(d));
const HIDDEN = 256, INTER = 512, VOCAB = 256, LAYERS = 2, HEADS = 4, KV_HEADS = 2, PROMPT = 40, FORCED = 32;

const owned: MlxArray[] = [];
afterAll(() => { for (const array of owned) array.dispose(); });
const keep = (array: MlxArray) => { owned.push(array); return array; };
const noise = (count: number, phase: number, scale: number, shift = 0) => Float32Array.from({ length: count },
  (_, i) => shift + Math.sin(i * 0.37 + phase) * scale + Math.cos(i * 0.11 + 2.3 * phase) * scale);

test("the generated sizes are the reference's four and the served ones beyond them", () => {
  expect([...TURBOQUANT_HEAD_DIMS]).toEqual([64, 96, 128, 256, 512]);
  expect(NEW).toEqual([96]);
});

test("the rotation at every generated size is orthogonal: its inverse recovers the input", () => {
  for (const d of TURBOQUANT_HEAD_DIMS) {
    using x = MlxArray.fromFloat32(noise(4 * d, d, 0.7), [1, 1, 4, d]);
    using rotated = fwht(x, true);
    using back = fwht(rotated, false);
    using error = ops.sub(back, x);
    using wide = error.astype(Dtype.float32);
    const worst = wide.toFloat32().reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    expect(worst, `head dimension ${d}`).toBeLessThan(1e-5);
  }
});

test("the 3-bit value round trip keeps the Lloyd-Max distortion of the existing sizes", () => {
  const relative = (d: number) => {
    using x = MlxArray.fromFloat32(noise(64 * d, 3 * d, 1, 0.1), [1, 2, 32, d]);
    const encoded = encodeValues(x, 3);
    try {
      using back = decodeValues(encoded.indices, encoded.scales, 3);
      using xf = x.astype(Dtype.float32), bf = back.astype(Dtype.float32);
      const a = xf.toFloat32(), b = bf.toFloat32();
      let error = 0, energy = 0;
      for (let i = 0; i < a.length; i++) { error += (a[i]! - b[i]!) ** 2; energy += a[i]! ** 2; }
      return error / energy;
    } finally { encoded.indices.dispose(); encoded.scales.dispose(); }
  };
  const existing = EXISTING.map(relative), fresh = NEW.map(relative);
  console.log(`turboquant 3-bit value round trip, relative squared error: ${JSON.stringify(Object.fromEntries(
    [...EXISTING, ...NEW].map((d, i) => [d, [...existing, ...fresh][i]])))}`);
  for (const value of fresh) {
    expect(value).toBeGreaterThan(Math.min(...existing) * 0.5);
    expect(value).toBeLessThan(Math.max(...existing) * 1.5);
  }
});

/** A llama-shaped graph with head dimension `d` over synthetic 4-bit weights. */
function graph(d: number, kv: KvStorageScheme): UniversalDenseModel {
  const tensors = new Map<string, MlxArray>();
  let seed = 1;
  const linear = (path: string, rows: number, cols: number, scale = 0.06) => {
    using wide = MlxArray.fromFloat32(noise(rows * cols, ++seed, scale), [rows, cols]);
    using w = wide.astype(Dtype.bfloat16);
    const q = ops.quantize(w, 64, 4);
    tensors.set(`${path}.weight`, keep(q.packed)); tensors.set(`${path}.scales`, keep(q.scales)); tensors.set(`${path}.biases`, keep(q.biases));
  };
  const norm = (path: string, width: number) => {
    using wide = MlxArray.fromFloat32(noise(width, ++seed, 0.05, 1), [width]);
    tensors.set(path, keep(wide.astype(Dtype.bfloat16)));
  };
  linear("model.embed_tokens", VOCAB, HIDDEN, 0.5);
  norm("model.norm.weight", HIDDEN);
  linear("lm_head", VOCAB, HIDDEN, 0.1);
  for (let layer = 0; layer < LAYERS; layer++) {
    const p = `model.layers.${layer}`;
    norm(`${p}.input_layernorm.weight`, HIDDEN);
    norm(`${p}.post_attention_layernorm.weight`, HIDDEN);
    linear(`${p}.self_attn.q_proj`, HEADS * d, HIDDEN, 0.12);
    linear(`${p}.self_attn.k_proj`, KV_HEADS * d, HIDDEN, 0.12);
    linear(`${p}.self_attn.v_proj`, KV_HEADS * d, HIDDEN);
    linear(`${p}.self_attn.o_proj`, HIDDEN, HEADS * d);
    linear(`${p}.mlp.gate_proj`, INTER, HIDDEN);
    linear(`${p}.mlp.up_proj`, INTER, HIDDEN);
    linear(`${p}.mlp.down_proj`, HIDDEN, INTER);
  }
  const weights = {
    tensorNames: [...tensors.keys()], info: (name: string) => ({ shape: tensors.get(name)!.shape }),
    has: (name: string) => tensors.has(name), tensor: (name: string) => tensors.get(name)!,
    shards: { files: new Map() }, setView: () => { throw new Error("canonical names need no view"); },
  } as unknown as Weights;
  const raw = { model_type: "llama", hidden_size: HIDDEN, num_hidden_layers: LAYERS, num_attention_heads: HEADS,
    num_key_value_heads: KV_HEADS, head_dim: d, intermediate_size: INTER, vocab_size: VOCAB, rms_norm_eps: 1e-6,
    rope_theta: 10000, tie_word_embeddings: false };
  const config = {
    modelDir: "", modelType: "llama", architectures: [], dtype: "bfloat16", kvQuant: null, hasVisionSidecar: false,
    eosTokenIds: [], raw, quantization: parseQuantization({ bits: 4, group_size: 64, mode: "affine" }),
    text: { hiddenSize: HIDDEN, numHiddenLayers: LAYERS, numAttentionHeads: HEADS, numKeyValueHeads: KV_HEADS, headDim: d,
      globalHeadDim: d, intermediateSize: INTER, rmsNormEps: 1e-6, vocabSize: VOCAB,
      ropeParameters: { full_attention: { ropeTheta: 10000 } }, tieWordEmbeddings: false },
  } as unknown as ModelConfig;
  return new UniversalDenseModel(weights, config, testComposition(kv));
}

/** Logits [1, FORCED, V] of the teacher-forced decode after the prompt, under `kv`. */
function forcedLogits(d: number, kv: KvStorageScheme): MlxArray {
  const model = graph(d, kv), caches: Cache[] = model.makeCache();
  const tokens = Array.from({ length: PROMPT + FORCED }, (_, i) => (i * 37 + 11) % VOCAB);
  try {
    using prompt = ops.fromInt32(tokens.slice(0, PROMPT), [1, PROMPT]);
    (model.multiSequencePrompt.at(-1)!.operation(prompt, caches, null) as MlxArray).dispose();
    const rows: MlxArray[] = [];
    try {
      for (let step = 0; step < FORCED; step++) {
        using id = ops.fromInt32([tokens[PROMPT + step]!], [1, 1]);
        using hidden = model.decode[1]!(id, caches, null) as MlxArray;
        rows.push(model.logits[1]!(hidden));
      }
      return ops.concatAxis(rows, 1);
    } finally { for (const row of rows) row.dispose(); }
  } finally { for (const cache of caches) cache.dispose(); }
}

test("teacher-forced KL of TurboQuant k8v3 against bf16: the new sizes sit within the existing sizes' range", () => {
  const turbo: KvStorageScheme = { kind: "turbo", kBits: 8, vBits: 3 };
  const table: Record<number, { klMean: number; klMax: number; argmaxAgree: number }> = {};
  for (const d of [...EXISTING, ...NEW]) {
    using reference = forcedLogits(d, { kind: "bf16" });
    using encoded = forcedLogits(d, turbo);
    const kl = klPerToken(reference, encoded);
    using a = ops.argmaxAxis(reference, -1), b = ops.argmaxAxis(encoded, -1);
    const x = a.toIntTokens(), y = b.toIntTokens();
    expect(kl.every(Number.isFinite), `head dimension ${d}`).toBe(true);
    table[d] = { klMean: kl.reduce((s, v) => s + v, 0) / kl.length, klMax: Math.max(...kl),
      argmaxAgree: x.filter((t, i) => t === y[i]).length / x.length };
  }
  console.log(`turboquant k8v3 vs bf16, teacher-forced decode KL (synthetic llama, ${FORCED} positions): ${JSON.stringify(table)}`);
  const existing = EXISTING.map(d => table[d]!.klMean);
  for (const d of NEW) {
    expect(table[d]!.klMean, `head dimension ${d}`).toBeGreaterThan(0);
    expect(table[d]!.klMean, `head dimension ${d}`).toBeLessThan(Math.max(...existing) * 2);
  }
}, 120_000);
