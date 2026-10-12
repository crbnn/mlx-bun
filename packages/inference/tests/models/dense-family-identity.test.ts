// D3 bit-identity gate for the dense family. MiniCPM5, Qwen3, Qwen3-MoE and
// the universal dense graph, whose blocks read through the cache for the phase
// their caller names, against the frozen pre-D3 graphs (minicpm5-reference.ts,
// qwen3-reference.ts, qwen3-moe-reference.ts, universal-dense-reference.ts,
// deleted with this test after the merge), on small synthetic weights. Every
// hidden output, every logit row and every cache tensor must agree to the bit:
// for bf16 KV, for affine 4- and 8-bit KV converted after the first forward by
// kv-maintenance with MLX_BUN_NO_FUSED_SDPA unset and set to 1, for TurboQuant
// k8v3 KV, and for delayed affine rows across (or, under the softcap read,
// below) their transition. The phase operations run their own cache set and are
// held to the same bits. The universal graph runs a llama descriptor with
// sliding layers (rotating caches) and a gemma2 descriptor whose softcapped
// read (`state/softcap-attention.ts`) replaces the graph's manual attention,
// all full attention and with synthetic sliding layers. MiniCPM5 adds the
// compiled-swiglu opt-out and its training forwards (flash attention off and
// on, padded batches, prefix-shared and segmented passes). No model, no
// download.
import { afterAll, describe, expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { parseQuantization, type ModelConfig } from "../../src/artifacts/config";
import type { Weights } from "../../src/artifacts/weights";
import type { BatchableCache, Cache, PaddedPrefillCache } from "../../src/contracts/mlx/cache";
import type { MlxPhaseForward, MlxPrefillChunk, MlxVerifyForward, MlxWidthTable } from "../../src/contracts/mlx/graph";
import type { RuntimeModel } from "../../src/models/factory";
import { setTrainingAttn } from "../../src/kernels/attention/flash";
import { createRuntimeConfig, runtimeConfig, withRuntimeConfig } from "../../src/runtime/config";
import { trainForward } from "../../src/scoring/full-sequence";
import { trainForward as frozenTrainForward } from "./full-sequence-reference";
import { evalPpl } from "../../src/scoring/perplexity";
import { createKvMaintenance } from "../../src/state/kv-maintenance";
import { composedAffineKernels } from "../../src/state/affine-attention";
import { settleAffine, settleTurbo } from "../state/convert-after-reference";
import { testComposition } from "./test-composition";
import { phaseHidden } from "./phase-forward";
import { ownedCacheLayout, prefillCacheLayout, targetCacheLayout } from "../../src/state/layout";
import { denseTurboQuantKernels, deferredTurboQuantKernels } from "../../src/state/turboquant-attention";
import type { KVCache } from "../../src/state/kv";
import { genericArgsFor, type UniversalArgs } from "../../src/models/universal/archs";
import { MiniCPM5Model } from "../../src/models/minicpm5/model";
import { Qwen3Model } from "../../src/models/qwen/qwen3";
import { Qwen3MoeModel } from "../../src/models/qwen/qwen3-moe";
import { UniversalDenseModel } from "../../src/models/universal/dense";
import { MiniCPM5Model as FrozenMiniCPM5Model } from "./minicpm5-reference";
import { Qwen3Model as FrozenQwen3Model } from "./qwen3-reference";
import { Qwen3MoeModel as FrozenQwen3MoeModel } from "./qwen3-moe-reference";
import { UniversalDenseModel as FrozenUniversalDenseModel } from "./universal-dense-reference";

const HIDDEN = 256, INTER = 384, MOE_INTER = 128, VOCAB = 256, LAYERS = 4, HEADS = 4, KV_HEADS = 2, HEAD_DIM = 64;
const WINDOW = 16, EXPERTS = 8;
const F = "full_attention", S = "sliding_attention";

const owned: MlxArray[] = [];
afterAll(() => { for (const array of owned) array.dispose(); });
const keep = (array: MlxArray) => { owned.push(array); return array; };

const noise = (count: number, phase: number, scale: number, shift = 0) => Float32Array.from({ length: count },
  (_, i) => shift + Math.sin(i * 0.37 + phase) * scale + Math.cos(i * 0.11 + 2.3 * phase) * scale);
const tensor = (values: Float32Array, shape: number[], dtype = Dtype.bfloat16): MlxArray => {
  using wide = MlxArray.fromFloat32(values, shape);
  return keep(wide.astype(dtype));
};

/** Synthetic tensors under the graphs' canonical names. `quantized` stores
 * affine 4-bit group-64 linears (`.scales`), otherwise bf16 weights. */
class Tensors {
  readonly map = new Map<string, MlxArray>();
  #seed = 1;
  linear(path: string, shape: number[], quantized: boolean, scale = 0.06): void {
    const w = tensor(noise(shape.reduce((a, b) => a * b, 1), ++this.#seed, scale), shape);
    if (!quantized) { this.map.set(`${path}.weight`, w); return; }
    const q = ops.quantize(w, 64, 4);
    this.map.set(`${path}.weight`, keep(q.packed));
    this.map.set(`${path}.scales`, keep(q.scales));
    this.map.set(`${path}.biases`, keep(q.biases));
  }
  norm(path: string, width: number, shift = 1): void {
    this.map.set(path, tensor(noise(width, ++this.#seed, 0.05, shift), [width]));
  }
  weights(): Weights {
    const tensors = this.map;
    return {
      tensorNames: [...tensors.keys()],
      info: (name: string) => ({ shape: tensors.get(name)!.shape }),
      has: (name: string) => tensors.has(name),
      tensor: (name: string) => {
        const found = tensors.get(name);
        if (!found) throw new Error(`synthetic artifact has no ${name}`);
        return found;
      },
      shards: { files: new Map() },
      setView: () => { throw new Error("canonical names need no view"); },
    } as unknown as Weights;
  }
}

const QUANT = { bits: 4, group_size: 64, mode: "affine" };
function textConfig(modelType: string, text: Record<string, unknown>, raw: Record<string, unknown> = {}): ModelConfig {
  return {
    modelDir: "", modelType, architectures: [], dtype: "bfloat16", kvQuant: null, hasVisionSidecar: false, eosTokenIds: [], raw,
    quantization: parseQuantization(QUANT),
    text: {
      hiddenSize: HIDDEN, numHiddenLayers: LAYERS, numAttentionHeads: HEADS, numKeyValueHeads: KV_HEADS, headDim: HEAD_DIM,
      intermediateSize: INTER, rmsNormEps: 1e-6, vocabSize: VOCAB, ropeParameters: { full_attention: { ropeTheta: 10000 } },
      tieWordEmbeddings: false, ...text,
    },
  } as unknown as ModelConfig;
}

/** The llama-shaped attention, MLP and norms every family below shares. */
function decoder(t: Tensors, qkNorm: boolean, mlp: (layer: number, prefix: string) => void): void {
  t.linear("model.embed_tokens", [VOCAB, HIDDEN], true, 0.5);
  t.norm("model.norm.weight", HIDDEN);
  for (let layer = 0; layer < LAYERS; layer++) {
    const prefix = `model.layers.${layer}`;
    t.norm(`${prefix}.input_layernorm.weight`, HIDDEN);
    t.norm(`${prefix}.post_attention_layernorm.weight`, HIDDEN);
    t.linear(`${prefix}.self_attn.q_proj`, [HEADS * HEAD_DIM, HIDDEN], true);
    t.linear(`${prefix}.self_attn.k_proj`, [KV_HEADS * HEAD_DIM, HIDDEN], true);
    t.linear(`${prefix}.self_attn.v_proj`, [KV_HEADS * HEAD_DIM, HIDDEN], true);
    t.linear(`${prefix}.self_attn.o_proj`, [HIDDEN, HEADS * HEAD_DIM], true);
    if (qkNorm) { t.norm(`${prefix}.self_attn.q_norm.weight`, HEAD_DIM); t.norm(`${prefix}.self_attn.k_norm.weight`, HEAD_DIM); }
    mlp(layer, `${prefix}.mlp`);
  }
}
const denseMlp = (t: Tensors) => (_layer: number, prefix: string) => {
  t.linear(`${prefix}.gate_proj`, [INTER, HIDDEN], true);
  t.linear(`${prefix}.up_proj`, [INTER, HIDDEN], true);
  t.linear(`${prefix}.down_proj`, [HIDDEN, INTER], true);
};

// ── The graphs ──────────────────────────────────────────────────────────────

/** What the gate drives on either graph: the head and the caches. */
interface Graph {
  makeCache(): Cache[];
  logitsFromHidden(h: MlxArray): MlxArray;
}
/** The old graph's deprecated entry point, which reads the phase from the input's shape. */
interface OldGraph extends Graph {
  forwardHidden(ids: MlxArray, cache: Cache[]): MlxArray;
}
/** The current graph's named phases. */
interface PhasedGraph extends Graph {
  readonly prefillChunk: MlxPrefillChunk;
  readonly prefillTail: MlxWidthTable<MlxVerifyForward>;
  readonly decode: MlxWidthTable<MlxPhaseForward>;
  readonly verify: MlxWidthTable<MlxVerifyForward>;
}
interface Pair { old: OldGraph; current: PhasedGraph }

type Scheme = { name: string; kvBits?: number; flag?: string; turbo?: { kBits: number; vBits: number } };
const BF16: Scheme = { name: "bf16" };
const CONVERTED: Scheme[] = [
  { name: "affine4 flag=unset", kvBits: 4 },
  { name: "affine4 flag=1", kvBits: 4, flag: "1" },
  { name: "affine8 flag=unset", kvBits: 8 },
  { name: "affine8 flag=1", kvBits: 8, flag: "1" },
  { name: "turbo k8v3", turbo: { kBits: 8, vBits: 3 } },
];

interface Family {
  name: string;
  build(): Pair;
  /** Schemes the graph attends: the softcapped read takes plain and TurboQuant
   * storage, never converted affine storage. */
  schemes: Scheme[];
  /** Affine rows delayed until this many positions; the softcapped read stays below it. */
  delayedStart: number;
}

function miniCpmPair(): Pair {
  const t = new Tensors();
  decoder(t, false, denseMlp(t));
  t.linear("lm_head", [VOCAB, HIDDEN], true, 0.05);
  const weights = t.weights(), config = textConfig("llama", {});
  return { old: new FrozenMiniCPM5Model(weights, config), current: new MiniCPM5Model(weights, config, testComposition()) };
}
function qwen3Pair(): Pair {
  const t = new Tensors();
  decoder(t, true, denseMlp(t));
  const weights = t.weights(), config = textConfig("qwen3", { tieWordEmbeddings: true, ropeParameters: {} });
  return { old: new FrozenQwen3Model(weights, config), current: new Qwen3Model(weights, config, testComposition()) };
}
function qwen3MoePair(): Pair {
  const t = new Tensors();
  decoder(t, true, (layer, prefix) => {
    if (layer === 0) { denseMlp(t)(layer, prefix); return; }
    t.linear(`${prefix}.gate`, [EXPERTS, HIDDEN], true);
    t.linear(`${prefix}.switch_mlp.gate_proj`, [EXPERTS, MOE_INTER, HIDDEN], true);
    t.linear(`${prefix}.switch_mlp.up_proj`, [EXPERTS, MOE_INTER, HIDDEN], true);
    t.linear(`${prefix}.switch_mlp.down_proj`, [EXPERTS, HIDDEN, MOE_INTER], true);
  });
  t.linear("lm_head", [VOCAB, HIDDEN], true, 0.05);
  const weights = t.weights(), config = textConfig("qwen3_moe", {
    numExperts: EXPERTS, topKExperts: 2, normTopkProb: true, decoderSparseStep: 1, mlpOnlyLayers: [0], ropeParameters: {} });
  return { old: new FrozenQwen3MoeModel(weights, config), current: new Qwen3MoeModel(weights, config, testComposition()) };
}
/** llama with alternating full and sliding layers (mistral-style layer types). */
function llamaSlidingPair(): Pair {
  const t = new Tensors();
  decoder(t, false, denseMlp(t));
  t.linear("lm_head", [VOCAB, HIDDEN], true, 0.05);
  const raw = { model_type: "llama", hidden_size: HIDDEN, num_hidden_layers: LAYERS, num_attention_heads: HEADS,
    num_key_value_heads: KV_HEADS, head_dim: HEAD_DIM, intermediate_size: INTER, vocab_size: VOCAB, rms_norm_eps: 1e-6,
    rope_theta: 10000, tie_word_embeddings: false, layer_types: [F, S, F, S], sliding_window: WINDOW };
  const weights = t.weights(), config = textConfig("llama", {}, raw);
  return { old: new FrozenUniversalDenseModel(weights, config), current: new UniversalDenseModel(weights, config, testComposition()) };
}
/** gemma2: softcapped attention scores over plain KV, bf16 (unquantized)
 * projections, every layer full attention unless `types` makes some sliding. */
function gemma2Pair(types: string[] | null = null): Pair {
  const t = new Tensors();
  t.linear("model.embed_tokens", [VOCAB, HIDDEN], true, 0.5);
  t.norm("model.norm.weight", HIDDEN, 0);
  for (let layer = 0; layer < LAYERS; layer++) {
    const prefix = `model.layers.${layer}`;
    for (const name of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"])
      t.norm(`${prefix}.${name}.weight`, HIDDEN, 0);
    t.linear(`${prefix}.self_attn.q_proj`, [HEADS * HEAD_DIM, HIDDEN], false);
    t.linear(`${prefix}.self_attn.k_proj`, [KV_HEADS * HEAD_DIM, HIDDEN], false);
    t.linear(`${prefix}.self_attn.v_proj`, [KV_HEADS * HEAD_DIM, HIDDEN], false);
    t.linear(`${prefix}.self_attn.o_proj`, [HIDDEN, HEADS * HEAD_DIM], false);
    t.linear(`${prefix}.mlp.gate_proj`, [INTER, HIDDEN], false);
    t.linear(`${prefix}.mlp.up_proj`, [INTER, HIDDEN], false);
    t.linear(`${prefix}.mlp.down_proj`, [HIDDEN, INTER], false);
  }
  const raw = { model_type: "gemma2", hidden_size: HIDDEN, num_hidden_layers: LAYERS, num_attention_heads: HEADS,
    num_key_value_heads: KV_HEADS, head_dim: HEAD_DIM, intermediate_size: INTER, vocab_size: VOCAB, rms_norm_eps: 1e-6,
    query_pre_attn_scalar: HEAD_DIM, attn_logit_softcapping: 30, final_logit_softcapping: 30 };
  const weights = t.weights(), config = textConfig("gemma2", {}, raw);
  const args: UniversalArgs = { ...genericArgsFor(config)!, ...(types ? { layerTypes: types, slidingWindow: WINDOW } : {}) };
  return { old: new FrozenUniversalDenseModel(weights, config, { ...args }), current: new UniversalDenseModel(weights, config, testComposition(), { ...args }) };
}

const FAMILIES: Family[] = [
  { name: "minicpm5", build: miniCpmPair, schemes: [BF16, ...CONVERTED], delayedStart: 40 },
  { name: "qwen3", build: qwen3Pair, schemes: [BF16, ...CONVERTED], delayedStart: 40 },
  { name: "qwen3-moe", build: qwen3MoePair, schemes: [BF16, ...CONVERTED], delayedStart: 40 },
  { name: "universal llama (sliding)", build: llamaSlidingPair, schemes: [BF16, ...CONVERTED], delayedStart: 40 },
  { name: "universal gemma2 softcap", build: () => gemma2Pair(), schemes: [BF16, CONVERTED[4]!], delayedStart: 80 },
  { name: "universal gemma2 softcap (sliding)", build: () => gemma2Pair([F, S, F, S]), schemes: [BF16, CONVERTED[4]!], delayedStart: 80 },
];

const pairs = new Map<string, Pair>();
const pair = (family: Family): Pair => {
  let found = pairs.get(family.name);
  if (!found) pairs.set(family.name, found = family.build());
  return found;
};

// ── Comparison ──────────────────────────────────────────────────────────────

const results = new Map<string, { worst: number; tensors: number }>();
const record = (label: string, diff: number, tensors: number) => {
  const prior = results.get(label) ?? { worst: 0, tensors: 0 };
  results.set(label, { worst: Math.max(prior.worst, diff), tensors: prior.tensors + tensors });
};
afterAll(() => {
  let comparisons = 0, worst = 0;
  for (const [label, entry] of results) {
    comparisons += entry.tensors; worst = Math.max(worst, entry.worst);
    console.log(`dense-identity ${label}: maxDiff ${entry.worst} over ${entry.tensors} tensors`);
  }
  console.log(`dense-identity total: maxDiff ${worst} over ${comparisons} tensors in ${results.size} labels`);
});

/** Largest absolute difference after requiring equal shape, dtype and bytes. */
function maxDiff(actual: MlxArray, expected: MlxArray, what: string): number {
  expect(actual.shape, what).toEqual(expected.shape);
  expect(actual.dtype, what).toBe(expected.dtype);
  using left = ops.contiguous(actual);
  using right = ops.contiguous(expected);
  const a = left.rawBytesView(), b = right.rawBytesView();
  if (Buffer.from(a.buffer, a.byteOffset, a.byteLength).equals(Buffer.from(b.buffer, b.byteOffset, b.byteLength))) return 0;
  using leftWide = left.astype(Dtype.float32);
  using rightWide = right.astype(Dtype.float32);
  const x = leftWide.toFloat32(), y = rightWide.toFloat32();
  let worst = 0;
  for (let i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(x[i]! - y[i]!));
  throw new Error(`${what}: bytes differ, maxDiff ${worst}`);
}

/** Every tensor of every layer's cache, with its signature, offset and row offsets. */
function stateDiff(label: string, actual: readonly Cache[], expected: readonly Cache[]): void {
  expect(actual.length).toBe(expected.length);
  let worst = 0, count = 0;
  for (let layer = 0; layer < expected.length; layer++) {
    const a = actual[layer]!, b = expected[layer]!;
    expect(a.signature(), `${label} layer ${layer}`).toBe(b.signature());
    expect(a.offset, `${label} layer ${layer}`).toBe(b.offset);
    expect((a as Partial<BatchableCache>).rowOffsets, `${label} layer ${layer}`).toEqual((b as Partial<BatchableCache>).rowOffsets);
    const left = a.state(), right = b.state();
    try {
      expect(left.length, `${label} layer ${layer}`).toBe(right.length);
      for (let i = 0; i < right.length; i++) { worst = Math.max(worst, maxDiff(left[i]!, right[i]!, `${label} layer ${layer} state ${i}`)); count++; }
    } finally {
      if (a.stateNeedsDispose) for (const array of left) array.dispose();
      if (b.stateNeedsDispose) for (const array of right) array.dispose();
    }
  }
  record(`${label} cache`, worst, count);
}

/** Hidden rows and their logits (both owned by the caller and disposed here). */
function outputDiff(label: string, model: Graph, actual: MlxArray, expected: MlxArray): void {
  try {
    const hidden = maxDiff(actual, expected, `${label} hidden`);
    using la = model.logitsFromHidden(actual);
    using lb = model.logitsFromHidden(expected);
    const logits = maxDiff(la, lb, `${label} logits`);
    using wide = lb.astype(Dtype.float32);
    expect(wide.toFloat32().every(Number.isFinite), `${label} logits finite`).toBe(true);
    record(`${label} hidden+logits`, Math.max(hidden, logits), 2);
  } finally { actual.dispose(); expected.dispose(); }
}

// ── Running both graphs ─────────────────────────────────────────────────────

const prompt = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 37 + salt * 11 + 5) % VOCAB);
const ids = (rows: number[][]) => ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);
const disposeAll = (...sets: Cache[][]) => { for (const set of sets) for (const cache of set) cache.dispose(); };

function withScheme<T>(scheme: Scheme, run: () => T): T {
  return withRuntimeConfig(createRuntimeConfig({ ...runtimeConfig().values, MLX_BUN_NO_FUSED_SDPA: scheme.flag }), run);
}

/** Convert after a forward, as the served path did before E3 (quantized_kv_start
 * 0): the convert-after reference, whose affine storage reads with the kernel set
 * the scheme's flag named (unset: OptiQ's tiled-causal set, `--fused-sdpa on`;
 * 1: the unfused port, `--fused-sdpa off`). */
function convert(scheme: Scheme, ...sets: Cache[][]): void {
  if (scheme.kvBits === undefined && !scheme.turbo) return;
  for (const caches of sets) {
    if (scheme.turbo) settleTurbo(caches, scheme.turbo.kBits, scheme.turbo.vBits, 0);
    else settleAffine(caches, scheme.kvBits!, 0, 64, composedAffineKernels(scheme.flag === "1" ? "unfused" : "tiled-causal"));
  }
}

/** The old graph's deprecated entry point and the phase a scheduler names for the same tokens on the current graph. */
function both(label: string, p: Pair, oldCaches: Cache[], newCaches: Cache[], rows: number[][]): void {
  using input = ids(rows);
  const expected = p.old.forwardHidden(input, oldCaches);
  const actual = phaseHidden(p.current, input, newCaches);
  outputDiff(label, p.current, actual, expected);
  stateDiff(label, newCaches, oldCaches);
}

/** The current graph's phase operation on its own cache set against the old
 * graph's deprecated entry point on a reference set. */
function phased(label: string, p: Pair, refCaches: Cache[], phaseCaches: Cache[], rows: number[][],
  operation: (input: MlxArray, caches: Cache[]) => MlxArray | Promise<MlxArray>): void {
  using input = ids(rows);
  const expected = p.old.forwardHidden(input, refCaches);
  const actual = operation(input, phaseCaches) as MlxArray;
  outputDiff(`${label} (phase operation)`, p.current, actual, expected);
  stateDiff(`${label} (phase operation)`, phaseCaches, refCaches);
}

/** Row layouts made by `layout` holding `sources` (one serial cache list per row). */
function rowsOf(sources: Cache[][], layout: (cache: Cache) => unknown): Cache[] {
  return sources[0]!.map((cache, layer) => {
    const rows = layout(cache) as BatchableCache;
    rows.mergeRows(sources.map(source => source[layer]!));
    return rows;
  });
}
const roundBegin = (...sets: Cache[][]) => { for (const set of sets) for (const cache of set) cache.specRoundBegin?.(); };
/** Keep `keep` of a five-position round: recurrent state replays, KV trims the rest. */
const rollback = (keep: number, ...sets: Cache[][]) => {
  for (const set of sets) for (const cache of set) {
    if (cache.specRoundRollback) cache.specRoundRollback(keep);
    else cache.trim(5 - keep, true);
  }
};

for (const family of FAMILIES) {
  describe(`D3 dense family: ${family.name}`, () => {
    for (const scheme of family.schemes) {
      const tag = `${family.name} ${scheme.name}`;

      test(`${tag}: prefill 37, decodes, window, verify, a rolled-back round, fill 1-4, phase operations`, () => withScheme(scheme, () => {
        const p = pair(family);
        const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
        const refCaches = p.old.makeCache(), phaseCaches = p.current.makeCache();
        try {
          const prefill = prompt(37, 1);
          both(`${tag} prefill37`, p, oldCaches, newCaches, [prefill]);
          phased(`${tag} prefillChunk37`, p, refCaches, phaseCaches, [prefill], (input, caches) => p.current.prefillChunk.gpu(input, caches, null));
          convert(scheme, oldCaches, newCaches, refCaches, phaseCaches);
          stateDiff(`${tag} converted`, newCaches, oldCaches);
          stateDiff(`${tag} converted (phase operation)`, phaseCaches, refCaches);

          for (let step = 0; step < 3; step++) {
            both(`${tag} decode`, p, oldCaches, newCaches, [[200 + step]]);
            phased(`${tag} decode`, p, refCaches, phaseCaches, [[200 + step]], (input, caches) => p.current.decode.one(input, caches));
          }
          both(`${tag} window5`, p, oldCaches, newCaches, [prompt(5, 2)]);
          phased(`${tag} prefillTail5`, p, refCaches, phaseCaches, [prompt(5, 2)], (input, caches) => p.current.prefillTail.fiveToEight(input, caches, null) as MlxArray);
          both(`${tag} verify4`, p, oldCaches, newCaches, [prompt(4, 3)]);
          phased(`${tag} verify4`, p, refCaches, phaseCaches, [prompt(4, 3)], (input, caches) => p.current.verify.four(input, caches, null));

          // A speculative round: arm, verify 5, keep 2, decode.
          roundBegin(oldCaches, newCaches, refCaches, phaseCaches);
          both(`${tag} spec verify5`, p, oldCaches, newCaches, [prompt(5, 4)]);
          phased(`${tag} spec verify5`, p, refCaches, phaseCaches, [prompt(5, 4)], (input, caches) => p.current.verify.fiveToEight(input, caches, null));
          rollback(2, oldCaches, newCaches, refCaches, phaseCaches);
          stateDiff(`${tag} spec rollback keep 2`, newCaches, oldCaches);
          both(`${tag} decode after rollback`, p, oldCaches, newCaches, [[77]]);

          // Committed fill spans: these graphs append them through the forward.
          for (const span of [1, 2, 3, 4]) both(`${tag} fill${span}`, p, oldCaches, newCaches, [prompt(span, 5 + span)]);
          both(`${tag} decode after fill`, p, oldCaches, newCaches, [[91]]);

          // The fill path's storage: target row layouts of one row.
          const oldRows = rowsOf([oldCaches], targetCacheLayout), newRows = rowsOf([newCaches], targetCacheLayout);
          try {
            stateDiff(`${tag} target rows`, newRows, oldRows);
            for (const span of [1, 2, 3, 4]) both(`${tag} fill${span} (target rows)`, p, oldRows, newRows, [prompt(span, 9 + span)]);
            both(`${tag} decode (target rows)`, p, oldRows, newRows, [[33]]);
          } finally { disposeAll(oldRows, newRows); }
        } finally { disposeAll(oldCaches, newCaches, refCaches, phaseCaches); }
      }), 120_000);

      test(`${tag}: prefill 200, decodes, a window across the 256-position step`, () => withScheme(scheme, () => {
        const p = pair(family);
        const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
        try {
          both(`${tag} prefill200`, p, oldCaches, newCaches, [prompt(200, 6)]);
          convert(scheme, oldCaches, newCaches);
          for (let step = 0; step < 2; step++) both(`${tag} decode`, p, oldCaches, newCaches, [[11 + step]]);
          both(`${tag} window60 across 256`, p, oldCaches, newCaches, [prompt(60, 7)]);
          both(`${tag} decode`, p, oldCaches, newCaches, [[19]]);
        } finally { disposeAll(oldCaches, newCaches); }
      }), 120_000);

      test(`${tag}: padded batched prefill of three unequal rows, batched decode, per-row rollback, batched fill`, () => withScheme(scheme, () => {
        const p = pair(family);
        const lengths = [9, 14, 6];
        const oldSources = lengths.map(() => p.old.makeCache()), newSources = lengths.map(() => p.current.makeCache());
        let oldRows: Cache[] = [], newRows: Cache[] = [];
        try {
          lengths.forEach((length, row) => both(`${tag} row ${row} prefill`, p, oldSources[row]!, newSources[row]!, [prompt(length, 11 + row)]));
          convert(scheme, ...oldSources, ...newSources);
          oldRows = rowsOf(oldSources, targetCacheLayout); newRows = rowsOf(newSources, targetCacheLayout);
          stateDiff(`${tag} merged rows`, newRows, oldRows);

          // Right-padded window: real lengths 5, 3 and 7 in a width of 7.
          const real = [5, 3, 7], width = 7;
          for (const set of [oldRows, newRows]) for (const cache of set)
            (cache as unknown as PaddedPrefillCache).preparePrefill({ lengths: real, rightPadding: real.map(n => width - n) });
          both(`${tag} padded prefill 5/3/7`, p, oldRows, newRows, real.map((n, row) => [...prompt(n, 14 + row), ...Array(width - n).fill(0)]));
          for (const set of [oldRows, newRows]) for (const cache of set) (cache as unknown as PaddedPrefillCache).finalizePrefill();
          stateDiff(`${tag} padded prefill finalized`, newRows, oldRows);

          for (let step = 0; step < 2; step++) both(`${tag} batched decode`, p, oldRows, newRows, [[40 + step], [50 + step], [60 + step]]);

          roundBegin(oldRows, newRows);
          both(`${tag} batched verify4`, p, oldRows, newRows, [prompt(4, 20), prompt(4, 21), prompt(4, 22)]);
          for (const set of [oldRows, newRows]) for (const cache of set)
            (cache as unknown as { specRoundRollback(keep: readonly number[]): void }).specRoundRollback([1, 3, 2]);
          stateDiff(`${tag} batched rollback keep 1/3/2`, newRows, oldRows);
          both(`${tag} batched decode after rollback`, p, oldRows, newRows, [[70], [71], [72]]);
          both(`${tag} batched fill (one position per row)`, p, oldRows, newRows, [[80], [81], [82]]);
        } finally { disposeAll(...oldSources, ...newSources, oldRows, newRows); }
      }), 120_000);

      test(`${tag}: the continuous-decoding row layout of three unequal rows, batched decode`, () => withScheme(scheme, () => {
        const p = pair(family);
        const lengths = [12, 5, 9];
        const oldSources = lengths.map(() => p.old.makeCache()), newSources = lengths.map(() => p.current.makeCache());
        let oldRows: Cache[] = [], newRows: Cache[] = [];
        try {
          lengths.forEach((length, row) => both(`${tag} row ${row} prefill`, p, oldSources[row]!, newSources[row]!, [prompt(length, 31 + row)]));
          convert(scheme, ...oldSources, ...newSources);
          oldRows = rowsOf(oldSources, cache => ownedCacheLayout(cache)!); newRows = rowsOf(newSources, cache => ownedCacheLayout(cache)!);
          stateDiff(`${tag} owned rows`, newRows, oldRows);
          for (let step = 0; step < 3; step++) both(`${tag} owned batched decode`, p, oldRows, newRows, [[3 + step], [5 + step], [7 + step]]);
        } finally { disposeAll(...oldSources, ...newSources, oldRows, newRows); }
      }), 120_000);
    }

    test(`${family.name} bf16: padded batched prefill of three unequal rows from empty prefill layouts`, () => {
      const p = pair(family), tag = `${family.name} bf16 empty`;
      const layouts = (model: Graph) => {
        const serial = model.makeCache();
        try { return serial.map(cache => prefillCacheLayout(cache) as unknown as Cache); }
        finally { disposeAll(serial); }
      };
      const oldRows = layouts(p.old), newRows = layouts(p.current);
      try {
        const real = [6, 2, 4], width = 6;
        for (const set of [oldRows, newRows]) for (const cache of set)
          (cache as unknown as PaddedPrefillCache).preparePrefill({ lengths: real, rightPadding: real.map(n => width - n) });
        both(`${tag} padded prefill 6/2/4`, p, oldRows, newRows, real.map((n, row) => [...prompt(n, 30 + row), ...Array(width - n).fill(0)]));
        for (const set of [oldRows, newRows]) for (const cache of set) (cache as unknown as PaddedPrefillCache).finalizePrefill();
        for (let step = 0; step < 2; step++) both(`${tag} batched decode`, p, oldRows, newRows, [[3 + step], [5 + step], [7 + step]]);
      } finally { disposeAll(oldRows, newRows); }
    }, 120_000);

    test(`${family.name} delayed affine4 (start ${family.delayedStart}): prefill 37, decode, fill 1-4, decode`, () => {
      const p = pair(family), tag = `${family.name} delayed affine4`;
      // The delayed row layouts read with the set the unset flag named (tiled-causal).
      const maintain = createKvMaintenance({ kind: "affine", bits: 4, groupSize: 64, quantizedKvStart: family.delayedStart,
        kernels: "tiled-causal" });
      const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
      maintain.preparePrefill!(oldCaches); maintain.preparePrefill!(newCaches);
      try {
        both(`${tag} prefill37`, p, oldCaches, newCaches, [prompt(37, 41)]);
        both(`${tag} decode`, p, oldCaches, newCaches, [[42]]);
        for (const span of [1, 2, 3, 4]) both(`${tag} fill${span}`, p, oldCaches, newCaches, [prompt(span, 43 + span)]);
        both(`${tag} decode after fill`, p, oldCaches, newCaches, [[29]]);
      } finally { disposeAll(oldCaches, newCaches); }
    }, 120_000);

    // Perplexity rows run over the stateless training caches (`TrainingCache`),
    // each composed with its layer's sliding window by scoring/full-sequence.ts.
    test(`${family.name}: full-sequence scoring, perplexity rows over the training caches`, () => {
      const p = pair(family), tag = `${family.name} scoring`;
      const rows = [Int32Array.from(prompt(33, 63)), Int32Array.from(prompt(33, 64)), Int32Array.from(prompt(33, 65))];
      const expected = evalPpl(p.old as unknown as RuntimeModel, rows, 2);
      const actual = evalPpl(p.current as unknown as RuntimeModel, rows, 2);
      expect(actual).toEqual(expected);
      expect(Number.isFinite(expected.ppl)).toBe(true);
      record(`${tag} perplexity (ppl, mean loss, standard error)`, Math.max(Math.abs(actual.ppl - expected.ppl),
        Math.abs(actual.meanLoss - expected.meanLoss), Math.abs(actual.standardError - expected.standardError)), 3);
    }, 120_000);

    test(`${family.name}: full-sequence scoring, a padded batch through the mask wrapper`, () => {
      const p = pair(family), tag = `${family.name} scoring`;
      const valid = [24, 19, 15];
      using input = ids(valid.map((n, row) => [...prompt(n, 66 + row), ...Array(24 - n).fill(0)]));
      using expectedLogits = frozenTrainForward(p.old as unknown as RuntimeModel, input, valid);
      using actualLogits = trainForward(p.current as unknown as RuntimeModel, input, valid);
      record(`${tag} padded batch logits`, maxDiff(actualLogits, expectedLogits, `${tag} padded batch logits`), 1);
    }, 120_000);
  });
}

// ── Composition facts ───────────────────────────────────────────────────────

describe("D3 dense family: what the graphs compose at construction", () => {
  test("makeCache composes one mask memo and the TurboQuant read each graph attended before", () => {
    const sets: [string, Pair, unknown][] = [["minicpm5", pair(FAMILIES[0]!), deferredTurboQuantKernels],
      ["qwen3", pair(FAMILIES[1]!), denseTurboQuantKernels], ["qwen3-moe", pair(FAMILIES[2]!), denseTurboQuantKernels],
      ["universal", pair(FAMILIES[3]!), denseTurboQuantKernels]];
    for (const [name, p, kernels] of sets) {
      const caches = p.current.makeCache();
      try {
        const masks = new Set(caches.map(cache => (cache as { masks?: unknown }).masks));
        expect(masks.size, name).toBe(1);
        for (const cache of caches.filter(cache => cache.signature() === "kv:plain"))
          expect((cache as KVCache).turboQuantKernels, name).toBe(kernels as KVCache["turboQuantKernels"]);
      } finally { disposeAll(caches); }
    }
  });

  test("the universal graph's adapter targets match the class test they replace", () => {
    for (const family of [FAMILIES[3]!, FAMILIES[4]!]) {
      const p = pair(family);
      const expected = (p.old as FrozenUniversalDenseModel).loraTargets(), actual = (p.current as UniversalDenseModel).loraTargets();
      expect([...actual.keys()], family.name).toEqual([...expected.keys()]);
      for (const [path, linear] of expected) expect(actual.get(path)!.w, path).toBe(linear.w);
    }
    expect((pair(FAMILIES[3]!).current as UniversalDenseModel).loraTargets().size).toBe(LAYERS * 7);
    expect((pair(FAMILIES[4]!).current as UniversalDenseModel).loraTargets().size).toBe(0);
  });

  test("array masks without a score softcap have no read and are refused at construction", () => {
    const t = new Tensors();
    decoder(t, false, denseMlp(t));
    t.linear("lm_head", [VOCAB, HIDDEN], true, 0.05);
    const raw = { model_type: "llama", hidden_size: HIDDEN, num_hidden_layers: LAYERS, num_attention_heads: HEADS,
      num_key_value_heads: KV_HEADS, head_dim: HEAD_DIM, intermediate_size: INTER, vocab_size: VOCAB, rms_norm_eps: 1e-6 };
    const config = textConfig("llama", {}, raw);
    expect(() => new UniversalDenseModel(t.weights(), config, testComposition(), { ...genericArgsFor(config)!, maskArray: true }))
      .toThrow("array masks without a score softcap");
  });
});

// ── MiniCPM5: the swiglu opt-out and the training forwards ──────────────────

describe("D3 dense family: MiniCPM5 construction choices and training forwards", () => {
  test("MLX_BUN_COMPILED_SWIGLU=0 is read once at construction: the unfused MLP, the same bits as the old per-call read", () => {
    const unfused = createRuntimeConfig({ ...runtimeConfig().values, MLX_BUN_COMPILED_SWIGLU: "0" });
    withRuntimeConfig(unfused, () => {
      const t = new Tensors();
      decoder(t, false, denseMlp(t));
      t.linear("lm_head", [VOCAB, HIDDEN], true, 0.05);
      const weights = t.weights(), config = textConfig("llama", {});
      const p: Pair = { old: new FrozenMiniCPM5Model(weights, config), current: new MiniCPM5Model(weights, config, testComposition()) };
      expect((p.current as MiniCPM5Model).layers.every(layer => layer.mlp.kind === "unfused")).toBe(true);
      const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
      try {
        both("minicpm5 unfused swiglu prefill200", p, oldCaches, newCaches, [prompt(200, 81)]);
        both("minicpm5 unfused swiglu decode", p, oldCaches, newCaches, [[82]]);
        both("minicpm5 unfused swiglu window5", p, oldCaches, newCaches, [prompt(5, 83)]);
      } finally { disposeAll(oldCaches, newCaches); }
    });
    expect((pair(FAMILIES[0]!).current as MiniCPM5Model).layers.every(layer => layer.mlp.kind === "compiled")).toBe(true);
  }, 120_000);

  for (const flash of [false, true]) test(`training forwards, flash attention ${flash ? "on" : "off"}: B=1, a padded batch, prefix-shared and segmented passes`, () => {
    const p = pair(FAMILIES[0]!), tag = `minicpm5 training flash=${flash ? "on" : "off"}`;
    const old = p.old as FrozenMiniCPM5Model, current = p.current as MiniCPM5Model;
    setTrainingAttn(flash ? "flash" : null);
    try {
      {
        using input = ids([prompt(48, 91)]);
        using expected = frozenTrainForward(old as unknown as RuntimeModel, input);
        using actual = trainForward(current as unknown as RuntimeModel, input);
        record(`${tag} B=1 logits`, maxDiff(actual, expected, `${tag} B=1 logits`), 1);
      }
      {
        const valid = [40, 33, 36];
        using input = ids(valid.map((n, row) => [...prompt(n, 92 + row), ...Array(40 - n).fill(0)]));
        using expected = frozenTrainForward(old as unknown as RuntimeModel, input, valid);
        using actual = trainForward(current as unknown as RuntimeModel, input, valid);
        record(`${tag} padded batch logits`, maxDiff(actual, expected, `${tag} padded batch logits`), 1);
      }
      const layout = { P: 20, Rc: 9, Rr: 11 };
      {
        using input = ids([prompt(layout.P + layout.Rc + layout.Rr, 95)]);
        using expected = old.trainable.prefixShared!.prefixSharedHidden(input, layout);
        using actual = current.trainable.prefixShared!.prefixSharedHidden(input, layout);
        record(`${tag} prefix-shared hidden`, maxDiff(actual, expected, `${tag} prefix-shared hidden`), 1);
      }
      for (const prefix of [undefined, layout]) {
        using input = ids([prompt(layout.P + layout.Rc + layout.Rr, 96)]);
        const run = (model: FrozenMiniCPM5Model | MiniCPM5Model) => {
          const pass = model.trainable.segmented!.begin(input, prefix);
          try {
            const first = pass.runRange(pass.input, 0, 2, new Map()).h;
            const second = pass.runRange(first, 2, LAYERS, new Map()).h;
            first.dispose();
            const out = model.finalNorm.forward(second);
            second.dispose();
            return out;
          } finally { pass.input.dispose(); pass.dispose(); }
        };
        using expected = run(old);
        using actual = run(current);
        const what = `${tag} segmented${prefix ? " prefix-shared" : ""} hidden`;
        record(what, maxDiff(actual, expected, what), 1);
      }
    } finally { setTrainingAttn(null); }
  }, 120_000);
});
