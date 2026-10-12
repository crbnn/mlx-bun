// The dense graphs no artifact on either machine holds, over small synthetic
// weights, held to their recorded digest (tests/fixtures/synthetic-digests/
// dense-family.json; tests/models/synthetic-digest.ts): Qwen3, Qwen3-MoE and the
// universal graph's gemma2 descriptor, whose softcapped read
// (`state/softcap-attention.ts`) serves all full attention and with synthetic
// sliding layers. Every hidden output, logit row and cache tensor: bf16 KV,
// affine 4- and 8-bit KV converted after the first forward with the tiled-causal
// and the unfused read, TurboQuant k8v3 KV, and delayed affine rows; the phase
// operations on their own cache set; row layouts, padded batches, rollbacks and
// fill spans; full-sequence scoring. The digest was recorded from the tree whose
// graphs equalled the pre-D3 graphs at maxDiff 0 on these cases, so it carries
// that check forward. MiniCPM5 and the universal llama descriptor have artifacts
// (their real-weight parity suites are their gates); their construction facts
// are checked below. No model, no download.
import { afterAll, describe, expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { parseQuantization, type ModelConfig } from "../../src/artifacts/config";
import type { Weights } from "../../src/artifacts/weights";
import type { BatchableCache, Cache, PaddedPrefillCache } from "../../src/contracts/mlx/cache";
import type { MlxPhaseForward, MlxPrefillChunk, MlxVerifyForward, MlxWidthTable } from "../../src/contracts/mlx/graph";
import type { RuntimeModel } from "../../src/models/factory";
import { createRuntimeConfig, runtimeConfig, withRuntimeConfig } from "../../src/runtime/config";
import { trainForward } from "../../src/scoring/full-sequence";
import { evalPpl } from "../../src/scoring/perplexity";
import { createKvMaintenance } from "../../src/state/layout";
import { composedAffineKernels } from "../../src/state/affine-attention";
import { settleAffine, settleTurbo } from "../state/convert-after-reference";
import { syntheticDigests, type DigestCase } from "./synthetic-digest";
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

/** What the digest drives: the head, the caches and the named phases. */
interface PhasedGraph {
  makeCache(): Cache[];
  logitsFromHidden(h: MlxArray): MlxArray;
  readonly prefillChunk: MlxPrefillChunk;
  readonly prefillTail: MlxWidthTable<MlxVerifyForward>;
  readonly decode: MlxWidthTable<MlxPhaseForward>;
  readonly verify: MlxWidthTable<MlxVerifyForward>;
}

type Scheme = { name: string; kvBits?: number; kernels?: "tiled-causal" | "unfused"; turbo?: { kBits: number; vBits: number } };
const BF16: Scheme = { name: "bf16" };
const CONVERTED: Scheme[] = [
  { name: "affine4 tiled-causal", kvBits: 4, kernels: "tiled-causal" },
  { name: "affine4 unfused", kvBits: 4, kernels: "unfused" },
  { name: "affine8 tiled-causal", kvBits: 8, kernels: "tiled-causal" },
  { name: "affine8 unfused", kvBits: 8, kernels: "unfused" },
  { name: "turbo k8v3", turbo: { kBits: 8, vBits: 3 } },
];

interface Family {
  name: string;
  build(): PhasedGraph;
  /** Schemes the graph attends: the softcapped read takes plain and TurboQuant
   * storage, never converted affine storage. */
  schemes: Scheme[];
  /** Affine rows delayed until this many positions; the softcapped read stays below it. */
  delayedStart: number;
}

function miniCpmGraph(): MiniCPM5Model {
  const t = new Tensors();
  decoder(t, false, denseMlp(t));
  t.linear("lm_head", [VOCAB, HIDDEN], true, 0.05);
  return new MiniCPM5Model(t.weights(), textConfig("llama", {}), testComposition());
}
function qwen3Graph(): Qwen3Model {
  const t = new Tensors();
  decoder(t, true, denseMlp(t));
  return new Qwen3Model(t.weights(), textConfig("qwen3", { tieWordEmbeddings: true, ropeParameters: {} }), testComposition());
}
function qwen3MoeGraph(): Qwen3MoeModel {
  const t = new Tensors();
  decoder(t, true, (layer, prefix) => {
    if (layer === 0) { denseMlp(t)(layer, prefix); return; }
    t.linear(`${prefix}.gate`, [EXPERTS, HIDDEN], true);
    t.linear(`${prefix}.switch_mlp.gate_proj`, [EXPERTS, MOE_INTER, HIDDEN], true);
    t.linear(`${prefix}.switch_mlp.up_proj`, [EXPERTS, MOE_INTER, HIDDEN], true);
    t.linear(`${prefix}.switch_mlp.down_proj`, [EXPERTS, HIDDEN, MOE_INTER], true);
  });
  t.linear("lm_head", [VOCAB, HIDDEN], true, 0.05);
  const config = textConfig("qwen3_moe", {
    numExperts: EXPERTS, topKExperts: 2, normTopkProb: true, decoderSparseStep: 1, mlpOnlyLayers: [0], ropeParameters: {} });
  return new Qwen3MoeModel(t.weights(), config, testComposition());
}
/** llama with alternating full and sliding layers (mistral-style layer types). */
function llamaSlidingGraph(): UniversalDenseModel {
  const t = new Tensors();
  decoder(t, false, denseMlp(t));
  t.linear("lm_head", [VOCAB, HIDDEN], true, 0.05);
  const raw = { model_type: "llama", hidden_size: HIDDEN, num_hidden_layers: LAYERS, num_attention_heads: HEADS,
    num_key_value_heads: KV_HEADS, head_dim: HEAD_DIM, intermediate_size: INTER, vocab_size: VOCAB, rms_norm_eps: 1e-6,
    rope_theta: 10000, tie_word_embeddings: false, layer_types: [F, S, F, S], sliding_window: WINDOW };
  return new UniversalDenseModel(t.weights(), textConfig("llama", {}, raw), testComposition());
}
/** gemma2: softcapped attention scores over plain KV, bf16 (unquantized)
 * projections, every layer full attention unless `types` makes some sliding. */
function gemma2Graph(types: string[] | null = null): UniversalDenseModel {
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
  const config = textConfig("gemma2", {}, raw);
  const args: UniversalArgs = { ...genericArgsFor(config)!, ...(types ? { layerTypes: types, slidingWindow: WINDOW } : {}) };
  return new UniversalDenseModel(t.weights(), config, testComposition(), { ...args });
}

const FAMILIES: Family[] = [
  { name: "qwen3", build: qwen3Graph, schemes: [BF16, ...CONVERTED], delayedStart: 40 },
  { name: "qwen3-moe", build: qwen3MoeGraph, schemes: [BF16, ...CONVERTED], delayedStart: 40 },
  { name: "universal gemma2 softcap", build: () => gemma2Graph(), schemes: [BF16, CONVERTED[4]!], delayedStart: 80 },
  { name: "universal gemma2 softcap (sliding)", build: () => gemma2Graph([F, S, F, S]), schemes: [BF16, CONVERTED[4]!], delayedStart: 80 },
];

const graphs = new Map<string, PhasedGraph>();
const graph = (family: Family): PhasedGraph => {
  let found = graphs.get(family.name);
  if (!found) graphs.set(family.name, found = family.build());
  return found;
};

// ── Running a graph into the digest ─────────────────────────────────────────

const digests = syntheticDigests("dense-family");
const prompt = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 37 + salt * 11 + 5) % VOCAB);
const ids = (rows: number[][]) => ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);
const disposeAll = (...sets: Cache[][]) => { for (const set of sets) for (const cache of set) cache.dispose(); };

/** Hidden rows, their logits and the caches after a forward, under `label`. */
function output(digest: DigestCase, label: string, g: PhasedGraph, hidden: MlxArray, caches: Cache[]): void {
  try {
    using logits = g.logitsFromHidden(hidden);
    using wide = logits.astype(Dtype.float32);
    expect(wide.toFloat32().every(Number.isFinite), `${label} logits finite`).toBe(true);
    digest.tensors(`${label} hidden`, hidden);
    digest.tensors(`${label} logits`, logits);
    digest.caches(`${label} caches`, caches);
  } finally { hidden.dispose(); }
}

/** The tokens of `rows` appended through the phase a scheduler names for them. */
function step(digest: DigestCase, label: string, g: PhasedGraph, caches: Cache[], rows: number[][]): void {
  using input = ids(rows);
  output(digest, label, g, phaseHidden(g, input, caches), caches);
}

/** A phase operation called directly on its own cache set. */
function phased(digest: DigestCase, label: string, g: PhasedGraph, caches: Cache[], rows: number[][],
  operation: (input: MlxArray, caches: Cache[]) => MlxArray | Promise<MlxArray>): void {
  using input = ids(rows);
  output(digest, `${label} (phase operation)`, g, operation(input, caches) as MlxArray, caches);
}

/** Convert after a forward (quantized_kv_start 0): the convert-after reference,
 * whose affine storage reads with the scheme's kernel set. */
function convert(scheme: Scheme, ...sets: Cache[][]): void {
  if (scheme.kvBits === undefined && !scheme.turbo) return;
  for (const caches of sets) {
    if (scheme.turbo) settleTurbo(caches, scheme.turbo.kBits, scheme.turbo.vBits, 0);
    else settleAffine(caches, scheme.kvBits!, 0, 64, composedAffineKernels(scheme.kernels!));
  }
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
  describe(`dense family: ${family.name}`, () => {
    for (const scheme of family.schemes) {
      const tag = `${family.name} ${scheme.name}`;

      digests.test(`${tag}: prefill 37, decodes, window, verify, a rolled-back round, fill 1-4, phase operations`, digest => {
        const g = graph(family);
        const caches = g.makeCache(), phaseCaches = g.makeCache();
        try {
          const prefill = prompt(37, 1);
          step(digest, "prefill37", g, caches, [prefill]);
          phased(digest, "prefillChunk37", g, phaseCaches, [prefill], (input, set) => g.prefillChunk.gpu(input, set, null));
          convert(scheme, caches, phaseCaches);
          digest.caches("converted", caches);
          digest.caches("converted (phase operation)", phaseCaches);

          for (let index = 0; index < 3; index++) {
            step(digest, `decode ${index}`, g, caches, [[200 + index]]);
            phased(digest, `decode ${index}`, g, phaseCaches, [[200 + index]], (input, set) => g.decode.one(input, set));
          }
          step(digest, "window5", g, caches, [prompt(5, 2)]);
          phased(digest, "prefillTail5", g, phaseCaches, [prompt(5, 2)], (input, set) => g.prefillTail.fiveToEight(input, set, null) as MlxArray);
          step(digest, "verify4", g, caches, [prompt(4, 3)]);
          phased(digest, "verify4", g, phaseCaches, [prompt(4, 3)], (input, set) => g.verify.four(input, set, null));

          // A speculative round: arm, verify 5, keep 2, decode.
          roundBegin(caches, phaseCaches);
          step(digest, "spec verify5", g, caches, [prompt(5, 4)]);
          phased(digest, "spec verify5", g, phaseCaches, [prompt(5, 4)], (input, set) => g.verify.fiveToEight(input, set, null));
          rollback(2, caches, phaseCaches);
          digest.caches("spec rollback keep 2", caches);
          step(digest, "decode after rollback", g, caches, [[77]]);

          // Committed fill spans: these graphs append them through the forward.
          for (const span of [1, 2, 3, 4]) step(digest, `fill${span}`, g, caches, [prompt(span, 5 + span)]);
          step(digest, "decode after fill", g, caches, [[91]]);

          // The fill path's storage: target row layouts of one row.
          const rows = rowsOf([caches], targetCacheLayout);
          try {
            digest.caches("target rows", rows);
            for (const span of [1, 2, 3, 4]) step(digest, `fill${span} (target rows)`, g, rows, [prompt(span, 9 + span)]);
            step(digest, "decode (target rows)", g, rows, [[33]]);
          } finally { disposeAll(rows); }
        } finally { disposeAll(caches, phaseCaches); }
      }, 120_000);

      digests.test(`${tag}: prefill 200, decodes, a window across the 256-position step`, digest => {
        const g = graph(family);
        const caches = g.makeCache();
        try {
          step(digest, "prefill200", g, caches, [prompt(200, 6)]);
          convert(scheme, caches);
          for (let index = 0; index < 2; index++) step(digest, `decode ${index}`, g, caches, [[11 + index]]);
          step(digest, "window60 across 256", g, caches, [prompt(60, 7)]);
          step(digest, "decode after", g, caches, [[19]]);
        } finally { disposeAll(caches); }
      }, 120_000);

      digests.test(`${tag}: padded batched prefill of three unequal rows, batched decode, per-row rollback, batched fill`, digest => {
        const g = graph(family);
        const lengths = [9, 14, 6];
        const sources = lengths.map(() => g.makeCache());
        let rows: Cache[] = [];
        try {
          lengths.forEach((length, row) => step(digest, `row ${row} prefill`, g, sources[row]!, [prompt(length, 11 + row)]));
          convert(scheme, ...sources);
          rows = rowsOf(sources, targetCacheLayout);
          digest.caches("merged rows", rows);

          // Right-padded window: real lengths 5, 3 and 7 in a width of 7.
          const real = [5, 3, 7], width = 7;
          for (const cache of rows) (cache as unknown as PaddedPrefillCache).preparePrefill({ lengths: real, rightPadding: real.map(n => width - n) });
          step(digest, "padded prefill 5/3/7", g, rows, real.map((n, row) => [...prompt(n, 14 + row), ...Array(width - n).fill(0)]));
          for (const cache of rows) (cache as unknown as PaddedPrefillCache).finalizePrefill();
          digest.caches("padded prefill finalized", rows);

          for (let index = 0; index < 2; index++) step(digest, `batched decode ${index}`, g, rows, [[40 + index], [50 + index], [60 + index]]);

          roundBegin(rows);
          step(digest, "batched verify4", g, rows, [prompt(4, 20), prompt(4, 21), prompt(4, 22)]);
          for (const cache of rows) (cache as unknown as { specRoundRollback(keep: readonly number[]): void }).specRoundRollback([1, 3, 2]);
          digest.caches("batched rollback keep 1/3/2", rows);
          step(digest, "batched decode after rollback", g, rows, [[70], [71], [72]]);
          step(digest, "batched fill (one position per row)", g, rows, [[80], [81], [82]]);
        } finally { disposeAll(...sources, rows); }
      }, 120_000);

      digests.test(`${tag}: the continuous-decoding row layout of three unequal rows, batched decode`, digest => {
        const g = graph(family);
        const lengths = [12, 5, 9];
        const sources = lengths.map(() => g.makeCache());
        let rows: Cache[] = [];
        try {
          lengths.forEach((length, row) => step(digest, `row ${row} prefill`, g, sources[row]!, [prompt(length, 31 + row)]));
          convert(scheme, ...sources);
          rows = rowsOf(sources, cache => ownedCacheLayout(cache)!);
          digest.caches("owned rows", rows);
          for (let index = 0; index < 3; index++) step(digest, `owned batched decode ${index}`, g, rows, [[3 + index], [5 + index], [7 + index]]);
        } finally { disposeAll(...sources, rows); }
      }, 120_000);
    }

    digests.test(`${family.name} bf16: padded batched prefill of three unequal rows from empty prefill layouts`, digest => {
      const g = graph(family);
      const serial = g.makeCache();
      let rows: Cache[] = [];
      try { rows = serial.map(cache => prefillCacheLayout(cache) as unknown as Cache); }
      finally { disposeAll(serial); }
      try {
        const real = [6, 2, 4], width = 6;
        for (const cache of rows) (cache as unknown as PaddedPrefillCache).preparePrefill({ lengths: real, rightPadding: real.map(n => width - n) });
        step(digest, "padded prefill 6/2/4", g, rows, real.map((n, row) => [...prompt(n, 30 + row), ...Array(width - n).fill(0)]));
        for (const cache of rows) (cache as unknown as PaddedPrefillCache).finalizePrefill();
        for (let index = 0; index < 2; index++) step(digest, `batched decode ${index}`, g, rows, [[3 + index], [5 + index], [7 + index]]);
      } finally { disposeAll(rows); }
    }, 120_000);

    digests.test(`${family.name} delayed affine4 (start ${family.delayedStart}): prefill 37, decode, fill 1-4, decode`, digest => {
      const g = graph(family);
      // The delayed row layouts read with the tiled-causal set.
      const maintain = createKvMaintenance({ kind: "affine", bits: 4, groupSize: 64, quantizedKvStart: family.delayedStart,
        kernels: "tiled-causal" });
      const caches = g.makeCache();
      maintain.preparePrefill!(caches);
      try {
        step(digest, "prefill37", g, caches, [prompt(37, 41)]);
        step(digest, "decode", g, caches, [[42]]);
        for (const span of [1, 2, 3, 4]) step(digest, `fill${span}`, g, caches, [prompt(span, 43 + span)]);
        step(digest, "decode after fill", g, caches, [[29]]);
      } finally { disposeAll(caches); }
    }, 120_000);

    // Perplexity rows run over the stateless training caches (`TrainingCache`),
    // each composed with its layer's sliding window by scoring/full-sequence.ts.
    digests.test(`${family.name}: full-sequence scoring, perplexity rows over the training caches`, digest => {
      const rows = [Int32Array.from(prompt(33, 63)), Int32Array.from(prompt(33, 64)), Int32Array.from(prompt(33, 65))];
      const result = evalPpl(graph(family) as unknown as RuntimeModel, rows, 2);
      expect(Number.isFinite(result.ppl)).toBe(true);
      digest.value("perplexity (ppl, mean loss, standard error)", result);
    }, 120_000);

    digests.test(`${family.name}: full-sequence scoring, a padded batch through the training caches`, digest => {
      const valid = [24, 19, 15];
      using input = ids(valid.map((n, row) => [...prompt(n, 66 + row), ...Array(24 - n).fill(0)]));
      using logits = trainForward(graph(family) as unknown as RuntimeModel, input, valid);
      digest.tensors("padded batch logits", logits);
    }, 120_000);
  });
}

// ── Composition facts ───────────────────────────────────────────────────────

describe("dense family: what the graphs compose at construction", () => {
  test("makeCache composes one mask memo and the TurboQuant read each graph attends", () => {
    const sets: [string, PhasedGraph, unknown][] = [["minicpm5", miniCpmGraph(), deferredTurboQuantKernels],
      ["qwen3", graph(FAMILIES[0]!), denseTurboQuantKernels], ["qwen3-moe", graph(FAMILIES[1]!), denseTurboQuantKernels],
      ["universal", llamaSlidingGraph(), denseTurboQuantKernels]];
    for (const [name, g, kernels] of sets) {
      const caches = g.makeCache();
      try {
        const masks = new Set(caches.map(cache => (cache as { masks?: unknown }).masks));
        expect(masks.size, name).toBe(1);
        for (const cache of caches.filter(cache => cache.signature() === "kv:plain"))
          expect((cache as KVCache).turboQuantKernels, name).toBe(kernels as KVCache["turboQuantKernels"]);
      } finally { disposeAll(caches); }
    }
  });

  test("the universal graph's adapter targets: every projection of a llama layer, none of the bf16 gemma2 layers", () => {
    expect(llamaSlidingGraph().loraTargets().size).toBe(LAYERS * 7);
    expect((graph(FAMILIES[2]!) as UniversalDenseModel).loraTargets().size).toBe(0);
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

  test("MiniCPM5 reads MLX_BUN_COMPILED_SWIGLU once at construction: 0 builds the unfused MLP, unset the compiled one", () => {
    const unfused = createRuntimeConfig({ ...runtimeConfig().values, MLX_BUN_COMPILED_SWIGLU: "0" });
    expect(withRuntimeConfig(unfused, () => miniCpmGraph()).layers.every(layer => layer.mlp.kind === "unfused")).toBe(true);
    expect(miniCpmGraph().layers.every(layer => layer.mlp.kind === "compiled")).toBe(true);
  });
});
