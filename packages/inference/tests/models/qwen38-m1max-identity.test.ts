// D2 bit-identity gate. The generated M1 Max graph (src/models/qwen/qwen38-27b-trellis-m1max.ts, an entire graph
// built from the composition) against the frozen hand-written TQ graph (qwen38-27b-trellis-tq-reference.ts, deleted
// with this test after the merge), on small synthetic weights laid out by the generated quant table: 48 DeltaNet and
// 16 attention layers with the artifact's 3- and 4-bit affine projections, Trellis gate/up at the artifact's widths
// (mixed in its ten layers) and its 3-bit block-interleaved and row-major downs, scaled down in width. For bf16,
// affine 4- and 8-bit and TurboQuant KV (each graph's own makeCache), every phase operation and width class the TQ
// graph served is held to the bit, hidden, logits and every cache tensor: prefillChunk.gpu; prefillTail at every
// class; decode at one row and at two, four, six and nine sequences; verify at 2, 3, 4, 5, 8 and 12 rows with capture
// and at several sequences; appendKnown and sequenceHidden at every class; the head at every width; the capturing
// mixed forward (decode, prompt and verify groups, packed and not); media positions and the vision prefill phases.
// The composition facts (capabilities, LoRA targets, no committed spans, the residual basis) equal the TQ graph's.
// The real artifact runs the same phases opt-in on the M1 Max (MLX_BUN_TEST_QWEN38_ARTIFACT=<snapshot>), where the
// reduce and scatter kernels take the representations that GPU family qualifies; elsewhere it skips, which is not a pass.
import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype, deviceArchitecture } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { parseQuantization, type ModelConfig } from "../../src/artifacts/config";
import { TURBOQUANT_FOLD_FILE } from "../../src/artifacts/turboquant-fold";
import type { Weights } from "../../src/artifacts/weights";
import type { BatchableCache, Cache } from "../../src/contracts/mlx/cache";
import { widthClass, type MlxLayerCapture, type MlxPhaseGroup, type MlxTokenPhases, type MlxWidthClass } from "../../src/contracts/mlx/graph";
import type { Composition, KvStorageScheme } from "../../src/contracts/portable/composition";
import { LAYERS, Qwen38TrellisM1Max } from "../../src/models/qwen/qwen38-27b-trellis-m1max";
import { targetCacheLayout } from "../../src/state/layout";
import { Qwen38TrellisTQ as FrozenTQ } from "./qwen38-27b-trellis-tq-reference";

const HIDDEN = 512, INTER = 256, VOCAB = 256, HEADS = 4, KV_HEADS = 2, HEAD_DIM = 256;
const HK = 4, HV = 8, DK = 64, DV = 64, KERNEL = 4;
const KEY_DIM = HK * DK, VALUE_DIM = HV * DV, CONV_DIM = 2 * KEY_DIM + VALUE_DIM;
const PREFIX = "language_model.model";
const CHUNK = 40;
const g13s = deviceArchitecture() === "applegpu_g13s";

const owned: MlxArray[] = [];
afterAll(() => { for (const array of owned) array.dispose(); });
const keep = (array: MlxArray) => { owned.push(array); return array; };

const noise = (count: number, phase: number, scale: number, shift = 0) => Float32Array.from({ length: count },
  (_, i) => shift + Math.sin(i * 0.37 + phase) * scale + Math.cos(i * 0.11 + 2.3 * phase) * scale);
const tensor = (values: Float32Array, shape: number[], dtype = Dtype.bfloat16): MlxArray => {
  using wide = MlxArray.fromFloat32(values, shape);
  return keep(wide.astype(dtype));
};

/** The synthetic artifact: every layer laid out as the generated table lists it. */
function artifact(): { weights: Weights; config: ModelConfig } {
  const tensors = new Map<string, MlxArray>();
  const quant: Record<string, unknown> = { bits: 4, group_size: 64, mode: "affine" };
  let seed = 1;
  const affine = (path: string, rows: number, cols: number, bits = 4, scale = 0.06) => {
    using w = tensor(noise(rows * cols, ++seed, scale), [rows, cols]);
    const q = ops.quantize(w, 64, bits);
    tensors.set(`${path}.weight`, keep(q.packed));
    tensors.set(`${path}.scales`, keep(q.scales));
    tensors.set(`${path}.biases`, keep(q.biases));
    quant[path] = { bits, group_size: 64, mode: "affine" };
  };
  const norm = (path: string, width: number) => tensors.set(path, tensor(noise(width, ++seed, 0.05, 1), [width]));
  /** Packed Trellis codes: row-major [stored, cols * k / 32] words, or the 3-bit block interleave [cols / 512, stored, 48]. */
  const trellis = (path: string, stored: number, cols: number, axis: 0 | 1, k: number, interleaved = false) => {
    let state = ++seed * 7919;
    const words = cols * k / 32;
    using ints = MlxArray.fromInt32(Int32Array.from({ length: stored * words },
      () => state = Math.imul(state, 1664525) + 1013904223), interleaved ? [cols / 512, stored, 48] : [stored, words]);
    tensors.set(`${path}.weight`, keep(ints.astype(Dtype.uint32)));
    tensors.set(`${path}.scales`, tensor(Float32Array.from({ length: stored }, (_, i) => (i % 17 + 1) / 400), [stored], Dtype.float16));
    quant[path] = { bits: k, group_size: 256, mode: "trellis", trellis: { L: 12, code: "1mad", axis } };
  };

  affine(`${PREFIX}.embed_tokens`, VOCAB, HIDDEN, 4, 0.5);
  affine("language_model.lm_head", VOCAB, HIDDEN, 4, 0.05);
  norm(`${PREFIX}.norm.weight`, HIDDEN);
  LAYERS.forEach((spec, layer) => {
    const prefix = `${PREFIX}.layers.${layer}`;
    norm(`${prefix}.input_layernorm.weight`, HIDDEN);
    norm(`${prefix}.post_attention_layernorm.weight`, HIDDEN);
    const [a, b, c, d, e] = spec.projections;
    if (spec.kind === "deltanet") {
      const p = `${prefix}.linear_attn`;
      affine(`${p}.in_proj_qkv`, CONV_DIM, HIDDEN, a);
      affine(`${p}.in_proj_z`, VALUE_DIM, HIDDEN, b);
      affine(`${p}.in_proj_a`, HV, HIDDEN, c);
      affine(`${p}.in_proj_b`, HV, HIDDEN, d);
      affine(`${p}.out_proj`, HIDDEN, VALUE_DIM, e);
      tensors.set(`${p}.conv1d.weight`, tensor(noise(CONV_DIM * KERNEL, ++seed, 0.3), [CONV_DIM, KERNEL, 1]));
      tensors.set(`${p}.A_log`, tensor(noise(HV, ++seed, 0.5), [HV], Dtype.float32));
      tensors.set(`${p}.dt_bias`, tensor(noise(HV, ++seed, 0.5), [HV]));
      norm(`${p}.norm.weight`, DV);
    } else {
      const p = `${prefix}.self_attn`;
      affine(`${p}.q_proj`, HEADS * HEAD_DIM * 2, HIDDEN, a);
      affine(`${p}.k_proj`, KV_HEADS * HEAD_DIM, HIDDEN, b);
      affine(`${p}.v_proj`, KV_HEADS * HEAD_DIM, HIDDEN, c);
      affine(`${p}.o_proj`, HIDDEN, HEADS * HEAD_DIM, d);
      norm(`${p}.q_norm.weight`, HEAD_DIM);
      norm(`${p}.k_norm.weight`, HEAD_DIM);
    }
    const m = `${prefix}.mlp`;
    trellis(`${m}.gate_proj`, INTER, HIDDEN, 1, spec.gate);
    trellis(`${m}.up_proj`, INTER, HIDDEN, 1, spec.up);
    trellis(`${m}.down_proj`, INTER, HIDDEN, 0, spec.down, spec.downLayout === "k3Interleaved");
  });
  const weights = {
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
  const config = {
    modelDir: "", modelType: "qwen3_5_text", architectures: [], dtype: "bfloat16", kvQuant: null,
    hasVisionSidecar: false, eosTokenIds: [], raw: {},
    quantization: parseQuantization(quant),
    text: {
      hiddenSize: HIDDEN, numHiddenLayers: LAYERS.length, numAttentionHeads: HEADS, numKeyValueHeads: KV_HEADS,
      headDim: HEAD_DIM, intermediateSize: INTER, rmsNormEps: 1e-6, vocabSize: VOCAB,
      linearNumKeyHeads: HK, linearNumValueHeads: HV, linearKeyHeadDim: DK, linearValueHeadDim: DV,
      linearConvKernelDim: KERNEL, fullAttentionInterval: 4, partialRotaryFactor: 0.25,
      ropeParameters: { full_attention: { ropeTheta: 10_000_000, ropeType: "default", partialRotaryFactor: 0.25, factor: 1 } },
      tieWordEmbeddings: false,
    },
  } as unknown as ModelConfig;
  return { weights, config };
}

const composition = (overrides: Partial<Composition> = {}): Composition => Object.freeze<Composition>({
  device: "applegpu_g13s", aneBridge: false, kv: { kind: "bf16" }, draftDepth: { adaptive: true, max: 7 },
  prefillChunkTokens: CHUNK, adapters: false, maxRows: 8, speculativeAffineKv: true, packedMixedMlp: true,
  vectorGreedySampling: true, ...overrides,
});

/** The KV schemes the composition serves, as `--kv-quant` composes them (fused SDPA unset). */
const SCHEMES: Readonly<Record<string, KvStorageScheme>> = Object.freeze({
  bf16: { kind: "bf16" },
  kv4: { kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" },
  kv8: { kind: "affine", bits: 8, groupSize: 64, kernels: "unfused" },
  turbo: { kind: "turbo", kBits: 8, vBits: 3 },
});

// ── Comparison ──────────────────────────────────────────────────────────────

const results = new Map<string, { worst: number; tensors: number }>();
const record = (label: string, diff: number, tensors: number) => {
  const prior = results.get(label) ?? { worst: 0, tensors: 0 };
  results.set(label, { worst: Math.max(prior.worst, diff), tensors: prior.tensors + tensors });
};
afterAll(() => {
  for (const [label, { worst, tensors }] of results) console.log(`qwen38-m1max-identity ${label}: maxDiff ${worst} over ${tensors} tensors`);
});

/** Largest absolute difference: 0 when shape, dtype and bytes agree; any other result throws. */
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

/** Every tensor of every layer's cache, with its signature, offsets and rows. */
function stateDiff(label: string, actual: readonly Cache[], expected: readonly Cache[]): void {
  expect(actual.length).toBe(expected.length);
  let count = 0;
  for (let layer = 0; layer < expected.length; layer++) {
    const a = actual[layer]!, b = expected[layer]!;
    expect(a.signature(), `${label} layer ${layer}`).toBe(b.signature());
    expect(a.offset, `${label} layer ${layer}`).toBe(b.offset);
    expect((a as Partial<BatchableCache>).rowOffsets, `${label} layer ${layer}`).toEqual((b as Partial<BatchableCache>).rowOffsets);
    const left = a.state(), right = b.state();
    try {
      expect(left.length, `${label} layer ${layer}`).toBe(right.length);
      for (let i = 0; i < right.length; i++) { maxDiff(left[i]!, right[i]!, `${label} layer ${layer} state ${i}`); count++; }
    } finally {
      if (a.stateNeedsDispose) for (const array of left) array.dispose();
      if (b.stateNeedsDispose) for (const array of right) array.dispose();
    }
  }
  record(`${label} cache`, 0, count);
}

/** Hidden rows and their logits through each graph's head of `width` (all owned by the caller and disposed here);
 *  `last` projects only the last position (a prompt chunk's logits at the real vocabulary). */
function outputDiff(label: string, actual: MlxArray, expected: MlxArray, actualHead: (h: MlxArray) => MlxArray,
  expectedHead: (h: MlxArray) => MlxArray, last = false): void {
  try {
    maxDiff(actual, expected, `${label} hidden`);
    const lastRow = (h: MlxArray) => { const L = h.shape[1]!; return h.slice([0, L - 1, 0], [h.shape[0]!, L, h.shape[2]!]); };
    using a = last ? lastRow(actual) : null, b = last ? lastRow(expected) : null;
    using la = actualHead(a ?? actual);
    using lb = expectedHead(b ?? expected);
    maxDiff(la, lb, `${label} logits`);
    using wide = lb.astype(Dtype.float32);
    expect(wide.toFloat32().every(Number.isFinite), `${label} logits finite`).toBe(true);
    record(`${label} hidden+logits`, 0, 2);
  } finally { actual.dispose(); expected.dispose(); }
}

function tapsDiff(label: string, actual: MlxArray[], expected: MlxArray[]): void {
  try {
    expect(actual.length, label).toBe(LAYERS.length + 1);
    expect(expected.length, label).toBe(LAYERS.length + 1);
    for (let i = 0; i < expected.length; i++) maxDiff(actual[i]!, expected[i]!, `${label} layer ${i}`);
    record(`${label} captured layers`, 0, expected.length);
  } finally { for (const array of [...actual, ...expected]) array.dispose(); }
}

/** A capture that copies every layer it is handed. */
function collecting(): { capture: MlxLayerCapture; taps: MlxArray[] } {
  const taps: MlxArray[] = [];
  return { taps, capture: (_layer, hidden) => { taps.push(ops.contiguous(hidden)); } };
}

// ── The two graphs ──────────────────────────────────────────────────────────

/** Both graphs' phase tables, called alike: every operation the TQ graph served runs synchronously. */
type Graph = MlxTokenPhases & Pick<Qwen38TrellisM1Max, "makeCache" | "sequenceHidden" | "prefillEmbeddings" |
  "prefillEmbeddingsAtPositions" | "prefillAtPositions" | "decodeAtPositions" | "embed" | "mixed">;
interface Pair { old: Graph; current: Graph; oldGraph: FrozenTQ; currentGraph: Qwen38TrellisM1Max }

const prompt = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 37 + salt * 11 + 5) % VOCAB);
const ids = (rows: number[][]) => ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);
const disposeAll = (...sets: Cache[][]) => { for (const set of sets) for (const cache of set) cache.dispose(); };

let synthetic: { weights: Weights; config: ModelConfig } | null = null;
const source = () => synthetic ??= artifact();
const pairs = new Map<string, Pair>();
function pair(scheme: string, overrides: Partial<Composition> = {}): Pair {
  const key = `${scheme} ${JSON.stringify(overrides)}`;
  let found = pairs.get(key);
  if (!found) {
    const { weights, config } = source();
    const facts = composition({ kv: SCHEMES[scheme]!, ...overrides });
    const oldGraph = new FrozenTQ(weights, config, facts), currentGraph = new Qwen38TrellisM1Max(weights, config, facts);
    found = { old: oldGraph as unknown as Graph, current: currentGraph as unknown as Graph, oldGraph, currentGraph };
    pairs.set(key, found);
  }
  return found;
}
afterAll(() => { for (const { currentGraph } of pairs.values()) currentGraph.dispose(); });

/** The storage a composed lego hands the row layouts (bf16 before its conversion, quantized or encoded after). */
const storage = (cache: Cache) => cache.composed?.storage() ?? cache;
/** Verification row layouts holding `sources` (one serial cache list per row), over each layer's storage. */
const rowsOf = (sources: Cache[][]): Cache[] => sources[0]!.map((cache, layer) => {
  const layout = targetCacheLayout(storage(cache)) as unknown as BatchableCache;
  layout.mergeRows(sources.map(rows => storage(rows[layer]!)));
  return layout;
});

/** Runs one operation on both graphs over their caches and compares hidden, the head of `width` and every cache;
 *  with `capture`, every captured layer too. */
function step(label: string, pairOf: Pair, caches: { old: Cache[]; current: Cache[] }, width: MlxWidthClass,
  run: (graph: Graph, state: Cache[], capture: MlxLayerCapture | null) => MlxArray | Promise<MlxArray>,
  options: { capture?: boolean; last?: boolean } = {}): void {
  const before = options.capture ? collecting() : null, after = options.capture ? collecting() : null;
  const expected = run(pairOf.old, caches.old, before?.capture ?? null) as MlxArray;
  const actual = run(pairOf.current, caches.current, after?.capture ?? null) as MlxArray;
  if (before && after) tapsDiff(label, after.taps, before.taps);
  outputDiff(label, actual, expected, pairOf.current.logits[width], pairOf.old.logits[width], options.last);
  stateDiff(label, caches.current, caches.old);
}

const input = (rows: number[][]) => keep(ids(rows));

describe("the composition the generated graph declares", () => {
  test("capabilities, LoRA targets, caches, no committed spans or verify costs, no Neural Engine operation: the TQ graph's", () => {
    for (const scheme of Object.keys(SCHEMES)) {
      const { oldGraph, currentGraph } = pair(scheme);
      expect(currentGraph.graphCapabilities).toEqual(oldGraph.graphCapabilities);
      expect([...currentGraph.loraTargets().keys()].sort()).toEqual([...oldGraph.loraTargets().keys()].sort());
      expect(currentGraph.loraTargets().size).toBe(48 * 5 + 16 * 4);
      expect((currentGraph as { knownSpans?: unknown }).knownSpans).toBeUndefined();
      expect(oldGraph.knownSpans).toBeUndefined();
      expect((currentGraph as { verifyRoundCosts?: unknown }).verifyRoundCosts).toBeUndefined();
      expect((oldGraph as { verifyRoundCosts?: unknown }).verifyRoundCosts).toBeUndefined();
      expect(currentGraph.prefillChunk.ane).toBeUndefined();
      expect(currentGraph.requiredDenseKvLayers).toEqual(oldGraph.requiredDenseKvLayers);
      const a = currentGraph.makeCache(), b = oldGraph.makeCache();
      try { expect(a.map(cache => `${cache.constructor.name} ${cache.signature()}`)).toEqual(b.map(cache => `${cache.constructor.name} ${cache.signature()}`)); }
      finally { disposeAll(a, b); }
    }
    const off = pair("bf16", { speculativeAffineKv: false });
    expect(off.currentGraph.graphCapabilities).toEqual(off.oldGraph.graphCapabilities);
    const adapted = pair("bf16", { adapters: true });
    expect(adapted.currentGraph.graphCapabilities).toEqual(adapted.oldGraph.graphCapabilities);
  });

  test("the residual basis its fold record declares, read once at construction, as the TQ graph read it", () => {
    const { weights, config } = source();
    const dir = mkdtempSync(join(tmpdir(), "qwen38-m1max-fold-"));
    const quiet = console.error;
    console.error = () => {};
    try {
      const at = { ...config, modelDir: dir } as ModelConfig;
      const declared = (graph: { draftTarget(caches: Cache[]): { residualBasis?: unknown } }) => graph.draftTarget([]).residualBasis;
      expect(declared(new Qwen38TrellisM1Max(weights, at, composition()))).toBeUndefined();
      writeFileSync(join(dir, TURBOQUANT_FOLD_FILE), JSON.stringify({ seed: 7, r1: true, hiddenSize: HIDDEN,
        final_gain: Array.from({ length: HIDDEN }, (_, i) => 1 + i / HIDDEN) }));
      const current = new Qwen38TrellisM1Max(weights, at, composition()), old = new FrozenTQ(weights, at, composition());
      expect(declared(current)).toEqual(declared(old));
      expect(declared(current)).toEqual({ r1Seed: 7, finalGain: Float32Array.from({ length: HIDDEN }, (_, i) => 1 + i / HIDDEN) });
      current.dispose();
    } finally { console.error = quiet; rmSync(dir, { recursive: true, force: true }); }
  });
});

for (const scheme of Object.keys(SCHEMES)) {
  describe(`Qwen3.8 M1 Max graph against the TQ graph, ${scheme} KV`, () => {
    test(`${scheme}: one sequence through every phase and width class`, () => {
      const p = pair(scheme);
      const caches = { old: p.old.makeCache(), current: p.current.makeCache() };
      try {
        step(`${scheme} prefillChunk.gpu ${CHUNK}`, p, caches, "wider", (g, s, c) => g.prefillChunk.gpu(input([prompt(CHUNK, 1)]), s, c), { capture: true });
        for (const rows of [13, 6, 4, 3, 2, 1])
          step(`${scheme} prefillTail ${rows}`, p, caches, widthClass(rows), (g, s, c) => g.prefillTail[widthClass(rows)](input([prompt(rows, 2 + rows)]), s, c), { capture: true });
        for (let n = 0; n < 3; n++) step(`${scheme} decode.one`, p, caches, "one", (g, s, c) => g.decode.one(input([[100 + n]]), s, c));
        for (const rows of [2, 3, 4, 5, 8, 12])
          step(`${scheme} verify ${rows}`, p, caches, widthClass(rows), (g, s, c) => g.verify[widthClass(rows)](input([prompt(rows, 20 + rows)]), s, c), { capture: true });
        for (const rows of [1, 2, 4, 6, 9])
          step(`${scheme} appendKnown ${rows}`, p, caches, widthClass(rows), (g, s) => g.appendKnown[widthClass(rows)](input([prompt(rows, 40 + rows)]), s));
        for (const rows of [3, 6, 10])
          step(`${scheme} sequenceHidden ${rows}`, p, caches, widthClass(rows), (g, s) => g.sequenceHidden(input([prompt(rows, 50 + rows)]), s, widthClass(rows)));
        for (const rows of [3, 6])
          step(`${scheme} prefillEmbeddings ${rows}`, p, caches, widthClass(rows), (g, s) => {
            using embeds = p.old.embed.encode(input([prompt(rows, 60 + rows)]));
            return g.prefillEmbeddings!(embeds, s, null, null, null, widthClass(rows));
          });
        // Media positions: [3, rows, tokens] request positions continuing the cache's text positions.
        const grid = (start: number, rows: number) =>
          keep(ops.fromInt32(Array.from({ length: 3 * rows }, (_, i) => start + (i % rows) + Math.floor(i / rows)), [3, 1, rows]));
        for (const rows of [6, 1]) {
          const start = caches.old[0]!.offset;
          const positions = grid(start, rows);
          step(`${scheme} media positions ${rows}`, p, caches, widthClass(rows), (g, s) => rows === 1
            ? g.decodeAtPositions(input([prompt(1, 70)]), s, positions, "one")
            : g.prefillAtPositions(input([prompt(rows, 71)]), s, positions, widthClass(rows)));
          if (rows === 1) continue;
          const next = grid(start + rows, rows);
          step(`${scheme} vision prefill ${rows}`, p, caches, widthClass(rows), (g, s) => {
            using embeds = p.old.embed.encode(input([prompt(rows, 72)]));
            return g.prefillEmbeddingsAtPositions(embeds, s, next, widthClass(rows));
          });
        }
        step(`${scheme} decode after every phase`, p, caches, "one", (g, s, c) => g.decode.one(input([[7]]), s, c), { capture: true });
        // The head alone at every width, on one hidden.
        for (const rows of [1, 2, 3, 4, 5, 8, 12]) {
          using hidden = p.old.embed.encode(input([prompt(rows, 80 + rows)]));
          using expected = p.old.logits[widthClass(rows)](hidden);
          using actual = p.current.logits[widthClass(rows)](hidden);
          maxDiff(actual, expected, `${scheme} head ${rows}`);
          record(`${scheme} logits[${widthClass(rows)}] ${rows} rows`, 0, 1);
        }
      } finally { disposeAll(caches.old, caches.current); }
    }, 600_000);

    test(`${scheme}: several sequences: decode at 2, 4, 6 and 9 rows, verify windows of several sequences, mixed rows`, () => {
      const p = pair(scheme);
      for (const sequences of [2, 4, 6, 9]) {
        const old: Cache[][] = [], current: Cache[][] = [];
        let oldRows: Cache[] = [], newRows: Cache[] = [];
        try {
          for (let row = 0; row < sequences; row++) {
            const length = 5 + 2 * row, single = { old: p.old.makeCache(), current: p.current.makeCache() };
            old.push(single.old); current.push(single.current);
            step(`${scheme} ${sequences} sequences prefill`, p, single, widthClass(length),
              (g, s) => g.prefillTail[widthClass(length)](input([prompt(length, 90 + row)]), s, null));
          }
          oldRows = rowsOf(old); newRows = rowsOf(current);
          const rows = { old: oldRows, current: newRows };
          const tokens = Array.from({ length: sequences }, (_, row) => [110 + row]);
          step(`${scheme} decode ${sequences} sequences`, p, rows, widthClass(sequences), (g, s, c) => g.decode[widthClass(sequences)](input(tokens), s, c), { capture: true });
          const depth = sequences <= 2 ? 2 : 3;
          const window = Array.from({ length: sequences }, (_, row) => prompt(depth, 120 + row));
          step(`${scheme} verify ${sequences}x${depth}`, p, rows, widthClass(sequences * depth),
            (g, s, c) => g.verify[widthClass(sequences * depth)](input(window), s, c), { capture: true });
        } finally { disposeAll(...old, ...current, oldRows, newRows); }
      }
      // Mixed rows: a decode group, a prompt group and a capturing verify group, each over its own caches, packed and not.
      for (const packedMixedMlp of [true, false]) {
        const q = pair(scheme, { packedMixedMlp });
        const sets = [0, 1, 2].map(() => ({ old: q.old.makeCache(), current: q.current.makeCache() }));
        try {
          for (const [row, set] of sets.entries())
            step(`${scheme} mixed prefill`, q, set, "fiveToEight", (g, s) => g.prefillTail.fiveToEight(input([prompt(7, 130 + row)]), s, null));
          const work = (graph: "old" | "current", taps: MlxLayerCapture): MlxPhaseGroup[] => [
            { phase: "decode", width: "one", ids: input([[140]]), state: sets[0]![graph], capture: null },
            { phase: "window", width: "fiveToEight", ids: input([prompt(5, 141)]), state: sets[1]![graph], capture: null },
            { phase: "verify", width: "twoToThree", ids: input([prompt(3, 142)]), state: sets[2]![graph], capture: taps },
          ];
          const before = collecting(), after = collecting();
          const expected = q.old.mixed!(work("old", before.capture), "wider");
          const actual = q.current.mixed!(work("current", after.capture), "wider");
          tapsDiff(`${scheme} mixed (packed ${packedMixedMlp}) verify group`, after.taps, before.taps);
          for (const [row, width] of (["one", "fiveToEight", "twoToThree"] as const).entries())
            outputDiff(`${scheme} mixed (packed ${packedMixedMlp}) group ${row}`, actual[row]!, expected[row]!, q.current.logits[width], q.old.logits[width]);
          for (const [row, set] of sets.entries()) stateDiff(`${scheme} mixed (packed ${packedMixedMlp}) group ${row}`, set.current, set.old);
        } finally { for (const set of sets) disposeAll(set.old, set.current); }
      }
    }, 600_000);
  });
}

// ── The real artifact (opt-in on the M1 Max: MLX_BUN_TEST_QWEN38_ARTIFACT=<mlx-bun/Qwen3.8-27B-Trellis-3.2bpw
// snapshot>, about 12 GB resident). Both graphs share one Weights. ──────────────────────────────────────────────

const ARTIFACT = process.env.MLX_BUN_TEST_QWEN38_ARTIFACT;

describe("Qwen3.8 M1 Max graph on the real artifact", () => {
  let shared: { weights: Weights; config: ModelConfig; view: Weights } | null = null;
  const open = async () => {
    if (shared) return shared;
    const { Weights } = await import("../../src/artifacts/weights");
    const { loadModelConfig } = await import("../../src/artifacts/config");
    const config = await loadModelConfig(ARTIFACT!);
    const weights = await Weights.open(ARTIFACT!);
    // The generated graph installs the checkpoint view; the TQ graph reads through it and must not install it again.
    const view = {
      tensor: (name: string) => weights.tensor(name), has: (name: string) => weights.has(name), info: (name: string) => weights.info(name),
      get tensorNames() { return weights.tensorNames; }, shards: weights.shards, setView: () => {},
    } as unknown as Weights;
    return shared = { weights, config, view };
  };
  afterAll(() => shared?.weights.dispose());
  const real = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 7919 + salt * 104729) % 151_000 + 1000);

  for (const scheme of Object.keys(SCHEMES))
    test.skipIf(!ARTIFACT || !g13s)(`${scheme}: 2048-token chunk, tails, decodes, verify 2 to 12 with capture, known tokens, two and four sequences`, async () => {
      const { weights, config, view } = await open();
      const facts = composition({ kv: SCHEMES[scheme]!, prefillChunkTokens: 2048 });
      const currentGraph = new Qwen38TrellisM1Max(weights, config, facts);
      const oldGraph = new FrozenTQ(view, config, facts);
      const p: Pair = { old: oldGraph as unknown as Graph, current: currentGraph as unknown as Graph, oldGraph, currentGraph };
      const caches = { old: p.old.makeCache(), current: p.current.makeCache() };
      try {
        step(`real ${scheme} prefillChunk.gpu 2048`, p, caches, "one", (g, s, c) => g.prefillChunk.gpu(input([real(2048, 1)]), s, c), { capture: true, last: true });
        for (const rows of [37, 6, 4, 3])
          step(`real ${scheme} prefillTail ${rows}`, p, caches, "one", (g, s, c) => g.prefillTail[widthClass(rows)](input([real(rows, 2 + rows)]), s, c), { last: true });
        for (let n = 0; n < 6; n++) step(`real ${scheme} decode.one`, p, caches, "one", (g, s, c) => g.decode.one(input([real(1, 10 + n)]), s, c), { capture: n === 0 });
        for (const rows of [2, 3, 4, 5, 8, 12])
          step(`real ${scheme} verify ${rows}`, p, caches, widthClass(rows), (g, s, c) => g.verify[widthClass(rows)](input([real(rows, 20 + rows)]), s, c), { capture: true });
        for (const rows of [1, 3, 6])
          step(`real ${scheme} appendKnown ${rows}`, p, caches, widthClass(rows), (g, s) => g.appendKnown[widthClass(rows)](input([real(rows, 40 + rows)]), s));
      } finally { disposeAll(caches.old, caches.current); }
      for (const sequences of [2, 4]) {
        const old: Cache[][] = [], current: Cache[][] = [];
        let oldRows: Cache[] = [], newRows: Cache[] = [];
        try {
          for (let row = 0; row < sequences; row++) {
            const length = 11 + 4 * row, single = { old: p.old.makeCache(), current: p.current.makeCache() };
            old.push(single.old); current.push(single.current);
            step(`real ${scheme} ${sequences} sequences prefill`, p, single, "one",
              (g, s) => g.prefillTail[widthClass(length)](input([real(length, 50 + row)]), s, null), { last: true });
          }
          oldRows = rowsOf(old); newRows = rowsOf(current);
          const rows = { old: oldRows, current: newRows };
          for (let n = 0; n < 2; n++)
            step(`real ${scheme} decode ${sequences} sequences`, p, rows, widthClass(sequences),
              (g, s, c) => g.decode[widthClass(sequences)](input(Array.from({ length: sequences }, (_, row) => real(1, 60 + 7 * n + row))), s, c));
          const window = Array.from({ length: sequences }, (_, row) => real(3, 70 + row));
          step(`real ${scheme} verify ${sequences}x3`, p, rows, widthClass(3 * sequences),
            (g, s, c) => g.verify[widthClass(3 * sequences)](input(window), s, c), { capture: true });
        } finally { disposeAll(...old, ...current, oldRows, newRows); }
      }
      currentGraph.dispose();
    }, 1_800_000);
});
