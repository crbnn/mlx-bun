// D2 Gemma 4 bit-identity gate. Each generated graph as scripts/gen-gemma4.ts
// now emits it (src/models/gemma4/generated/*.ts: an entire graph, composed from
// blocks.ts, reading every cache through its own read) against the class the
// generator emitted before D2 (gemma4-generated-reference/*.ts: it extends
// Gemma4Model, checks its caches against the per-layer KV literals it folded,
// attends their quantized storage itself on a match and runs the generic forward
// otherwise; frozen as the oracle and deleted with this test after the merge).
// Synthetic weights in each composition's shape, read from its committed fixture
// (tests/fixtures/gemma4-graphs): the layer types, the KV-sharing layers, the
// per-layer inputs (e4b), keys equal to values (12B, 26B), the routed experts
// (26B), a layer scalar on every layer and the artifact's per-layer kv_config;
// small widths and a 16-position sliding window, so prompts of 37 and 200 wrap
// the rings. Every hidden output, logit row, captured layer and cache tensor must
// agree to the bit under bf16 caches and under the composition's kv_config
// (converted after the first forward as the served path does, and before it as
// the real-weight parity test does), with MLX_BUN_NO_FUSED_SDPA unset (the
// tiled-causal kernel set the conversion composes; the oracle's tiled dispatch)
// and set to 1 (unfused): prefill 37 and 200, decodes, windows across the sliding
// boundary and the 256-position step, verify with layer capture (the verify
// phase operation, the mixed lane's capture and the deprecated hidden tap), a
// speculative rollback, fill spans of 1 to 4 on solo caches and target rows, two
// rows in one forward, two groups in one mixed forward, aligned batched rows,
// vision prefill, the phase operations against the oracle's deprecated forward,
// the compiled decode step (e4b whole-graph, 12B segmented; the 26B declares
// none) and the assistant drafter's donor read. Under kv_config the oracle's
// unrolled path must have run. Since E3 both graphs are built for the bf16
// composition and the test converts their plain caches itself, in the
// convert-after order the bf16-first legos keep inside their appends, reading
// with the kernel set the flag names; the flag itself now steers only the
// oracle's frozen dispatch (quantized-sdpa-reference.ts). No model, no download.
import { afterAll, describe, expect, test } from "bun:test";
import { join } from "node:path";
import { readdirSync } from "node:fs";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { loadModelConfig, parseQuantization, type ModelConfig } from "../../src/artifacts/config";
import type { Weights } from "../../src/artifacts/weights";
import type { BatchableCache, Cache } from "../../src/contracts/mlx/cache";
import type { Composition as CompositionRecord } from "../../src/contracts/portable/composition";
import type { MlxCompiledDecodeStep, MlxLayerCapture } from "../../src/contracts/mlx/graph";
import type { Gemma4Model } from "../../src/models/gemma4/model";
import type { TokenGroup } from "../../src/contracts/mlx/token-work";
import { createRuntimeConfig, runtimeConfig, withRuntimeConfig } from "../../src/runtime/config";
import { composedAffineKernels, type AffineKernelSet } from "../../src/state/affine-attention";
import { convertToAffine, transitionDue } from "../../src/state/bf16-first-kv";
import { targetCacheLayout } from "../../src/state/layout";
import { testComposition } from "./test-composition";
import { CompiledDecode } from "../../src/models/gemma4/compiled-decode";
import * as current12b from "../../src/models/gemma4/generated/gemma4-12b";
import * as current26b from "../../src/models/gemma4/generated/gemma4-26b";
import * as currentE4b from "../../src/models/gemma4/generated/gemma4-e4b";
import * as oracle12b from "./gemma4-generated-reference/gemma4-12b";
import * as oracle26b from "./gemma4-generated-reference/gemma4-26b";
import * as oracleE4b from "./gemma4-generated-reference/gemma4-e4b";

const HIDDEN = 256, INTER = 512, EXPERT_INTER = 128, VOCAB = 256, HEADS = 4, KV_HEADS = 2;
const HEAD_DIM = 256, GLOBAL_HEAD_DIM = 512, WINDOW = 16, PER_LAYER = 64, EXPERTS = 4, TOP_K = 2;
const PREFIX = "language_model.model";
const FIXTURES = join(import.meta.dir, "../fixtures/gemma4-graphs");

/** Every oracle class extends the generic graph. */
type OracleGraph = Gemma4Model;
/** What the gate calls on a generated graph (the 26B declares no compiled decode step). */
type CurrentGraph = Pick<InstanceType<typeof currentE4b.GeneratedGemma4>, "makeCache" | "forwardHidden" | "forwardEmbeddings" |
  "forwardHiddenMixed" | "prefillChunk" | "prefillTail" | "decode" | "verify" | "logits" | "draftTarget" | "hiddenTap"> &
  { compiledDecodeStep?(): MlxCompiledDecodeStep };
type Graph = OracleGraph | CurrentGraph;
interface Composition {
  readonly stem: string;
  readonly repo: string;
  readonly oracle: { readonly GeneratedGemma4: new (weights: Weights, config: ModelConfig, composition: CompositionRecord) => OracleGraph;
    readonly generatedForwardUses: number };
  readonly current: { readonly GeneratedGemma4: new (weights: Weights, config: ModelConfig, composition: CompositionRecord) => CurrentGraph };
}
const COMPOSITIONS: readonly Composition[] = [
  { stem: "gemma4-e4b", repo: "gemma-4-e4b-it-OptiQ-4bit", oracle: oracleE4b, current: currentE4b },
  { stem: "gemma4-12b", repo: "gemma-4-12B-it-OptiQ-4bit", oracle: oracle12b, current: current12b },
  { stem: "gemma4-26b", repo: "gemma-4-26B-A4B-it-OptiQ-4bit", oracle: oracle26b, current: current26b },
];

const owned: MlxArray[] = [];
afterAll(() => { for (const array of owned) array.dispose(); });
const keep = (array: MlxArray) => { owned.push(array); return array; };
const noise = (count: number, phase: number, scale: number, shift = 0) => Float32Array.from({ length: count },
  (_, i) => shift + Math.sin(i * 0.37 + phase) * scale + Math.cos(i * 0.11 + 2.3 * phase) * scale);
const tensor = (values: Float32Array, shape: number[], dtype = Dtype.bfloat16): MlxArray => {
  using wide = MlxArray.fromFloat32(values, shape);
  return keep(wide.astype(dtype));
};

/** The fixture's config: the composition's structure and its kv_config. */
async function fixtureConfig(repo: string): Promise<ModelConfig> {
  const snapshots = join(FIXTURES, `models--mlx-community--${repo}`, "snapshots");
  return loadModelConfig(join(snapshots, readdirSync(snapshots)[0]!));
}

/** Synthetic weights and config in the composition's shape. */
function artifact(real: ModelConfig, seed0: number): { weights: Weights; config: ModelConfig } {
  const t = real.text;
  const layers = t.numHiddenLayers, donors = layers - t.numKvSharedLayers, moe = t.enableMoeBlock;
  const perLayer = t.hiddenSizePerLayerInput > 0 ? PER_LAYER : 0;
  const globalKv = t.attentionKEqV ? (moe ? 2 : 1) : KV_HEADS;
  const tensors = new Map<string, MlxArray>();
  let seed = seed0;
  const affine = (path: string, rows: number, cols: number, scale = 0.06, experts = 0) => {
    using w = tensor(noise((experts || 1) * rows * cols, ++seed, scale), [(experts || 1) * rows, cols]);
    const q = ops.quantize(w, 64, 4);
    const shaped = (array: MlxArray) => experts ? ops.reshape(array, [experts, rows, array.shape[1]!]) : array;
    for (const [name, array] of [["weight", q.packed], ["scales", q.scales], ["biases", q.biases]] as const) {
      const out = shaped(array);
      if (out !== array) array.dispose();
      tensors.set(`${path}.${name}`, keep(out));
    }
  };
  const norm = (path: string, width: number) => tensors.set(path, tensor(noise(width, ++seed, 0.05, 1), [width]));
  affine(`${PREFIX}.embed_tokens`, VOCAB, HIDDEN, 0.05);
  norm(`${PREFIX}.norm.weight`, HIDDEN);
  if (perLayer) {
    affine(`${PREFIX}.embed_tokens_per_layer`, VOCAB, layers * perLayer, 0.05);
    affine(`${PREFIX}.per_layer_model_projection`, layers * perLayer, HIDDEN, 0.05);
    norm(`${PREFIX}.per_layer_projection_norm.weight`, perLayer);
  }
  for (let layer = 0; layer < layers; layer++) {
    const prefix = `${PREFIX}.layers.${layer}`, full = t.layerTypes[layer] === "full_attention";
    const headDim = full ? GLOBAL_HEAD_DIM : HEAD_DIM, kEqV = full && t.attentionKEqV;
    const kvHeads = kEqV ? globalKv : KV_HEADS, p = `${prefix}.self_attn`;
    affine(`${p}.q_proj`, HEADS * headDim, HIDDEN);
    affine(`${p}.o_proj`, HIDDEN, HEADS * headDim);
    norm(`${p}.q_norm.weight`, headDim);
    if (layer < donors) {
      affine(`${p}.k_proj`, kvHeads * headDim, HIDDEN);
      if (!kEqV) affine(`${p}.v_proj`, kvHeads * headDim, HIDDEN);
      norm(`${p}.k_norm.weight`, headDim);
    }
    affine(`${prefix}.mlp.gate_proj`, INTER, HIDDEN);
    affine(`${prefix}.mlp.up_proj`, INTER, HIDDEN);
    affine(`${prefix}.mlp.down_proj`, HIDDEN, INTER);
    for (const name of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"])
      norm(`${prefix}.${name}.weight`, HIDDEN);
    tensors.set(`${prefix}.layer_scalar`, tensor(Float32Array.of(0.75 + 0.01 * layer), [1]));
    if (perLayer) {
      affine(`${prefix}.per_layer_input_gate`, perLayer, HIDDEN);
      affine(`${prefix}.per_layer_projection`, HIDDEN, perLayer);
      norm(`${prefix}.post_per_layer_input_norm.weight`, HIDDEN);
    }
    if (moe) {
      affine(`${prefix}.router.proj`, EXPERTS, HIDDEN, 0.3);
      norm(`${prefix}.router.scale`, HIDDEN);
      tensors.set(`${prefix}.router.per_expert_scale`, tensor(noise(EXPERTS, ++seed, 0.2, 1), [EXPERTS]));
      affine(`${prefix}.experts.switch_glu.gate_proj`, EXPERT_INTER, HIDDEN, 0.06, EXPERTS);
      affine(`${prefix}.experts.switch_glu.up_proj`, EXPERT_INTER, HIDDEN, 0.06, EXPERTS);
      affine(`${prefix}.experts.switch_glu.down_proj`, HIDDEN, EXPERT_INTER, 0.06, EXPERTS);
      for (const name of ["post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2"])
        norm(`${prefix}.${name}.weight`, HIDDEN);
    }
  }
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
  } as unknown as Weights;
  const config = {
    ...real, modelDir: "", hasVisionSidecar: false,
    quantization: parseQuantization({ bits: 4, group_size: 64, mode: "affine" }),
    text: {
      ...t, hiddenSize: HIDDEN, numAttentionHeads: HEADS, numKeyValueHeads: KV_HEADS, headDim: HEAD_DIM,
      numGlobalKeyValueHeads: globalKv, globalHeadDim: GLOBAL_HEAD_DIM, intermediateSize: INTER, vocabSize: VOCAB,
      slidingWindow: WINDOW, hiddenSizePerLayerInput: perLayer, vocabSizePerLayerInput: VOCAB,
      numExperts: moe ? EXPERTS : 0, topKExperts: moe ? TOP_K : 0,
    },
  } as ModelConfig;
  return { weights, config };
}

// ── Comparison ──────────────────────────────────────────────────────────────

const results = new Map<string, { worst: number; tensors: number }>();
const record = (label: string, diff: number, tensors: number) => {
  const prior = results.get(label) ?? { worst: 0, tensors: 0 };
  results.set(label, { worst: Math.max(prior.worst, diff), tensors: prior.tensors + tensors });
};
afterAll(() => {
  for (const [label, { worst, tensors }] of results)
    console.log(`gemma4-generated-identity ${label}: maxDiff ${worst} over ${tensors} tensors`);
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

/** Every tensor of every layer's cache, with its signature and offsets. */
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
function outputDiff(label: string, model: CurrentGraph, oracle: OracleGraph, actual: MlxArray, expected: MlxArray): void {
  try {
    const hidden = maxDiff(actual, expected, `${label} hidden`);
    using la = model.logits.one(actual);
    using lb = oracle.logitsFromHidden(expected);
    const logits = maxDiff(la, lb, `${label} logits`);
    using wide = lb.astype(Dtype.float32);
    expect(wide.toFloat32().every(Number.isFinite), `${label} logits finite`).toBe(true);
    record(`${label} hidden+logits`, Math.max(hidden, logits), 2);
  } finally { actual.dispose(); expected.dispose(); }
}

function tapsDiff(label: string, actual: MlxArray[], expected: MlxArray[]): void {
  try {
    expect(actual.length, label).toBe(expected.length);
    expect(actual.length, label).toBeGreaterThan(0);
    let worst = 0;
    for (let i = 0; i < expected.length; i++) worst = Math.max(worst, maxDiff(actual[i]!, expected[i]!, `${label} tap ${i}`));
    record(`${label} captured layers`, worst, expected.length);
  } finally { for (const array of [...actual, ...expected]) array.dispose(); }
}

// ── Running both graphs ─────────────────────────────────────────────────────

const prompt = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 37 + salt * 11 + 5) % VOCAB);
const ids = (rows: number[][]) => ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);
const disposeAll = (...sets: Cache[][]) => { for (const set of sets) for (const cache of set) cache.dispose(); };

interface Pair { c: Composition; config: ModelConfig; old: OracleGraph; current: CurrentGraph }
const pairs = new Map<string, Pair>();
async function pair(c: Composition): Promise<Pair> {
  let found = pairs.get(c.stem);
  if (!found) {
    const real = await fixtureConfig(c.repo);
    const { weights, config } = artifact(real, 100 * (COMPOSITIONS.indexOf(c) + 1));
    // Both built for the bf16 composition, so each makeCache holds plain caches the test converts after the first forward.
    found = { c, config, old: new c.oracle.GeneratedGemma4(weights, config, testComposition()),
      current: new c.current.GeneratedGemma4(weights, config, testComposition()) };
    pairs.set(c.stem, found);
  }
  return found;
}

type Scheme = { name: string; kvConfig: boolean; flag: string | undefined };
const SCHEMES: Scheme[] = [
  { name: "bf16", kvConfig: false, flag: undefined },
  { name: "kv_config flag=unset", kvConfig: true, flag: undefined },
  { name: "kv_config flag=1", kvConfig: true, flag: "1" },
];
/** The flag as the oracle reads it (its frozen dispatch, `quantized-sdpa-reference.ts`); the current
 * graph reads through its caches, whose kernels the test composes from the same flag (`flagKernels`). */
const withFlag = <T>(flag: string | undefined, run: () => T): T =>
  withRuntimeConfig(createRuntimeConfig({ ...runtimeConfig().values, MLX_BUN_NO_FUSED_SDPA: flag }), run);
/** The affine read kernels the flag named before E3: unset, the tiled-causal set (`--fused-sdpa on`); 1, unfused. */
const flagKernels = (flag: string | undefined): AffineKernelSet => composedAffineKernels(flag === "1" ? "unfused" : "tiled-causal");

/** Convert after a forward, as the served path did before E3 (quantized_kv_start 0; the bf16-first legos keep
 *  this order inside their appends): every plain cache the kv_config lists converts, reading with the flag's kernels. */
function convert(p: Pair, scheme: Scheme, ...sets: Cache[][]): void {
  if (!scheme.kvConfig) return;
  for (const caches of sets) caches.forEach((cache, layer) => {
    const entry = p.config.kvQuant!.find(item => item.layerIdx === layer);
    if (entry && transitionDue(cache.offset, 0))
      caches[layer] = convertToAffine(cache as unknown as { offset: number; toQuantized(g: number, b: number, k: AffineKernelSet): Cache },
        entry.groupSize, entry.bits, 0, flagKernels(scheme.flag));
  });
}

/** Caches converted to the kv_config before any forward (the parity test's serve scenario), reading with the flag's kernels. */
function preconverted(p: Pair, model: Graph, flag: string | undefined): Cache[] {
  const caches = model.makeCache();
  for (let layer = 0; layer < caches.length; layer++) {
    const entry = p.config.kvQuant!.find(item => item.layerIdx === layer);
    if (entry) caches[layer] = (caches[layer] as unknown as { toQuantized(g: number, b: number, k: AffineKernelSet): Cache })
      .toQuantized(entry.groupSize, entry.bits, flagKernels(flag));
  }
  return caches;
}

/** The oracle's and the current graph's call on the same input. */
function both(label: string, p: Pair, oldCaches: Cache[], newCaches: Cache[],
  run: (model: Graph, caches: Cache[]) => MlxArray, current?: (caches: Cache[]) => MlxArray): void {
  const expected = run(p.old, oldCaches);
  const actual = current ? current(newCaches) : run(p.current, newCaches);
  outputDiff(label, p.current, p.old, actual, expected);
  stateDiff(label, newCaches, oldCaches);
}

const forwardHidden = (rows: number[][]) => (model: Graph, caches: Cache[]) => {
  using input = ids(rows);
  return model.forwardHidden(input, caches);
};
const operation = (rows: number[][], run: (input: MlxArray, caches: Cache[]) => MlxArray) => (caches: Cache[]) => {
  using input = ids(rows);
  return run(input, caches);
};

/** A capturing forward; returns the final hidden and every captured layer (copies). */
function captured(run: (capture: MlxLayerCapture) => MlxArray): { hidden: MlxArray; taps: MlxArray[] } {
  const taps: MlxArray[] = [];
  const hidden = run((_layer, h) => { taps.push(ops.contiguous(h)); });
  return { hidden, taps };
}
const mixedVerify = (model: Graph, caches: Cache[], rows: number[][]) => captured(capture => {
  using input = ids(rows);
  const group: TokenGroup = { ids: input, cache: caches, preserveTokenGeometry: true, captureLayer: capture };
  return model.forwardHiddenMixed([group])[0]!;
});

/** Target row layouts holding `sources` (one serial cache list per row), the fill and speculative paths' storage. */
function rowsOf(sources: Cache[][]): Cache[] {
  return sources[0]!.map((cache, layer) => {
    const layout = targetCacheLayout(cache) as unknown as BatchableCache;
    layout.mergeRows(sources.map(source => source[layer]!));
    return layout;
  });
}

/** The assistant drafter's donor read over each graph's caches: its positions,
 *  embedding and both donor reads attended with one query each. */
function donorReadDiff(label: string, p: Pair, oldCaches: Cache[], newCaches: Cache[]): void {
  const read = (model: Graph, caches: Cache[]) => {
    const rows = model.draftTarget(caches).assistantRows!;
    using token = ids([[17]]);
    const embedded = rows.embed(token);
    const donors = rows.readDonors();
    try {
      using qs = tensor(noise(HEADS * HEAD_DIM, 3, 0.5), [1, HEADS, 1, HEAD_DIM]);
      using qf = tensor(noise(HEADS * GLOBAL_HEAD_DIM, 4, 0.5), [1, HEADS, 1, GLOBAL_HEAD_DIM]);
      return { positions: donors.positions, outs: [embedded, donors.sliding.attend(qs, 1, WINDOW), donors.full.attend(qf, 1, null)] };
    } finally { donors.dispose(); }
  };
  const expected = read(p.old, oldCaches), actual = read(p.current, newCaches);
  expect(actual.positions, label).toEqual(expected.positions);
  let worst = 0;
  try { actual.outs.forEach((out, i) => { worst = Math.max(worst, maxDiff(out, expected.outs[i]!, `${label} donor ${i}`)); }); }
  finally { for (const out of [...actual.outs, ...expected.outs]) out.dispose(); }
  record(`${label} assistant donor read`, worst, actual.outs.length);
}

/** Image prompt: merged embeddings, the image-position mask (positions 5 to 24, across the sliding window) and the spliced ids. */
function visionPrefill(model: Graph, caches: Cache[], length: number, salt: number): MlxArray {
  using input = ids([prompt(length, salt)]);
  using wide = MlxArray.fromFloat32(noise(length * HIDDEN, salt, 0.04), [1, length, HIDDEN]);
  using embeddings = wide.astype(Dtype.bfloat16);
  using flags = MlxArray.fromInt32(Int32Array.from({ length }, (_, i) => i >= 5 && i < 25 ? 1 : 0), [length]);
  using image = flags.astype(Dtype.bool);
  return model.forwardEmbeddings(embeddings, caches, image, input, image);
}

/** Runs `body`, then checks the oracle's unrolled path ran under the kv_config and never under bf16. */
function expectOraclePath(p: Pair, scheme: Scheme, body: () => void): void {
  const before = p.c.oracle.generatedForwardUses;
  body();
  const used = p.c.oracle.generatedForwardUses - before;
  if (scheme.kvConfig) expect(used).toBeGreaterThan(0);
  else expect(used).toBe(0);
}

test("the generated graphs extend nothing", () => {
  for (const c of COMPOSITIONS) expect(Object.getPrototypeOf(c.current.GeneratedGemma4.prototype)).toBe(Object.prototype);
});

for (const c of COMPOSITIONS) {
  describe(`${c.stem}: the entire generated graph against the pre-D2 generated class`, () => {
    for (const scheme of SCHEMES) {
      const tag = `${c.stem} ${scheme.name}`;

      test(`${tag}: prefill 37, decodes, window, verify with capture, rollback, fill 1-4, phase operations, donor read`, async () => {
        const p = await pair(c);
        withFlag(scheme.flag, () => expectOraclePath(p, scheme, () => {
          const layers = p.config.text.numHiddenLayers;
          const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
          // The phase operations run their own cache set against the oracle's deprecated forward on a reference set.
          const refCaches = p.old.makeCache(), phaseCaches = p.current.makeCache();
          const phased = (label: string, rows: number[][], run: (input: MlxArray, caches: Cache[]) => MlxArray) =>
            both(`${label} (phase operation)`, p, refCaches, phaseCaches, forwardHidden(rows), operation(rows, run));
          try {
            const prefill = prompt(37, 1);
            both(`${tag} prefill37`, p, oldCaches, newCaches, forwardHidden([prefill]));
            phased(`${tag} prefillChunk37`, [prefill], (input, caches) => p.current.prefillChunk.gpu(input, caches) as MlxArray);
            convert(p, scheme, oldCaches, newCaches, refCaches, phaseCaches);
            stateDiff(`${tag} converted`, newCaches, oldCaches);
            stateDiff(`${tag} converted (phase operation)`, phaseCaches, refCaches);
            for (let step = 0; step < 3; step++) {
              const token = [[(200 + step) % VOCAB]];
              both(`${tag} decode`, p, oldCaches, newCaches, forwardHidden(token));
              phased(`${tag} decode`, token, (input, caches) => p.current.decode.one(input, caches) as MlxArray);
            }
            both(`${tag} window5`, p, oldCaches, newCaches, forwardHidden([prompt(5, 2)]));
            phased(`${tag} prefillTail5`, [prompt(5, 2)], (input, caches) => p.current.prefillTail(input, caches));
            donorReadDiff(tag, p, oldCaches, newCaches);
            {
              // The verify phase with its capture against the oracle's verify (its inherited generic forward).
              const rows = [prompt(4, 3)];
              using input = ids(rows);
              const expected = captured(capture => p.old.verify.four(input, oldCaches, capture) as MlxArray);
              const actual = captured(capture => p.current.verify.four(input, newCaches, capture) as MlxArray);
              expect(actual.taps.length).toBe(layers + 1);
              tapsDiff(`${tag} verify4`, actual.taps, expected.taps);
              outputDiff(`${tag} verify4`, p.current, p.old, actual.hidden, expected.hidden);
              stateDiff(`${tag} verify4`, newCaches, oldCaches);
              // The same window through the verify phase without a capture against the oracle's deprecated forward.
              phased(`${tag} verify4`, rows, (window, caches) => p.current.verify.four(window, caches, null) as MlxArray);
            }
            {
              // The mixed lane's capturing verify of 5, rolled back to 2 right after its write, then a decode.
              const rows = [prompt(5, 4)];
              const expected = mixedVerify(p.old, oldCaches, rows);
              const actual = mixedVerify(p.current, newCaches, rows);
              expect(actual.taps.length).toBe(layers + 1);
              tapsDiff(`${tag} spec verify5 (mixed capture)`, actual.taps, expected.taps);
              outputDiff(`${tag} spec verify5 (mixed capture)`, p.current, p.old, actual.hidden, expected.hidden);
            }
            for (const set of [oldCaches, newCaches]) for (const cache of set) cache.trim(3, true);
            stateDiff(`${tag} spec rollback keep 2`, newCaches, oldCaches);
            both(`${tag} decode after rollback`, p, oldCaches, newCaches, forwardHidden([[77]]));
            {
              // The deprecated hidden tap (DSpark's binding) on a window.
              const tap = (model: Graph, caches: Cache[]) => {
                const captures = new Map<number, MlxArray>();
                model.hiddenTap = { layers: new Set([0, 3, layers - 1, layers]), captured: captures };
                try {
                  using input = ids([prompt(3, 6)]);
                  return { hidden: model.forwardHidden(input, caches), taps: [...captures.entries()].sort(([a], [b]) => a - b).map(([, a]) => a) };
                } finally { model.hiddenTap = null; }
              };
              const expected = tap(p.old, oldCaches), actual = tap(p.current, newCaches);
              expect(actual.taps.length).toBe(4);
              tapsDiff(`${tag} hidden tap`, actual.taps, expected.taps);
              outputDiff(`${tag} hidden tap`, p.current, p.old, actual.hidden, expected.hidden);
              stateDiff(`${tag} hidden tap`, newCaches, oldCaches);
            }
            for (const span of [1, 2, 3, 4])
              both(`${tag} fill${span}`, p, oldCaches, newCaches, forwardHidden([prompt(span, 5 + span)]));
            both(`${tag} decode after fill`, p, oldCaches, newCaches, forwardHidden([[91]]));
            const oldRows = rowsOf([oldCaches]), newRows = rowsOf([newCaches]);
            try {
              stateDiff(`${tag} target rows`, newRows, oldRows);
              for (const span of [1, 2, 3, 4])
                both(`${tag} fill${span} (target rows)`, p, oldRows, newRows, forwardHidden([prompt(span, 9 + span)]));
              both(`${tag} decode (target rows)`, p, oldRows, newRows, forwardHidden([[33]]));
            } finally { disposeAll(oldRows, newRows); }
          } finally { disposeAll(oldCaches, newCaches, refCaches, phaseCaches); }
        }));
      }, 300_000);

      test(`${tag}: prefill 200, decodes, a window across the 256-position step`, async () => {
        const p = await pair(c);
        withFlag(scheme.flag, () => expectOraclePath(p, scheme, () => {
          const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
          try {
            both(`${tag} prefill200`, p, oldCaches, newCaches, forwardHidden([prompt(200, 6)]));
            convert(p, scheme, oldCaches, newCaches);
            for (let step = 0; step < 2; step++) both(`${tag} decode`, p, oldCaches, newCaches, forwardHidden([[11 + step]]));
            both(`${tag} window60 across 256`, p, oldCaches, newCaches, forwardHidden([prompt(60, 7)]));
            both(`${tag} decode`, p, oldCaches, newCaches, forwardHidden([[19]]));
          } finally { disposeAll(oldCaches, newCaches); }
        }));
      }, 300_000);

      test(`${tag}: prefill 12, a window of 8 across the sliding boundary, decodes`, async () => {
        const p = await pair(c);
        withFlag(scheme.flag, () => expectOraclePath(p, scheme, () => {
          const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
          try {
            both(`${tag} prefill12`, p, oldCaches, newCaches, forwardHidden([prompt(12, 8)]));
            convert(p, scheme, oldCaches, newCaches);
            both(`${tag} window8 across the sliding boundary`, p, oldCaches, newCaches, forwardHidden([prompt(8, 9)]));
            for (let step = 0; step < 2; step++) both(`${tag} decode`, p, oldCaches, newCaches, forwardHidden([[23 + step]]));
          } finally { disposeAll(oldCaches, newCaches); }
        }));
      }, 300_000);

      test(`${tag}: vision prefill with a bidirectional image mask, then decodes`, async () => {
        const p = await pair(c);
        withFlag(scheme.flag, () => {
          const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
          try {
            both(`${tag} vision prefill37`, p, oldCaches, newCaches, (model, caches) => visionPrefill(model, caches, 37, 21));
            convert(p, scheme, oldCaches, newCaches);
            for (let step = 0; step < 2; step++) both(`${tag} decode after vision`, p, oldCaches, newCaches, forwardHidden([[41 + step]]));
            both(`${tag} window3 after vision`, p, oldCaches, newCaches, forwardHidden([prompt(3, 22)]));
          } finally { disposeAll(oldCaches, newCaches); }
        });
      }, 300_000);

      test(`${tag}: two groups packed in one mixed forward`, async () => {
        const p = await pair(c);
        withFlag(scheme.flag, () => {
          const oldA = p.old.makeCache(), newA = p.current.makeCache(), oldB = p.old.makeCache(), newB = p.current.makeCache();
          try {
            both(`${tag} group A prefill`, p, oldA, newA, forwardHidden([prompt(13, 8)]));
            both(`${tag} group B prefill`, p, oldB, newB, forwardHidden([prompt(21, 9)]));
            convert(p, scheme, oldA, newA, oldB, newB);
            const run = (model: Graph, a: Cache[], b: Cache[]) => {
              using first = ids([[23]]);
              using second = ids([prompt(3, 10)]);
              return model.forwardHiddenMixed([{ ids: first, cache: a }, { ids: second, cache: b }]);
            };
            const expected = run(p.old, oldA, oldB), actual = run(p.current, newA, newB);
            outputDiff(`${tag} mixed group A (decode)`, p.current, p.old, actual[0]!, expected[0]!);
            outputDiff(`${tag} mixed group B (window)`, p.current, p.old, actual[1]!, expected[1]!);
            stateDiff(`${tag} mixed group A`, newA, oldA);
            stateDiff(`${tag} mixed group B`, newB, oldB);
          } finally { disposeAll(oldA, newA, oldB, newB); }
        });
      }, 300_000);

      test(`${tag}: aligned rows of one prompt: batched decode, verify with capture, rollback`, async () => {
        const p = await pair(c);
        withFlag(scheme.flag, () => {
          const layers = p.config.text.numHiddenLayers;
          const oldSources = [0, 1, 2].map(() => p.old.makeCache()), newSources = [0, 1, 2].map(() => p.current.makeCache());
          let oldRows: Cache[] = [], newRows: Cache[] = [];
          try {
            for (let row = 0; row < 3; row++)
              both(`${tag} aligned row ${row} prefill`, p, oldSources[row]!, newSources[row]!, forwardHidden([prompt(21, 3)]));
            convert(p, scheme, ...oldSources, ...newSources);
            oldRows = rowsOf(oldSources); newRows = rowsOf(newSources);
            for (let step = 0; step < 2; step++)
              both(`${tag} aligned batched decode`, p, oldRows, newRows, forwardHidden([[40 + step], [40 + step], [40 + step]]));
            for (const set of [oldRows, newRows]) for (const cache of set) cache.specRoundBegin?.();
            const rows = [prompt(3, 20), prompt(3, 20), prompt(3, 21)];
            {
              const expected = mixedVerify(p.old, oldRows, rows), actual = mixedVerify(p.current, newRows, rows);
              expect(actual.taps.length).toBe(layers + 1);
              tapsDiff(`${tag} aligned batched verify3`, actual.taps, expected.taps);
              outputDiff(`${tag} aligned batched verify3`, p.current, p.old, actual.hidden, expected.hidden);
            }
            for (const set of [oldRows, newRows]) for (const cache of set)
              (cache as unknown as { specRoundRollback(keep: readonly number[]): void }).specRoundRollback([3, 3, 1]);
            stateDiff(`${tag} aligned batched rollback keep 3/3/1`, newRows, oldRows);
            both(`${tag} aligned batched decode after rollback`, p, oldRows, newRows, forwardHidden([[70], [71], [72]]));
          } finally { disposeAll(...oldSources, ...newSources, oldRows, newRows); }
        });
      }, 300_000);

      if ("compiledDecodeStep" in c.current.GeneratedGemma4.prototype)
        test(`${tag}: compiled decode step, growing and ring phases`, async () => {
          const p = await pair(c);
          withFlag(scheme.flag, () => {
            const label = `${tag} compiled`;
            const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
            const before = CompiledDecode.stepsExecuted;
            try {
              both(`${label} prefill10`, p, oldCaches, newCaches, forwardHidden([prompt(10, 61)]));
              convert(p, scheme, oldCaches, newCaches);
              for (let step = 0; step < 9; step++) {
                using signed = ops.fromInt32([(100 + 7 * step) % VOCAB], [1]);
                using token = signed.astype(Dtype.uint32);
                const expected = p.old.compiledDecodeStep().step(token, oldCaches);
                const actual = p.current.compiledDecodeStep!().step(token, newCaches);
                ops.evalAll([expected.logits, ...expected.evalWith, actual.logits, ...actual.evalWith]);
                try { record(`${label} logits`, maxDiff(actual.logits, expected.logits, `${label} step ${step} logits`), 1); }
                finally { expected.logits.dispose(); actual.logits.dispose(); }
                stateDiff(`${label} step`, newCaches, oldCaches);
              }
              expect(CompiledDecode.stepsExecuted - before).toBe(18);
              expect(CompiledDecode.unexpectedRetraces).toBe(0);
              both(`${label} decode after compiled`, p, oldCaches, newCaches, forwardHidden([[17]]));
            } finally {
              disposeAll(oldCaches, newCaches);
              CompiledDecode.release(p.old);
              CompiledDecode.release(p.current);
            }
          });
        }, 300_000);
    }

    // The real-weight parity test's serve scenario: every kv_config cache converted
    // before the first forward, so the oracle's unrolled path runs the prefill too.
    for (const flag of [undefined, "1"])
      test(`${c.stem} kv_config converted before the first forward, flag=${flag ?? "unset"}: prefill 37, decodes, two rows in one forward`, async () => {
        const p = await pair(c);
        const tag = `${c.stem} preconverted flag=${flag ?? "unset"}`;
        withFlag(flag, () => expectOraclePath(p, { name: "", kvConfig: true, flag }, () => {
          const oldCaches = preconverted(p, p.old, flag), newCaches = preconverted(p, p.current, flag);
          const oldPair = preconverted(p, p.old, flag), newPair = preconverted(p, p.current, flag);
          try {
            const before = p.c.oracle.generatedForwardUses;
            both(`${tag} prefill37`, p, oldCaches, newCaches, forwardHidden([prompt(37, 31)]));
            expect(p.c.oracle.generatedForwardUses - before).toBe(1);
            for (let step = 0; step < 3; step++) both(`${tag} decode`, p, oldCaches, newCaches, forwardHidden([[50 + step]]));
            both(`${tag} window6`, p, oldCaches, newCaches, forwardHidden([prompt(6, 32)]));
            both(`${tag} two rows in one forward`, p, oldPair, newPair, forwardHidden([prompt(16, 33), prompt(16, 34)]));
          } finally { disposeAll(oldCaches, newCaches, oldPair, newPair); }
        }));
      }, 300_000);
  });
}
