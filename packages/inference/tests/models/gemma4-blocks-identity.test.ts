// D3 Gemma 4 bit-identity gate. The Gemma 4 graph whose attention blocks read
// through the cache for the phase their caller names (models/gemma4/model.ts)
// against the frozen pre-D3 graph (gemma4-reference.ts, with its compiled
// decode step and training declaration; all three deleted with this test after
// the merge), on small synthetic weights in two layouts: a dense one in the
// 12B's shape (sliding and full layers, keys equal to values on the full
// layers, the segmented compiled step) and one in the e4b's shape (per-layer
// inputs and three KV-sharing layers that attend their donors' reads, the
// whole-graph compiled step). The sliding window is 16 positions, so prompts
// of 37 and 200 wrap the rings. Every hidden output, logit row and cache
// tensor must agree to the bit for bf16 KV, affine 4- and 8-bit KV converted
// after the first forward by the batch preparation (state/layout) with MLX_BUN_NO_FUSED_SDPA unset
// (tiled-causal kernel set) and set to 1 (unfused set), and TurboQuant k8v3
// (the deferred set Gemma4Model.makeCache composes): decodes, windows across
// the sliding boundary and the 256-position step, verify windows with layer
// capture through the mixed verify group and the verify phase operation, fill spans
// of 1 to 4 on solo caches and target rows, vision prefill with a
// bidirectional image mask, two groups in one packed mixed forward, padded
// batched prefill with batched decode and per-row rollback, the delayed
// prefill-cohort layouts across their transition and the decode layouts made
// from the rows they convert, the compiled decode step in both forms, paged KV
// at one row and gathered bf16 paged rows at three, and the training forwards
// (full-sequence scoring, the padded mask wrapper, prefix-shared, segmented
// and gradient checkpointed). No model, no download.
import { afterAll, describe, expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { parseQuantization, type ModelConfig } from "../../src/artifacts/config";
import type { Weights } from "../../src/artifacts/weights";
import type { BatchableCache, Cache, PaddedPrefillCache, SharedKv } from "../../src/contracts/mlx/cache";
import type { TokenGroup } from "./frozen-token-groups";
import { phaseHidden } from "./phase-forward";
import { widthClass } from "../../src/contracts/mlx/graph";
import type { RuntimeModel } from "../../src/models/factory";
import { createRuntimeConfig, runtimeConfig, withRuntimeConfig } from "../../src/runtime/config";
import { trainForward } from "../../src/scoring/full-sequence";
import { trainForward as frozenTrainForward } from "./full-sequence-reference";
import { createKvMaintenance } from "../../src/state/layout";
import { composedAffineKernels } from "../../src/state/affine-attention";
import { settleAffine, settleTurbo } from "../state/convert-after-reference";
import { testComposition } from "./test-composition";
import { KVCache } from "../../src/state/kv";
import { prefillCacheLayout, targetCacheLayout } from "../../src/state/layout";
import { PagedKVCache } from "../../src/state/paged/cache";
import { PagedKvRows } from "../../src/state/paged/rows";
import { deferredTurboQuantKernels } from "../../src/state/turboquant-attention";
import type { TurboQuantKVCache } from "../../src/state/turboquant-kv";
import { CompiledDecode } from "../../src/models/gemma4/compiled-decode";
import { Gemma4Model } from "../../src/models/gemma4/model";
import { CompiledDecode as FrozenCompiledDecode } from "./gemma4-compiled-decode-reference";
import { Gemma4Model as FrozenGemma4Model } from "./gemma4-reference";

const HIDDEN = 256, INTER = 512, VOCAB = 256, HEADS = 4, KV_HEADS = 2, HEAD_DIM = 256, GLOBAL_HEAD_DIM = 512;
const WINDOW = 16, PER_LAYER = 64;
const S = "sliding_attention", F = "full_attention";

const owned: MlxArray[] = [];
afterAll(() => { for (const array of owned) array.dispose(); });
const keep = (array: MlxArray) => { owned.push(array); return array; };

const noise = (count: number, phase: number, scale: number, shift = 0) => Float32Array.from({ length: count },
  (_, i) => shift + Math.sin(i * 0.37 + phase) * scale + Math.cos(i * 0.11 + 2.3 * phase) * scale);
const tensor = (values: Float32Array, shape: number[], dtype = Dtype.bfloat16): MlxArray => {
  using wide = MlxArray.fromFloat32(values, shape);
  return keep(wide.astype(dtype));
};

type Kind = "dense" | "shared";
interface Layout { readonly types: readonly string[]; readonly sharedLayers: number; readonly kEqV: boolean; readonly perLayer: number }
const LAYOUTS: Record<Kind, Layout> = {
  // The 12B's shape: no sharing, no per-layer inputs, one global KV head with
  // keys equal to values on the full layers (the segmented compiled step).
  dense: { types: [S, S, F, S, S, F], sharedLayers: 0, kEqV: true, perLayer: 0 },
  // The e4b's shape: per-layer inputs and three sharers (layers 5, 6 and 7
  // attend donors 2, 4 and 2), the whole-graph compiled step.
  shared: { types: [S, S, F, S, S, F, S, F], sharedLayers: 3, kEqV: false, perLayer: PER_LAYER },
};

/** The synthetic artifact: tensors under the graph's names and the config it reads. */
function artifact(kind: Kind): { weights: Weights; config: ModelConfig } {
  const { types, sharedLayers, kEqV, perLayer } = LAYOUTS[kind];
  const tensors = new Map<string, MlxArray>();
  let seed = kind === "dense" ? 1 : 1000;
  const affine = (path: string, rows: number, cols: number, scale = 0.06) => {
    using w = tensor(noise(rows * cols, ++seed, scale), [rows, cols]);
    const q = ops.quantize(w, 64, 4);
    tensors.set(`${path}.weight`, keep(q.packed));
    tensors.set(`${path}.scales`, keep(q.scales));
    tensors.set(`${path}.biases`, keep(q.biases));
  };
  const norm = (path: string, width: number) => tensors.set(path, tensor(noise(width, ++seed, 0.05, 1), [width]));
  const layers = types.length, donors = layers - sharedLayers;
  affine("model.embed_tokens", VOCAB, HIDDEN, 0.05);
  norm("model.norm.weight", HIDDEN);
  if (perLayer) {
    affine("model.embed_tokens_per_layer", VOCAB, layers * perLayer, 0.05);
    affine("model.per_layer_model_projection", layers * perLayer, HIDDEN, 0.05);
    norm("model.per_layer_projection_norm.weight", perLayer);
  }
  for (let layer = 0; layer < layers; layer++) {
    const prefix = `model.layers.${layer}`, full = types[layer] === F;
    const headDim = full ? GLOBAL_HEAD_DIM : HEAD_DIM;
    const kvHeads = full && kEqV ? 1 : KV_HEADS;
    const p = `${prefix}.self_attn`;
    affine(`${p}.q_proj`, HEADS * headDim, HIDDEN);
    affine(`${p}.o_proj`, HIDDEN, HEADS * headDim);
    norm(`${p}.q_norm.weight`, headDim);
    if (layer < donors) {
      affine(`${p}.k_proj`, kvHeads * headDim, HIDDEN);
      if (!(full && kEqV)) affine(`${p}.v_proj`, kvHeads * headDim, HIDDEN);
      norm(`${p}.k_norm.weight`, headDim);
    }
    affine(`${prefix}.mlp.gate_proj`, INTER, HIDDEN);
    affine(`${prefix}.mlp.up_proj`, INTER, HIDDEN);
    affine(`${prefix}.mlp.down_proj`, HIDDEN, INTER);
    for (const name of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"])
      norm(`${prefix}.${name}.weight`, HIDDEN);
    tensors.set(`${prefix}.layer_scalar`, tensor(Float32Array.of(0.75 + 0.05 * layer), [1]));
    if (perLayer) {
      affine(`${prefix}.per_layer_input_gate`, perLayer, HIDDEN);
      affine(`${prefix}.per_layer_projection`, HIDDEN, perLayer);
      norm(`${prefix}.post_per_layer_input_norm.weight`, HIDDEN);
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
    modelDir: "", modelType: "gemma4_text", architectures: [], dtype: "bfloat16", kvQuant: null,
    hasVisionSidecar: false, eosTokenIds: [], raw: {},
    quantization: parseQuantization({ bits: 4, group_size: 64, mode: "affine" }),
    text: {
      hiddenSize: HIDDEN, numHiddenLayers: layers, numAttentionHeads: HEADS, numKeyValueHeads: KV_HEADS,
      headDim: HEAD_DIM, numGlobalKeyValueHeads: kEqV ? 1 : KV_HEADS, globalHeadDim: GLOBAL_HEAD_DIM,
      attentionKEqV: kEqV, intermediateSize: INTER, rmsNormEps: 1e-6, vocabSize: VOCAB, slidingWindow: WINDOW,
      layerTypes: [...types], hiddenSizePerLayerInput: perLayer, vocabSizePerLayerInput: VOCAB,
      numKvSharedLayers: sharedLayers, enableMoeBlock: false, numExperts: 0, topKExperts: 0,
      ropeParameters: {
        sliding_attention: { ropeTheta: 10_000, ropeType: "default", partialRotaryFactor: 1, factor: 1 },
        full_attention: { ropeTheta: 1_000_000, ropeType: "proportional", partialRotaryFactor: 0.25, factor: 1 },
      },
      finalLogitSoftcapping: 30, tieWordEmbeddings: true,
    },
  } as unknown as ModelConfig;
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
    console.log(`gemma4-identity ${label}: maxDiff ${worst} over ${tensors} tensors`);
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
function outputDiff(label: string, model: Gemma4Model, actual: MlxArray, expected: MlxArray): void {
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

type Graph = Gemma4Model | FrozenGemma4Model;
const prompt = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 37 + salt * 11 + 5) % VOCAB);
const ids = (rows: number[][]) => ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);

type Scheme = { name: string; kvBits: number | null; flag: string | undefined; turbo?: { kBits: number; vBits: number } };
const SCHEMES: Scheme[] = [
  { name: "bf16", kvBits: null, flag: undefined },
  { name: "affine4 flag=unset", kvBits: 4, flag: undefined },
  { name: "affine4 flag=1", kvBits: 4, flag: "1" },
  { name: "affine8 flag=unset", kvBits: 8, flag: undefined },
  { name: "affine8 flag=1", kvBits: 8, flag: "1" },
  // The deferred TurboQuant read Gemma4Model.makeCache composes, which the
  // conversion hands the converted full-attention caches.
  { name: "turbo k8v3", kvBits: null, flag: undefined, turbo: { kBits: 8, vBits: 3 } },
];

/** The frozen and current graphs over one synthetic artifact. */
interface Pair { kind: Kind; old: FrozenGemma4Model; current: Gemma4Model }
const pairs = new Map<Kind, Pair>();
function pair(kind: Kind): Pair {
  let found = pairs.get(kind);
  if (!found) {
    const { weights, config } = artifact(kind);
    found = { kind, old: new FrozenGemma4Model(weights, config), current: new Gemma4Model(weights, config, testComposition()) };
    pairs.set(kind, found);
  }
  return found;
}

function withScheme<T>(scheme: Scheme, run: () => T): T {
  return withRuntimeConfig(createRuntimeConfig({ ...runtimeConfig().values, MLX_BUN_NO_FUSED_SDPA: scheme.flag }), run);
}

/** Convert after a forward, as the served path did before E3 (quantized_kv_start
 * 0): the convert-after reference, whose affine storage reads with the kernel set
 * the scheme's flag named (unset: tiled-causal, `--fused-sdpa on`; 1: unfused). */
function convert(scheme: Scheme, ...sets: Cache[][]): void {
  if (scheme.kvBits === null && !scheme.turbo) return;
  for (const caches of sets) {
    if (scheme.turbo) settleTurbo(caches, scheme.turbo.kBits, scheme.turbo.vBits, 0);
    else settleAffine(caches, scheme.kvBits!, 0, 64, composedAffineKernels(flagKernels(scheme.flag)));
  }
}
/** The affine read kernels the flag named before E3. */
const flagKernels = (flag: string | undefined) => flag === "1" ? "unfused" as const : "tiled-causal" as const;

/** The old graph's and the current graph's call on the same input. */
function both(label: string, p: Pair, oldCaches: Cache[], newCaches: Cache[], run: (model: Graph, caches: Cache[]) => MlxArray): void {
  const expected = run(p.old, oldCaches);
  const actual = run(p.current, newCaches);
  outputDiff(label, p.current, actual, expected);
  stateDiff(label, newCaches, oldCaches);
}

/** The old graph's deprecated forward, which reads the phase from the input's shape, against the
 *  current graph's named phase for the same tokens (`phaseHidden`: one per sequence a decode, more a tail). */
const forwardHidden = (rows: number[][]) => (model: Graph, caches: Cache[]) => {
  using input = ids(rows);
  return model instanceof FrozenGemma4Model ? model.forwardHidden(input, caches) : phaseHidden(model, input, caches);
};

/** A capturing verify forward through a mixed forward (the old graph's
 *  geometry-preserving token group, the current graph's verify group); returns
 *  the final hidden and every captured layer (copies). */
function mixedVerify(model: Graph, caches: Cache[], rows: number[][]): { hidden: MlxArray; taps: MlxArray[] } {
  using input = ids(rows);
  const taps: MlxArray[] = [];
  const capture = (_layer: number, hidden: MlxArray) => { taps.push(ops.contiguous(hidden)); };
  if (model instanceof FrozenGemma4Model) {
    const group: TokenGroup = { ids: input, cache: caches, preserveTokenGeometry: true, captureLayer: capture };
    const [hidden] = model.forwardHiddenMixed([group]);
    return { hidden: hidden!, taps };
  }
  const width = widthClass(rows.length * rows[0]!.length);
  const [hidden] = model.mixed([{ phase: "verify", width, ids: input, state: caches, capture }], width);
  return { hidden: hidden!, taps };
}

function tapsDiff(label: string, layers: number, actual: MlxArray[], expected: MlxArray[]): void {
  try {
    expect(actual.length, label).toBe(layers + 1);
    expect(expected.length, label).toBe(layers + 1);
    let worst = 0;
    for (let i = 0; i < expected.length; i++) worst = Math.max(worst, maxDiff(actual[i]!, expected[i]!, `${label} layer ${i}`));
    record(`${label} captured layers`, worst, expected.length);
  } finally { for (const array of [...actual, ...expected]) array.dispose(); }
}

/** Target row layouts holding `sources` (one serial cache list per row), the
 *  fill and speculative paths' storage. */
function rowsOf(sources: Cache[][]): Cache[] {
  return sources[0]!.map((cache, layer) => {
    const layout = targetCacheLayout(cache) as unknown as BatchableCache;
    layout.mergeRows(sources.map(source => source[layer]!));
    return layout;
  });
}

/** Image prompt: merged embeddings, the image-position mask (positions 5 to
 *  24, across the sliding window) and the spliced ids. */
function visionPrefill(model: Graph, caches: Cache[], length: number, salt: number): MlxArray {
  const tokens = prompt(length, salt);
  using input = ids([tokens]);
  using wide = MlxArray.fromFloat32(noise(length * HIDDEN, salt, 0.04), [1, length, HIDDEN]);
  using embeddings = wide.astype(Dtype.bfloat16);
  using flags = MlxArray.fromInt32(Int32Array.from({ length }, (_, i) => i >= 5 && i < 25 ? 1 : 0), [length]);
  using image = flags.astype(Dtype.bool);
  return model instanceof FrozenGemma4Model ? model.forwardEmbeddings(embeddings, caches, image, input, image)
    : model.prefillEmbeddings(embeddings, caches, image, input, image);
}

const disposeAll = (...sets: Cache[][]) => { for (const set of sets) for (const cache of set) cache.dispose(); };

for (const kind of ["dense", "shared"] as const) {
  const layers = LAYOUTS[kind].types.length;
  describe(`Gemma 4 blocks read through the cache: ${kind} layout`, () => {
    for (const scheme of SCHEMES) {
      const tag = `${kind} ${scheme.name}`;

      test(`${tag}: prefill 37, decodes, window, verify with capture, rollback, fill 1-4, phase operations`, () => withScheme(scheme, () => {
        const p = pair(kind);
        const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
        // The phase operations run their own cache set against the old graph's
        // deprecated entry point on a reference set.
        const refCaches = p.old.makeCache(), phaseCaches = p.current.makeCache();
        const phased = (label: string, rows: number[][], operation: (input: MlxArray, caches: Cache[]) => MlxArray) => {
          using input = ids(rows);
          const expected = p.old.forwardHidden(input, refCaches);
          const actual = operation(input, phaseCaches);
          outputDiff(`${label} (phase operation)`, p.current, actual, expected);
          stateDiff(`${label} (phase operation)`, phaseCaches, refCaches);
        };
        try {
          const prefill = prompt(37, 1);
          both(`${tag} prefill37`, p, oldCaches, newCaches, forwardHidden([prefill]));
          phased(`${tag} prefillChunk37`, [prefill], (input, caches) => p.current.prefillChunk.gpu(input, caches, null) as MlxArray);
          convert(scheme, oldCaches, newCaches, refCaches, phaseCaches);
          stateDiff(`${tag} converted`, newCaches, oldCaches);
          stateDiff(`${tag} converted (phase operation)`, phaseCaches, refCaches);
          if (scheme.turbo) expect((newCaches[2] as TurboQuantKVCache).turboQuantKernels).toBe(deferredTurboQuantKernels);

          for (let step = 0; step < 3; step++) {
            const token = [[(200 + step) % VOCAB]];
            both(`${tag} decode`, p, oldCaches, newCaches, forwardHidden(token));
            phased(`${tag} decode`, token, (input, caches) => p.current.decode.one(input, caches, null));
          }

          both(`${tag} window5`, p, oldCaches, newCaches, forwardHidden([prompt(5, 2)]));
          phased(`${tag} prefillTail5`, [prompt(5, 2)], (input, caches) => p.current.prefillTail.fiveToEight(input, caches, null) as MlxArray);

          {
            const rows = [prompt(4, 3)];
            const expected = mixedVerify(p.old, oldCaches, rows);
            const actual = mixedVerify(p.current, newCaches, rows);
            tapsDiff(`${tag} verify4`, layers, actual.taps, expected.taps);
            outputDiff(`${tag} verify4`, p.current, actual.hidden, expected.hidden);
            stateDiff(`${tag} verify4`, newCaches, oldCaches);
            // `verify` against the same capturing forward on the reference set.
            const reference = mixedVerify(p.old, refCaches, rows);
            using input = ids(rows);
            const taps: MlxArray[] = [];
            const hidden = p.current.verify.four(input, phaseCaches, (_layer, h) => { taps.push(ops.contiguous(h)); }) as MlxArray;
            tapsDiff(`${tag} verify4 (phase operation)`, layers, taps, reference.taps);
            outputDiff(`${tag} verify4 (phase operation)`, p.current, hidden, reference.hidden);
            stateDiff(`${tag} verify4 (phase operation)`, phaseCaches, refCaches);
          }

          // A speculative verify of 5 rolled back to 2 right after its write
          // (the rings' concatenated tail is sliced off), then a decode.
          {
            const rows = [prompt(5, 4)];
            const expected = mixedVerify(p.old, oldCaches, rows);
            const actual = mixedVerify(p.current, newCaches, rows);
            tapsDiff(`${tag} spec verify5`, layers, actual.taps, expected.taps);
            outputDiff(`${tag} spec verify5`, p.current, actual.hidden, expected.hidden);
          }
          for (const set of [oldCaches, newCaches]) for (const cache of set) cache.trim(3, true);
          stateDiff(`${tag} spec rollback keep 2`, newCaches, oldCaches);
          both(`${tag} decode after rollback`, p, oldCaches, newCaches, forwardHidden([[77]]));

          // Fill spans: known tokens appended through the ordinary forward.
          for (const span of [1, 2, 3, 4])
            both(`${tag} fill${span}`, p, oldCaches, newCaches, forwardHidden([prompt(span, 5 + span)]));
          both(`${tag} decode after fill`, p, oldCaches, newCaches, forwardHidden([[91]]));

          // The fill path's storage: target row layouts of one row.
          const oldRows = rowsOf([oldCaches]), newRows = rowsOf([newCaches]);
          try {
            stateDiff(`${tag} target rows`, newRows, oldRows);
            for (const span of [1, 2, 3, 4])
              both(`${tag} fill${span} (target rows)`, p, oldRows, newRows, forwardHidden([prompt(span, 9 + span)]));
            both(`${tag} decode (target rows)`, p, oldRows, newRows, forwardHidden([[33]]));
          } finally { disposeAll(oldRows, newRows); }
        } finally { disposeAll(oldCaches, newCaches, refCaches, phaseCaches); }
      }), 120_000);

      test(`${tag}: prefill 200, decodes, a window across the 256-position step`, () => withScheme(scheme, () => {
        const p = pair(kind);
        const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
        try {
          both(`${tag} prefill200`, p, oldCaches, newCaches, forwardHidden([prompt(200, 6)]));
          convert(scheme, oldCaches, newCaches);
          for (let step = 0; step < 2; step++)
            both(`${tag} decode`, p, oldCaches, newCaches, forwardHidden([[11 + step]]));
          both(`${tag} window60 across 256`, p, oldCaches, newCaches, forwardHidden([prompt(60, 7)]));
          both(`${tag} decode`, p, oldCaches, newCaches, forwardHidden([[19]]));
        } finally { disposeAll(oldCaches, newCaches); }
      }), 120_000);

      test(`${tag}: vision prefill with a bidirectional image mask, then decodes`, () => withScheme(scheme, () => {
        const p = pair(kind);
        const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
        try {
          both(`${tag} vision prefill37`, p, oldCaches, newCaches, (model, caches) => visionPrefill(model, caches, 37, 21));
          convert(scheme, oldCaches, newCaches);
          for (let step = 0; step < 2; step++)
            both(`${tag} decode after vision`, p, oldCaches, newCaches, forwardHidden([[41 + step]]));
          both(`${tag} window3 after vision`, p, oldCaches, newCaches, forwardHidden([prompt(3, 22)]));
        } finally { disposeAll(oldCaches, newCaches); }
      }), 120_000);

      test(`${tag}: two groups packed in one mixed forward`, () => withScheme(scheme, () => {
        const p = pair(kind);
        const oldA = p.old.makeCache(), newA = p.current.makeCache(), oldB = p.old.makeCache(), newB = p.current.makeCache();
        try {
          both(`${tag} group A prefill`, p, oldA, newA, forwardHidden([prompt(13, 8)]));
          both(`${tag} group B prefill`, p, oldB, newB, forwardHidden([prompt(21, 9)]));
          convert(scheme, oldA, newA, oldB, newB);
          const run = (model: Graph, a: Cache[], b: Cache[]) => {
            using first = ids([[23]]);
            using second = ids([prompt(3, 10)]);
            return model instanceof FrozenGemma4Model ? model.forwardHiddenMixed([{ ids: first, cache: a }, { ids: second, cache: b }])
              : model.mixed([{ phase: "decode", width: "one", ids: first, state: a, capture: null },
                { phase: "window", width: "twoToThree", ids: second, state: b, capture: null }], "four");
          };
          const expected = run(p.old, oldA, oldB);
          const actual = run(p.current, newA, newB);
          outputDiff(`${tag} mixed group A (decode)`, p.current, actual[0]!, expected[0]!);
          outputDiff(`${tag} mixed group B (window)`, p.current, actual[1]!, expected[1]!);
          stateDiff(`${tag} mixed group A`, newA, oldA);
          stateDiff(`${tag} mixed group B`, newB, oldB);
        } finally { disposeAll(oldA, newA, oldB, newB); }
      }), 120_000);

      // A speculative cohort of one prompt: the rows stay aligned (no padding),
      // so the row layouts' own masks are plain causal.
      test(`${tag}: aligned rows of one prompt: batched decode, verify with capture, rollback`, () => withScheme(scheme, () => {
        const p = pair(kind);
        const oldSources = [0, 1, 2].map(() => p.old.makeCache()), newSources = [0, 1, 2].map(() => p.current.makeCache());
        let oldRows: Cache[] = [], newRows: Cache[] = [];
        try {
          for (let row = 0; row < 3; row++)
            both(`${tag} aligned row ${row} prefill`, p, oldSources[row]!, newSources[row]!, forwardHidden([prompt(21, 3)]));
          convert(scheme, ...oldSources, ...newSources);
          oldRows = rowsOf(oldSources); newRows = rowsOf(newSources);
          for (let step = 0; step < 2; step++)
            both(`${tag} aligned batched decode`, p, oldRows, newRows, forwardHidden([[40 + step], [40 + step], [40 + step]]));
          for (const set of [oldRows, newRows]) for (const cache of set) cache.specRoundBegin?.();
          {
            const rows = [prompt(3, 20), prompt(3, 20), prompt(3, 21)];
            const expected = mixedVerify(p.old, oldRows, rows);
            const actual = mixedVerify(p.current, newRows, rows);
            tapsDiff(`${tag} aligned batched verify3`, layers, actual.taps, expected.taps);
            outputDiff(`${tag} aligned batched verify3`, p.current, actual.hidden, expected.hidden);
          }
          for (const set of [oldRows, newRows]) for (const cache of set)
            (cache as unknown as { specRoundRollback(keep: readonly number[]): void }).specRoundRollback([3, 3, 1]);
          stateDiff(`${tag} aligned batched rollback keep 3/3/1`, newRows, oldRows);
          both(`${tag} aligned batched decode after rollback`, p, oldRows, newRows, forwardHidden([[70], [71], [72]]));
        } finally { disposeAll(...oldSources, ...newSources, oldRows, newRows); }
      }), 120_000);

      test(`${tag}: padded batched prefill of three unequal rows, batched decode, batched rollback`, () => withScheme(scheme, () => {
        const p = pair(kind);
        const lengths = [9, 20, 6];
        const oldSources = lengths.map(() => p.old.makeCache()), newSources = lengths.map(() => p.current.makeCache());
        let oldRows: Cache[] = [], newRows: Cache[] = [];
        try {
          lengths.forEach((length, row) =>
            both(`${tag} row ${row} prefill`, p, oldSources[row]!, newSources[row]!, forwardHidden([prompt(length, 11 + row)])));
          convert(scheme, ...oldSources, ...newSources);
          oldRows = rowsOf(oldSources); newRows = rowsOf(newSources);
          stateDiff(`${tag} merged rows`, newRows, oldRows);

          // Right-padded window: real lengths 5, 3 and 7 in a width of 7.
          const real = [5, 3, 7], width = 7;
          for (const set of [oldRows, newRows]) for (const cache of set)
            (cache as unknown as PaddedPrefillCache).preparePrefill({ lengths: real, rightPadding: real.map(n => width - n) });
          const window = real.map((n, row) => [...prompt(n, 14 + row), ...Array(width - n).fill(0)]);
          both(`${tag} padded prefill 5/3/7`, p, oldRows, newRows, forwardHidden(window));
          for (const set of [oldRows, newRows]) for (const cache of set) (cache as unknown as PaddedPrefillCache).finalizePrefill();
          stateDiff(`${tag} padded prefill finalized`, newRows, oldRows);

          for (let step = 0; step < 2; step++)
            both(`${tag} batched decode`, p, oldRows, newRows, forwardHidden([[40 + step], [50 + step], [60 + step]]));

          for (const set of [oldRows, newRows]) for (const cache of set) cache.specRoundBegin?.();
          {
            const rows = [prompt(4, 20), prompt(4, 21), prompt(4, 22)];
            const expected = mixedVerify(p.old, oldRows, rows);
            const actual = mixedVerify(p.current, newRows, rows);
            tapsDiff(`${tag} batched verify4`, layers, actual.taps, expected.taps);
            outputDiff(`${tag} batched verify4`, p.current, actual.hidden, expected.hidden);
          }
          for (const set of [oldRows, newRows]) for (const cache of set)
            (cache as unknown as { specRoundRollback(keep: readonly number[]): void }).specRoundRollback([1, 3, 2]);
          stateDiff(`${tag} batched rollback keep 1/3/2`, newRows, oldRows);
          both(`${tag} batched decode after rollback`, p, oldRows, newRows, forwardHidden([[70], [71], [72]]));
          both(`${tag} batched fill (one position per row)`, p, oldRows, newRows, forwardHidden([[80], [81], [82]]));
        } finally { disposeAll(...oldSources, ...newSources, oldRows, newRows); }
      }), 120_000);
    }

    test(`${kind} bf16: padded batched prefill of three unequal rows from empty layouts`, () => {
      const p = pair(kind);
      const layouts = (model: Graph) => {
        const serial = model.makeCache();
        try { return serial.map(cache => targetCacheLayout(cache) as unknown as Cache); }
        finally { disposeAll(serial); }
      };
      const oldRows = layouts(p.old), newRows = layouts(p.current);
      try {
        const real = [6, 2, 4], width = 6;
        for (const set of [oldRows, newRows]) for (const cache of set)
          (cache as unknown as PaddedPrefillCache).preparePrefill({ lengths: real, rightPadding: real.map(n => width - n) });
        const window = real.map((n, row) => [...prompt(n, 30 + row), ...Array(width - n).fill(0)]);
        both(`${kind} bf16 empty padded prefill 6/2/4`, p, oldRows, newRows, forwardHidden(window));
        for (const set of [oldRows, newRows]) for (const cache of set) (cache as unknown as PaddedPrefillCache).finalizePrefill();
        for (let step = 0; step < 2; step++)
          both(`${kind} bf16 empty batched decode`, p, oldRows, newRows, forwardHidden([[3 + step], [5 + step], [7 + step]]));
      } finally { disposeAll(oldRows, newRows); }
    }, 120_000);

    // The served prefill cohort: the graph's caches wrapped by the scheme's
    // maintenance (bf16 until 8 positions, then quantized), each layer's empty
    // prefill layout, padded prefill and decodes across the transition.
    const delayed: { name: string; options: Parameters<typeof createKvMaintenance>[0] }[] = [
      { name: "delayed affine4", options: { kind: "affine", bits: 4, groupSize: 64, quantizedKvStart: 8, kernels: "tiled-causal" } },
      { name: "delayed affine8", options: { kind: "affine", bits: 8, groupSize: 64, quantizedKvStart: 8, kernels: "tiled-causal" } },
      { name: "delayed turbo k8v3", options: { kind: "turbo", kBits: 8, vBits: 3, quantizedKvStart: 8 } },
    ];
    for (const { name, options } of delayed)
      test(`${kind} ${name}: padded prefill cohort from empty layouts, decodes across the transition, verify, rollback`, () => {
        const p = pair(kind), tag = `${kind} ${name}`;
        const maintain = createKvMaintenance(options);
        const layouts = (model: Graph) => {
          const serial = model.makeCache();
          maintain.preparePrefill!(serial);
          try { return serial.map(cache => prefillCacheLayout(cache) as unknown as Cache); }
          finally { disposeAll(serial); }
        };
        const oldRows = layouts(p.old), newRows = layouts(p.current);
        try {
          const real = [6, 2, 4], width = 6;
          for (const set of [oldRows, newRows]) for (const cache of set)
            (cache as unknown as PaddedPrefillCache).preparePrefill({ lengths: real, rightPadding: real.map(n => width - n) });
          const window = real.map((n, row) => [...prompt(n, 52 + row), ...Array(width - n).fill(0)]);
          both(`${tag} padded prefill 6/2/4`, p, oldRows, newRows, forwardHidden(window));
          for (const set of [oldRows, newRows]) for (const cache of set) (cache as unknown as PaddedPrefillCache).finalizePrefill();
          for (let step = 0; step < 6; step++)
            both(`${tag} batched decode`, p, oldRows, newRows, forwardHidden([[3 + step], [5 + step], [7 + step]]));
          for (const set of [oldRows, newRows]) for (const cache of set) cache.specRoundBegin?.();
          {
            const verify = [prompt(4, 55), prompt(4, 56), prompt(4, 57)];
            const expected = mixedVerify(p.old, oldRows, verify);
            const actual = mixedVerify(p.current, newRows, verify);
            tapsDiff(`${tag} batched verify4`, layers, actual.taps, expected.taps);
            outputDiff(`${tag} batched verify4`, p.current, actual.hidden, expected.hidden);
          }
          for (const set of [oldRows, newRows]) for (const cache of set)
            (cache as unknown as { specRoundRollback(keep: readonly number[]): void }).specRoundRollback([1, 3, 2]);
          stateDiff(`${tag} batched rollback keep 1/3/2`, newRows, oldRows);
          both(`${tag} batched decode after rollback`, p, oldRows, newRows, forwardHidden([[70], [71], [72]]));
        } finally { disposeAll(oldRows, newRows); }
      }, 120_000);

    // The compiled decode step: the current step (layers reading their trace
    // stand-ins through appendDecode, closures keyed by slot signatures alone)
    // against the frozen step (masks built per layer, closures keyed on the
    // flag), through the growing phase into the ring phase.
    for (const scheme of SCHEMES.filter(item => !item.turbo))
      test(`${kind} ${scheme.name}: compiled decode step, growing and ring phases`, () => withScheme(scheme, () => {
        const p = pair(kind), tag = `${kind} ${scheme.name} compiled`;
        const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
        const oldSteps = FrozenCompiledDecode.stepsExecuted, newSteps = CompiledDecode.stepsExecuted;
        try {
          both(`${tag} prefill10`, p, oldCaches, newCaches, forwardHidden([prompt(10, 61)]));
          convert(scheme, oldCaches, newCaches);
          for (let step = 0; step < 9; step++) {
            using signed = ops.fromInt32([(100 + 7 * step) % VOCAB], [1]);
            using token = signed.astype(Dtype.uint32);
            const expected = FrozenCompiledDecode.for(p.old).step(token, oldCaches);
            const actual = CompiledDecode.for(p.current).step(token, newCaches);
            ops.evalAll([expected.logits, ...expected.evalWith, actual.logits, ...actual.evalWith]);
            try { record(`${tag} logits`, maxDiff(actual.logits, expected.logits, `${tag} step ${step} logits`), 1); }
            finally { expected.logits.dispose(); actual.logits.dispose(); }
            stateDiff(`${tag} step`, newCaches, oldCaches);
          }
          expect(FrozenCompiledDecode.stepsExecuted - oldSteps).toBe(9);
          expect(CompiledDecode.stepsExecuted - newSteps).toBe(9);
          expect(CompiledDecode.unexpectedRetraces).toBe(0);
          // The ordinary decode continues from the compiled steps' state.
          both(`${tag} decode after compiled`, p, oldCaches, newCaches, forwardHidden([[17]]));
        } finally {
          disposeAll(oldCaches, newCaches);
          FrozenCompiledDecode.release(p.old);
          CompiledDecode.release(p.current);
        }
      }), 120_000);

    // Paged KV, one row: the full-attention caches replaced by block pools
    // (gathered reads), as the paged request policy does.
    for (const quantization of [undefined, { bits: 4 as const, groupSize: 64 }, { bits: 8 as const, groupSize: 64 }])
      test(`${kind} paged ${quantization ? `affine${quantization.bits}` : "bf16"}: prefill 37, decodes, window, verify`, () => {
        const p = pair(kind), tag = `${kind} paged ${quantization ? `affine${quantization.bits}` : "bf16"}`;
        const paged = (model: Graph) => model.makeCache().map(cache => {
          if (!(cache instanceof KVCache)) return cache;
          cache.dispose();
          return new PagedKVCache(512, PagedKVCache.DEFAULT_BLOCK_SIZE, false, quantization);
        });
        const oldCaches = paged(p.old), newCaches = paged(p.current);
        try {
          both(`${tag} prefill37`, p, oldCaches, newCaches, forwardHidden([prompt(37, 71)]));
          for (let step = 0; step < 2; step++)
            both(`${tag} decode`, p, oldCaches, newCaches, forwardHidden([[21 + step]]));
          both(`${tag} window5`, p, oldCaches, newCaches, forwardHidden([prompt(5, 72)]));
          const rows = [prompt(4, 73)];
          const expected = mixedVerify(p.old, oldCaches, rows);
          const actual = mixedVerify(p.current, newCaches, rows);
          tapsDiff(`${tag} verify4`, layers, actual.taps, expected.taps);
          outputDiff(`${tag} verify4`, p.current, actual.hidden, expected.hidden);
        } finally { disposeAll(oldCaches, newCaches); }
      }, 120_000);

    // Paged bf16 rows at B > 1: the batched paged cache joins the rows
    // left-padded into one fused SDPA under the padding mask for decode and
    // windows, as the old path did (B1f; B1a's per-row read measured 0.03
    // here).
    test(`${kind} paged bf16 rows at B = 3: decode, window`, () => {
      const p = pair(kind);
      const rows = (model: Graph) => model.makeCache().map(cache => {
        if (!(cache instanceof KVCache)) return targetCacheLayout(cache) as unknown as Cache;
        cache.dispose();
        return new PagedKvRows(1024, PagedKVCache.DEFAULT_BLOCK_SIZE, false) as unknown as Cache;
      });
      const pagedSources = (model: Graph, lengths: number[]) => lengths.map((length, row) => {
        const caches = model.makeCache().map(cache => {
          if (!(cache instanceof KVCache)) return cache;
          cache.dispose();
          return new PagedKVCache(1024, PagedKVCache.DEFAULT_BLOCK_SIZE, false);
        });
        forwardHidden([prompt(length, 81 + row)])(model, caches).dispose();
        return caches;
      });
      const lengths = [90, 300, 37];
      const oldSources = pagedSources(p.old, lengths), newSources = pagedSources(p.current, lengths);
      const oldRows = rows(p.old), newRows = rows(p.current);
      try {
        for (const [set, sources] of [[oldRows, oldSources], [newRows, newSources]] as const)
          set.forEach((layout, layer) => (layout as unknown as BatchableCache).mergeRows(sources.map(source => source[layer]!)));
        for (const [label, rows] of [["decode", [[40], [50], [60]]], ["window4", [prompt(4, 84), prompt(4, 85), prompt(4, 86)]],
          ["decode after window", [[41], [51], [61]]]] as const)
          both(`${kind} paged bf16 rows B=3 ${label}`, p, oldRows, newRows, forwardHidden(rows.map(row => [...row])));
      } finally { disposeAll(...oldSources, ...newSources, oldRows, newRows); }
    }, 120_000);

    // Rows a prefill cohort converts inside its delayed affine layouts carry
    // the kernel set the layouts took when the batch preparation built them
    // (servedAffineKernels: tiled-causal with MLX_BUN_NO_FUSED_SDPA unset,
    // unfused with it set to 1), and the decode group's layouts made from those
    // rows keep it. The old graph attended them through quantizedSdpa under the
    // same flag. B1f; before it the layouts composed the unfused set and the
    // flag-unset window measured 0.77.
    for (const flag of [undefined, "1"])
      test(`${kind} delayed affine4 cohort rows in the decode layouts, flag=${flag ?? "unset"}`, () =>
        withScheme({ name: "", kvBits: 4, flag }, () => {
          const p = pair(kind);
          const maintain = createKvMaintenance({ kind: "affine", bits: 4, groupSize: 64, kernels: flagKernels(flag) });
          const cohort = (model: Graph) => {
            const serial = model.makeCache();
            maintain.preparePrefill!(serial);
            try { return serial.map(cache => prefillCacheLayout(cache) as unknown as Cache); }
            finally { disposeAll(serial); }
          };
          const oldCohort = cohort(p.old), newCohort = cohort(p.current);
          let oldRows: Cache[] = [], newRows: Cache[] = [];
          const extracted: Cache[][] = [];
          try {
            for (const set of [oldCohort, newCohort]) for (const cache of set)
              { (cache as unknown as PaddedPrefillCache).preparePrefill({ lengths: [7, 7, 7] }); cache.prefillMaintenance?.beginPrefill(); }
            both(`${kind} delayed cohort prefill 7 (flag ${flag ?? "unset"})`, p, oldCohort, newCohort,
              forwardHidden([prompt(7, 101), prompt(7, 101), prompt(7, 101)]));
            for (const set of [oldCohort, newCohort]) for (const cache of set) {
              // The cohort commits each row at its planned boundary, then ends prefill.
              cache.prefillMaintenance?.commitPrefill([0, 1, 2]); cache.prefillMaintenance?.endPrefill();
              (cache as unknown as PaddedPrefillCache).finalizePrefill();
            }
            // The cohort's rows join the decode group's layouts (`targetCacheLayout`).
            const rowsOfCohort = (set: Cache[]) => {
              const sources = [0, 1, 2].map(row => set.map(cache => (cache as unknown as BatchableCache).extractRow(row)));
              extracted.push(...sources);
              return rowsOf(sources);
            };
            oldRows = rowsOfCohort(oldCohort); newRows = rowsOfCohort(newCohort);
            // The rows converted at the cohort's boundary: affine decode layouts.
            expect(newRows.map(cache => cache.signature())).toEqual(oldRows.map(cache => cache.signature()));
            expect(newRows.map(cache => cache.signature().split(":").slice(0, 2).join(":")))
              .toEqual(LAYOUTS[kind].types.slice(0, newRows.length).map(type => type === S ? "kv:rotating-affine" : "kv:batched-quant"));
            for (const [label, rows] of [["decode", [[40], [40], [40]]], ["window3", [prompt(3, 102), prompt(3, 102), prompt(3, 103)]],
              ["decode after window", [[41], [41], [41]]]] as const)
              both(`${kind} delayed affine4 cohort rows B=3 ${label} (flag ${flag ?? "unset"})`, p, oldRows, newRows,
                forwardHidden(rows.map(row => [...row])));
          } finally { disposeAll(oldCohort, newCohort, oldRows, newRows, ...extracted); }
        }), 120_000);

    test(`${kind}: MLX_BUN_COMPILED_GEGLU=0 is read once at construction: spelled-out GeGLU, the same bits as the old per-call read`, () => {
      const spelled = createRuntimeConfig({ ...runtimeConfig().values, MLX_BUN_COMPILED_GEGLU: "0" });
      withRuntimeConfig(spelled, () => {
        const { weights, config } = artifact(kind);
        const p: Pair = { kind, old: new FrozenGemma4Model(weights, config), current: new Gemma4Model(weights, config, testComposition()) };
        expect(p.current.layers.every(layer => !layer.mlp.compiled && (layer.experts?.switchGlu.compiled ?? false) === false)).toBe(true);
        const oldCaches = p.old.makeCache(), newCaches = p.current.makeCache();
        try {
          both(`${kind} spelled-out geglu prefill 37`, p, oldCaches, newCaches, forwardHidden([prompt(37, 71)]));
          both(`${kind} spelled-out geglu decode`, p, oldCaches, newCaches, forwardHidden([[72]]));
          both(`${kind} spelled-out geglu window 5`, p, oldCaches, newCaches, forwardHidden([prompt(5, 73)]));
        } finally { disposeAll(oldCaches, newCaches); }
      });
      expect(pair(kind).current.layers.every(layer => layer.mlp.compiled)).toBe(true);
    }, 120_000);

    test(`${kind} training forwards: full-sequence scoring, padded batch, prefix-shared, segmented, checkpointed`, () => {
      const p = pair(kind), tag = `${kind} training`;
      // 49 positions: past the sliding window, so the sliding layers' training
      // masks are window-bound arrays.
      const tokens = prompt(49, 91);
      using input = ids([tokens]);
      {
        using expected = frozenTrainForward(p.old as unknown as RuntimeModel, input);
        using actual = trainForward(p.current as unknown as RuntimeModel, input);
        record(`${tag} full-sequence logits`, maxDiff(actual, expected, `${tag} full-sequence logits`), 1);
      }
      {
        const valid = [24, 19, 15];
        using padded = ids(valid.map((n, row) => [...prompt(n, 92 + row), ...Array(24 - n).fill(0)]));
        using expected = frozenTrainForward(p.old as unknown as RuntimeModel, padded, valid);
        using actual = trainForward(p.current as unknown as RuntimeModel, padded, valid);
        record(`${tag} padded batch logits`, maxDiff(actual, expected, `${tag} padded batch logits`), 1);
      }
      {
        // Prompt 20, chosen 17, rejected 12: 49 positions, past the window.
        const layout = { P: 20, Rc: 17, Rr: 12 };
        using expected = p.old.trainable.prefixShared!.prefixSharedHidden(input, layout);
        using actual = p.current.trainable.prefixShared!.prefixSharedHidden(input, layout);
        record(`${tag} prefix-shared hidden`, maxDiff(actual, expected, `${tag} prefix-shared hidden`), 1);
      }
      {
        const run = (model: Graph) => {
          const segmented = model.trainable.segmented!;
          const pass = segmented.begin(input);
          try {
            // Two segments, threading the reused donors' keys and values.
            const middle = Math.ceil(layers / 2);
            const first = pass.runRange(pass.input, 0, middle, new Map());
            const second = pass.runRange(first.h, middle, layers, first.donorKvOut);
            const out = segmented.finalNorm(second.h);
            for (const shared of [...first.donorKvOut.values(), ...second.donorKvOut.values()] as SharedKv[])
              if (shared.kind === "plain") { shared.keys.dispose(); shared.values.dispose(); shared.offsetArr?.dispose(); }
            first.h.dispose(); second.h.dispose();
            return out;
          } finally { pass.input.dispose(); pass.dispose(); }
        };
        using expected = run(p.old);
        using actual = run(p.current);
        record(`${tag} segmented hidden`, maxDiff(actual, expected, `${tag} segmented hidden`), 1);
      }
      for (const splitMlp of [false, true]) {
        const run = (model: Graph, forward: typeof trainForward) => {
          const checkpoint = model.trainable.gradCheckpoint!.enable([], splitMlp);
          try { return forward(model as unknown as RuntimeModel, input); }
          finally { checkpoint.end(); }
        };
        using expected = run(p.old, frozenTrainForward);
        using actual = run(p.current, trainForward);
        record(`${tag} checkpointed logits (split MLP ${splitMlp})`, maxDiff(actual, expected, `${tag} checkpointed logits`), 1);
      }
    }, 120_000);
  });
}
