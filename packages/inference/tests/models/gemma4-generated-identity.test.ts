import { phaseRowLimit } from "../../src/models/graph";
// The written Gemma graphs against the generic family over deterministic
// synthetic weights: exact hidden/logits/cache bytes through prompt, decode,
// verify, ring wrap and the 256-position cache boundary. No artifact download.

import { afterAll, describe, expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { parseQuantization, type ModelConfig } from "../../src/artifacts/config";
import type { Weights } from "../../src/artifacts/weights";
import type { BatchableCache, Cache } from "../../src/contracts/mlx/cache";
import type { MlxTokenPhases } from "../../src/contracts/mlx/graph";
import type { KvStorageScheme } from "../../src/contracts/portable/composition";
import type { ServedComposition } from "../../src/models/implementation";
import { Gemma4Model } from "../../src/models/gemma4/model";
import { phaseHidden } from "./phase-forward";
import { testComposition } from "./test-composition";
import * as current12b from "../../src/models/gemma4/gemma4-12b";
import * as current26b from "../../src/models/gemma4/gemma4-26b";
import * as currentE4b from "../../src/models/gemma4/gemma4-e4b";

const HIDDEN = 256, INTER = 512, EXPERT_INTER = 128, VOCAB = 256, HEADS = 4, KV_HEADS = 2;
const HEAD_DIM = 256, GLOBAL_HEAD_DIM = 512, WINDOW = 16, PER_LAYER = 64, EXPERTS = 16, TOP_K = 8;
const PREFIX = "language_model.model";
const owned: MlxArray[] = [];
afterAll(() => { for (const array of owned) array.dispose(); });
const keep = (array: MlxArray) => { owned.push(array); return array; };
const noise = (count: number, phase: number, scale: number, shift = 0) => Float32Array.from({ length: count },
  (_, i) => shift + Math.sin(i * 0.37 + phase) * scale + Math.cos(i * 0.11 + 2.3 * phase) * scale);
const tensor = (values: Float32Array, shape: number[], dtype = Dtype.bfloat16): MlxArray => {
  using wide = MlxArray.fromFloat32(values, shape);
  return keep(wide.astype(dtype));
};

/** Independent artifact facts, recorded before the graph-file rewrite.
 * All three artifacts have full attention at indices 5, 11, 17, ... .
 * Cache widths are the original artifact kv_config, independent of the file. */
const ARTIFACT_FACTS = {
  e4b: { layers: 42, shared: 18, kEqV: false, input: PER_LAYER, moe: false, caches: 24, kv8: [3, 4, 9, 10, 11, 14] },
  twelve: { layers: 48, shared: 0, kEqV: true, input: 0, moe: false, caches: 48, kv8: [3, 6, 7, 8, 10, 12] },
  moe: { layers: 30, shared: 0, kEqV: true, input: 0, moe: true, caches: 30, kv8: [0, 3, 4, 5, 8, 9, 11, 12] },
} as const;
const artifactFacts = (repo: string) => repo.includes("e4b") ? ARTIFACT_FACTS.e4b : repo.includes("26B") ? ARTIFACT_FACTS.moe : ARTIFACT_FACTS.twelve;
function artifactScheme(repo: string): KvStorageScheme & { readonly kind: "affine-layers" } {
  const facts = artifactFacts(repo), kv8 = new Set<number>(facts.kv8);
  return { kind: "affine-layers", kernels: "tiled-causal", layers: Array.from({ length: facts.caches }, (_, layer) => ({ layer, bits: kv8.has(layer) ? 8 : 4, groupSize: 64 })) };
}
async function fixtureConfig(repo: string): Promise<ModelConfig> {
  const facts = artifactFacts(repo);
  return {
    modelDir: "", hasVisionSidecar: false, raw: {},
    quantization: parseQuantization({ bits: 4, group_size: 64, mode: "affine" }),
    text: {
      numHiddenLayers: facts.layers, numKvSharedLayers: facts.shared,
      layerTypes: Array.from({ length: facts.layers }, (_, layer) => layer % 6 === 5 ? "full_attention" : "sliding_attention"),
      attentionKEqV: facts.kEqV, hiddenSizePerLayerInput: facts.input,
      enableMoeBlock: facts.moe, tieWordEmbeddings: true,
      rmsNormEps: 1e-6, finalLogitSoftcapping: 30,
      ropeParameters: {
        sliding_attention: { ropeType: "default", ropeTheta: 10000, partialRotaryFactor: 1, factor: 1 },
        full_attention: { ropeType: "proportional", ropeTheta: 1000000, partialRotaryFactor: 0.25, factor: 1 },
      },
    },
  } as unknown as ModelConfig;
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

// ── The graphs ──────────────────────────────────────────────────────────────

type Graph = MlxTokenPhases & { makeCache(): Cache[]; logitsFromHidden(hidden: MlxArray): MlxArray };
interface Generated {
  readonly COMPOSITION: ServedComposition;
  readonly GeneratedGemma4: { new (weights: Weights, config: ModelConfig): Graph & { readonly cacheLayers: readonly MlxCacheLayer[] }; readonly prototype: object };

}
const COMPOSITIONS: readonly { stem: string; repo: string; module: Generated }[] = [
  { stem: "gemma4-e4b", repo: "gemma-4-e4b-it-OptiQ-4bit", module: currentE4b as unknown as Generated },
  { stem: "gemma4-12b", repo: "gemma-4-12B-it-OptiQ-4bit", module: current12b as unknown as Generated },
  { stem: "gemma4-26b", repo: "gemma-4-26B-A4B-it-OptiQ-4bit", module: current26b as unknown as Generated },
];

const prompt = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 37 + salt * 11 + 5) % VOCAB);
const ids = (rows: number[][]) => ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);
const disposeAll = (...sets: Cache[][]) => { for (const set of sets) for (const cache of set) cache.dispose(); };

/** Hidden rows and their logits through each graph's head at the class of those rows (owned, disposed here). */
function outputDiff(label: string, model: Graph, generic: Graph, actual: MlxArray, expected: MlxArray, rows: number): void {
  try {
    const width = rows;
    const hidden = maxDiff(actual, expected, `${label} hidden`);
    using la = model.logitsFromHidden(actual);
    using lb = generic.logitsFromHidden(expected);
    const logits = maxDiff(la, lb, `${label} logits`);
    using wide = lb.astype(Dtype.float32);
    expect(wide.toFloat32().every(Number.isFinite), `${label} logits finite`).toBe(true);
    record(`${label} hidden+logits`, Math.max(hidden, logits), 2);
  } finally { actual.dispose(); expected.dispose(); }
}

/** The same tokens through the phase a scheduler names for them, on both graphs. */
function both(label: string, model: Graph, generic: Graph, mine: Cache[], theirs: Cache[], rows: number[][]): void {
  using input = ids(rows);
  outputDiff(label, model, generic, phaseHidden(model, input, mine), phaseHidden(generic, input, theirs), rows.length * rows[0]!.length);
  stateDiff(label, mine, theirs);
}

test("the generated graphs extend nothing", () => {
  for (const c of COMPOSITIONS) expect(Object.getPrototypeOf(c.module.GeneratedGemma4.prototype)).toBe(Object.prototype);
});

for (const c of COMPOSITIONS) {
  describe(`${c.stem}: the generated graph against the generic Gemma 4 graph, kv_config composition`, () => {
    let graphs: { model: Graph; generic: Graph } | null = null;
    const pair = async () => {
      if (graphs) return graphs;
      const { weights, config } = artifact(await fixtureConfig(c.repo), 1);
      const kv = artifactScheme(c.repo);
      expect(c.module.COMPOSITION.kv![0]).toEqual({ kind: "affine-layers", layers: kv.layers });
      const composition = testComposition(kv);
      return graphs = { model: bindGraphCaches(new c.module.GeneratedGemma4(weights, config), composition), generic: bindGraphCaches(new Gemma4Model(weights, config, phaseRowLimit(composition)), composition) };

    };

    test(`${c.stem}: prefill 37, decodes, a window across the sliding boundary, verify 4, decodes`, async () => {
      const { model, generic } = await pair();
      const mine = model.makeCache(), theirs = generic.makeCache();
      try {
        both("prefill 37", model, generic, mine, theirs, [prompt(37, 1)]);
        for (let step = 0; step < 3; step++) both(`decode ${step}`, model, generic, mine, theirs, [[200 + step]]);
        both("window 8", model, generic, mine, theirs, [prompt(8, 2)]);
        both("verify 4", model, generic, mine, theirs, [prompt(4, 3)]);
        for (let step = 0; step < 2; step++) both(`decode after ${step}`, model, generic, mine, theirs, [[90 + step]]);
      } finally { disposeAll(mine, theirs); }
    }, 120_000);

    test(`${c.stem}: prefill 200, decodes, a window across the 256-position step`, async () => {
      const { model, generic } = await pair();
      const mine = model.makeCache(), theirs = generic.makeCache();
      try {
        both("prefill 200", model, generic, mine, theirs, [prompt(200, 6)]);
        for (let step = 0; step < 2; step++) both(`decode ${step}`, model, generic, mine, theirs, [[11 + step]]);
        both("window 60 across 256", model, generic, mine, theirs, [prompt(60, 7)]);
        both("decode after", model, generic, mine, theirs, [[19]]);
      } finally { disposeAll(mine, theirs); }
    }, 120_000);

    test(`${c.stem}: known spans above the verify row cap stay one fixed forward`,async()=>{
      const {model,generic}=await pair();
      const mine=model.makeCache(),theirs=generic.makeCache();
      try {
        using ids=ops.fromInt32(prompt(129,8),[1,129]);
        const left=knownOperationFor(model.appendKnown,129)(ids,mine) as MlxArray;
        const right=knownOperationFor(generic.appendKnown,129)(ids,theirs) as MlxArray;
        try { maxDiff(left,right,`${c.stem} known129`);stateDiff(`${c.stem} known129`,mine,theirs); }
        finally {left.dispose();right.dispose();}
      } finally {disposeAll(mine,theirs);}
    },120_000);

    if (c.stem !== "gemma4-26b") test(`${c.stem}: the literal plain trace preserves compiled decode logits and cache`, async () => {
      const { model, generic } = await pair();
      const compiled = model as Graph & { compiledDecodeStep(): MlxCompiledDecodeStep; releaseCompiledDecode(): void };
      const control = generic as Graph & { compiledDecodeStep(): MlxCompiledDecodeStep; releaseCompiledDecode(): void };
      const mine = model.makeCache(), theirs = generic.makeCache();
      try {
        both("compiled prefix", model, generic, mine, theirs, [prompt(37, 9)]);
        const left = compiled.compiledDecodeStep(), right = control.compiledDecodeStep();
        for (const token of [200, 201, 202]) {
          using raw = ops.fromInt32([token], [1]);
          using id = raw.astype(Dtype.uint32);
          const candidate = left.step(id, mine), expected = right.step(id, theirs);
          ops.evalAll([candidate.logits, ...candidate.evalWith]); ops.evalAll([expected.logits, ...expected.evalWith]);
          try { maxDiff(candidate.logits, expected.logits, `${c.stem} compiled logits`); stateDiff(`${c.stem} compiled state`, mine, theirs); }
          finally { candidate.logits.dispose(); expected.logits.dispose(); }
        }
      } finally { compiled.releaseCompiledDecode(); control.releaseCompiledDecode(); disposeAll(mine, theirs); }
    }, 120_000);
  });
}

import type { MlxCacheLayer } from "../../src/contracts/mlx/graph";

import { bindGraphCaches } from "../../src/models/cache-composition";

import type { MlxCompiledDecodeStep } from "../../src/contracts/mlx/graph";

import { knownOperationFor } from "../../src/generation/phase-dispatch";
