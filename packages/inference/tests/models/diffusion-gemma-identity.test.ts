// D3 bit-identity gate for DiffusionGemma. The graph whose blocks read through
// the cache (models/diffusion-gemma/model.ts: the window read for prefill and
// continuation, the block read for the canvas pass, the encoder window read for
// the vision prompt) against the frozen pre-D3 graph (diffusion-gemma-reference.ts,
// deleted with this test after the merge), on small synthetic weights: six
// layers (sliding, sliding, full, sliding, sliding, full) with the real model's
// head shapes (sliding 256, full 512 with k reused as v and partial rotary), a
// sliding window of 16 so prompts cross it, the parallel dense and 8-expert MoE
// branch, self-conditioning, the tied 4-bit head and a two-layer vision tower
// fed synthetic pixels. Text prefill, continuation (extendPrefill), the canvas
// pass at several seeded denoising steps with self-conditioning feedback, the
// vision prefill, the D1 entry point and seeded generation through the
// denoising engine (both samplers, text and image): every logit, every soft
// embedding and every cache tensor must agree to the bit, and generation must
// produce the same tokens. No model, no download.
import { afterAll, expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { parseQuantization, type ModelConfig } from "../../src/artifacts/config";
import type { Weights } from "../../src/artifacts/weights";
import type { Cache } from "../../src/contracts/mlx/cache";
import { DenoisingKeys, denoiseSync, type DiffusionGenOptions } from "../../src/generation/diffusion";
import { bindLegacyDenoisingModel } from "../../src/generation/bindings/denoising";
import { disposeResources } from "../../src/runtime/resources";
import { DiffusionGemmaModel } from "../../src/models/diffusion-gemma/model";
import { DiffusionGemmaModel as FrozenDiffusionGemmaModel } from "./diffusion-gemma-reference";

const HIDDEN = 128, VOCAB = 256, HEADS = 4, SLIDING_KV = 2, SLIDING_DIM = 256, FULL_KV = 1, FULL_DIM = 512;
const INTER = 128, EXPERTS = 8, TOP_K = 2, MOE_INTER = 64, WINDOW = 16, CANVAS = 24;
const LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention",
  "sliding_attention", "sliding_attention", "full_attention"];
const IMAGE = 250, BOI = 251, EOI = 252;
const VISION = { hidden: 128, inter: 128, layers: 2, heads: 8, headDim: 72, patch: 8, pool: 3, output: 9, positions: 16 };

const owned: MlxArray[] = [];
afterAll(() => { for (const array of owned) array.dispose(); });
const keep = (array: MlxArray) => { owned.push(array); return array; };

const noise = (count: number, phase: number, scale: number, shift = 0) => Float32Array.from({ length: count },
  (_, i) => shift + Math.sin(i * 0.37 + phase) * scale + Math.cos(i * 0.11 + 2.3 * phase) * scale);
const tensor = (values: Float32Array, shape: number[], dtype = Dtype.bfloat16): MlxArray => {
  using wide = MlxArray.fromFloat32(values, shape);
  return keep(wide.astype(dtype));
};

/** The synthetic artifact: tensors under the checkpoint's names and the config the graph reads. */
function artifact(): { weights: Weights; config: ModelConfig } {
  const tensors = new Map<string, MlxArray>();
  let seed = 1;
  const affine = (path: string, shape: number[], scale = 0.06) => {
    using w = tensor(noise(shape.reduce((a, b) => a * b, 1), ++seed, scale), shape);
    const q = ops.quantize(w, 64, 4);
    tensors.set(`${path}.weight`, keep(q.packed));
    tensors.set(`${path}.scales`, keep(q.scales));
    tensors.set(`${path}.biases`, keep(q.biases));
  };
  const plain = (path: string, shape: number[], scale: number, shift = 0) =>
    tensors.set(path, tensor(noise(shape.reduce((a, b) => a * b, 1), ++seed, scale, shift), shape));
  const norm = (path: string, width: number) => plain(path, [width], 0.05, 1);

  affine("model.decoder.embed_tokens", [VOCAB, HIDDEN], 0.5);
  norm("model.decoder.norm.weight", HIDDEN);
  const sc = "model.decoder.self_conditioning";
  norm(`${sc}.pre_norm.weight`, HIDDEN);
  affine(`${sc}.gate_proj`, [INTER, HIDDEN]);
  affine(`${sc}.up_proj`, [INTER, HIDDEN]);
  affine(`${sc}.down_proj`, [HIDDEN, INTER]);
  LAYER_TYPES.forEach((type, layer) => {
    const sliding = type === "sliding_attention";
    const [kv, dim] = sliding ? [SLIDING_KV, SLIDING_DIM] : [FULL_KV, FULL_DIM];
    const prefix = `model.decoder.layers.${layer}`;
    affine(`${prefix}.self_attn.q_proj`, [HEADS * dim, HIDDEN]);
    affine(`${prefix}.self_attn.k_proj`, [kv * dim, HIDDEN]);
    if (sliding) affine(`${prefix}.self_attn.v_proj`, [kv * dim, HIDDEN]);
    affine(`${prefix}.self_attn.o_proj`, [HIDDEN, HEADS * dim], 0.03);
    norm(`${prefix}.self_attn.q_norm.weight`, dim);
    norm(`${prefix}.self_attn.k_norm.weight`, dim);
    affine(`${prefix}.mlp.gate_proj`, [INTER, HIDDEN]);
    affine(`${prefix}.mlp.up_proj`, [INTER, HIDDEN]);
    affine(`${prefix}.mlp.down_proj`, [HIDDEN, INTER]);
    affine(`${prefix}.router.proj`, [EXPERTS, HIDDEN], 0.3);
    plain(`${prefix}.router.scale`, [HIDDEN], 0.1, 1);
    plain(`${prefix}.router.per_expert_scale`, [EXPERTS], 0.2, 1);
    affine(`${prefix}.experts.gate_up_proj`, [EXPERTS, 2 * MOE_INTER, HIDDEN]);
    affine(`${prefix}.experts.down_proj`, [EXPERTS, HIDDEN, MOE_INTER]);
    for (const name of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
      "post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2"])
      norm(`${prefix}.${name}.weight`, HIDDEN);
    plain(`${prefix}.layer_scalar`, [1], 0.05, 0.8);
    plain(`model.encoder.language_model.layers.${layer}.layer_scalar`, [1], 0.05, 0.7);
  });

  const vp = "model.encoder.vision_tower.";
  const v = VISION, vAttn = v.heads * v.headDim;
  affine(`${vp}patch_embedder.input_proj`, [v.hidden, 3 * v.patch * v.patch]);
  plain(`${vp}patch_embedder.position_embedding_table`, [2, v.positions, v.hidden], 0.2);
  for (let layer = 0; layer < v.layers; layer++) {
    const prefix = `${vp}encoder.layers.${layer}`;
    for (const name of ["q_proj", "k_proj", "v_proj"]) affine(`${prefix}.self_attn.${name}.linear`, [vAttn, v.hidden]);
    affine(`${prefix}.self_attn.o_proj.linear`, [v.hidden, vAttn], 0.03);
    norm(`${prefix}.self_attn.q_norm.weight`, v.headDim);
    norm(`${prefix}.self_attn.k_norm.weight`, v.headDim);
    affine(`${prefix}.mlp.gate_proj.linear`, [v.inter, v.hidden]);
    affine(`${prefix}.mlp.up_proj.linear`, [v.inter, v.hidden]);
    plain(`${prefix}.mlp.down_proj.linear.weight`, [v.hidden, v.inter], 0.05);
    for (const name of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"])
      norm(`${prefix}.${name}.weight`, v.hidden);
  }
  plain(`${vp}std_bias`, [v.hidden], 0.1);
  plain(`${vp}std_scale`, [v.hidden], 0.1, 1);
  affine("model.encoder.embed_vision.embedding_projection", [HIDDEN, v.hidden]);

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
    modelDir: "", modelType: "diffusion_gemma", architectures: [], dtype: "bfloat16", kvQuant: null,
    hasVisionSidecar: false, eosTokenIds: [],
    raw: { image_token_id: IMAGE, vision_config: {
      hidden_size: v.hidden, intermediate_size: v.inter, num_hidden_layers: v.layers, num_attention_heads: v.heads,
      num_key_value_heads: v.heads, head_dim: v.headDim, rope_parameters: { rope_theta: 100 }, rms_norm_eps: 1e-6,
      patch_size: v.patch, pooling_kernel_size: v.pool, default_output_length: v.output,
      position_embedding_size: v.positions, standardize: true } },
    quantization: parseQuantization({ bits: 4, group_size: 64, mode: "affine" }),
    text: {
      hiddenSize: HIDDEN, numHiddenLayers: LAYER_TYPES.length, layerTypes: LAYER_TYPES, rmsNormEps: 1e-6,
      slidingWindow: WINDOW, headDim: SLIDING_DIM, globalHeadDim: FULL_DIM, numAttentionHeads: HEADS,
      numKeyValueHeads: SLIDING_KV, numGlobalKeyValueHeads: FULL_KV, intermediateSize: INTER, vocabSize: VOCAB,
      numExperts: EXPERTS, topKExperts: TOP_K, moeIntermediateSize: MOE_INTER, canvasLength: CANVAS,
      finalLogitSoftcapping: 30,
      ropeParameters: {
        sliding_attention: { ropeType: "default", ropeTheta: 10_000, partialRotaryFactor: 1, factor: 1 },
        full_attention: { ropeType: "proportional", ropeTheta: 1_000_000, partialRotaryFactor: 0.25, factor: 1 },
      },
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
    console.log(`diffusion-gemma-identity ${label}: maxDiff ${worst} over ${tensors} tensors`);
});

/** 0 when shape, dtype and bytes are equal; otherwise throws with the largest difference. */
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

/** Every tensor of every layer's cache, with its signature and offset, recorded under `category`. */
function stateDiff(category: string, label: string, actual: readonly Cache[], expected: readonly Cache[]): void {
  expect(actual.length).toBe(expected.length);
  let worst = 0, count = 0;
  for (let layer = 0; layer < expected.length; layer++) {
    const a = actual[layer]!, b = expected[layer]!;
    expect(a.signature(), `${label} layer ${layer}`).toBe(b.signature());
    expect(a.offset, `${label} layer ${layer}`).toBe(b.offset);
    const left = a.state(), right = b.state();
    expect(left.length, `${label} layer ${layer}`).toBe(right.length);
    expect(right.length, `${label} layer ${layer}`).toBeGreaterThan(0);
    for (let i = 0; i < right.length; i++) { worst = Math.max(worst, maxDiff(left[i]!, right[i]!, `${label} layer ${layer} state ${i}`)); count++; }
  }
  record(category, worst, count);
}

/** Owned arrays from both graphs, disposed here, recorded under `category`. */
function outputDiff(category: string, label: string, actual: MlxArray, expected: MlxArray): void {
  try {
    const diff = maxDiff(actual, expected, label);
    using wide = expected.astype(Dtype.float32);
    expect(wide.toFloat32().every(Number.isFinite), `${label} finite`).toBe(true);
    record(category, diff, 1);
  } finally { actual.dispose(); expected.dispose(); }
}

const { weights, config } = artifact();
const current = new DiffusionGemmaModel(weights, config);
const frozen = new FrozenDiffusionGemmaModel(weights, config);

const prompt = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 37 + salt * 11 + 5) % 240);
/** Channel-first synthetic pixels in [0, 1]: a 9x9 patch grid, nine soft tokens. */
function pixels(salt: number): MlxArray {
  const side = VISION.patch * VISION.pool * 3;
  return MlxArray.fromFloat32(Float32Array.from({ length: 3 * side * side },
    (_, i) => 0.5 + 0.5 * Math.sin(i * 0.013 * salt + salt)), [1, 3, side, side]);
}
/** A text prompt around one image of `soft` soft tokens. */
const visionPrompt = (before: number, soft: number, after: number) =>
  [...prompt(before, 3), BOI, ...Array(soft).fill(IMAGE), EOI, ...prompt(after, 4)];

/** The engine's canvas step with self-conditioning feedback: decoder logits at
 * the scheduled temperature, the argmax as the next canvas where the draw
 * accepts it, and the next soft embeddings, on both graphs, compared at every
 * step. Leaves both cache sets as they were (the canvas pass stores nothing). */
function canvasSteps(label: string, caches: { current: Cache[]; frozen: Cache[] }, seed: bigint, steps: number): void {
  const keys = new DenoisingKeys(seed);
  try { canvasStepsWith(keys, label, caches, steps); } finally { keys.dispose(); }
}
function canvasStepsWith(keys: DenoisingKeys, label: string, caches: { current: Cache[]; frozen: Cache[] }, steps: number): void {
  const kind = label.split(" ")[0]!;
  using weightCurrent = current.dequantEmbedWeight();
  using weightFrozen = frozen.dequantEmbedWeight();
  outputDiff("dequantized embedding", `${label} embed weight`, ops.contiguous(weightCurrent), ops.contiguous(weightFrozen));
  let canvas = keys.randint(VOCAB, [1, CANVAS]);
  let feedback: { current: MlxArray; frozen: MlxArray } | null = null;
  try {
    for (let step = 1; step <= steps; step++) {
      using lc = current.decoderLogits(canvas, caches.current, feedback?.current ?? null);
      using lf = frozen.decoderLogits(canvas, caches.frozen, feedback?.frozen ?? null);
      record(`${kind} canvas decoder logits`, maxDiff(lc, lf, `${label} step ${step} logits`), 1);
      using temperature = ops.scalarLike(0.4 + 0.1 * step, lc);
      using pc = ops.div(lc, temperature);
      using pf = ops.div(lf, temperature);
      const next = { current: current.softEmbeddings(pc, weightCurrent), frozen: frozen.softEmbeddings(pf, weightFrozen) };
      if (feedback) { feedback.current.dispose(); feedback.frozen.dispose(); }
      feedback = next;
      record(`${kind} canvas soft embeddings`, maxDiff(next.current, next.frozen, `${label} step ${step} soft embeddings`), 1);
      using argmax = ops.argmaxAxis(pc, -1);
      using ids = argmax.astype(Dtype.int32);
      using noiseIds = keys.randint(VOCAB, [1, CANVAS]);
      using half = keys.randint(2, [1, CANVAS]);
      using zero = ops.scalarLike(0, half);
      using accept = ops.greater(half, zero);
      const nextCanvas = ops.where(accept, ids, noiseIds);
      canvas.dispose();
      canvas = nextCanvas;
      stateDiff(`${kind} cache after a canvas pass`, `${label} step ${step} after canvas pass`, caches.current, caches.frozen);
    }
  } finally {
    canvas.dispose();
    if (feedback) { feedback.current.dispose(); feedback.frozen.dispose(); }
  }
}

/** Continuation over an accepted block of `length` tokens on both cache sets. */
function extend(label: string, caches: { current: Cache[]; frozen: Cache[] }, length: number, salt: number): void {
  using block = ops.fromInt32(prompt(length, salt), [1, length]);
  current.extendPrefill(block, caches.current);
  frozen.extendPrefill(block, caches.frozen);
  stateDiff(`${label.split(" ")[0]} extendPrefill cache`, `${label} extendPrefill ${length}`, caches.current, caches.frozen);
}

test("text: prefill, canvas passes and continuations are bit-identical across the window", () => {
  // 5 < window - 1, 15 = window - 1 (no block mask), 16 and 37 past it (block mask and cut).
  for (const length of [2, 5, 15, 16, 37]) {
    const ids = prompt(length, length);
    const caches = { current: current.prefill(ids), frozen: frozen.prefill(ids) };
    try {
      stateDiff("text prefill cache", `text prefill ${length}`, caches.current, caches.frozen);
      canvasSteps(`text ${length}`, caches, BigInt(length), 4);
      // Continuations: a full canvas block, then a short one that lands the ring mid-buffer.
      extend(`text ${length}`, caches, CANVAS, 7);
      canvasSteps(`text ${length} +${CANVAS}`, caches, BigInt(length) + 100n, 3);
      extend(`text ${length} +${CANVAS}`, caches, 5, 9);
      canvasSteps(`text ${length} +${CANVAS}+5`, caches, BigInt(length) + 200n, 2);
    } finally { disposeResources([...caches.current, ...caches.frozen]); }
  }
}, 120_000);

test("vision: the encoder window over synthetic pixels is bit-identical, inside and past the sliding window", () => {
  for (const [before, after] of [[0, 1], [2, 3], [8, 13]] as const) {
    const ids = visionPrompt(before, VISION.output, after);
    using image = pixels(before + 1);
    const caches = { current: current.prefillVision(ids, image), frozen: frozen.prefillVision(ids, image) };
    try {
      stateDiff("vision prefill cache", `vision prefill ${ids.length}`, caches.current, caches.frozen);
      canvasSteps(`vision ${ids.length}`, caches, BigInt(ids.length), 3);
      extend(`vision ${ids.length}`, caches, CANVAS, 11);
      canvasSteps(`vision ${ids.length} +${CANVAS}`, caches, BigInt(ids.length) + 50n, 2);
    } finally { disposeResources([...caches.current, ...caches.frozen]); }
  }
  // The longest prompt crosses the window, where the encoder read differs from a windowed one.
  expect(visionPrompt(8, VISION.output, 13).length).toBeGreaterThan(WINDOW);
}, 120_000);

test("D1 entry and the training forward: forwardCanvasLogits and forwardCanvasLogitsArr are bit-identical", () => {
  for (const length of [6, 29]) {
    const ids = prompt(length, 2 * length), canvas = prompt(CANVAS, length);
    outputDiff("forwardCanvasLogits logits", `forwardCanvasLogits ${length}`,
      current.forwardCanvasLogits(ids, canvas), frozen.forwardCanvasLogits(ids, canvas));
    using canvasArr = ops.fromInt32(canvas, [1, CANVAS]);
    outputDiff("forwardCanvasLogitsArr logits", `forwardCanvasLogitsArr ${length}`,
      current.forwardCanvasLogitsArr(ids, canvasArr), frozen.forwardCanvasLogitsArr(ids, canvasArr));
  }
}, 120_000);

/** One seeded run through the denoising engine, recording every step's logits and the final state. */
function generate(model: DiffusionGemmaModel | FrozenDiffusionGemmaModel, ids: number[], options: DiffusionGenOptions) {
  const { graph } = bindLegacyDenoisingModel(model);
  const logits: MlxArray[] = [];
  const closed: Cache[][] = [];
  const result = denoiseSync({ ...graph,
    decoderLogits: (canvas, cache, feedback) => {
      const out = graph.decoderLogits(canvas, cache, feedback);
      logits.push(ops.contiguous(out));
      return out;
    },
    closeState: (cache) => { closed.push(cache); },
  }, ids, options);
  expect(closed).toHaveLength(1);
  return { result, logits, state: closed[0]! };
}

test("seeded generation: the denoising engine produces identical tokens, step logits and final caches", () => {
  const runs: { label: string; ids: number[]; image?: () => MlxArray }[] = [
    { label: "text 9", ids: prompt(9, 1) },
    { label: "text 30", ids: prompt(30, 5) },
    { label: "vision", ids: visionPrompt(4, VISION.output, 6), image: () => pixels(3) },
  ];
  for (const run of runs) for (const sampler of ["confidence-threshold", "entropy-bound"] as const) for (const seed of [0n, 11n]) {
    const image = run.image?.();
    const options: DiffusionGenOptions = { maxTokens: 2 * CANVAS + 10, maxDenoisingSteps: 4, minCanvasLength: 8,
      sampler, seed, eosTokenIds: [], visionPixels: image };
    const label = `generate ${run.label} ${sampler} seed ${seed}`;
    const a = generate(current, run.ids, options);
    const b = generate(frozen, run.ids, options);
    try {
      expect(a.result, label).toEqual(b.result);
      expect(a.result.blocks.length, label).toBeGreaterThan(1);
      expect(a.logits.length, label).toBe(b.logits.length);
      let worst = 0;
      for (let i = 0; i < b.logits.length; i++) worst = Math.max(worst, maxDiff(a.logits[i]!, b.logits[i]!, `${label} step ${i} logits`));
      record("generation step logits", worst, b.logits.length);
      stateDiff("generation final cache", label, a.state, b.state);
    } finally {
      disposeResources([...a.logits, ...b.logits, ...a.state, ...b.state]);
      image?.dispose();
    }
  }
}, 120_000);
