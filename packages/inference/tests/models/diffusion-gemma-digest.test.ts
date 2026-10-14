import { testGraph } from "./test-composition";
import { phaseRowLimit } from "@mlx-bun/inference/models/graph";
// DiffusionGemma's graph over small synthetic weights, held to its recorded
// digest (tests/fixtures/synthetic-digests/diffusion-gemma.json;
// tests/models/synthetic-digest.ts): six layers (sliding, sliding, full,
// sliding, sliding, full) with the real model's head shapes (sliding 256, full
// 512 with k reused as v and partial rotary), a sliding window of 16 so prompts
// cross it, the parallel dense and 8-expert MoE branch, self-conditioning, the
// tied 4-bit head and a two-layer vision tower fed synthetic pixels. Text
// prefill, continuation (extendPrefill), the canvas pass at several seeded
// denoising steps with self-conditioning feedback, the vision prefill, the D1
// entry point and seeded generation through the denoising engine (both
// samplers, text and image): every logit, every soft embedding, every cache
// tensor and the generated tokens. The digest was recorded from the tree whose
// graph equalled the pre-D3 graph at maxDiff 0 on these cases, so it carries that
// check forward. Then the composition's legos: paged bf16 against the bf16 lego
// at maxDiff 0, the quantized forms by KL. No model, no download.
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
import { syntheticDigests, type DigestCase } from "./synthetic-digest";
import { testComposition } from "./test-composition";
import type { KvStorageScheme } from "../../src/contracts/portable/composition";


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

// ── Digest ──────────────────────────────────────────────────────────────────

const digests = syntheticDigests("diffusion-gemma");
const { weights, config } = artifact();
const current = testGraph(new DiffusionGemmaModel(weights, config), testComposition());

/** Same shape, dtype and bytes; both owned by the caller and disposed here. */
function expectSame(label: string, actual: MlxArray, expected: MlxArray): void {
  try {
    expect(actual.shape, label).toEqual(expected.shape);
    expect(actual.dtype, label).toBe(expected.dtype);
    using left = ops.contiguous(actual), right = ops.contiguous(expected);
    expect(Buffer.from(left.rawBytes()).equals(Buffer.from(right.rawBytes())), `${label} maxDiff 0`).toBe(true);
  } finally { actual.dispose(); expected.dispose(); }
}

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
 * accepts it, and the next soft embeddings, each step handed to the digest.
 * Leaves the cache set as it was (the canvas pass stores nothing). */
function canvasSteps(digest: DigestCase, label: string, caches: Cache[], seed: bigint, steps: number): void {
  const keys = new DenoisingKeys(seed);
  try { canvasStepsWith(digest, keys, label, caches, steps); } finally { keys.dispose(); }
}
function canvasStepsWith(digest: DigestCase, keys: DenoisingKeys, label: string, caches: Cache[], steps: number): void {
  using weight = current.dequantEmbedWeight();
  digest.tensors(`${label} embed weight`, weight);
  let canvas = keys.randint(VOCAB, [1, CANVAS]);
  let feedback: MlxArray | null = null;
  try {
    for (let step = 1; step <= steps; step++) {
      using logits = current.decoderLogits(canvas, caches, feedback);
      digest.tensors(`${label} step ${step} logits`, logits);
      using temperature = ops.scalarLike(0.4 + 0.1 * step, logits);
      using scaled = ops.div(logits, temperature);
      const next = current.softEmbeddings(scaled, weight);
      feedback?.dispose();
      feedback = next;
      digest.tensors(`${label} step ${step} soft embeddings`, next);
      using argmax = ops.argmaxAxis(scaled, -1);
      using ids = argmax.astype(Dtype.int32);
      using noiseIds = keys.randint(VOCAB, [1, CANVAS]);
      using half = keys.randint(2, [1, CANVAS]);
      using zero = ops.scalarLike(0, half);
      using accept = ops.greater(half, zero);
      const nextCanvas = ops.where(accept, ids, noiseIds);
      canvas.dispose();
      canvas = nextCanvas;
      digest.caches(`${label} step ${step} after canvas pass`, caches);
    }
  } finally {
    canvas.dispose();
    feedback?.dispose();
  }
}

/** Continuation over an accepted block of `length` tokens. */
function extend(digest: DigestCase, label: string, caches: Cache[], length: number, salt: number): void {
  using block = ops.fromInt32(prompt(length, salt), [1, length]);
  current.extendPrefill(block, caches);
  digest.caches(`${label} extendPrefill ${length}`, caches);
}

// 5 < window - 1, 15 = window - 1 (no block mask), 16 and 37 past it (block mask and cut).
for (const length of [2, 5, 15, 16, 37]) digests.test(`text ${length}: prefill, canvas passes and continuations across the window`, digest => {
  const caches = bindLegacyDenoisingModel(current).graph.prefill(prompt(length, length));
  try {
    digest.caches(`text prefill ${length}`, caches);
    canvasSteps(digest, `text ${length}`, caches, BigInt(length), 4);
    // Continuations: a full canvas block, then a short one that lands the ring mid-buffer.
    extend(digest, `text ${length}`, caches, CANVAS, 7);
    canvasSteps(digest, `text ${length} +${CANVAS}`, caches, BigInt(length) + 100n, 3);
    extend(digest, `text ${length} +${CANVAS}`, caches, 5, 9);
    canvasSteps(digest, `text ${length} +${CANVAS}+5`, caches, BigInt(length) + 200n, 2);
  } finally { disposeResources(caches); }
}, 120_000);

for (const [before, after] of [[0, 1], [2, 3], [8, 13]] as const) {
  digests.test(`vision ${before}+${after}: the encoder window over synthetic pixels, inside and past the sliding window`, digest => {
    const ids = visionPrompt(before, VISION.output, after);
    using image = pixels(before + 1);
    const caches = bindLegacyDenoisingModel(current).graph.prefill(ids,image);
    try {
      digest.caches(`vision prefill ${ids.length}`, caches);
      canvasSteps(digest, `vision ${ids.length}`, caches, BigInt(ids.length), 3);
      extend(digest, `vision ${ids.length}`, caches, CANVAS, 11);
      canvasSteps(digest, `vision ${ids.length} +${CANVAS}`, caches, BigInt(ids.length) + 50n, 2);
    } finally { disposeResources(caches); }
  }, 120_000);
}
test("the longest vision prompt crosses the window, where the encoder read differs from a windowed one", () => {
  expect(visionPrompt(8, VISION.output, 13).length).toBeGreaterThan(WINDOW);
});

digests.test("D1 entry and the training forward: forwardCanvasLogits and forwardCanvasLogitsArr", digest => {
  for (const length of [6, 29]) {
    const ids = prompt(length, 2 * length), canvas = prompt(CANVAS, length);
    using logits = current.forwardCanvasLogits(ids, canvas);
    digest.tensors(`forwardCanvasLogits ${length}`, logits);
    using canvasArr = ops.fromInt32(canvas, [1, CANVAS]);
    using logitsArr = current.forwardCanvasLogitsArr(ids, canvasArr);
    digest.tensors(`forwardCanvasLogitsArr ${length}`, logitsArr);
  }
}, 120_000);

/** One seeded run through the denoising engine, recording every step's logits and the final state. */
function generate(ids: number[], options: DiffusionGenOptions) {
  const { graph } = bindLegacyDenoisingModel(current);
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

const runs: { label: string; ids: number[]; image?: () => MlxArray }[] = [
  { label: "text 9", ids: prompt(9, 1) },
  { label: "text 30", ids: prompt(30, 5) },
  { label: "vision", ids: visionPrompt(4, VISION.output, 6), image: () => pixels(3) },
];
for (const run of runs) digests.test(`seeded generation, ${run.label}: the denoising engine's tokens, step logits and final caches`, digest => {
  for (const sampler of ["confidence-threshold", "entropy-bound"] as const) for (const seed of [0n, 11n]) {
    const image = run.image?.();
    const options: DiffusionGenOptions = { maxTokens: 2 * CANVAS + 10, maxDenoisingSteps: 4, minCanvasLength: 8,
      sampler, seed, eosTokenIds: [], visionPixels: image };
    const label = `${sampler} seed ${seed}`;
    const a = generate(run.ids, options);
    try {
      expect(a.result.blocks.length, label).toBeGreaterThan(1);
      digest.value(`${label} result`, a.result);
      digest.tensors(`${label} step logits`, ...a.logits);
      digest.caches(`${label} final cache`, a.state);
    } finally {
      disposeResources([...a.logits, ...a.state]);
      image?.dispose();
    }
  }
}, 120_000);

// The composition's KV scheme and paging (B4): makeCache composes the legos
// they name. Bf16 pages read the bf16 lego's bits, so the paged graph's canvas
// logits equal the bf16 lego's at maxDiff 0 over text and vision prompts and
// continuations. The quantized forms have no oracle: their canvas logits are
// compared with the bf16 graph's on the same canvases, by KL, and reported.
const composedGraph = (kv: KvStorageScheme, paged: boolean) =>
  testGraph(new DiffusionGemmaModel(weights, config), testComposition(kv, paged ? { pagedKv: { blockSize: 16, direct: false } } : {}));

test("the composition's paging: paged bf16 canvas passes equal the bf16 lego's", () => {
  const paged = composedGraph({ kind: "bf16" }, true);
  for (const length of [5, 37]) {
    const ids = prompt(length, length);
    const caches = { paged: bindLegacyDenoisingModel(paged).graph.prefill(ids), plain: bindLegacyDenoisingModel(current).graph.prefill(ids) };
    try {
      expect(caches.paged.filter(cache => cache.signature() === "kv:paged").length).toBe(2);
      const step = (label: string, salt: number) => {
        using canvas = ops.fromInt32(prompt(CANVAS, salt), [1, CANVAS]);
        expectSame(label, paged.decoderLogits(canvas, caches.paged, null), current.decoderLogits(canvas, caches.plain, null));
      };
      step(`paged text ${length}`, 3);
      using block = ops.fromInt32(prompt(CANVAS, 7), [1, CANVAS]);
      paged.extendPrefill(block, caches.paged);
      current.extendPrefill(block, caches.plain);
      step(`paged text ${length} +${CANVAS}`, 4);
    } finally { disposeResources([...caches.paged, ...caches.plain]); }
  }
  const ids = visionPrompt(8, VISION.output, 13);
  using image = pixels(9);
  const caches = { paged: bindLegacyDenoisingModel(paged).graph.prefill(ids,image), plain: bindLegacyDenoisingModel(current).graph.prefill(ids,image) };
  try {
    using canvas = ops.fromInt32(prompt(CANVAS, 5), [1, CANVAS]);
    expectSame("paged vision", paged.decoderLogits(canvas, caches.paged, null), current.decoderLogits(canvas, caches.plain, null));
  } finally { disposeResources([...caches.paged, ...caches.plain]); }
}, 120_000);

/** KL(bf16 ‖ form) per canvas position, from two logit tensors [1, positions, vocab]. */
function klRows(reference: MlxArray, form: MlxArray): { mean: number; max: number; agree: number } {
  using a = reference.astype(Dtype.float32), b = form.astype(Dtype.float32);
  const x = a.toFloat32(), y = b.toFloat32(), positions = x.length / VOCAB;
  let sum = 0, max = 0, agree = 0;
  for (let p = 0; p < positions; p++) {
    const row = (v: Float32Array) => Array.from(v.subarray(p * VOCAB, (p + 1) * VOCAB));
    const lx = row(x), ly = row(y);
    const lse = (r: number[]) => { const m = Math.max(...r); return m + Math.log(r.reduce((s, v) => s + Math.exp(v - m), 0)); };
    const zx = lse(lx), zy = lse(ly);
    let kl = 0;
    for (let i = 0; i < VOCAB; i++) kl += Math.exp(lx[i]! - zx) * ((lx[i]! - zx) - (ly[i]! - zy));
    sum += kl; max = Math.max(max, kl);
    agree += Number(lx.indexOf(Math.max(...lx)) === ly.indexOf(Math.max(...ly)));
  }
  return { mean: sum / positions, max, agree: agree / positions };
}

test("the composition's KV scheme: quantized and paged quantized forms serve the canvas, reported by KL against bf16", () => {
  const reference = composedGraph({ kind: "bf16" }, false);
  const forms: [string, KvStorageScheme, boolean][] = [
    ["affine 4", { kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" }, false],
    ["affine 8", { kind: "affine", bits: 8, groupSize: 64, kernels: "unfused" }, false],
    ["affine per-layer (full layers 8-bit)", { kind: "affine-layers", layers: [{ layer: 2, bits: 8, groupSize: 64 }, { layer: 5, bits: 8, groupSize: 64 }], kernels: "tiled-causal" }, false],
    ["TurboQuant 8/3", { kind: "turbo", kBits: 8, vBits: 3 }, false],
    ["paged affine 4", { kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" }, true],
    ["paged TurboQuant 8/3", { kind: "turbo", kBits: 8, vBits: 3 }, true],
  ];
  const report: Record<string, unknown> = {};
  for (const [name, kv, paged] of forms) {
    const graph = composedGraph(kv, paged);
    const totals = { mean: 0, max: 0, agree: 0, passes: 0 };
    for (const length of [5, 37]) {
      const ids = prompt(length, length);
      const caches = { form: bindLegacyDenoisingModel(graph).graph.prefill(ids), reference: bindLegacyDenoisingModel(reference).graph.prefill(ids) };
      try {
        for (const salt of [3, 4]) {
          using canvas = ops.fromInt32(prompt(CANVAS, salt), [1, CANVAS]);
          using x = reference.decoderLogits(canvas, caches.reference, null);
          using y = graph.decoderLogits(canvas, caches.form, null);
          const kl = klRows(x, y);
          expect(Number.isFinite(kl.mean), name).toBe(true);
          totals.mean += kl.mean; totals.max = Math.max(totals.max, kl.max); totals.agree += kl.agree; totals.passes++;
          using block = ops.fromInt32(prompt(CANVAS, salt + 10), [1, CANVAS]);
          graph.extendPrefill(block, caches.form);
          reference.extendPrefill(block, caches.reference);
        }
      } finally { disposeResources([...caches.form, ...caches.reference]); }
    }
    report[name] = { klMean: totals.mean / totals.passes, klMax: totals.max, argmaxAgree: totals.agree / totals.passes };
  }
  console.log(`diffusion-gemma composed KV (synthetic weights, canvas KL vs bf16): ${JSON.stringify(report)}`);
}, 240_000);

test("declared denoising ranges preserve prompt, canvas and continuation arithmetic across the expert threshold", () => {
  const binding = bindLegacyDenoisingModel(current).graph;
  for (const length of [1,31,32,37]) {
    const ids = prompt(length,length);
    const candidate = binding.prefill(ids), control = current.makeCache();
    current.encodePrompt(ids,control);
    const sameState = () => {
      expect(candidate.length).toBe(control.length);
      for (let i=0; i<control.length; i++) {
        expect(candidate[i]!.offset).toBe(control[i]!.offset);
        const a=candidate[i]!.state(), b=control[i]!.state();
        try {
          expect(a.length).toBe(b.length);
          for (let j=0;j<a.length;j++) {
            using left=ops.contiguous(a[j]!), right=ops.contiguous(b[j]!);
            expect(Buffer.from(left.rawBytes()).equals(Buffer.from(right.rawBytes()))).toBe(true);
          }
        } finally {
          if(candidate[i]!.stateNeedsDispose) disposeResources(a);
          if(control[i]!.stateNeedsDispose) disposeResources(b);
        }
      }
    };
    try {
      sameState();
      for (const rows of [1,7,31,32,37]) {
        using canvas=ops.fromInt32(prompt(rows,3),[1,rows]);
        using hidden=current.decodeCanvasArr(canvas,control);
        expectSame(`range canvas ${rows}`,binding.decoderLogits(canvas,candidate,null),current.logitsFromHidden(hidden));
        sameState();
      }
      for(const rows of [1,31,32]) {
        using tokens=ops.fromInt32(prompt(rows,4),[1,rows]);
        binding.extendPrefill(tokens,candidate); current.extendPrefill(tokens,control); sameState();
      }
    } finally { disposeResources([...candidate,...control]); }
  }
},120_000);
