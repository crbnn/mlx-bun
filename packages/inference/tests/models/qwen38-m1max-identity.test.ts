// The generated M1 Max graph (src/models/qwen/qwen38-27b-trellis-m1max.ts, an entire graph built from the
// composition) on small synthetic weights laid out by the generated quant table: 48 DeltaNet and 16 attention layers
// with the artifact's 3- and 4-bit affine projections, Trellis gate/up at the artifact's widths (mixed in its ten
// layers) and its 3-bit block-interleaved and row-major downs, scaled down in width. The composition facts it
// declares (no LoRA targets unless the composition mounts adapters, no committed spans, no Neural Engine operation,
// its measured verify costs, the residual basis its fold record names, the caches of each KV scheme). Where the graph
// runs the generic blocks (a whole prompt chunk, prompt tails and verify windows wider than eight rows, every
// operation under the adapters composition) it is held to the generic Qwen 3.5 graph of its family over the same
// weights under bf16, affine 4- and 8-bit and TurboQuant KV, hidden, logits and every cache tensor, at maxDiff 0. Its
// row kernels (threadgroup-codebook at one row; factored, simdgroup-matrix and affine verify kernels at two to eight)
// have no generic counterpart: the generator's drift test (qwen38-generated-sync), the kernel tests, the
// real-artifact parity suite on the M1 Max (tests/parity/qwen-trellis-parity.test.ts) and the app digests are their
// gates. No model, no download.
import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { parseQuantization, type ModelConfig } from "../../src/artifacts/config";
import { TURBOQUANT_FOLD_FILE } from "../../src/artifacts/turboquant-fold";
import type { Weights } from "../../src/artifacts/weights";
import type { BatchableCache, Cache } from "../../src/contracts/mlx/cache";
import type { MlxWidthClass } from "../../src/contracts/mlx/graph";
import type { Composition, KvStorageScheme } from "../../src/contracts/portable/composition";
import { LAYERS, Qwen38TrellisM1Max } from "../../src/models/qwen/qwen38-27b-trellis-m1max";
import { Qwen35Model } from "../../src/models/qwen/qwen3_5";
import { QWEN38_COMPOSITIONS } from "../../scripts/gen-qwen38";

const HIDDEN = 512, INTER = 256, VOCAB = 256, HEADS = 4, KV_HEADS = 2, HEAD_DIM = 256;
const HK = 4, HV = 8, DK = 64, DV = 64, KERNEL = 4;
const KEY_DIM = HK * DK, VALUE_DIM = HV * DV, CONV_DIM = 2 * KEY_DIM + VALUE_DIM;
const PREFIX = "language_model.model";
const CHUNK = 40;


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
  device: "applegpu_g13s", aneBridge: false, kv: { kind: "bf16" }, pagedKv: null, draftDepth: { adaptive: true, max: 7 },
  prefillChunkTokens: CHUNK, adapters: false, maxRows: 8, packedMixedMlp: true,
  vectorGreedySampling: true, compiledDecode: true, mixedPrefill: false, mixedTokenBudget: 256, prefillTailSplit: true,
  grammarJump: false, greedyDraft: false, compiledGeglu: true, compiledSwiglu: true, ...overrides,
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

// ── The graphs ──────────────────────────────────────────────────────────────

const prompt = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 37 + salt * 11 + 5) % VOCAB);
const ids = (rows: number[][]) => ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);
const disposeAll = (...sets: Cache[][]) => { for (const set of sets) for (const cache of set) cache.dispose(); };

let synthetic: { weights: Weights; config: ModelConfig } | null = null;
const source = () => synthetic ??= artifact();
const built = new Map<string, { current: Qwen38TrellisM1Max; generic: Qwen35Model }>();
function graphs(scheme: string, overrides: Partial<Composition> = {}): { current: Qwen38TrellisM1Max; generic: Qwen35Model } {
  const key = `${scheme} ${JSON.stringify(overrides)}`;
  let found = built.get(key);
  if (!found) {
    const { weights, config } = source();
    const facts = composition({ kv: SCHEMES[scheme]!, ...overrides });
    found = { current: new Qwen38TrellisM1Max(weights, config, facts), generic: new Qwen35Model(weights, config, facts) };
    built.set(key, found);
  }
  return found;
}
afterAll(() => { for (const { current } of built.values()) current.dispose(); });

describe("the composition the generated graph declares", () => {
  test("no LoRA targets without adapters, caches, no committed spans, no Neural Engine operation, the verify costs its composition measured", () => {
    for (const scheme of Object.keys(SCHEMES)) {
      const { current, generic } = graphs(scheme);
      expect(current.graphCapabilities.adapters).toEqual({ batched: false, mountable: false });
      expect(current.loraTargets().size).toBe(0);
      expect((current as { knownSpans?: unknown }).knownSpans).toBeUndefined();
      expect([...current.verifyRoundCosts]).toEqual(QWEN38_COMPOSITIONS["qwen38-27b-trellis-m1max"]!.verifyRoundCosts!.costs.map(([rows, cost]) => [rows, cost]));
      expect(current.prefillChunk.ane).toBeUndefined();
      const a = current.makeCache(), b = generic.makeCache();
      try { expect(a.map(cache => `${cache.constructor.name} ${cache.signature()}`)).toEqual(b.map(cache => `${cache.constructor.name} ${cache.signature()}`)); }
      finally { disposeAll(a, b); }
    }
  });

  test("the residual basis its fold record declares, read once at construction", () => {
    const { weights, config } = source();
    const dir = mkdtempSync(join(tmpdir(), "qwen38-m1max-fold-"));
    const quiet = console.error;
    console.error = () => {};
    try {
      const at = { ...config, modelDir: dir } as ModelConfig;
      const declared = (graph: { draftTarget(caches: Cache[]): { residualBasis?: unknown } }) => graph.draftTarget([]).residualBasis;
      const without = new Qwen38TrellisM1Max(weights, at, composition());
      expect(declared(without)).toBeUndefined();
      without.dispose();
      writeFileSync(join(dir, TURBOQUANT_FOLD_FILE), JSON.stringify({ seed: 7, r1: true, hiddenSize: HIDDEN,
        final_gain: Array.from({ length: HIDDEN }, (_, i) => 1 + i / HIDDEN) }));
      const current = new Qwen38TrellisM1Max(weights, at, composition());
      expect(declared(current)).toEqual({ r1Seed: 7, finalGain: Float32Array.from({ length: HIDDEN }, (_, i) => 1 + i / HIDDEN) });
      current.dispose();
    } finally { console.error = quiet; rmSync(dir, { recursive: true, force: true }); }
  });
});

for (const scheme of Object.keys(SCHEMES)) {
  test(`${scheme} KV: where both run the generic blocks, the M1 Max graph equals the generic Qwen 3.5 graph (a prompt chunk, a tail and a verify window wider than eight rows)`, () => {
    const { current, generic } = graphs(scheme);
    const mine = current.makeCache(), theirs = generic.makeCache();
    const step = (label: string, width: MlxWidthClass, rows: number[][],
      run: (graph: Qwen38TrellisM1Max | Qwen35Model, input: MlxArray, caches: Cache[]) => MlxArray | Promise<MlxArray>) => {
      using input = ids(rows);
      outputDiff(`${scheme} ${label}`, run(current, input, mine) as MlxArray, run(generic, input, theirs) as MlxArray,
        current.logits[width], generic.logits[width]);
      stateDiff(`${scheme} ${label}`, mine, theirs);
    };
    try {
      step(`prefillChunk.gpu ${CHUNK}`, "wider", [prompt(CHUNK, 1)], (g, input, caches) => g.prefillChunk.gpu(input, caches, null));
      step("prefillTail.wider 13", "wider", [prompt(13, 3)], (g, input, caches) => g.prefillTail.wider(input, caches, null));
      step("verify.wider 12", "wider", [prompt(12, 5)], (g, input, caches) => g.verify.wider(input, caches, null));
    } finally { disposeAll(mine, theirs); }
  }, 180_000);
}

test("the adapters composition: every operation runs the generic blocks", () => {
  const { current: adapted, generic } = graphs("bf16", { adapters: true });
  expect(adapted.graphCapabilities.adapters).toEqual({ batched: true, mountable: true });
  expect(adapted.loraTargets().size).toBe(48 * 5 + 16 * 4);
  expect([...adapted.loraTargets().keys()].sort()).toEqual([...generic.loraTargets().keys()].sort());
  const mine = adapted.makeCache(), theirs = generic.makeCache();
  try {
    const steps: [string, MlxWidthClass, number[][], (graph: Qwen38TrellisM1Max | Qwen35Model, input: MlxArray, caches: Cache[]) => MlxArray][] = [
      ["prefillChunk.gpu", "wider", [prompt(CHUNK, 90)], (g, input, caches) => g.prefillChunk.gpu(input, caches, null) as MlxArray],
      ["decode.one", "one", [[7]], (g, input, caches) => g.decode.one(input, caches, null) as MlxArray],
      ["verify.twoToThree 3", "twoToThree", [prompt(3, 91)], (g, input, caches) => g.verify.twoToThree(input, caches, null) as MlxArray],
      ["verify.four 4", "four", [prompt(4, 93)], (g, input, caches) => g.verify.four(input, caches, null) as MlxArray],
      ["verify.fiveToEight 6", "fiveToEight", [prompt(6, 92)], (g, input, caches) => g.verify.fiveToEight(input, caches, null) as MlxArray],
    ];
    for (const [label, width, rows, run] of steps) {
      using input = ids(rows);
      outputDiff(`adapters ${label}`, run(adapted, input, mine), run(generic, input, theirs), adapted.logits[width], generic.logits[width]);
      stateDiff(`adapters ${label}`, mine, theirs);
    }
  } finally { disposeAll(mine, theirs); }
}, 180_000);
