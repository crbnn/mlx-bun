// D1 bit-identity gate. The generated M4 Pro graph (src/models/qwen/qwen38-27b-trellis-m4pro.ts, an entire graph
// built from the composition) against the frozen pre-D1 graph (qwen38-27b-trellis-m4pro-reference.ts, deleted with
// this test after the merge), on small synthetic weights laid out by the generated quant table: 48 DeltaNet and 16
// attention layers with the artifact's 3- and 4-bit affine projections, Trellis gate/up at the artifact's widths
// (mixed in its ten layers) and its 3-bit block-interleaved and row-major downs, scaled down in width. For bf16 KV
// and for 4-bit group-64 KV (Kv4Head256Cache under both graphs, so both read the same kernels), every phase
// operation and width class the old graph served is held to the bit, logits and every cache tensor:
// prefillChunk.gpu and prefillTail against the old prefill plan, decode.one against the one-row plan, verify at
// widths 2, 3, 4, 5 and 8 with capture against the old verify forward (the four set is the old four-row plan:
// shared-row gate/up, matrix-unit down; the two downs agree at these synthetic widths, so the real-artifact run is
// the check that separates them), each head width against the old head, and the deprecated entry points
// (forwardHidden at every width, the capturing mixed forward, media positions, committed fills, two sequences)
// against the same calls on the old graph. The Neural Engine chunk operation is gated against the old ANE plan
// with MLX_BUN_TEST_ANE=1 (unset skips, which is not a pass). No model, no download.
import { afterAll, describe, expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype, deviceArchitecture } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { parseQuantization, type ModelConfig } from "../../src/artifacts/config";
import type { Weights } from "../../src/artifacts/weights";
import type { BatchableCache, Cache } from "../../src/contracts/mlx/cache";
import { widthClass, type MlxLayerCapture, type MlxWidthTable } from "../../src/contracts/mlx/graph";
import type { Composition, KvStorageScheme } from "../../src/contracts/portable/composition";
import { AneProgram } from "../../src/kernels/ane/linear";
import { LAYERS, Qwen38TrellisM4Pro, anePrograms, type AneWidths } from "../../src/models/qwen/qwen38-27b-trellis-m4pro";
import { unfusedAffineKernels } from "../../src/state/affine-attention";
import { AttentionMasks } from "../../src/state/attention-read";
import { Bf16FirstKv4Head256Cache, Bf16FirstQuantizedKVCache, Bf16FirstTurboQuantKVCache, convertToAffine } from "../../src/state/bf16-first-kv";
import { KVCache } from "../../src/state/kv";
import { Kv4Head256Cache } from "../../src/state/kv4-head256-cache";
import { targetCacheLayout } from "../../src/state/layout";
import { SSMCache } from "../../src/state/ssm";
import { Qwen38TrellisM4Pro as FrozenM4Pro } from "./qwen38-27b-trellis-m4pro-reference";
import { phaseHidden } from "./phase-forward";

const HIDDEN = 512, INTER = 256, VOCAB = 256, HEADS = 4, KV_HEADS = 2, HEAD_DIM = 256;
const HK = 4, HV = 8, DK = 64, DV = 64, KERNEL = 4;
const KEY_DIM = HK * DK, VALUE_DIM = HV * DV, CONV_DIM = 2 * KEY_DIM + VALUE_DIM;
const PREFIX = "language_model.model";
const ANE = !!process.env.MLX_BUN_TEST_ANE;
const g16s = deviceArchitecture() === "applegpu_g16s";

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
  device: "applegpu_g16s", aneBridge: false, kv: { kind: "bf16" }, pagedKv: null, draftDepth: { adaptive: true, max: 7 },
  prefillChunkTokens: CHUNK, adapters: false, maxRows: 8, packedMixedMlp: true,
  vectorGreedySampling: true, compiledDecode: true, mixedPrefill: false, mixedTokenBudget: 256, prefillTailSplit: true,
  grammarJump: false, greedyDraft: false, compiledGeglu: true, ...overrides,
});
/** The M4 Pro composition's KV scheme (`--kv-quant 4`, fused SDPA unset). */
const KV4: KvStorageScheme = Object.freeze({ kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" });

/** Kv4Head256Cache storage, quantized from token zero, for every attention layer: the storage the D1 gate holds both
 *  graphs to (each reads its packed keys through the KV4 kernels), as the graph's makeCache composed it before E3
 *  made the composition's lego bf16-first. One mask memo; the uniform composition's affine set for several rows. */
function kv4Caches(): Cache[] {
  const masks = new AttentionMasks();
  return LAYERS.map(layer => layer.kind === "deltanet" ? new SSMCache()
    : new Kv4Head256Cache(unfusedAffineKernels(4, 64, Dtype.bfloat16), masks));
}

// ── Comparison ──────────────────────────────────────────────────────────────

const results = new Map<string, { worst: number; tensors: number }>();
const record = (label: string, diff: number, tensors: number) => {
  const prior = results.get(label) ?? { worst: 0, tensors: 0 };
  results.set(label, { worst: Math.max(prior.worst, diff), tensors: prior.tensors + tensors });
};
afterAll(() => {
  for (const [label, { worst, tensors }] of results) console.log(`qwen38-m4pro-identity ${label}: maxDiff ${worst} over ${tensors} tensors`);
});

/** Largest absolute difference: 0 when shape, dtype and bytes agree; with `gate`, any other result throws. */
function maxDiff(actual: MlxArray, expected: MlxArray, what: string, gate = true): number {
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
  if (gate) throw new Error(`${what}: bytes differ, maxDiff ${worst}`);
  return worst;
}

/** Every tensor of every layer's cache, with its signature, offsets and rows. */
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

/** Hidden rows and their logits through the heads given (all owned by the caller and disposed here). */
function outputDiff(label: string, actual: MlxArray, expected: MlxArray, actualHead: (h: MlxArray) => MlxArray,
  expectedHead: (h: MlxArray) => MlxArray): void {
  try {
    const hidden = maxDiff(actual, expected, `${label} hidden`);
    using la = actualHead(actual);
    using lb = expectedHead(expected);
    const logits = maxDiff(la, lb, `${label} logits`);
    using wide = lb.astype(Dtype.float32);
    expect(wide.toFloat32().every(Number.isFinite), `${label} logits finite`).toBe(true);
    record(`${label} hidden+logits`, Math.max(hidden, logits), 2);
  } finally { actual.dispose(); expected.dispose(); }
}

function tapsDiff(label: string, actual: MlxArray[], expected: MlxArray[]): void {
  try {
    expect(actual.length, label).toBe(LAYERS.length + 1);
    expect(expected.length, label).toBe(LAYERS.length + 1);
    let worst = 0;
    for (let i = 0; i < expected.length; i++) worst = Math.max(worst, maxDiff(actual[i]!, expected[i]!, `${label} layer ${i}`));
    record(`${label} captured layers`, worst, expected.length);
  } finally { for (const array of [...actual, ...expected]) array.dispose(); }
}

// ── The two graphs ──────────────────────────────────────────────────────────

const CHUNK = 40;
const prompt = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 37 + salt * 11 + 5) % VOCAB);
const ids = (rows: number[][]) => ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);
const disposeAll = (...sets: Cache[][]) => { for (const set of sets) for (const cache of set) cache.dispose(); };

type Scheme = "bf16" | "kv4";
const SCHEMES: Scheme[] = ["bf16", "kv4"];

/** The install step for a graph's Neural Engine programs at `widths` (the artifact's by default): construction only
 *  loads them. The synthetic artifact's: in_proj_z and out_proj VALUE_DIM wide, o_proj HEADS × HEAD_DIM. */
const SYNTHETIC_ANE_WIDTHS: AneWidths = { hidden: HIDDEN, intermediate: INTER,
  projections: [[HIDDEN, VALUE_DIM], [VALUE_DIM, HIDDEN], [HEADS * HEAD_DIM, HIDDEN]] };
const compileAne = (record: Composition, widths?: AneWidths) => { for (const spec of anePrograms(record, widths)) AneProgram.compile(spec); };

let synthetic: { weights: Weights; config: ModelConfig } | null = null;
const source = () => synthetic ??= artifact();
let pair: { old: FrozenM4Pro; current: Qwen38TrellisM4Pro } | null = null;
function graphs(): { old: FrozenM4Pro; current: Qwen38TrellisM4Pro } {
  if (!pair) {
    const { weights, config } = source();
    const old = new FrozenM4Pro(weights, config, false);
    // The old createAppend reads the 27B geometry from its config; nothing in a text forward does.
    (old as { config: ModelConfig }).config = { ...config, text: { ...config.text,
      hiddenSize: 5120, intermediateSize: 17408, headDim: 256, numAttentionHeads: 24, numKeyValueHeads: 4 } };
    pair = { old, current: new Qwen38TrellisM4Pro(weights, config, composition()) };
  }
  return pair;
}

/** Caches for `scheme`: the new graph's own composition, and the same storage for the old graph, whose plans read
 *  Kv4Head256Cache's packed keys through the kernels the lego reads with. */
function caches(scheme: Scheme): { old: Cache[]; current: Cache[] } {
  const { current } = graphs();
  if (scheme === "bf16") return { old: current.makeCache(), current: current.makeCache() };
  return { old: kv4Caches(), current: kv4Caches() };
}

const oldHead = (h: MlxArray) => graphs().old.logitsFromHidden(h);
const widthOf = (rows: number): keyof MlxWidthTable<unknown> =>
  rows === 1 ? "one" : rows <= 3 ? "twoToThree" : rows === 4 ? "four" : rows <= 8 ? "fiveToEight" : "wider";
const headAt = (rows: number) => graphs().current.logits[widthOf(rows)];

/** A capture that copies every layer it is handed. */
function collecting(): { capture: MlxLayerCapture; taps: MlxArray[] } {
  const taps: MlxArray[] = [];
  return { taps, capture: (_layer, hidden) => { taps.push(ops.contiguous(hidden)); } };
}

describe("composition", () => {
  test("makeCache composes the lego the composition's KV scheme names; the graph declares what the composition built", () => {
    const { current } = graphs();
    const { weights, config } = source();
    const composed = (kv: KvStorageScheme) => new Qwen38TrellisM4Pro(weights, config, composition({ kv })).makeCache();
    const plain = current.makeCache(), kv4 = composed(KV4), fused = composed({ ...KV4, kernels: "tiled-causal" });
    const other = composed({ kind: "affine", bits: 8, groupSize: 64, kernels: "unfused" });
    const turbo = composed({ kind: "turbo", kBits: 8, vBits: 3 });
    try {
      for (const [layer, spec] of LAYERS.entries()) {
        if (spec.kind === "deltanet") for (const set of [plain, kv4, fused, other, turbo]) expect(set[layer]).toBeInstanceOf(SSMCache);
        else {
          expect(plain[layer]).toBeInstanceOf(KVCache);
          expect(other[layer]).toBeInstanceOf(Bf16FirstQuantizedKVCache);
          expect(other[layer]).not.toBeInstanceOf(Bf16FirstKv4Head256Cache);
          expect(turbo[layer]).toBeInstanceOf(Bf16FirstTurboQuantKVCache);
          // The KV4 lego, its several-row reads the composition's affine set.
          for (const [set, plainSet] of [[kv4, unfusedAffineKernels], [fused, "tiled-causal"]] as const) {
            expect(set[layer]).toBeInstanceOf(Bf16FirstKv4Head256Cache);
            const lego = set[layer] as Bf16FirstKv4Head256Cache;
            expect(lego.start).toBe(0);
            if (plainSet === unfusedAffineKernels) expect(lego.plain).toBe(unfusedAffineKernels);
            else expect(lego.plain).not.toBe(unfusedAffineKernels);
          }
        }
      }
      // One mask memo per makeCache call.
      for (const set of [plain, kv4, turbo])
        expect(new Set(set.filter(cache => !(cache instanceof SSMCache)).map(cache => (cache as { masks?: unknown }).masks)).size).toBe(1);
    } finally { disposeAll(plain, kv4, fused, other, turbo); }
    expect(current.graphCapabilities.adapters).toEqual({ batched: false, mountable: false });
    expect(current.loraTargets().size).toBe(0);
    expect(current.prefillChunk.ane).toBeUndefined();
    expect(LAYERS.filter(layer => layer.downLayout === "k3Interleaved").length).toBe(40);
    expect(LAYERS.filter(layer => layer.kind === "deltanet").length).toBe(48);
  });
});

for (const scheme of SCHEMES) {
  describe(`Qwen3.8 M4 Pro graph, ${scheme} KV`, () => {
    test(`${scheme}: prefill chunk, tail, decode at one row, verify at 2, 3, 4, 5 and 8 with capture, every head width`, () => {
      const { old, current } = graphs();
      const set = caches(scheme);
      try {
        {
          using input = ids([prompt(CHUNK, 1)]);
          const expected = old.forwardHidden(input, set.old);
          outputDiff(`${scheme} prefillChunk.gpu ${CHUNK}`, current.prefillChunk.gpu(input, set.current, null) as MlxArray, expected,
            current.logits.wider, oldHead);
          stateDiff(`${scheme} prefillChunk.gpu ${CHUNK}`, set.current, set.old);
        }
        {
          using input = ids([prompt(13, 2)]);
          const expected = old.forwardHidden(input, set.old);
          outputDiff(`${scheme} prefillTail 13`, (current.prefillTail.wider(input, set.current, null) as MlxArray), expected, current.logits.wider, oldHead);
          stateDiff(`${scheme} prefillTail 13`, set.current, set.old);
        }
        for (let step = 0; step < 3; step++) {
          using input = ids([[100 + step]]);
          const expected = old.forwardHidden(input, set.old);
          outputDiff(`${scheme} decode.one`, current.decode.one(input, set.current, null), expected, current.logits.one, oldHead);
          stateDiff(`${scheme} decode.one`, set.current, set.old);
        }
        for (const width of [2, 3, 4, 5, 8]) {
          using input = ids([prompt(width, 3 + width)]);
          const before = collecting(), after = collecting();
          const expected = old.verify[widthOf(width)](input, set.old, before.capture) as MlxArray;
          const actual = current.verify[widthOf(width)](input, set.current, after.capture) as MlxArray;
          tapsDiff(`${scheme} verify ${width}`, after.taps, before.taps);
          outputDiff(`${scheme} verify ${width}`, actual, expected, headAt(width), oldHead);
          stateDiff(`${scheme} verify ${width}`, set.current, set.old);
        }
        // A window wider than eight rows takes the generic blocks and the plain head, as the old prefill plan did.
        {
          using input = ids([prompt(12, 20)]);
          const before = collecting(), after = collecting();
          const expected = old.verify.wider(input, set.old, before.capture) as MlxArray;
          const actual = current.verify.wider(input, set.current, after.capture) as MlxArray;
          tapsDiff(`${scheme} verify.wider 12`, after.taps, before.taps);
          outputDiff(`${scheme} verify.wider 12`, actual, expected, current.logits.wider, oldHead);
          stateDiff(`${scheme} verify.wider 12`, set.current, set.old);
        }
        // The head alone at every width, on one hidden.
        for (const rows of [1, 2, 3, 4, 5, 8, 12]) {
          using input = ids([prompt(rows, 30 + rows)]);
          using hidden = old.embed.encode(input);
          using expected = old.logitsFromHidden(hidden);
          using actual = headAt(rows)(hidden);
          record(`${scheme} logits[${widthOf(rows)}] ${rows} rows`, maxDiff(actual, expected, `${scheme} head ${rows}`), 1);
        }
      } finally { disposeAll(set.old, set.current); }
    }, 180_000);

    test(`${scheme}: the phases a scheduler names reach the same blocks as the old graph's deprecated entry points`, () => {
      const { old, current } = graphs();
      const set = caches(scheme);
      /** The head for a hidden's rows, as the scheduler indexes `logits`. */
      const headFor = (h: MlxArray) => current.logits[widthClass(h.size / h.shape.at(-1)!)](h);
      // One sequence: a decode for one token, a prompt tail for more (`phaseHidden`), against the old shape-read forward.
      const both = (label: string, rows: number[][]) => {
        using input = ids(rows);
        const expected = old.forwardHidden(input, set.old);
        outputDiff(label, phaseHidden(current, input, set.current), expected, headFor, oldHead);
        stateDiff(label, set.current, set.old);
      };
      try {
        both(`${scheme} prompt tail 37`, [prompt(37, 5)]);
        for (const width of [1, 2, 3, 4, 5, 8, 13]) both(`${scheme} phase ${width}`, [prompt(width, 6 + width)]);
        // The capturing mixed forward (the speculative binding's work path): a verify group of three rows.
        {
          using input = ids([prompt(3, 40)]);
          const before = collecting(), after = collecting();
          const [expected] = old.forwardHiddenMixed([{ ids: input, cache: set.old, captureLayer: before.capture, preserveTokenGeometry: true }]);
          const [actual] = current.mixed([{ phase: "verify", width: "twoToThree", ids: input, state: set.current, capture: after.capture }],
            "twoToThree");
          tapsDiff(`${scheme} mixed verify 3 (capture)`, after.taps, before.taps);
          outputDiff(`${scheme} mixed verify 3 (capture)`, actual!, expected!, headFor, oldHead);
          stateDiff(`${scheme} mixed verify 3 (capture)`, set.current, set.old);
        }
        // The one capture path E2 keeps (the binding's copy of each tapped layer from a prompt tail's capture)
        // against the speculative binding's layer tap around a deprecated forward.
        {
          using input = ids([prompt(3, 41)]);
          const layers = new Set([0, 31, 63]);
          const tapOld = { layers, captured: new Map<number, MlxArray>() };
          const captured = new Map<number, MlxArray>();
          old.hiddenTap = tapOld;
          const expected = old.forwardHidden(input, set.old);
          old.hiddenTap = null;
          const actual = current.prefillTail.twoToThree(input, set.current, (layer, h) => {
            if (layers.has(layer)) captured.set(layer, ops.contiguous(h));
          });
          try {
            expect([...captured.keys()].sort()).toEqual([...tapOld.captured.keys()].sort());
            let worst = 0;
            for (const [layer, copy] of tapOld.captured) worst = Math.max(worst, maxDiff(captured.get(layer)!, copy, `${scheme} capture layer ${layer}`));
            record(`${scheme} capture against hiddenTap, captured layers`, worst, tapOld.captured.size);
          } finally { for (const copy of [...tapOld.captured.values(), ...captured.values()]) copy.dispose(); }
          outputDiff(`${scheme} capture against hiddenTap, forward`, actual, expected, headFor, oldHead);
          stateDiff(`${scheme} capture against hiddenTap, forward`, set.current, set.old);
        }
        // Media positions: [3, rows, tokens] request positions, continuing the cache's text positions. The continuation
        // takes the decode read at one position and the window read above, as the old shape-read forward did.
        for (const width of [6, 1]) {
          using input = ids([prompt(width, 50 + width)]);
          const start = set.old[0]!.offset;
          using positions = ops.fromInt32(Array.from({ length: 3 * width }, (_, i) => start + (i % width) + Math.floor(i / width)), [3, 1, width]);
          const expected = old.forwardHiddenAtPositions(input, set.old, positions);
          const actual = width === 1 ? current.decodeAtPositions(input, set.current, positions, "one")
            : current.prefillAtPositions(input, set.current, positions, widthClass(width));
          outputDiff(`${scheme} media continuation ${width}`, actual, expected, headFor, oldHead);
          stateDiff(`${scheme} media continuation ${width}`, set.current, set.old);
          // The vision prefill phase over embeddings is a window (a prompt of one position carries no image).
          if (width === 1) continue;
          using embeds = old.embed.encode(input);
          const e2 = old.forwardEmbeddingsAtPositions(embeds, set.old, positions);
          const a2 = current.prefillEmbeddingsAtPositions(embeds, set.current, positions, widthClass(width));
          outputDiff(`${scheme} vision prefill ${width}`, a2, e2, headFor, oldHead);
          stateDiff(`${scheme} vision prefill ${width}`, set.current, set.old);
        }
        // Known tokens through appendKnown (committed spans) against the old graph's createAppend (which needs the M4 Pro device).
        if (g16s) for (const span of [1, 2, 3, 4]) {
          using input = ids([prompt(span, 60 + span)]);
          const expected = old.createAppend({ hasAdapters: false, pagedKv: false })!.forwardHidden(input, set.old) as MlxArray;
          const actual = current.appendKnown[widthClass(span)](input, set.current) as MlxArray;
          outputDiff(`${scheme} appendKnown ${span}`, actual, expected, headFor, oldHead);
          stateDiff(`${scheme} appendKnown ${span}`, set.current, set.old);
        }
        both(`${scheme} decode after known tokens`, [[91]]);
      } finally { disposeAll(set.old, set.current); }
    }, 180_000);
  });
}

describe("Qwen3.8 M4 Pro graph, several sequences (bf16 KV)", () => {
  /** Row layouts holding `sources` (one serial cache list per row). */
  const rowsOf = (sources: Cache[][]): Cache[] => sources[0]!.map((cache, layer) => {
    const layout = targetCacheLayout(cache) as unknown as BatchableCache;
    layout.mergeRows(sources.map(rows => rows[layer]!));
    return layout;
  });

  test("two sequences: each prefill as the old graph ran it; the width tables run the row kernels for both sequences' rows", () => {
    const { old, current } = graphs();
    const oldSources = [caches("bf16").old, caches("bf16").old], newSources = [caches("bf16").current, caches("bf16").current];
    let newRows: Cache[] = [];
    try {
      for (const [row, length] of [9, 14].entries()) {
        using input = ids([prompt(length, 80 + row)]);
        outputDiff(`two sequences prefill ${row}`, phaseHidden(current, input, newSources[row]!), old.forwardHidden(input, oldSources[row]!),
          current.logits.wider, oldHead);
        stateDiff(`two sequences prefill ${row}`, newSources[row]!, oldSources[row]!);
      }
      newRows = rowsOf(newSources);
      // No old graph served these (its deprecated forward took the generic blocks for several sequences): the
      // one-kernel layers take both sequences' rows, decode at two rows and a window of six.
      {
        using input = ids([[60], [61]]);
        using hidden = current.decode.twoToThree(input, newRows, null);
        using logits = current.logits.twoToThree(hidden);
        using wide = logits.astype(Dtype.float32);
        expect(wide.toFloat32().every(Number.isFinite)).toBe(true);
      }
      {
        using input = ids([prompt(3, 84), prompt(3, 85)]);
        using hidden = current.verify.fiveToEight(input, newRows, null);
        using logits = current.logits.fiveToEight(hidden);
        using wide = logits.astype(Dtype.float32);
        expect(wide.toFloat32().every(Number.isFinite)).toBe(true);
      }
    } finally { disposeAll(...oldSources, ...newSources, newRows); }
  }, 180_000);
});

describe("Qwen3.8 M4 Pro graph, adapters composition", () => {
  test("every operation runs the generic blocks, which the old graph ran under a mounted adapter", () => {
    const { weights, config } = source();
    const { old } = graphs();
    const adapted = new Qwen38TrellisM4Pro(weights, config, composition({ adapters: true, aneBridge: true }));
    expect(adapted.graphCapabilities.adapters).toEqual({ batched: true, mountable: true });
    expect(adapted.loraTargets().size).toBe(48 * 5 + 16 * 4);
    expect(adapted.prefillChunk.ane).toBeUndefined();
    const oldCaches = adapted.makeCache(), newCaches = adapted.makeCache();
    // The old graph's generic forward: what it ran under a mounted adapter (or the profiling hook, used here).
    const generic = <T>(run: () => T): T => {
      const scope = globalThis as Record<string, unknown>;
      scope.__deltaProf = {};
      try { return run(); } finally { delete scope.__deltaProf; }
    };
    try {
      const steps: [string, number[][], (input: MlxArray) => MlxArray][] = [
        ["prefillChunk.gpu", [prompt(CHUNK, 90)], input => adapted.prefillChunk.gpu(input, newCaches, null) as MlxArray],
        ["decode.one", [[7]], input => adapted.decode.one(input, newCaches, null)],
        ["verify.twoToThree 3", [prompt(3, 91)], input => adapted.verify.twoToThree(input, newCaches, null) as MlxArray],
        ["verify.four 4", [prompt(4, 93)], input => adapted.verify.four(input, newCaches, null) as MlxArray],
        ["verify.fiveToEight 6", [prompt(6, 92)], input => adapted.verify.fiveToEight(input, newCaches, null) as MlxArray],
      ];
      for (const [label, rows, run] of steps) {
        using input = ids(rows);
        const expected = generic(() => old.forwardHidden(input, oldCaches));
        const actual = run(input);
        outputDiff(`adapters ${label}`, actual, expected, adapted.logits.one, h => old.lmHead!.forward(h));
        stateDiff(`adapters ${label}`, newCaches, oldCaches);
      }
    } finally { disposeAll(oldCaches, newCaches); }
  }, 180_000);
});

// ── The real artifact (opt-in: MLX_BUN_TEST_QWEN38_ARTIFACT=<mlx-bun/Qwen3.8-27B-Trellis-3.2bpw snapshot>, 11 GB
// resident; with MLX_BUN_TEST_ANE=1 the Neural Engine chunk too). Both graphs share one Weights. ──────────────────

const ARTIFACT = process.env.MLX_BUN_TEST_QWEN38_ARTIFACT;

describe("Qwen3.8 M4 Pro graph on the real artifact", () => {
  test.skipIf(!ARTIFACT || !g16s)("2048-token chunk, tail, 8 decodes, verify 3, 8 and 4 (synthetic proposals), the old graph's plans to the bit", async () => {
    const { Weights } = await import("../../src/artifacts/weights");
    const { loadModelConfig } = await import("../../src/artifacts/config");
    const config = await loadModelConfig(ARTIFACT!);
    const weights = await Weights.open(ARTIFACT!);
    try {
      // The new graph installs the checkpoint view; the old graphs read through it and must not install it again.
      const record = composition({ aneBridge: ANE, prefillChunkTokens: 2048 });
      compileAne(record);
      const current = new Qwen38TrellisM4Pro(weights, config, record);
      const shared = {
        tensor: (name: string) => weights.tensor(name), has: (name: string) => weights.has(name), info: (name: string) => weights.info(name),
        get tensorNames() { return weights.tensorNames; }, shards: weights.shards, setView: () => {},
      } as unknown as Weights;
      const old = new FrozenM4Pro(shared, config, false);
      const head = (h: MlxArray) => old.logitsFromHidden(h);
      const last = (h: MlxArray) => { const L = h.shape[1]!; return h.slice([0, L - 1, 0], [1, L, h.shape[2]!]); };
      const lastLogits = (project: (h: MlxArray) => MlxArray) => (h: MlxArray) => { using row = last(h); return project(row); };
      const oldCaches = old.makeCache(), newCaches = current.makeCache();
      const real = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 7919 + salt * 104729) % 151_000 + 1000);
      try {
        {
          using input = ids([real(2048, 1)]);
          const expected = old.forwardHidden(input, oldCaches);
          outputDiff("real prefillChunk.gpu 2048", current.prefillChunk.gpu(input, newCaches, null) as MlxArray, expected,
            lastLogits(current.logits.one), lastLogits(head));
          stateDiff("real prefillChunk.gpu 2048", newCaches, oldCaches);
        }
        {
          using input = ids([real(37, 2)]);
          const expected = old.forwardHidden(input, oldCaches);
          outputDiff("real prefillTail 37", (current.prefillTail.wider(input, newCaches, null) as MlxArray), expected, lastLogits(current.logits.one), lastLogits(head));
          stateDiff("real prefillTail 37", newCaches, oldCaches);
        }
        for (let step = 0; step < 8; step++) {
          using input = ids([[real(1, 10 + step)[0]!]]);
          const expected = old.forwardHidden(input, oldCaches);
          outputDiff("real decode.one", current.decode.one(input, newCaches, null), expected, current.logits.one, head);
          stateDiff("real decode.one", newCaches, oldCaches);
        }
        // Verify 4 runs last, on the input and cache state where the 2-to-4 class measured 0.078 against the old plan.
        for (const [width, salt] of [[3, 23], [8, 28], [4, 30]] as const) {
          using input = ids([real(width, salt)]);
          const before = collecting(), after = collecting();
          const expected = old.verify[widthOf(width)](input, oldCaches, before.capture) as MlxArray;
          const actual = current.verify[widthOf(width)](input, newCaches, after.capture) as MlxArray;
          tapsDiff(`real verify ${width}`, after.taps, before.taps);
          outputDiff(`real verify ${width}`, actual, expected, current.logits[widthOf(width)], head);
          stateDiff(`real verify ${width}`, newCaches, oldCaches);
        }
      } finally { disposeAll(oldCaches, newCaches); }
      if (ANE) {
        const oldAne = new FrozenM4Pro(shared, config, true);
        const oldSet = oldAne.makeCache(), newSet = current.makeCache();
        try {
          using input = ids([real(2048, 40)]);
          const expected = oldAne.forwardHidden(input, oldSet);
          outputDiff("real prefillChunk.ane 2048", current.prefillChunk.ane!(input, newSet, null) as MlxArray, expected,
            lastLogits(current.logits.one), lastLogits(head));
          stateDiff("real prefillChunk.ane 2048", newSet, oldSet);
        } finally { disposeAll(oldSet, newSet); }
      }
    } finally { weights.dispose(); }
  }, 1_800_000);

  // E3: the composition's KV scheme (`--kv-quant 4`: affine 4-bit group 64) composes the bf16-first KV4 lego in the
  // graph's own makeCache. Driven as the scheduler drives one row: the prefill chunk the composition is built for
  // (read bf16; the lego converts inside that append, the old path converted after the forward), the tail, decodes and
  // verify at 3 and 8, against the old graph over plain caches converted at the same point, whose plans read the packed
  // keys through the cores it injected.
  test.skipIf(!ARTIFACT || !g16s)("composed KV4: the bf16-first lego through prefill, its conversion, decodes and verify 3 and 8, the old graph's injected cores to the bit", async () => {
    const { Weights } = await import("../../src/artifacts/weights");
    const { loadModelConfig } = await import("../../src/artifacts/config");
    const config = await loadModelConfig(ARTIFACT!);
    const weights = await Weights.open(ARTIFACT!);
    try {
      const current = new Qwen38TrellisM4Pro(weights, config, composition({ kv: KV4, prefillChunkTokens: 2048 }));
      const shared = {
        tensor: (name: string) => weights.tensor(name), has: (name: string) => weights.has(name), info: (name: string) => weights.info(name),
        get tensorNames() { return weights.tensorNames; }, shards: weights.shards, setView: () => {},
      } as unknown as Weights;
      const old = new FrozenM4Pro(shared, config, false);
      const head = (h: MlxArray) => old.logitsFromHidden(h);
      const last = (h: MlxArray) => { const L = h.shape[1]!; return h.slice([0, L - 1, 0], [1, L, h.shape[2]!]); };
      const lastLogits = (project: (h: MlxArray) => MlxArray) => (h: MlxArray) => { using row = last(h); return project(row); };
      const oldCaches = old.makeCache(), newCaches = current.makeCache();
      const storage = (set: Cache[]) => set.map(cache => cache.composed?.storage() ?? cache);
      const real = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 7919 + salt * 104729) % 151_000 + 1000);
      try {
        expect(newCaches.filter(cache => cache instanceof Bf16FirstKv4Head256Cache)).toHaveLength(LAYERS.length - 48);
        {
          using input = ids([real(2048, 1)]);
          const expected = old.forwardHidden(input, oldCaches);
          outputDiff("composed kv4 prefillChunk.gpu 2048", current.prefillChunk.gpu(input, newCaches, null) as MlxArray, expected,
            lastLogits(current.logits.one), lastLogits(head));
          // The conversion point: the old path's conversion after the first forward (quantized_kv_start 0).
          for (const [layer, cache] of oldCaches.entries())
            if (cache instanceof KVCache) oldCaches[layer] = convertToAffine(cache, 64, 4, 0, unfusedAffineKernels);
          expect(storage(newCaches).filter(cache => cache.signature() === "kv:quant:4:64")).toHaveLength(LAYERS.length - 48);
          stateDiff("composed kv4 conversion point", storage(newCaches), oldCaches);
        }
        {
          using input = ids([real(37, 2)]);
          const expected = old.forwardHidden(input, oldCaches);
          outputDiff("composed kv4 prefillTail 37", (current.prefillTail.wider(input, newCaches, null) as MlxArray), expected, lastLogits(current.logits.one), lastLogits(head));
          stateDiff("composed kv4 prefillTail 37", storage(newCaches), oldCaches);
        }
        for (let step = 0; step < 4; step++) {
          using input = ids([[real(1, 10 + step)[0]!]]);
          const expected = old.forwardHidden(input, oldCaches);
          outputDiff("composed kv4 decode.one", current.decode.one(input, newCaches, null), expected, current.logits.one, head);
          stateDiff("composed kv4 decode.one", storage(newCaches), oldCaches);
        }
        for (const [width, salt] of [[3, 23], [8, 28]] as const) {
          using input = ids([real(width, salt)]);
          const before = collecting(), after = collecting();
          const expected = old.verify[widthOf(width)](input, oldCaches, before.capture) as MlxArray;
          const actual = current.verify[widthOf(width)](input, newCaches, after.capture) as MlxArray;
          tapsDiff(`composed kv4 verify ${width}`, after.taps, before.taps);
          outputDiff(`composed kv4 verify ${width}`, actual, expected, current.logits[widthOf(width)], head);
          stateDiff(`composed kv4 verify ${width}`, storage(newCaches), oldCaches);
        }
      } finally { disposeAll(oldCaches, newCaches); }
    } finally { weights.dispose(); }
  }, 1_800_000);
});

describe("Qwen3.8 M4 Pro graph, Neural Engine chunk operation (MLX_BUN_TEST_ANE=1)", () => {
  for (const scheme of SCHEMES) test.skipIf(!ANE)(`${scheme}: prefillChunk.ane against the old ANE plan, prefillChunk.gpu against the old prefill plan, at 2048 rows`, () => {
    const { weights, config } = source();
    const record = composition({ aneBridge: true, prefillChunkTokens: 2048 });
    compileAne(record, SYNTHETIC_ANE_WIDTHS);
    const oldAne = new FrozenM4Pro(weights, config, true);
    const { old } = graphs();
    const current = new Qwen38TrellisM4Pro(weights, config, record);
    expect(current.prefillChunk.ane).toBeDefined();
    const make = () => scheme === "kv4" ? kv4Caches() : current.makeCache();
    const sets = [make(), make(), make(), make()];
    try {
      using input = ids([prompt(2048, 100)]);
      const ane = oldAne.forwardHidden(input, sets[0]!);
      outputDiff(`${scheme} prefillChunk.ane 2048`, current.prefillChunk.ane!(input, sets[1]!, null) as MlxArray, ane, current.logits.wider, oldHead);
      stateDiff(`${scheme} prefillChunk.ane 2048`, sets[1]!, sets[0]!);
      const gpu = old.forwardHidden(input, sets[2]!);
      outputDiff(`${scheme} prefillChunk.gpu 2048`, current.prefillChunk.gpu(input, sets[3]!, null) as MlxArray, gpu, current.logits.wider, oldHead);
      stateDiff(`${scheme} prefillChunk.gpu 2048`, sets[3]!, sets[2]!);
      // Decode after the Neural Engine chunk, as after the old one.
      using next = ids([[5]]);
      outputDiff(`${scheme} decode.one after the ANE chunk`, current.decode.one(next, sets[1]!, null), oldAne.forwardHidden(next, sets[0]!),
        current.logits.one, oldHead);
    } finally { disposeAll(...sets); }
  }, 600_000);
});
