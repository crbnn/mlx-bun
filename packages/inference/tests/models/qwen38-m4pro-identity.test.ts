import { MlxPhaseDispatch } from "../../src/generation/phase-dispatch";
// The generated M4 Pro graph (src/models/qwen/qwen38-27b-trellis-m4pro.ts, an entire graph built from the
// composition) on small synthetic weights laid out by the generated quant table: 48 DeltaNet and 16 attention layers
// with the artifact's 3- and 4-bit affine projections, Trellis gate/up at the artifact's widths (mixed in its ten
// layers) and its 3-bit block-interleaved and row-major downs, scaled down in width. What the composition builds:
// makeCache composes the lego the KV scheme names and the graph declares what it built. Where the graph runs the
// generic blocks (a whole prompt chunk, windows wider than eight rows, every operation under the adapters
// composition) it is held to the generic Qwen 3.5 graph of its family over the same weights, hidden, logits and
// every cache tensor, at maxDiff 0. Its row kernels have no generic counterpart: the generator's drift test
// (qwen38-generated-sync), the real-artifact parity suite (tests/parity/qwen-trellis-parity.test.ts) and the app
// digests are their gates. No model, no download.
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { parseQuantization, type ModelConfig } from "../../src/artifacts/config";
import type { Weights } from "../../src/artifacts/weights";
import type { BatchableCache, Cache } from "../../src/contracts/mlx/cache";
import type { Composition, KvStorageScheme } from "../../src/contracts/portable/composition";
import { anePrograms, Qwen38TrellisM4Pro } from "../../src/models/qwen/qwen38-27b-trellis-m4pro";
import { LAYERS } from "./qwen38-layout";
import { AneProgram } from "../../src/kernels/ane/linear";
import { bindGraphCaches, composeGraphCaches } from "../../src/models/cache-composition";
import { Qwen35Model } from "../../src/models/qwen/qwen3_5";
import { unfusedAffineKernels } from "../../src/state/affine-attention";
import { Bf16FirstKv4Head256Cache, Bf16FirstQuantizedKVCache, Bf16FirstTurboQuantKVCache } from "../../src/state/bf16-first-kv";
import { KVCache } from "../../src/state/kv";
import { SSMCache } from "../../src/state/ssm";

const HIDDEN = 512, INTER = 256, VOCAB = 256, HEADS = 4, KV_HEADS = 2, HEAD_DIM = 256;
const HK = 4, HV = 16, DK = 64, DV = 64, KERNEL = 4;
const KEY_DIM = HK * DK, VALUE_DIM = HV * DV, CONV_DIM = 2 * KEY_DIM + VALUE_DIM;
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
      tieWordEmbeddings: false,maxPositionEmbeddings:4096,
    },
  } as unknown as ModelConfig;
  return { weights, config };
}

const composition = (overrides: Partial<Composition> = {}): Composition => Object.freeze<Composition>({
  device: "applegpu_g16s", kv: { kind: "bf16" }, pagedKv: null, draftDepth: { adaptive: true, max: 7 },
  prefillChunkTokens: CHUNK, adapters: false, maxRows: 8, packedMixedMlp: true,
  vectorGreedySampling: true, compiledDecode: true, mixedPrefill: false, mixedTokenBudget: 256, prefillTailSplit: true,
  grammarJump: false, greedyDraft: false, compiledGeglu: true, compiledSwiglu: true, ...overrides,
});
/** The M4 Pro composition's KV scheme (`--kv-quant 4`, fused SDPA unset). */
const KV4: KvStorageScheme = Object.freeze({ kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" });
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

// ── The graphs ──────────────────────────────────────────────────────────────

const CHUNK = 40;
const prompt = (n: number, salt: number) => Array.from({ length: n }, (_, i) => (i * 37 + salt * 11 + 5) % VOCAB);
const ids = (rows: number[][]) => ops.fromInt32(rows.flat(), [rows.length, rows[0]!.length]);
const disposeAll = (...sets: Cache[][]) => { for (const set of sets) for (const cache of set) cache.dispose(); };

let synthetic: { weights: Weights; config: ModelConfig } | null = null;
const source = () => synthetic ??= artifact();
let built: (Qwen38TrellisM4Pro & {makeCache():Cache[]}) | null = null;
beforeAll(() => {
  for (const program of anePrograms({hidden:HIDDEN,intermediate:INTER,projections:[[HIDDEN,VALUE_DIM],[VALUE_DIM,HIDDEN]]})) AneProgram.compile(program);
});
afterAll(() => { built?.dispose(); });
const graph = () => built ??= bindGraphCaches(new Qwen38TrellisM4Pro(source().weights,source().config),composition());

describe("composition", () => {
  test("makeCache composes the lego the composition's KV scheme names; the graph declares what the composition built", () => {
    const current = graph();
    const { weights, config } = source();
    const composed = (kv: KvStorageScheme) => composeGraphCaches(current,composition({kv}));
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

    expect(current.loraTargets().size).toBe(48*5+16*4+64*3);
    expect(current.prompt.filter(range => range.unit === "ane").map(({from,to})=>[from,to])).toEqual([[640,1024],[1025,1536],[1537,2048]]);
    expect(LAYERS.filter(layer => layer.downLayout === "k3Interleaved").length).toBe(40);
    expect(LAYERS.filter(layer => layer.kind === "deltanet").length).toBe(48);
  });
});

describe("the M4 Pro graph against the generic Qwen 3.5 graph where both run the generic blocks (bf16 KV)", () => {
  test("a known grammar span beyond the verify cap retains one full committed forward",()=>{
    const current=graph(),{weights,config}=source(),generic=new Qwen35Model(weights,config,composition());
    const mine=current.makeCache(),theirs=generic.makeCache();
    try {
      using input=ids([prompt(80,71)]);
      expect(new MlxPhaseDispatch(current).known(80)).toBe(new MlxPhaseDispatch(current).known(262144));
      outputDiff("known span 80",new MlxPhaseDispatch(current).known(80)(input,mine) as MlxArray,
        generic.forwardLayers(generic.embed.encode(input),theirs,"committed",undefined,null,80,false),
        current.logitsFromHidden.bind(current),generic.logitsFromHidden.bind(generic));
      stateDiff("known span 80",mine,theirs);
    } finally {disposeAll(mine,theirs);generic.dispose();}
  },180_000);
  test("a whole prompt chunk, a prompt tail and a window wider than eight rows", () => {
    const current = graph(), { weights, config } = source();
    const generic = new Qwen35Model(weights, config, composition());
    const mine = current.makeCache(), theirs = generic.makeCache();
    try {
      {
        using input = ids([prompt(CHUNK, 1)]);
        outputDiff(`prefillChunk.gpu ${CHUNK}`, current.multiSequencePrompt.at(-1)!.operation(input, mine, null) as MlxArray,
          generic.multiSequencePrompt.at(-1)!.operation(input, theirs, null) as MlxArray, current.logits[9]!, generic.logits[9]!);
        stateDiff(`prefillChunk.gpu ${CHUNK}`, mine, theirs);
      }
      {
        using input = ids([prompt(CHUNK, 2)]);
        outputDiff("prompt tail expanded", new MlxPhaseDispatch(current).prompt("tail",1,CHUNK)(input,mine,null) as MlxArray,
          new MlxPhaseDispatch(generic).prompt("tail",1,CHUNK)(input,theirs,null) as MlxArray,current.logits[CHUNK]!,generic.logits[CHUNK]!);
        stateDiff("prompt tail expanded",mine,theirs);
      }
      {
        using input=ids([prompt(CHUNK,3)]);
        outputDiff("verify expanded",current.verify[CHUNK]!(input,mine,null) as MlxArray,
          generic.verify[CHUNK]!(input,theirs,null) as MlxArray,current.logits[CHUNK]!,generic.logits[CHUNK]!);
        stateDiff("verify expanded",mine,theirs);
      }
    } finally { disposeAll(mine, theirs); }
  }, 180_000);

  test("a request adapter overlays the written projection without rebuilding phase bindings", () => {
    const current=graph(),{weights,config}=source(),generic=new Qwen35Model(weights,config,composition());
    const path=`${PREFIX}.layers.0.linear_attn.in_proj_z`;
    const left=current.loraTargets().get(path)!,right=generic.loraTargets().get(path)!;
    const adapter={a:tensor(noise(HIDDEN*2,99,0.02),[HIDDEN,2],Dtype.float32),b:tensor(noise(2*VALUE_DIM,101,0.02),[2,VALUE_DIM],Dtype.float32),rank:2,scale:0.5};
    const operation=current.verify[CHUNK];
    left.adapters=new Map([["synthetic",adapter]]);left.loraState=current.loraState;
    right.adapters=new Map([["synthetic",adapter]]);right.loraState=generic.loraState;
    current.loraState.active=["synthetic"];generic.loraState.active=["synthetic"];
    const mine=current.makeCache(),theirs=generic.makeCache();
    try {
      using input=ids([prompt(CHUNK,90)]);
      outputDiff("request adapter expanded",current.verify[CHUNK]!(input,mine,null) as MlxArray,generic.verify[CHUNK]!(input,theirs,null) as MlxArray,current.logits[CHUNK]!,generic.logits[CHUNK]!);
      stateDiff("request adapter expanded",mine,theirs);
      expect(current.verify[CHUNK]).toBe(operation);
    } finally {
      current.loraState.active=[];generic.loraState.active=[];
      left.adapters=null;left.loraState=null;right.adapters=null;right.loraState=null;
      disposeAll(mine,theirs);generic.dispose();
    }
  },180_000);
});
