import { testGraph, type TestGraph } from "./test-composition";
import { phaseRowLimit } from "../../src/models/graph";
import { MlxPhaseDispatch } from "../../src/generation/phase-dispatch";
import { bindMlxAutoregressiveGraph } from "../../src/generation/bindings/autoregressive";
// Written M1 operations on a small synthetic artifact with all 64 quantization
// layouts. Expanded GPU windows preserve exact hidden, head, and cache tensors
// under every KV scheme at257+rows. Tuned codebook/MMA, dense prompt projections
// and ANE splits are held to fixed-layer tests and current-main artifact gates.
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
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
import type { MlxLayerCapture } from "../../src/contracts/mlx/graph";
import type { DraftProjectionRange } from "../../src/contracts/mlx/draft-target";
import { Affine4RowsLinear, AffineSplitMmaLinear } from "../../src/layers/affine-verify-linear";
import { QuantizedLinear } from "../../src/layers/quantized-linear";

import type { Composition, KvStorageScheme } from "../../src/contracts/portable/composition";
import { anePrograms, Qwen38TrellisM1Max } from "../../src/models/qwen/qwen38-27b-trellis-m1max";
import { AneProgram } from "../../src/kernels/ane/linear";
import { LAYERS } from "./qwen38-layout";
import { bindGraphCaches, composeGraphCaches } from "../../src/models/cache-composition";
import { Qwen35Model } from "../../src/models/qwen/qwen3_5";
import type { TrellisMlp } from "../../src/models/qwen/blocks";

const HIDDEN = 512, INTER = 256, VOCAB = 256, HEADS = 4, KV_HEADS = 2, HEAD_DIM = 256;
const HK = 4, HV = 16, DK = 64, DV = 64, KERNEL = 4;
const KEY_DIM = HK * DK, VALUE_DIM = HV * DV, CONV_DIM = 2 * KEY_DIM + VALUE_DIM;
const PREFIX = "language_model.model";
const CHUNK = 257;

beforeAll(()=>{
  for (const program of anePrograms({hidden:HIDDEN,intermediate:INTER,
    projections:[[HIDDEN,CONV_DIM],[HIDDEN,VALUE_DIM],[VALUE_DIM,HIDDEN],[HIDDEN,HEADS*HEAD_DIM*2]]})) AneProgram.compile(program);
},180_000);

const owned: MlxArray[] = [];
afterAll(() => { for (const array of owned) array.dispose(); });
const keep = (array: MlxArray) => { owned.push(array); return array; };

const noise = (count: number, phase: number, scale: number, shift = 0) => Float32Array.from({ length: count },
  (_, i) => shift + Math.sin(i * 0.37 + phase) * scale + Math.cos(i * 0.11 + 2.3 * phase) * scale);
const tensor = (values: Float32Array, shape: number[], dtype = Dtype.bfloat16): MlxArray => {
  using wide = MlxArray.fromFloat32(values, shape);
  return keep(wide.astype(dtype));
};

/** The synthetic artifact: every layer laid out from the independent artifact layout. */
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
  device: "applegpu_g13s", kv: { kind: "bf16" }, pagedKv: null, draftDepth: { adaptive: true, max: 7 },
  prefillChunkTokens: CHUNK, maxRows: 8,
  vectorGreedySampling: true, mixedPrefill: false, mixedTokenBudget: 256, prefillTailSplit: true,
  grammarJump: false, greedyDraft: false, ...overrides,
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

/** Hidden rows and their logits through the declared head (all owned by the caller and disposed here);
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
type BoundM1=Qwen38TrellisM1Max & {makeCache():Cache[]};
const built = new Map<string, { current: BoundM1; generic: TestGraph<Qwen35Model> }>();
function graphs(scheme: string, overrides: Partial<Composition> = {}): { current: BoundM1; generic: TestGraph<Qwen35Model> } {
  const key = `${scheme} ${JSON.stringify(overrides)}`;
  let found = built.get(key);
  if (!found) {
    const { weights, config } = source();
    const facts = composition({ kv: SCHEMES[scheme]!, ...overrides });
    const current=bindGraphCaches(new Qwen38TrellisM1Max(weights,config),facts);
    const generic=testGraph(new Qwen35Model(weights,config,phaseRowLimit(facts)),facts);
    // Compare operations over identical caches, including the M1 folded reads
    // and block128 KV4 cache now declared by the optimized graph.
    generic.makeCache=()=>composeGraphCaches(current,facts);
    found={current,generic};
    built.set(key, found);
  }
  return found;
}
afterAll(() => { for (const { current,generic } of built.values()) {current.dispose();generic.dispose();} });

describe("the written graph declarations", () => {
  test("universal targets, folded caches, tuned rows, ANE programs and measured draft facts", () => {
    for (const scheme of Object.keys(SCHEMES)) {
      const { current, generic } = graphs(scheme);
      const expected=LAYERS.flatMap((spec,layer)=>{
        const prefix=`${PREFIX}.layers.${layer}`;
        const projections=spec.kind==="deltanet"?["in_proj_qkv","in_proj_z","in_proj_a","in_proj_b","out_proj"]:["q_proj","k_proj","v_proj","o_proj"];
        const block=spec.kind==="deltanet"?"linear_attn":"self_attn";
        return [...projections.map(name=>`${prefix}.${block}.${name}`),...(["gate_proj","up_proj","down_proj"] as const).map(name=>`${prefix}.mlp.${name}`)];
      });
      expect([...current.loraTargets().keys()].sort()).toEqual(expected.sort());
      expect(current.loraTargets().size).toBe(48 * 5 + 16 * 4 + 64 * 3);
      expect((current as { knownSpans?: unknown }).knownSpans).toBeUndefined();
      expect([...current.verifyRoundCosts]).toEqual([[1,1],[2,1.77],[3,1.95],[4,1.98],[5,2],[6,2.03],[7,2.04],[8,2.05],[16,3.46],[32,6.4]]);
      expect(current.wiredWeights).toBe(true);
      expect([...current.adaptiveDraftWidths]).toEqual([["mtp",4]]);
      expect(current.prompt.filter(range => range.unit === "ane").map(({from,to})=>[from,to])).toEqual([
        [224,256],[257,320],[321,384],[385,448],[449,512],[513,576],[577,640],[641,704],[705,768],
        [769,896],[897,1024],[1025,1280],[1281,1536],[1537,1792],[1793,2048],
      ]);
      expect(anePrograms()).toHaveLength(43);
      expect(current.sequence.map(({from,to})=>[from,to])).toEqual([[1,1],[2,3],[4,8],[9,256],[257,512],[513,Infinity]]);
      expect(current.decode[9]).toBe(current.decode[256]);
      expect(current.decode[257]).toBe(current.decode[262144]);
      expect(current.verify[9]).toBe(current.verify[256]);
      expect(current.verify[257]).toBe(current.verify[262144]);
      expect(current.logits[256]).not.toBe(current.logits[257]);
      expect(current.logits[257]).toBe(current.logits[262144]);
      const a = current.makeCache(), b = generic.makeCache();
      try { expect(a.map(cache => `${cache.constructor.name} ${cache.signature()}`)).toEqual(b.map(cache => `${cache.constructor.name} ${cache.signature()}`)); }
      finally { disposeAll(a, b); }
    }
  });

  test("draft context projections keep the stock one-row operation and unbounded split MMA rows",()=>{
    const {current}=graphs("bf16"),table=current.draftTarget([]).draftProjections!;
    expect(current.draftTarget([]).draftProjections).toBe(table);
    const ranges=table as readonly DraftProjectionRange[];
    expect(ranges.map(({from,to})=>[from,to])).toEqual([[1,1],[2,3],[4,Infinity]]);
    using source=ops.randomNormal([64,512],Dtype.bfloat16,0,.05,null);
    const q=ops.quantize(source,64,4),linear=new QuantizedLinear(q.packed,q.scales,q.biases,{bits:4,groupSize:64,mode:"affine"});
    const cases:[number,{forward(x:MlxArray):MlxArray}][]=[[1,linear],[3,new Affine4RowsLinear(linear)],
      [4,new AffineSplitMmaLinear(linear)],[8,new AffineSplitMmaLinear(linear)],[12,new AffineSplitMmaLinear(linear)],[257,new AffineSplitMmaLinear(linear)]];
    try {
      for (const [rows,reference] of cases) {
        using x=ops.randomNormal([1,rows,512],Dtype.bfloat16,0,1,null);
        using actual=ranges.find(range=>range.from<=rows&&range.to>=rows)!.operation(q.packed,q.scales,q.biases)(x);
        using expected=reference.forward(x);
        maxDiff(actual,expected,`draft context rows${rows}`);
      }
    } finally {q.packed.dispose();q.scales.dispose();q.biases.dispose();}
  });

  test("the residual basis its fold record declares, read once at construction", () => {
    const { weights, config } = source();
    const dir = mkdtempSync(join(tmpdir(), "qwen38-m1max-fold-"));
    const quiet = console.error;
    console.error = () => {};
    try {
      const at = { ...config, modelDir: dir } as ModelConfig;
      const declared = (graph: { draftTarget(caches: Cache[]): { residualBasis?: unknown } }) => graph.draftTarget([]).residualBasis;
      const without = new Qwen38TrellisM1Max(weights, at);
      expect(declared(without)).toBeUndefined();
      without.dispose();
      writeFileSync(join(dir, TURBOQUANT_FOLD_FILE), JSON.stringify({ seed: 7, r1: true, hiddenSize: HIDDEN,
        final_gain: Array.from({ length: HIDDEN }, (_, i) => 1 + i / HIDDEN) }));
      const current = new Qwen38TrellisM1Max(weights, at);
      expect(declared(current)).toEqual({ r1Seed: 7, finalGain: Float32Array.from({ length: HIDDEN }, (_, i) => 1 + i / HIDDEN) });
      current.dispose();
    } finally { console.error = quiet; rmSync(dir, { recursive: true, force: true }); }
  });
});

for (const scheme of Object.keys(SCHEMES)) {
  test(`${scheme} KV: expanded GPU windows equal generic Qwen hidden, logits and caches`, () => {
    const { current, generic } = graphs(scheme);
    const mine = current.makeCache(), theirs = generic.makeCache();
    const step = (label: string, totalRows: number, rows: number[][],
      run: (graph: Qwen38TrellisM1Max | Qwen35Model, input: MlxArray, caches: Cache[]) => MlxArray | Promise<MlxArray>) => {
      using input = ids(rows);
      outputDiff(`${scheme} ${label}`, run(current, input, mine) as MlxArray, run(generic, input, theirs) as MlxArray,
        current.logits[totalRows]!, generic.logits[totalRows]!);
      stateDiff(`${scheme} ${label}`, mine, theirs);
    };
    try {
      // The artifact fixes dense prompt projections at257+rows. At512 inputs
      // those GEMMs still split K, so this compact fixture cannot use generic
      // quantized projections as a bit-exact oracle for those prompt bindings.
      step("verify257",257,[prompt(257,1)],(g,input,caches)=>g.verify[257]!(input,caches,null));
      step("verify258 after append",258,[prompt(258,5)],(g,input,caches)=>g.verify[258]!(input,caches,null));
    } finally { disposeAll(mine, theirs); }
  }, 180_000);
}

/** The existing main gate's tolerance: wider MMA reassociates these small
 * synthetic weights. Exact arithmetic is compared to main by artifact digests. */
function relativeDiff(actual:MlxArray,expected:MlxArray):number {
  expect(actual.shape).toEqual(expected.shape);
  using a=actual.astype(Dtype.float32),b=expected.astype(Dtype.float32);
  const x=a.toFloat32(),y=b.toFloat32();
  let worst=0,scale=0;
  for (let i=0;i<x.length;i++) {worst=Math.max(worst,Math.abs(x[i]!-y[i]!));scale=Math.max(scale,Math.abs(y[i]!));}
  return worst/scale;
}

test("tuned9..256 rows retain the current-main first-layer reassociation gate",()=>{
  const {current,generic}=graphs("bf16");
  for (const [phase,rows] of [["tail",13],["tail",40],["tail",256],["verify",12]] as const) {
    const mine=current.makeCache(),theirs=generic.makeCache(),captures:[MlxArray[],MlxArray[]]=[[],[]];
    const capture=(into:MlxArray[]):MlxLayerCapture=>(_,hidden)=>{into.push(ops.contiguous(hidden));};
    try {
      using input=ids([prompt(rows,rows)]);
      const run=(graph:Qwen38TrellisM1Max|Qwen35Model,cache:Cache[],tap:MlxLayerCapture)=>
        phase==="verify"?graph.verify[rows]!(input,cache,tap):graph.multiSequencePrompt.find(range=>range.from<=rows&&range.to>=rows)!.operation(input,cache,tap);
      using actual=run(current,mine,capture(captures[0])) as MlxArray,expected=run(generic,theirs,capture(captures[1])) as MlxArray;
      const worst=Math.max(...captures[0].slice(0,4).map((hidden,layer)=>relativeDiff(hidden,captures[1][layer]!)));
      expect(worst,`${phase} rows${rows}: first4layers`).toBeLessThan(5e-2);
      using logits=current.logits[rows]!(actual),wide=logits.astype(Dtype.float32);
      expect(wide.toFloat32().every(Number.isFinite)).toBe(true);
    } finally {disposeAll(mine,theirs);for (const hidden of captures.flat()) hidden.dispose();}
  }
},180_000);

test("a long known span keeps one window and reaches beyond verify rows",()=>{
  const {current,generic}=graphs("bf16"),mine=current.makeCache(),theirs=generic.makeCache();
  try {
    using input=ids([prompt(300,71)]);
    expect(new MlxPhaseDispatch(current).known(300)).toBe(new MlxPhaseDispatch(current).known(262144));
    outputDiff("known span300",new MlxPhaseDispatch(current).known(300)(input,mine) as MlxArray,
      new MlxPhaseDispatch(generic).known(300)(input,theirs) as MlxArray,current.logitsFromHidden.bind(current),generic.logitsFromHidden.bind(generic));
    stateDiff("known span300",mine,theirs);
  } finally {disposeAll(mine,theirs);}
},180_000);

test("prepared embeddings use the declared window operation and remain borrowed",()=>{
  const {current,generic}=graphs("bf16"),mine=current.makeCache(),theirs=generic.makeCache();
  try {
    using input=ids([prompt(40,72)]),embeddings=current.embed.encode(input);
    const before=Buffer.from(embeddings.rawBytes());
    const actual=bindMlxAutoregressiveGraph(current).prefillEmbeddings!(embeddings,mine,null,input,null,40);
    const expected=bindMlxAutoregressiveGraph(generic).prefillEmbeddings!(embeddings,theirs,null,input,null,40);
    outputDiff("plain embeddings 40",actual,expected,current.logitsFromHidden.bind(current),generic.logitsFromHidden.bind(generic));
    stateDiff("plain embeddings 40",mine,theirs);expect(Buffer.from(embeddings.rawBytes())).toEqual(before);
  } finally {disposeAll(mine,theirs);}
},180_000);

test("request adapters on gate, up and down overlay the fixed MLP without changing phase bindings",()=>{
  const {current,generic}=graphs("bf16"),mine=current.makeCache(),theirs=generic.makeCache();
  const mlp=generic.layers[0]!.mlp as TrellisMlp;
  const right=[mlp.gate,mlp.up,mlp.down];
  const left=[current.loraTargets().get(`${PREFIX}.layers.0.mlp.gate_proj`)!,
    current.loraTargets().get(`${PREFIX}.layers.0.mlp.up_proj`)!,current.loraTargets().get(`${PREFIX}.layers.0.mlp.down_proj`)!];
  const operation=current.verify[257];
  for (let i=0;i<3;i++) {
    const projection=left[i]!;
    const adapter={a:tensor(noise(projection.inFeatures*2,91+i,0.02),[projection.inFeatures,2],Dtype.float32),
      b:tensor(noise(2*projection.outFeatures,94+i,0.02),[2,projection.outFeatures],Dtype.float32),rank:2,scale:0.5};
    projection.adapters=new Map([["synthetic",adapter]]);projection.loraState=current.loraState;
    right[i]!.adapters=new Map([["synthetic",adapter]]);right[i]!.loraState=generic.loraState;
  }
  current.loraState.active=["synthetic"];generic.loraState.active=["synthetic"];
  try {
    using input=ids([prompt(257,91)]);
    outputDiff("request MLP adapters",current.verify[257]!(input,mine,null),generic.verify[257]!(input,theirs,null) as MlxArray,
      current.logitsFromHidden.bind(current),generic.logitsFromHidden.bind(generic));
    stateDiff("request MLP adapters",mine,theirs);expect(current.verify[257]).toBe(operation);
  } finally {
    current.loraState.active=[];generic.loraState.active=[];
    for (const projection of [...left,...right]) {projection.adapters=null;projection.loraState=null;}
    disposeAll(mine,theirs);
  }
},180_000);

test("packed prompt and decode groups preserve hidden, capture order and independent cache states",()=>{
  const {current,generic}=graphs("bf16");
  const mine=[current.makeCache(),current.makeCache()],theirs=[generic.makeCache(),generic.makeCache()];
  const leftCaptures:number[][]=[[],[]],rightCaptures:number[][]=[[],[]];
  using prefill=ids([prompt(257,81)]);
  using decode=ids([prompt(1,82)]);
  const left=[{phase:"window" as const,rows:257,ids:prefill,state:mine[0]!,capture:(layer:number)=>leftCaptures[0]!.push(layer)},
    {phase:"decode" as const,rows:1,ids:decode,state:mine[1]!,capture:(layer:number)=>leftCaptures[1]!.push(layer)}];
  const right=[{...left[0]!,state:theirs[0]!,capture:(layer:number)=>rightCaptures[0]!.push(layer)},
    {...left[1]!,state:theirs[1]!,capture:(layer:number)=>rightCaptures[1]!.push(layer)}];
  let a:MlxArray[]=[],b:MlxArray[]=[];
  try {
    a=current.mixed.at(-1)!.operation(left,258);
    b=generic.mixed.at(-1)!.operation(right,258);
    for (let row=0;row<2;row++) {
      outputDiff(`packed group ${row}`,a[row]!,b[row]!,current.logitsFromHidden.bind(current),generic.logitsFromHidden.bind(generic));
      a[row]=null as unknown as MlxArray;b[row]=null as unknown as MlxArray;
      stateDiff(`packed group ${row}`,mine[row]!,theirs[row]!);
      expect(leftCaptures[row]).toEqual(Array.from({length:65},(_,layer)=>layer));
      expect(leftCaptures[row]).toEqual(rightCaptures[row]);
    }
  } finally {for (const hidden of [...a,...b]) hidden?.dispose();disposeAll(...mine,...theirs);}
},180_000);
