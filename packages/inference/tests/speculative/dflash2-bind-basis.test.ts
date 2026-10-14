// The DFlash 2 drafter loaded in its trained basis and given the target's basis
// when a draft group binds, on small synthetic weights whose 4-bit projections
// fit the simdgroup matrix unit: every proposal holds its depth's tokens, the
// 4-bit drafter runs the matrix unit exactly where the width class of the rows
// names it (main's per-call row rule: five to eight rows), for a draft block and
// for a context projection (fc, each layer's k and v) alike, and a drafter keeps
// the basis it was first bound to. Context chunks are 12 and 2×12 rows, plus a
// commit of 4, 2×5 or 6 rows. The
// rotated Trellis graphs' declaration (the basis their constructor reads once
// from turboquant_fold.json and the ports their draftTarget hands out), over a
// synthetic artifact's graph, carries the basis the record holds, and the
// drafter bound through it proposes exactly what the same basis given directly
// proposes. The drafter's numbers on the real artifact are the teacher-forced
// DFlash 2 blocks the parity suites hash.
import { afterAll, beforeAll, expect, spyOn, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { loadModelConfig } from "../../src/artifacts/config";
import { writeShardedSafetensors, type NamedTensor } from "../../src/artifacts/safetensors-writer";
import { TURBOQUANT_FOLD_FILE } from "../../src/artifacts/turboquant-fold";
import { Weights } from "../../src/artifacts/weights";
import { Qwen35Model } from "../../src/models/qwen/qwen3_5";
import { testComposition } from "../models/test-composition";
import { QWEN38_TRELLIS_M1MAX_GRAPH } from "../../src/models/qwen/qwen38-27b-trellis-m1max";
import { qwenDraftTarget, qwenDraftVocabulary, qwenResidualBasis } from "../../src/models/qwen/media-input";
import { Affine4MmaLinear, Affine4RowsLinear, AffineSplitMmaLinear } from "../../src/layers/affine-verify-linear";
import type { DraftProjection } from "../../src/contracts/mlx/draft-projection";
import { atEveryWidth, byWidth } from "../../src/models/graph";
import { widthClass, type MlxWidthClass } from "../../src/contracts/mlx/graph";
import type { DraftProjectionKernel, ResidualBasis, TargetView } from "../../src/contracts/mlx/draft-target";
import { QuantizedLinear } from "../../src/layers/quantized-linear";
import type { DraftPrefillGroup, GroupedDraftProvider } from "../../src/generation/speculative/source";
import { Dflash2Provider } from "../../src/generation/speculative/sources/dflash2-source";
import { Dflash2Drafter } from "../../src/models/speculative/dflash2";

// Every 4-bit projection fits the matrix unit: K a multiple of 256, N of 8.
const H = 1024, HEADS = 8, KV_HEADS = 2, HEAD_DIM = 128, INTER = 2048, LAYERS = 2, BLOCK = 8;
const TAPS = [1, 3], TARGET_LAYERS = 4, V = 64, RANK = 16, GROUP = 16, TOP_K = 4, MASK = 63;
const CONTEXT = 12, SEED = 7;

let dir: string;
const resident: MlxArray[] = [];
let projection: DraftProjection;
/** Pre-selection logits of each draft, read back as the head returns them. */
let captured: Float32Array[] = [];
let basis: ResidualBasis;

function normal(seed: number): () => number {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
}

const draw = normal(20261010);
function bf16(shape: number[], scale: number, offset = 0): MlxArray {
  const data = new Float32Array(shape.reduce((a, b) => a * b, 1));
  for (let i = 0; i < data.length; i++) data[i] = offset + scale * draw();
  using f = MlxArray.fromFloat32(data, shape);
  return f.astype(Dtype.bfloat16);
}

beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "mlx-bun-dflash2-gate-"));
  writeFileSync(join(dir, "config.json"), JSON.stringify({
    architectures: ["DFlash2DraftModel"], hidden_size: H, num_hidden_layers: LAYERS,
    num_attention_heads: HEADS, num_key_value_heads: KV_HEADS, head_dim: HEAD_DIM, rms_norm_eps: 1e-6,
    rope_parameters: { rope_type: "default", rope_theta: 1_000_000 }, num_target_layers: TARGET_LAYERS,
    dflash_config: { block_size: BLOCK, mask_token_id: MASK, conv_group_size: GROUP, conv_kernel_size: 2,
      selector_top_k: TOP_K, target_layer_ids: TAPS },
  }));
  const tensors: NamedTensor[] = [];
  const add = (name: string, array: MlxArray) => tensors.push({ name, array });
  add("fc.weight", bf16([H, TAPS.length * H], 1 / Math.sqrt(TAPS.length * H)));
  add("hidden_norm.weight", bf16([H], 0.1, 1));
  add("norm.weight", bf16([H], 0.1, 1));
  add("candidate_selector.hidden_projection.weight", bf16([RANK, H], 1 / Math.sqrt(H)));
  add("candidate_selector.predecessor_codebook", bf16([V, RANK], 0.3));
  add("candidate_selector.successor_codebook", bf16([V, RANK], 0.3));
  for (let i = 0; i < LAYERS; i++) {
    const p = `layers.${i}`;
    add(`${p}.input_layernorm.weight`, bf16([H], 0.1, 1));
    add(`${p}.post_attention_layernorm.weight`, bf16([H], 0.1, 1));
    add(`${p}.self_attn.q_proj.weight`, bf16([HEADS * HEAD_DIM, H], 1 / Math.sqrt(H)));
    add(`${p}.self_attn.k_proj.weight`, bf16([KV_HEADS * HEAD_DIM, H], 1 / Math.sqrt(H)));
    add(`${p}.self_attn.v_proj.weight`, bf16([KV_HEADS * HEAD_DIM, H], 1 / Math.sqrt(H)));
    add(`${p}.self_attn.o_proj.weight`, bf16([H, HEADS * HEAD_DIM], 1 / Math.sqrt(HEADS * HEAD_DIM)));
    add(`${p}.self_attn.q_norm.weight`, bf16([HEAD_DIM], 0.1, 1));
    add(`${p}.self_attn.k_norm.weight`, bf16([HEAD_DIM], 0.1, 1));
    add(`${p}.mlp.gate_proj.weight`, bf16([INTER, H], 1 / Math.sqrt(H)));
    add(`${p}.mlp.up_proj.weight`, bf16([INTER, H], 1 / Math.sqrt(H)));
    add(`${p}.mlp.down_proj.weight`, bf16([H, INTER], 1 / Math.sqrt(INTER)));
    for (const site of ["attention_conv", "mlp_conv"]) {
      add(`${p}.${site}.base_kernel`, bf16([2, 2, H], 0.5));
      add(`${p}.${site}.kernel_projection.weight`, bf16([4 * (H / GROUP), H], 0.1 / Math.sqrt(H)));
    }
  }
  writeShardedSafetensors(dir, tensors);
  for (const { array } of tensors) array.dispose();

  // The target: an embedding and a head over the residual basis it declares.
  const table = bf16([V, H], 1), head = bf16([V, H], 1 / Math.sqrt(H));
  const headT = ops.transposeAxes(head, [1, 0]);
  resident.push(table, head, headT);
  projection = {
    embed: { encode: ids => ops.takeAxis(table, ids, 0), scales: { dtype: Dtype.bfloat16 } },
    // One head at every width.
    logits: atEveryWidth((hidden: MlxArray) => {
      const logits = ops.matmul(hidden, headT);
      captured.push(logits.toFloat32());
      return logits;
    }),
  };
  const gain = new Float32Array(H);
  for (let i = 0; i < H; i++) gain[i] = 0.5 + Math.abs(draw());
  basis = Object.freeze({ r1Seed: SEED, finalGain: gain });
});

afterAll(() => {
  for (const array of resident) array.dispose();
  rmSync(dir, { recursive: true, force: true });
});

const targetOf = (residualBasis: ResidualBasis | undefined): TargetView => Object.freeze({
  identity: {}, hiddenLayerTaps: { layerCount: TARGET_LAYERS, projection }, ...(residualBasis ? { residualBasis } : {}),
});

interface Run { tokens: number[][]; logits: Float32Array }

/** Prefill `anchors.length` rows with the same taps, commit `accepted`, then draft each depth. */
function drafts(groups: GroupedDraftProvider, target: TargetView, taps: MlxArray, commit: MlxArray,
  anchors: readonly number[], accepted: readonly number[], depths: readonly number[]): Run[] {
  const B = anchors.length;
  // Projected rows serve prefill, commit and draft from one group.
  const rows = groups.openPrefill({ target, checkpoints: Array(B).fill(null) }) as DraftPrefillGroup & {
    commit(accepted: readonly number[], hidden: MlxArray): void; draft(pending: readonly number[], depth: number): number[][];
  };
  try {
    using tokens = ops.zeros([B, CONTEXT], Dtype.int32);
    rows.prefill(tokens, taps);
    rows.commit(accepted, commit);
    rows.materialize();
    return depths.map(depth => {
      captured = [];
      const proposed = rows.draft(anchors, depth);
      expect(captured.length).toBe(1);
      return { tokens: proposed, logits: captured[0]! };
    });
  } finally { rows.dispose(); }
}

function maxDiff(a: Float32Array, b: Float32Array): number {
  expect(a.length).toBe(b.length);
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    expect(Number.isFinite(a[i]!) && Number.isFinite(b[i]!)).toBe(true);
    worst = Math.max(worst, Math.abs(a[i]! - b[i]!));
  }
  return worst;
}

const DEPTHS = [1, 3, 7];
const CASES = [
  { anchors: [5], accepted: [3], contextOnMatrixUnit: false }, // blocks of 2, 4, 8 rows; commit of 4
  { anchors: [5, 11], accepted: [4, 1], contextOnMatrixUnit: false }, // blocks of 4, 8, 16 rows; commit of 2×5, rows end apart
  { anchors: [5], accepted: [5], contextOnMatrixUnit: true }, // blocks of 2, 4, 8 rows; commit of 6
];

for (const [bits, folded] of [[4, true], [4, false], [0, true], [0, false]] as const) {
  test(`basis at bind (${bits ? "4-bit" : "BF16"}, ${folded ? "R1 fold" : "trained basis"}): every proposal holds its depth, the matrix unit runs for one block width and a commit of five to eight rows, depths 1, 3, 7`, async () => {
    const provider = await Dflash2Provider.load(dir, { bits });
    const mma = spyOn(Affine4MmaLinear.prototype, "forward");
    try {
      const target = targetOf(folded ? basis : undefined);
      for (const { anchors, accepted, contextOnMatrixUnit } of CASES) {
        const B = anchors.length, width = Math.max(...accepted) + 1;
        using taps = bf16([B, CONTEXT, TAPS.length * H], 1);
        using commit = bf16([B, width, TAPS.length * H], 1);
        const calls = mma.mock.calls.length;
        const runs = drafts(provider.grouped, target, taps, commit, anchors, accepted, DEPTHS);
        for (const [i, depth] of DEPTHS.entries()) {
          expect(runs[i]!.logits.every(Number.isFinite)).toBe(true);
          expect(runs[i]!.tokens.every(row => row.length === depth)).toBe(true);
          expect(runs[i]!.logits.length).toBe(B * depth * V);
        }
        // The 4-bit drafter ran the matrix unit for exactly one block width (B·G = 8 rows: q, k, v, o, gate, up
        // and down per layer) and, where the commit appended five to eight rows, for its context projection
        // (fc, then each layer's k and v); the prefill chunks (12 and 24 rows) and the other commits never do.
        expect(mma.mock.calls.length - calls).toBe(bits ? LAYERS * 7 + (contextOnMatrixUnit ? 1 + 2 * LAYERS : 0) : 0);
      }
    } finally {
      mma.mockRestore();
      provider.dispose();
    }
  });
}

test("a drafter keeps the basis and kernels it was first bound to and refuses others", async () => {
  const drafter = await Dflash2Drafter.load(dir, { bits: 4 });
  try {
    drafter.bindTarget(basis, null);
    drafter.bindTarget(basis, null);
    expect(() => drafter.bindTarget(null, null))
      .toThrow("this DFlash 2 drafter is bound to a target with a different residual basis");
    expect(() => drafter.bindTarget({ r1Seed: SEED, finalGain: basis.finalGain }, null))
      .toThrow("this DFlash 2 drafter is bound to a target with a different residual basis");
    expect(() => drafter.bindTarget(basis, declared().table))
      .toThrow("this DFlash 2 drafter is bound to a target with different projection kernels");
  } finally { drafter.dispose(); }
});

/** A target's declared drafter kernels: one per width class, each MLX's quantized matmul over the drafter's weight,
 *  counting the projections built and the forwards run per class. */
function declared() {
  const built = new Map<MlxWidthClass, number>(), runs = new Map<MlxWidthClass, number>();
  const table = byWidth<DraftProjectionKernel>(width => (w, scales, biases) => {
    built.set(width, (built.get(width) ?? 0) + 1);
    const linear = new QuantizedLinear(w, scales, biases, { bits: 4, groupSize: 64, mode: "affine" });
    return x => { runs.set(width, (runs.get(width) ?? 0) + 1); return linear.forward(x); };
  });
  return { table, built, runs };
}

test("a target that declares its composition's kernels runs every 4-bit projection on the one its class names, the drafter's own rule nowhere", async () => {
  const kernels = declared();
  const provider = await Dflash2Provider.load(dir, { bits: 4 });
  const mma = spyOn(Affine4MmaLinear.prototype, "forward");
  try {
    const target = Object.freeze({ ...targetOf(basis), draftProjections: kernels.table });
    // Built once per projection and class when the first group binds: fc and seven per layer.
    using taps = bf16([1, CONTEXT, TAPS.length * H], 1);
    using commit = bf16([1, 6, TAPS.length * H], 1);
    const runs = drafts(provider.grouped, target, taps, commit, [5], [5], DEPTHS);
    for (const [i, depth] of DEPTHS.entries()) expect(runs[i]!.tokens[0]!.length).toBe(depth);
    for (const width of ["one", "twoToThree", "four", "fiveToEight", "wider"] as const) expect(kernels.built.get(width)).toBe(1 + 7 * LAYERS);
    // Prefill (12 rows) and the commit (6) project the context; blocks of 2, 4 and 8 rows draft.
    expect(kernels.runs.get("wider")).toBe(1 + 2 * LAYERS);
    expect(kernels.runs.get("fiveToEight")).toBe(1 + 2 * LAYERS + 7 * LAYERS);
    expect(kernels.runs.get("twoToThree")).toBe(7 * LAYERS);
    expect(kernels.runs.get("four")).toBe(7 * LAYERS);
    expect(mma.mock.calls.length).toBe(0);
  } finally {
    mma.mockRestore();
    provider.dispose();
  }
});

test("context projection runs the matrix unit at five to eight rows and MLX's quantized matmul at every other width; BF16 never runs it", async () => {
  for (const bits of [4, 0]) {
    const drafter = await Dflash2Drafter.load(dir, { bits });
    drafter.bindTarget(null, null);
    const mma = spyOn(Affine4MmaLinear.prototype, "forward");
    try {
      // B × L context rows in every class: 1, 2, 4, 6 (one row of six and two of three), 8, 9 and 12.
      for (const [B, L] of [[1, 1], [1, 2], [1, 4], [1, 6], [2, 3], [1, 8], [1, 9], [2, 6]] as const) {
        const width = widthClass(B * L), calls = mma.mock.calls.length;
        using taps = bf16([B, L, TAPS.length * H], 1);
        using context = drafter.projectContext[width](taps);
        const kv = drafter.projectContextKVRows[width](context, 0);
        ops.evalAll(kv.flatMap(pair => [pair.k, pair.v]));
        for (const pair of kv) { pair.k.dispose(); pair.v.dispose(); }
        // fc, then each layer's k and v.
        expect(mma.mock.calls.length - calls).toBe(bits && width === "fiveToEight" ? 1 + 2 * LAYERS : 0);
      }
    } finally {
      mma.mockRestore();
      drafter.dispose();
    }
  }
});

/** Width of every head, value and MLP projection of the synthetic target. */
const QWEN_DIM = 64;

/** A Qwen 3.5 artifact at the drafter's target geometry (hidden H, TARGET_LAYERS layers, the last one full
 *  attention, V tokens), 4-bit group-64, in the graph's own tensor names. Only its embedding and head run. */
function writeQwenTarget(target: string): void {
  const tensors: NamedTensor[] = [];
  const add = (name: string, array: MlxArray) => tensors.push({ name, array });
  const linear = (path: string, out: number, inFeatures: number) => {
    using w = bf16([out, inFeatures], 1 / Math.sqrt(inFeatures));
    const q = ops.quantize(w, 64, 4);
    ops.evalAll([q.packed, q.scales, q.biases]);
    add(`${path}.weight`, q.packed); add(`${path}.scales`, q.scales); add(`${path}.biases`, q.biases);
  };
  const P = "language_model.model";
  linear(`${P}.embed_tokens`, V, H);
  linear("language_model.lm_head", V, H);
  add(`${P}.norm.weight`, bf16([H], 0.1, 1));
  for (let i = 0; i < TARGET_LAYERS; i++) {
    const p = `${P}.layers.${i}`;
    add(`${p}.input_layernorm.weight`, bf16([H], 0.1, 1));
    add(`${p}.post_attention_layernorm.weight`, bf16([H], 0.1, 1));
    if (i < TARGET_LAYERS - 1) {
      linear(`${p}.linear_attn.in_proj_qkv`, 3 * QWEN_DIM, H);
      linear(`${p}.linear_attn.in_proj_z`, QWEN_DIM, H);
      linear(`${p}.linear_attn.in_proj_a`, 1, H);
      linear(`${p}.linear_attn.in_proj_b`, 1, H);
      linear(`${p}.linear_attn.out_proj`, H, QWEN_DIM);
      add(`${p}.linear_attn.conv1d.weight`, bf16([3 * QWEN_DIM, 4, 1], 0.2));
      add(`${p}.linear_attn.A_log`, bf16([1], 0.1, 0.5));
      add(`${p}.linear_attn.dt_bias`, bf16([1], 0.1, 1));
      add(`${p}.linear_attn.norm.weight`, bf16([QWEN_DIM], 0.05, 1));
    } else {
      linear(`${p}.self_attn.q_proj`, 2 * QWEN_DIM, H);
      linear(`${p}.self_attn.k_proj`, QWEN_DIM, H);
      linear(`${p}.self_attn.v_proj`, QWEN_DIM, H);
      linear(`${p}.self_attn.o_proj`, H, QWEN_DIM);
      add(`${p}.self_attn.q_norm.weight`, bf16([QWEN_DIM], 0.1, 1));
      add(`${p}.self_attn.k_norm.weight`, bf16([QWEN_DIM], 0.1, 1));
    }
    linear(`${p}.mlp.gate_proj`, QWEN_DIM, H);
    linear(`${p}.mlp.up_proj`, QWEN_DIM, H);
    linear(`${p}.mlp.down_proj`, H, QWEN_DIM);
  }
  writeShardedSafetensors(target, tensors);
  for (const { array } of tensors) array.dispose();
  writeFileSync(join(target, "config.json"), JSON.stringify({
    model_type: "qwen3_5_text", hidden_size: H, num_hidden_layers: TARGET_LAYERS, num_attention_heads: 1,
    num_key_value_heads: 1, head_dim: QWEN_DIM, vocab_size: V, intermediate_size: QWEN_DIM, rms_norm_eps: 1e-6,
    max_position_embeddings: 512, tie_word_embeddings: false, eos_token_id: 1, linear_num_value_heads: 1,
    linear_num_key_heads: 1, linear_key_head_dim: QWEN_DIM, linear_value_head_dim: QWEN_DIM, linear_conv_kernel_dim: 4,
    full_attention_interval: TARGET_LAYERS, attn_output_gate: true,
    rope_parameters: { rope_type: "default", rope_theta: 1e7, partial_rotary_factor: 0.25 },
    quantization: { bits: 4, group_size: 64 },
  }));
}

test("the rotated Trellis graphs' declaration carries the basis its turboquant_fold.json records, and the drafter bound through it proposes exactly what that basis given directly proposes", async () => {
  const target = mkdtempSync(join(tmpdir(), "mlx-bun-dflash2-target-"));
  const foldPath = join(target, TURBOQUANT_FOLD_FILE);
  const record = { source: "synthetic", generatedAt: "2026-10-11T00:00:00.000Z", seed: SEED, r1: true, r2: false,
    hiddenSize: H, gamma_convention: "stored+1 folded in; folded norms written as zeros" };
  const errors = spyOn(console, "error").mockImplementation(() => {});
  const fileLines = () => errors.mock.calls.map(args => String(args[0])).filter(line => line.includes(foldPath));
  let weights: Weights | null = null;
  const providers: Dflash2Provider[] = [];
  try {
    writeQwenTarget(target);
    const config = await loadModelConfig(target);
    weights = await Weights.open(target);

    // A Trellis graph's declaration: the basis its constructor reads once from the fold record
    // (`qwenResidualBasis`) and the ports every draftTarget call hands out over the graph (`qwenDraftTarget`),
    // here over the Qwen 3.5 graph of the synthetic artifact, whose embedding and head the drafter reads.
    const trellisTarget = (model: Qwen35Model) => {
      const residualBasis = qwenResidualBasis(QWEN38_TRELLIS_M1MAX_GRAPH, target);
      return { draftTarget: (_caches: unknown[]): TargetView => qwenDraftTarget(model, residualBasis) };
    };
    // A record without final_gain declares nothing and names the file once, at construction; so does a missing record.
    writeFileSync(foldPath, JSON.stringify(record));
    const generic = new Qwen35Model(weights, config, testComposition());
    expect("residualBasis" in trellisTarget(generic).draftTarget([])).toBe(false);
    expect(fileLines()).toEqual([`[graph] ${QWEN38_TRELLIS_M1MAX_GRAPH}: ${foldPath} records no final_gain; DFlash 2 drafts ` +
      "unrotated (output exact, acceptance collapses) until it records the final-norm gain the fold moved into lm_head"]);
    unlinkSync(foldPath);
    expect("residualBasis" in trellisTarget(generic).draftTarget([])).toBe(false);
    expect(fileLines()).toHaveLength(2);
    expect(fileLines()[1]).toContain(`${foldPath} is missing;`);
    // A stream the record says is not rotated has no basis to declare and nothing to report.
    writeFileSync(foldPath, JSON.stringify({ ...record, r1: false, final_gain: Array.from(basis.finalGain) }));
    expect("residualBasis" in trellisTarget(generic).draftTarget([])).toBe(false);

    // The record with final_gain: the Trellis declaration carries it; the generic graph on the same artifact declares nothing.
    writeFileSync(foldPath, JSON.stringify({ ...record, final_gain: Array.from(basis.finalGain) }));
    const model = new Qwen35Model(weights, config, testComposition());
    const graph = trellisTarget(model);
    expect("residualBasis" in model.draftTarget([])).toBe(false);
    expect(fileLines()).toHaveLength(2);
    unlinkSync(foldPath); // read once, at construction: the declaration no longer depends on the file
    const head = model.logitsFromHidden.bind(model);
    spyOn(model, "logitsFromHidden").mockImplementation(hidden => {
      const logits = head(hidden);
      captured.push(logits.toFloat32());
      return logits;
    });
    const declared = graph.draftTarget([]);
    expect(declared.residualBasis).toEqual({ r1Seed: SEED, finalGain: basis.finalGain });
    expect(graph.draftTarget([]).residualBasis).toBe(declared.residualBasis!);
    const direct: TargetView = Object.freeze({ identity: {}, hiddenLayerTaps: declared.hiddenLayerTaps,
      residualBasis: Object.freeze({ r1Seed: SEED, finalGain: Float32Array.from(basis.finalGain) }) });
    const unrotated: TargetView = Object.freeze({ identity: {}, hiddenLayerTaps: declared.hiddenLayerTaps });

    const load = async () => { const provider = await Dflash2Provider.load(dir, { bits: 4 }); providers.push(provider); return provider; };
    const [throughGraph, given, trained] = [await load(), await load(), await load()];
    for (const { anchors, accepted } of CASES) {
      const B = anchors.length, width = Math.max(...accepted) + 1;
      using taps = bf16([B, CONTEXT, TAPS.length * H], 1);
      using commit = bf16([B, width, TAPS.length * H], 1);
      const run = (provider: Dflash2Provider, view: TargetView) =>
        drafts(provider.grouped, view, taps, commit, anchors, accepted, DEPTHS);
      const bound = run(throughGraph, declared), expected = run(given, direct), control = run(trained, unrotated);
      for (const [i, depth] of DEPTHS.entries()) {
        expect(maxDiff(bound[i]!.logits, expected[i]!.logits)).toBe(0);
        expect(bound[i]!.tokens).toEqual(expected[i]!.tokens);
        expect(bound[i]!.tokens.every(row => row.length === depth)).toBe(true);
        expect(bound[i]!.logits.length).toBe(B * depth * V);
        // Negative control: the same drafter in its trained basis drafts differently.
        expect(maxDiff(bound[i]!.logits, control[i]!.logits)).toBeGreaterThan(0);
      }
    }
  } finally {
    errors.mockRestore();
    for (const provider of providers) provider.dispose();
    weights?.dispose();
    rmSync(target, { recursive: true, force: true });
  }
});

/** Each row's top `k` token ids of logits [rows × V] read back flat. */
function topIds(logits: Float32Array, k: number): number[] {
  const ids: number[] = [];
  for (let row = 0; row < logits.length / V; row++) {
    const scores = Array.from(logits.subarray(row * V, (row + 1) * V), (score, id) => ({ score, id }));
    ids.push(...scores.sort((a, b) => b.score - a.score).slice(0, k).map(entry => entry.id));
  }
  return ids;
}

test("a target that declares a draft vocabulary drafts among its rows: a list holding every candidate of the full head proposes the full head's drafts, any list only its own ids", async () => {
  const provider = await Dflash2Provider.load(dir, { bits: 4 });
  try {
    const { anchors, accepted } = CASES[1]!;
    const B = anchors.length, width = Math.max(...accepted) + 1;
    using taps = bf16([B, CONTEXT, TAPS.length * H], 1);
    using commit = bf16([B, width, TAPS.length * H], 1);
    const full = drafts(provider.grouped, targetOf(basis), taps, commit, anchors, accepted, DEPTHS);
    // The head over a list: the full head's logits at the listed columns (captured whole, as the full head is).
    const listedTarget = (list: readonly number[]): TargetView => {
      using index = ops.fromInt32([...list], [list.length]);
      const ids = index.astype(Dtype.uint32), columns = ops.fromInt32([...list], [list.length]);
      resident.push(ids, columns);
      const vocabulary = { ids, logits: atEveryWidth((hidden: MlxArray) => {
        using logits = projection.logits.one(hidden);
        return ops.takeAxis(logits, columns, -1);
      }) };
      return Object.freeze({ identity: {}, residualBasis: basis,
        hiddenLayerTaps: { layerCount: TARGET_LAYERS, projection: { ...projection, vocabulary } } });
    };
    const candidates = new Set(full.flatMap(run => topIds(run.logits, TOP_K)));
    const holding = [...new Set([...candidates, 0, V - 2])].sort((a, b) => a - b);
    expect(holding.length).toBeLessThan(V);
    const listed = drafts(provider.grouped, listedTarget(holding), taps, commit, anchors, accepted, DEPTHS);
    for (const [i, run] of listed.entries()) expect(run.tokens).toEqual(full[i]!.tokens);
    // Every other id: whatever the full head would draft, only listed ids are proposed.
    const even = Array.from({ length: V / 2 }, (_, i) => 2 * i);
    const restricted = drafts(provider.grouped, listedTarget(even), taps, commit, anchors, accepted, DEPTHS);
    for (const [i, depth] of DEPTHS.entries())
      expect(restricted[i]!.tokens.every(row => row.length === depth && row.every(id => id % 2 === 0))).toBe(true);
    expect(full.some(run => run.tokens.flat().some(id => id % 2 === 1))).toBe(true);
  } finally { provider.dispose(); }
});

test("the Qwen graphs' draft vocabulary is the head's listed rows on the head's layers, their logits the full head's at those rows", () => {
  const artifact = mkdtempSync(join(tmpdir(), "mlx-bun-draft-vocabulary-"));
  const OUT = 256, owned: { dispose(): void }[] = [];
  const list = Array.from({ length: 96 }, (_, i) => (i * 37 + 11) % OUT);
  try {
    mkdirSync(join(artifact, "mtp"));
    writeFileSync(join(artifact, "mtp", "draft_vocab.json"), JSON.stringify({ ids: [...list, list[0]] }));
    using weight = bf16([OUT, H], 1 / Math.sqrt(H));
    const q = ops.quantize(weight, 64, 4);
    const head = new QuantizedLinear(q.packed, q.scales, q.biases, { bits: 4, groupSize: 64, mode: "affine" });
    owned.push(q.packed, q.scales, q.biases);
    // The M1 Max head's layers per width class.
    const kernels = (linear: QuantizedLinear) => {
      const rows = new Affine4RowsLinear(linear), split = new AffineSplitMmaLinear(linear);
      return Object.freeze({ one: (x: MlxArray) => linear.forward(x), twoToThree: (x: MlxArray) => rows.forward(x),
        four: (x: MlxArray) => split.forward(x), fiveToEight: (x: MlxArray) => split.forward(x), wider: (x: MlxArray) => split.forward(x) });
    };
    expect(() => qwenDraftVocabulary(head, artifact, "mtp/draft_vocab.json", 200, kernels, owned)).toThrow(/outside the target vocabulary 200/);
    const vocabulary = qwenDraftVocabulary(head, artifact, "mtp/draft_vocab.json", OUT, kernels, owned);
    const sorted = [...new Set(list)].sort((a, b) => a - b);
    expect(vocabulary.ids.dtype).toBe(Dtype.uint32);
    expect(vocabulary.ids.toIntTokens()).toEqual(sorted);
    const fullHead = kernels(head);
    using columns = ops.fromInt32(sorted, [sorted.length]);
    for (const rows of [1, 2, 3, 4, 7, 12]) {
      const width = widthClass(rows);
      using hidden = bf16([rows, H], 1);
      using all = fullHead[width](hidden);
      using taken = ops.takeAxis(all, columns, -1);
      using expected = ops.contiguous(taken);
      using actual = vocabulary.logits[width](hidden);
      expect(actual.shape).toEqual([rows, sorted.length]);
      expect(maxDiff(actual.toFloat32(), expected.toFloat32())).toBe(0);
    }
  } finally {
    for (const array of owned) array.dispose();
    rmSync(artifact, { recursive: true, force: true });
  }
});
