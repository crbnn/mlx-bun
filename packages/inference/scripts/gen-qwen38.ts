// Qwen3.8-27B Trellis graph generator: compiles one composition (the artifact's
// graph fingerprint, the GPU family, the KV scheme its cache lego serves, the
// draft depth, prefill chunk size and rows per forward it was gated for, and
// whether the Neural Engine bridge is part of it) into an entire graph file:
// every layer's blocks for every phase and width class, chosen from the
// artifact's quant table and the device's kernel table, written out as data and
// straight-line construction. The graph extends nothing and calls no other
// graph.
//
//   bun packages/inference/scripts/gen-qwen38.ts <source> <composition> [--out <dir>]
//   bun packages/inference/scripts/gen-qwen38.ts --describe <snapshot-dir>
//
// <source> is an artifact snapshot directory (config.json, the safetensors index
// and the shard headers; no tensor bytes are read) or the description JSON that
// `--describe` prints for one. A change to the blocks, layers, kernels or this
// template is a regeneration, never a hand edit: tests/models/qwen38-generated-sync.test.ts
// regenerates every composition from tests/fixtures/qwen38-graphs and fails when a
// committed file differs. Where a regenerated graph runs the generic blocks, its identity test holds it to the
// generic Qwen 3.5 graph (tests/models/qwen38-m4pro-identity.test.ts, qwen38-m1max-identity.test.ts); its row
// kernels are the real-artifact parity suite's (tests/parity/qwen-trellis-parity.test.ts) and the app digests'.

import { parseArgs } from "node:util";
import { loadModelConfig, quantFor } from "../src/artifacts/config";

/** The facts that fix one generated graph. The fingerprint and device choose the
 * layer and kernel tables written into the file; `kv` names the scheme whose
 * cache lego `makeCache` composes, or every scheme; `mixedGateUp` chooses the
 * mixed-width gate/up layer where the device's table names the reduce kernels;
 * `verifyRoundCosts` are the numbers measured for this composition on its host,
 * `wiredWeights` a residency it measured there.
 * `draftDepth`, `prefillChunkTokens`, `maxRows` and `aneBridge` record what the
 * file was generated and gated for (the device's table writes the Neural Engine
 * chunk operation, which the constructor builds when the record it is handed has
 * the bridge). Its exported `COMPOSITION`, the loader's table entry, constrains
 * the fingerprint, the device and, when the composition names them, the KV
 * schemes (bf16 and `kv`) and the row bound; the constructor builds from the
 * record it is handed. */
export interface Qwen38Composition {
  /** Output file stem under src/models/qwen. */
  readonly stem: string;
  /** The graph id the class reports. */
  readonly graph: string;
  /** The exported constant holding `graph`. */
  readonly graphConstant: string;
  readonly className: string;
  /** Human name of the artifact, for the header. */
  readonly artifact: string;
  readonly fingerprint: string;
  readonly device: string;
  /** bf16 and one affine scheme, whose cache lego the device's table names (`kvLego`); or `"every"`: the table entry
   *  constrains none, the scheme of the device's `kvLego` composes that lego and each other scheme its own
   *  (`composedFullAttentionCache`). */
  readonly kv: { readonly kind: "affine"; readonly bits: number; readonly groupSize: number } | "every";
  readonly draftDepth: { readonly adaptive: true; readonly max: number } | number;
  readonly prefillChunkTokens: number;
  /** The most rows per forward the table entry admits; null admits any number. */
  readonly maxRows: number | null;
  readonly aneBridge: boolean;
  /** Layers whose gate and up widths differ, at one row, on the codebook kernel. `"split"`, the default and measured
   *  best: `CodebookGateUpRow` with MLX's compiled-SwiGLU tail, bit-identical to the two projections plus a separate
   *  SwiGLU and one kernel instead of three. `"fused"`: the same kernel with the same-width kernel's float32 sigmoid
   *  (last bits differ). `"off"`: those layers keep the generic blocks' two projections and separate SwiGLU, the A/B
   *  control. Only a device whose table names a layer that takes the tail (`TAILED`) reads it. */
  readonly mixedGateUp?: "split" | "fused" | "off";
  /** A verify round's cost by its rows (draft depth + 1) relative to one plain decode step, measured for this
   *  composition on `host`: the DFlash 2 drafter's round time at each fixed depth over the plain decode step's time,
   *  on this graph and artifact (scripts/bench-mtp.ts, greedy). The scheduler's adaptive draft gate weighs them
   *  against the decayed acceptance rate. Absent: the graph declares none, and `--num-draft-tokens adaptive` is
   *  refused on it. */
  readonly verifyRoundCosts?: { readonly host: string; readonly costs: readonly (readonly [number, number])[] };
  /** The graph declares its weights wired for each generation (`MlxModelMemory.wiredWeights`), measured for this
   *  composition on its host, where their share of the recommended working set alone would leave them unwired.
   *  Absent: the share decides. */
  readonly wiredWeights?: true;
}

export const QWEN38_COMPOSITIONS: Readonly<Record<string, Qwen38Composition>> = Object.freeze({
  "qwen38-27b-trellis-m4pro": {
    stem: "qwen38-27b-trellis-m4pro",
    graph: "qwen3.8-27b-trellis-3.2bpw-m4pro",
    graphConstant: "QWEN38_TRELLIS_M4PRO_GRAPH",
    className: "Qwen38TrellisM4Pro",
    artifact: "mlx-bun/Qwen3.8-27B-Trellis-3.2bpw",
    fingerprint: "cfa205c8f5af046a",
    device: "applegpu_g16s",
    kv: { kind: "affine", bits: 4, groupSize: 64 },
    draftDepth: { adaptive: true, max: 7 },
    prefillChunkTokens: 2048,
    maxRows: 8,
    aneBridge: true,
    // Measured 2026-10-09 on the pre-refactor tree: DFlash 2 rounds at depths 1..7 over plain decode's 68.5 ms per
    // token, bf16 KV, greedy; six to eight rows (131.8 to 137.5 ms, not monotone) declared at the depth-7 plateau.
    verifyRoundCosts: { host: "M4 Pro 24 GB", costs: [[1, 1], [2, 1.3], [3, 1.46], [4, 1.68], [5, 1.82], [6, 1.92], [7, 1.93], [8, 1.94]] },
  },
  // This artifact on the M1 Max GPU, under every KV scheme and any number of rows per forward.
  "qwen38-27b-trellis-m1max": {
    stem: "qwen38-27b-trellis-m1max",
    graph: "qwen3.8-27b-trellis-3.2bpw-m1max",
    graphConstant: "QWEN38_TRELLIS_M1MAX_GRAPH",
    className: "Qwen38TrellisM1Max",
    artifact: "mlx-bun/Qwen3.8-27B-Trellis-3.2bpw",
    fingerprint: "cfa205c8f5af046a",
    device: "applegpu_g13s",
    kv: "every",
    draftDepth: { adaptive: true, max: 7 },
    prefillChunkTokens: 2048,
    maxRows: null,
    aneBridge: true,
    mixedGateUp: "split",
    // Measured 2026-10-13 on this graph and artifact, with the one-row codebook kernels' bit-field reads, the split-K
    // affine kernels at four to eight rows and the drafter's own kernels this graph declares: scripts/bench-mtp.ts,
    // the DFlash 2 drafter at depths 1..7 against the depth-0 plain arm (54.5-55.9 ms per token), bf16 KV, greedy,
    // two prompts, 128 tokens, four sessions of two repetitions, whose ratios span under 2.5% at every width.
    verifyRoundCosts: { host: "M1 Max 32 GB", costs: [[1, 1], [2, 2.01], [3, 2.42], [4, 2.56], [5, 2.69], [6, 2.78], [7, 2.87], [8, 2.96]] },
    // Measured 2026-10-13 on the M1 Max 32 GB, where the 11.96e9 B weights are under half the 26.8e9 B recommended
    // set: wired, bit-identical, paired A/B, GPU prefill 1k/8k 136/142 -> 143/145 tok/s; with the Neural Engine
    // split 171/173 -> 172/177 and the first GPU step after the split 0.15-1.4 s -> 61-69 ms; one-row decode
    // unchanged (60.0 ms).
    wiredWeights: true,
  },
});

export const GENERATED_DIR = new URL("../src/models/qwen/", import.meta.url).pathname;

/** One decoder layer as the quant table and the stored shapes give it. */
export interface Qwen38LayerSpec {
  readonly kind: "deltanet" | "attention";
  /** Affine width of each projection, in `DELTANET_PROJECTIONS` or `ATTENTION_PROJECTIONS` order. */
  readonly projections: readonly number[];
  readonly gate: number;
  readonly up: number;
  readonly down: number;
  readonly downLayout: "rowMajor" | "k3Interleaved";
}

/** The widths a layer's Neural Engine programs are built at: the MLP's hidden and
 * intermediate widths (config) and each projection's input and output width by
 * name (its stored weight and quantization width). Every layer of a kind has the
 * same widths. */
export interface Qwen38Widths {
  readonly hidden: number;
  readonly intermediate: number;
  readonly projections: Readonly<Record<string, readonly [number, number]>>;
}

export interface Qwen38Description {
  /** Where the description was read: repo@revision. */
  readonly source: string;
  readonly fingerprint: string;
  readonly widths: Qwen38Widths;
  readonly layers: readonly Qwen38LayerSpec[];
}

export const DELTANET_PROJECTIONS = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"] as const;
export const ATTENTION_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"] as const;

/** The width classes up to eight rows (`MlxWidthTable` without `wider`, which is the generic blocks). */
const NARROW = ["one", "twoToThree", "four", "fiveToEight"] as const;
type Narrow = (typeof NARROW)[number];
/** The rows of each class, fewest and most. */
const CLASS_ROWS: Readonly<Record<Narrow, readonly [number, number]>> = { one: [1, 1], twoToThree: [2, 3], four: [4, 4], fiveToEight: [5, 8] };
const ROW_WORDS = ["", "one", "two", "three", "four", "five", "six", "seven", "eight"];
const rowsText = ([fewest, most]: readonly [number, number]) => fewest === most
  ? `${ROW_WORDS[fewest]} row${fewest === 1 ? "" : "s"}` : `${ROW_WORDS[fewest]} to ${ROW_WORDS[most]} rows`;

/** One width class's one-kernel MLP layers. */
interface MlpLayers {
  readonly sameGateUp: string; readonly mixedGateUp: string; readonly rowMajorDown: string; readonly k3InterleavedDown: string;
}

/** Each device's kernels and how its operations evaluate: the one-kernel layer of every width class. `null`
 * projections keep the generic blocks' QuantizedLinear (MLX's own qmv/qmm); a `null` MLP class runs the generic
 * blocks' TrellisMlp at that class. */
interface DeviceKernels {
  readonly label: string;
  readonly mlp: Readonly<Record<Narrow, MlpLayers | null>>;
  readonly projections: Readonly<Record<Narrow, { readonly 3: string; readonly 4: string } | null>>;
  /** The head of each class (`null`: QuantizedLinear); `one` and `wider` keep QuantizedLinear. `null`: QuantizedLinear
   *  at every width. */
  readonly head: Readonly<Record<Exclude<Narrow, "one">, string | null>> | null;
  /** Submit the graph every N layers at decode and verify widths up to 8; `null`: those operations are one lazy graph. */
  readonly asyncEvery: number | null;
  /** Verify windows above eight rows: each layer and its recurrent state evaluated before the next (`bounded`), or
   *  one lazy graph (`lazy`). */
  readonly widerVerify: "bounded" | "lazy";
  /** Known tokens append through the committed read on the generic blocks, and the graph declares `knownSpans`;
   *  false: as windows on the block set of their width, one lazy graph. */
  readonly knownSpans: boolean;
  /** Mixed rows: each group on its own phase's operation for its width (`perGroup`), or the generic blocks with the
   *  prompt and decode groups' MLP rows packed into one call (`packed`). */
  readonly mixed: "perGroup" | "packed";
  /** The cache lego the affine scheme it serves (`KV_LEGOS`) composes, the scheme of a composition naming one or
   *  that scheme among every scheme, and the tokens per threadgroup of its one-row decode kernel (absent: the lego's
   *  own); `null`: each scheme composes its own lego (`composedFullAttentionCache`). */
  readonly kvLego: { readonly name: string; readonly decodeBlock?: number } | null;
  /** The causal window read of the composition's bf16 caches, a `DenseRead` of state/attention-read; `null`: the
   *  stock fused SDPA (`sdpaRead`). */
  readonly bf16WindowRead: string | null;
  /** The Neural Engine operations: the channel fraction, the split layers, and the row counts their programs are built
   *  for (ascending), the smallest of which a one-sequence prompt piece of `minRows` positions or more runs. `affine`
   *  splits the named `projections`; none (`null`, `[]`): the MLP alone is split. */
  readonly ane: { readonly fraction: number; readonly minRows: number; readonly rows: readonly number[];
    readonly rowMajorMlp: string; readonly k3InterleavedMlp: string; readonly affine: string | null; readonly projections: readonly string[] } | null;
  /** The attention and DeltaNet projections of the prompt operations (prefillChunk.gpu, prefillTail.wider, and the
   *  Neural Engine operations' projections it does not split), when the composition mounts no adapters: a layer of
   *  `PROMPT_PROJECTION_MODULES`; `null` keeps the generic blocks' QuantizedLinear. */
  readonly promptProjection: string | null;
  /** The layer a bound drafter's own 4-bit projections run at each width class of their rows (`null` at a class:
   *  QuantizedLinear, as at \`wider\`), declared on the graph's draft target (\`TargetView.draftProjections\`); \`null\`:
   *  the graph declares none and the drafter keeps its own rule. */
  readonly drafter: Readonly<Record<Narrow, string | null>> | null;
  /** Every operation above eight rows of at most \`maxRows\` rows (decode and verify windows, prompt tails, known-token
   *  appends, a mixed forward's packed MLP rows, the head) on these layers, which take any number of rows: the MLP's,
   *  the attention and DeltaNet projections' and the head's. More rows run the generic blocks. \`maxRows\` is the
   *  boundary measured on the whole forward. \`null\`: every operation above eight rows runs the generic blocks. */
  readonly wider: { readonly maxRows: number; readonly mlp: MlpLayers; readonly projections: { readonly 3: string; readonly 4: string };
    readonly head: string } | null;
}

/** The affine scheme each KV cache lego of state/bf16-first-kv serves. */
const KV_LEGOS: Readonly<Record<string, { readonly bits: number; readonly groupSize: number }>> = Object.freeze({
  Bf16FirstKv4Head256Cache: { bits: 4, groupSize: 64 },
});

/** The module under src/layers each prompt projection layer lives in. */
const PROMPT_PROJECTION_MODULES: Readonly<Record<string, string>> = Object.freeze({ DequantizedLinear: "dequantized-linear" });

/** The module under src/layers each one-kernel MLP layer lives in. */
const LAYER_MODULES: Readonly<Record<string, string>> = Object.freeze({
  FactoredGateUpRow: "trellis-gate-up", FactoredGateUpRows: "trellis-gate-up", FactoredMixedGateUp: "trellis-gate-up",
  MmaGateUp: "trellis-gate-up", MmaGateUpWide: "trellis-gate-up",
  FactoredDownRow: "trellis-down", FactoredDownRows: "trellis-down", FactoredDownK3iRow: "trellis-down",
  FactoredDownK3iRows: "trellis-down", MmaDown: "trellis-down", MmaDownK3i: "trellis-down",
  MmaDownWide: "trellis-down", MmaDownWideK3i: "trellis-down",
  CodebookGateUpRow: "trellis-reduce-gate-up", CodebookDownRow: "trellis-scatter-down", CodebookDownK3iRow: "trellis-scatter-down",
});

/** Gate/up layers that take the composition's mixed-width SwiGLU tail (`mixedGateUp`) as their third argument. */
const TAILED = new Set(["CodebookGateUpRow"]);

const KERNELS: Readonly<Record<string, DeviceKernels>> = Object.freeze({
  applegpu_g16s: {
    label: "M4 Pro GPU",
    mlp: {
      one: { sameGateUp: "FactoredGateUpRow", mixedGateUp: "FactoredMixedGateUp", rowMajorDown: "FactoredDownRow", k3InterleavedDown: "FactoredDownK3iRow" },
      twoToThree: { sameGateUp: "FactoredGateUpRows", mixedGateUp: "FactoredMixedGateUp", rowMajorDown: "FactoredDownRows", k3InterleavedDown: "FactoredDownK3iRows" },
      // Measured on M4 Pro: at four rows the matrix-unit down overtakes the shared-row one; gate/up stays shared-row.
      four: { sameGateUp: "FactoredGateUpRows", mixedGateUp: "FactoredMixedGateUp", rowMajorDown: "MmaDown", k3InterleavedDown: "MmaDownK3i" },
      fiveToEight: { sameGateUp: "MmaGateUp", mixedGateUp: "MmaGateUp", rowMajorDown: "MmaDown", k3InterleavedDown: "MmaDownK3i" },
    },
    projections: {
      one: null,
      twoToThree: { 3: "Affine3RowsLinear", 4: "Affine4RowsLinear" },
      four: { 3: "Affine3RowsLinear", 4: "Affine4RowsLinear" },
      fiveToEight: { 3: "Affine3MmaLinear", 4: "Affine4MmaLinear" },
    },
    head: { twoToThree: "Affine4RowsLinear", four: "Affine4RowsLinear", fiveToEight: "Affine4MmaLinear" },
    asyncEvery: 4,
    widerVerify: "bounded",
    knownSpans: true,
    mixed: "perGroup",
    kvLego: { name: "Bf16FirstKv4Head256Cache" },
    bf16WindowRead: null,
    // Main's Neural Engine prefill plan (9593d881, from a8a5ba45): every one-sequence prompt piece of 640 to 2048
    // positions, a chunk or a tail, on the smallest of the programs built every 512 positions up to 2048.
    ane: { fraction: 0.55, minRows: 640, rows: [1024, 1536, 2048],
      rowMajorMlp: "AneTrellisMlpSplit", k3InterleavedMlp: "AneTrellisMlpSplitK3i", affine: "AneAffineSplit",
      projections: ["in_proj_z", "out_proj", "o_proj"] },
    promptProjection: null,
    drafter: null,
    wider: null,
  },
  // The M1 Max GPU's kernels and evaluation, measured there per class (forward + head, one sequence, bf16 KV, each
  // layer's per-call shader time and the whole forward): the threadgroup-codebook gate/up and down kernels at one row,
  // with MLX's QuantizedLinear; at two to three rows the factored kernels and the affine row kernels, which beat the
  // matrix-unit ones there; at four the matrix-unit gate/up and affine set (the shared-row gate/up and affine row
  // kernels fall behind at four rows here) with the factored down; at five to eight the matrix-unit set, about 5.6x
  // faster than the generic blocks' prefill tiles and 1.2x than QuantizedLinear for the projections and head. Graph
  // submission every 4 layers takes about 4 ms off every width (2 and 8 within noise of it); wider verify windows one
  // lazy graph (1.8% faster than bounded at 12 and 16 rows); mixed rows packed on the generic blocks; known tokens as
  // windows (the committed read's attention boundaries are qualified on the M4 Pro only); prompt pieces with their
  // projections on the dense GEMM and, from 640 positions, their MLP split with the Neural Engine.
  applegpu_g13s: {
    label: "M1 Max GPU",
    mlp: {
      one: { sameGateUp: "CodebookGateUpRow", mixedGateUp: "CodebookGateUpRow", rowMajorDown: "CodebookDownRow", k3InterleavedDown: "CodebookDownK3iRow" },
      twoToThree: { sameGateUp: "FactoredGateUpRows", mixedGateUp: "FactoredMixedGateUp", rowMajorDown: "FactoredDownRows", k3InterleavedDown: "FactoredDownK3iRows" },
      four: { sameGateUp: "MmaGateUp", mixedGateUp: "MmaGateUp", rowMajorDown: "FactoredDownRows", k3InterleavedDown: "FactoredDownK3iRows" },
      fiveToEight: { sameGateUp: "MmaGateUp", mixedGateUp: "MmaGateUp", rowMajorDown: "MmaDown", k3InterleavedDown: "MmaDownK3i" },
    },
    projections: {
      one: null,
      twoToThree: { 3: "Affine3RowsLinear", 4: "Affine4RowsLinear" },
      four: { 3: "AffineSplitMmaLinear", 4: "AffineSplitMmaLinear" },
      fiveToEight: { 3: "AffineSplitMmaLinear", 4: "AffineSplitMmaLinear" },
    },
    head: { twoToThree: "Affine4RowsLinear", four: "AffineSplitMmaLinear", fiveToEight: "AffineSplitMmaLinear" },
    asyncEvery: 4,
    widerVerify: "lazy",
    knownSpans: false,
    mixed: "packed",
    // Affine 4-bit group 64 KV on the KV4 lego (one-row decode kernel, folded-GQA windows) in place of the unfused
    // port, measured 2026-10-13 (forward + head, one sequence, 16 steps, two sessions each) at 512, 4K, 8.5K and 16K
    // positions: decode 60.3/63.4/67.8/74.3 -> 59.9/61.3/63.5/66.9 ms, verify windows of 4 rows
    // 128.4/143.5/159.9/190.9 -> 125.6/131.9/136.3/145.6, of 8 rows 146.4/154.4/165.0/188.2 -> 144.0/147.4/153.4/164.5.
    // Its decode kernel at 128 tokens per threadgroup (the M4 Pro's 512 leaves four threadgroups per layer at 512
    // positions): decode 61.3/62.0/64.0/65.1 ms at 512 -> 59.9/61.4/63.5/66.9 at 128 (256: 60.4/61.8/63.2/66.0).
    kvLego: { name: "Bf16FirstKv4Head256Cache", decodeBlock: 128 },
    // bf16 verify windows of six to eight rows (36 to 48 query vectors per KV head, more than MLX's vector kernel takes)
    // with GQA folded into rows, bit for bit MLX's fallback, which reads every KV head once per query head: eight rows
    // 146.7/159.3/177.7/207.9 -> 142.9/149.8/159.3/172.1 ms at 512/4K/8.5K/16K positions (forward + head, two
    // sessions each; logits bit-identical at every width).
    bf16WindowRead: "foldedWindowRead",
    // Measured 2026-10-13 on one layer's MLP at 1024 and 2048 rows: the Neural Engine's 34% of the channels (5888,
    // 23 tiles of 256, which it runs faster than 5760) beside the GPU's 66% in 0.70 of the GPU-only time (fractions
    // 0.25 to 0.45 swept; the Neural Engine at about 4 TFLOPS against the GPU's 8; 2% faster prompts than 0.31).
    // Projections stay whole on the dense GEMM (promptProjection): a split there, its GPU share on MLX's quantized
    // matmul, gains nothing over it.
    ane: { fraction: 0.34, minRows: 640, rows: [1024, 1536, 2048],
      rowMajorMlp: "AneTrellisMlpSplit", k3InterleavedMlp: "AneTrellisMlpSplitK3i", affine: null, projections: [] },
    // MLX's quantized matmul bit for bit from 640 rows, 1.27x to 1.38x faster (DequantizedLinear).
    promptProjection: "DequantizedLinear",
    // The DFlash 2 drafter's 4-bit projections (its five layers' attention and MLP, fc and the context k and v), each
    // kernel per row count: MLX's quantized matmul at one row, the affine row kernel at two to three (MLP 4.48 ->
    // 3.57 ms at two rows), the split-K matrix-unit kernel at four to eight (MLP 8.49 -> 4.46 ms at four rows,
    // 16.2 -> 4.45 at eight on the matrix unit before it).
    drafter: { one: null, twoToThree: "Affine4RowsLinear", four: "AffineSplitMmaLinear", fiveToEight: "AffineSplitMmaLinear" },
    // Nine to 256 rows on the matrix unit's any-row kernels, each decoded weight fragment feeding several 8-row blocks
    // (one Qwen MLP, 3-bit, gate/up + down: 16 rows 1.0 + 0.85 ms against 10.9 + 1.7 on the prefill tiles and weight
    // expansion; 64 rows 3.7 + 2.5 against 5.3 + 2.9), the split-K affine kernel for the projections and head (16 rows
    // 2x MLX's quantized matmul, level with it from 32). The whole forward, one sequence's verify window, bf16 KV at
    // 512 positions: 16 rows 0.89 -> 0.19 s, 32 0.89 -> 0.34, 64 0.77 -> 0.63, 128 1.45 -> 1.22, 256 2.48 -> 2.38;
    // past 256 rows weight expansion and MLX's GEMM win (384 rows 3.64 against 3.51 s).
    wider: { maxRows: 256,
      mlp: { sameGateUp: "MmaGateUpWide", mixedGateUp: "MmaGateUpWide", rowMajorDown: "MmaDownWide", k3InterleavedDown: "MmaDownWideK3i" },
      projections: { 3: "AffineSplitMmaLinear", 4: "AffineSplitMmaLinear" }, head: "AffineSplitMmaLinear" },
  },
});

/** Break a `// ` comment sentence onto further `// ` lines at 120 columns. */
function wrapComment(text: string): string {
  const lines = [""];
  for (const word of text.split(" "))
    if (lines.at(-1)!.length && 3 + lines.at(-1)!.length + 1 + word.length > 120) lines.push(word);
    else lines[lines.length - 1] = lines.at(-1)!.length ? `${lines.at(-1)} ${word}` : word;
  return lines.join("\n// ");
}

// ── Description: the artifact facts the template needs ────────────────────────

/** The repo@revision of a hub-layout snapshot path, else the path itself. */
function sourceOf(dir: string): string {
  const match = dir.replace(/\/+$/, "").match(/models--([^/]+)--([^/]+)\/snapshots\/([^/]+)$/);
  return match ? `${match[1]}/${match[2]}@${match[3]}` : dir;
}

/** Shapes of the named tensors, from the shard headers only. */
async function storedShapes(dir: string, names: readonly string[]): Promise<Map<string, number[]>> {
  const index = await Bun.file(`${dir}/model.safetensors.index.json`).json() as { weight_map: Record<string, string> };
  const headers = new Map<string, Record<string, { shape: number[] }>>();
  const shapes = new Map<string, number[]>();
  for (const name of names) {
    const file = index.weight_map[name];
    if (!file) continue;
    if (!headers.has(file)) {
      const shard = Bun.file(`${dir}/${file}`);
      const length = Number(new DataView(await shard.slice(0, 8).arrayBuffer()).getBigUint64(0, true));
      headers.set(file, JSON.parse(await shard.slice(8, 8 + length).text()));
    }
    shapes.set(name, headers.get(file)![name]!.shape);
  }
  return shapes;
}

/** Read an artifact snapshot into the description the template consumes. */
export async function describeArtifact(dir: string): Promise<Qwen38Description> {
  const config = await loadModelConfig(dir);
  // The table's fingerprint lives beside the Qwen checkpoint view, whose module loads the native library.
  const { qwen35GraphFingerprint } = await import("../src/models/qwen/checkpoint");
  const t = config.text;
  const downNames = Array.from({ length: t.numHiddenLayers }, (_, i) => [
    `language_model.model.layers.${i}.mlp.down_proj.weight`, `model.language_model.layers.${i}.mlp.down_proj.weight`]).flat();
  // The first DeltaNet and the first attention layer: every layer of a kind has their projection widths.
  const firstOf = { deltanet: 0, attention: t.fullAttentionInterval - 1 };
  const projectionPaths = [
    ...DELTANET_PROJECTIONS.map(name => [name, `layers.${firstOf.deltanet}.linear_attn.${name}`] as const),
    ...ATTENTION_PROJECTIONS.map(name => [name, `layers.${firstOf.attention}.self_attn.${name}`] as const),
  ];
  const projectionNames = projectionPaths.flatMap(([, path]) => [`language_model.model.${path}.weight`, `model.language_model.${path}.weight`]);
  const shapes = await storedShapes(dir, [...downNames, ...projectionNames]);
  const bits = (path: string) => {
    const spec = quantFor(config.quantization, path);
    if (!spec) throw new Error(`${path}: no quantization entry`);
    return spec.bits;
  };
  const projections = Object.fromEntries(projectionPaths.map(([name, path], j): [string, [number, number]] => {
    // A packed affine weight is [outputs, inputs · bits / 32] words.
    const shape = shapes.get(projectionNames[2 * j]!) ?? shapes.get(projectionNames[2 * j + 1]!);
    if (!shape || shape.length !== 2) throw new Error(`${path}: no stored 2-D weight`);
    return [name, [shape[1]! * 32 / bits(`language_model.model.${path}`), shape[0]!]];
  }));
  const widths: Qwen38Widths = { hidden: t.hiddenSize, intermediate: t.intermediateSize, projections };
  const layers = Array.from({ length: t.numHiddenLayers }, (_, i): Qwen38LayerSpec => {
    const prefix = `language_model.model.layers.${i}`;
    const kind = (i + 1) % t.fullAttentionInterval !== 0 ? "deltanet" : "attention";
    const projections = kind === "deltanet"
      ? DELTANET_PROJECTIONS.map(name => bits(`${prefix}.linear_attn.${name}`))
      : ATTENTION_PROJECTIONS.map(name => bits(`${prefix}.self_attn.${name}`));
    const shape = shapes.get(downNames[2 * i]!) ?? shapes.get(downNames[2 * i + 1]!);
    if (!shape) throw new Error(`layer ${i}: no stored down_proj weight`);
    return { kind, projections, gate: bits(`${prefix}.mlp.gate_proj`), up: bits(`${prefix}.mlp.up_proj`),
      down: bits(`${prefix}.mlp.down_proj`), downLayout: shape.length === 3 ? "k3Interleaved" : "rowMajor" };
  });
  return { source: sourceOf(dir), fingerprint: qwen35GraphFingerprint(config), widths, layers };
}

export async function readDescription(source: string): Promise<Qwen38Description> {
  return source.endsWith(".json") ? await Bun.file(source).json() as Qwen38Description : describeArtifact(source);
}

// ── Template ─────────────────────────────────────────────────────────────────

/** The read a phase operation's layers take (`ReadPhase` without the committed read). */
type ReadPhaseName = "window" | "decode";

const list = (names: readonly string[]) => `[${names.map(name => JSON.stringify(name)).join(", ")}]`;
const depthText = (depth: Qwen38Composition["draftDepth"]) => typeof depth === "number" ? `${depth}` : `adaptive up to ${depth.max}`;

/** The generated module's text for composition `stem` of `compositions` (the committed table unless a test hands
 *  another). Every difference between two outputs is a fact of the composition or its device's table, resolved here. */
export function emitQwen38Graph(description: Qwen38Description, stem: string,
  compositions: Readonly<Record<string, Qwen38Composition>> = QWEN38_COMPOSITIONS): string {
  const c = compositions[stem];
  if (!c) throw new Error(`no composition ${stem} (known: ${Object.keys(compositions).join(", ")})`);
  const k = KERNELS[c.device];
  if (!k) throw new Error(`${stem}: no kernel table for device ${c.device}`);
  if (description.fingerprint !== c.fingerprint)
    throw new Error(`${stem}: the source's graph fingerprint ${description.fingerprint} is not the composition's ${c.fingerprint}`);
  const kvLego = k.kvLego?.name ?? null, ane = k.ane, head = k.head, asyncEvery = k.asyncEvery;
  /** The lego's decode block argument, where the device's table names one. */
  const decodeBlock = k.kvLego?.decodeBlock === undefined ? "" : `, ${k.kvLego.decodeBlock}`;
  /** The bf16 caches' window read argument, where the device's table names one. */
  const windowRead = k.bf16WindowRead === null ? "" : `, ${k.bf16WindowRead}`;
  /** The bf16 cache's read, in the header's prose. */
  const plainRead = k.bf16WindowRead === null ? "the stock SDPA"
    : `the stock SDPA, its windows of six to eight rows folding GQA into rows (${k.bf16WindowRead})`;
  /** The affine scheme the device's KV lego serves. */
  const legoKv = kvLego === null ? null : KV_LEGOS[kvLego];
  if (legoKv === undefined) throw new Error(`${stem}: no KV scheme for cache lego ${kvLego}`);
  if (c.kv !== "every" && (legoKv?.bits !== c.kv.bits || legoKv.groupSize !== c.kv.groupSize))
    throw new Error(`${stem}: an affine KV composition composes the lego its device's table names for that scheme (${c.device}: ${kvLego ?? "none"})`);
  if ((c.kv === "every") !== (c.maxRows === null))
    throw new Error(`${stem}: this template constrains the KV schemes and the rows together, or neither`);
  if (c.aneBridge && !ane) throw new Error(`${stem}: ${c.device}'s table builds no Neural Engine operation`);
  if (ane && (ane.affine === null) !== (ane.projections.length === 0))
    throw new Error(`${stem}: ${c.device}'s Neural Engine table names a projection split layer exactly when it splits projections`);
  /** The Neural Engine operations split projections as well as the MLP. */
  const splitsProjections = !!ane && ane.projections.length > 0;
  const promptProjection = k.promptProjection;
  if (promptProjection !== null && !PROMPT_PROJECTION_MODULES[promptProjection])
    throw new Error(`${stem}: no module for prompt projection layer ${promptProjection}`);
  /** Each Neural Engine operation's row range: from the device's fewest positions, or one above the previous
   *  program's, up to its own program's. */
  const aneRanges = (ane?.rows ?? []).map((rows, j, all) => [j === 0 ? ane!.minRows : all[j - 1]! + 1, rows] as const);
  if (aneRanges.some(([minRows, rows]) => minRows > rows))
    throw new Error(`${stem}: ${c.device}'s Neural Engine programs ascend from at least the fewest positions a piece runs`);
  const one = k.mlp.one;
  if (!one) throw new Error(`${stem}: ${c.device}'s table names no one-row MLP layers`);
  if (c.mixedGateUp !== undefined && !NARROW.some(cls => TAILED.has(k.mlp[cls]?.mixedGateUp ?? "")))
    throw new Error(`${stem}: mixedGateUp chooses the tail of a mixed-width layer, which ${c.device}'s table does not name`);
  const mixedTail = c.mixedGateUp ?? "split";
  const layers = description.layers;
  const interleaved = layers.filter(layer => layer.downLayout === "k3Interleaved").length;
  const mixed = layers.map((layer, i) => layer.gate !== layer.up ? i : -1).filter(i => i >= 0);
  const deltanet = layers.filter(layer => layer.kind === "deltanet").length;
  const table = layers.map((layer, i) =>
    `  /* ${String(i).padStart(2)} */ { kind: ${JSON.stringify(layer.kind)}, projections: [${layer.projections.join(", ")}], ` +
    `gate: ${layer.gate}, up: ${layer.up}, down: ${layer.down}, downLayout: ${JSON.stringify(layer.downLayout)} },`).join("\n");

  /** A class whose layers are the generic blocks' (TrellisMlp at that class). */
  const generic = (cls: Narrow) => k.mlp[cls] === null;
  const wider = k.wider;
  if (wider && !head) throw new Error(`${stem}: ${c.device}'s wider layers take the head on its table's head layers`);
  const sets = [...NARROW.flatMap(cls => k.mlp[cls] ? [k.mlp[cls]!] : []), ...(wider ? [wider.mlp] : [])];
  /** The reduce mixed-width layer with `mixedGateUp: "off"`: those layers keep the generic blocks. */
  const offMixed = (m: MlpLayers) => TAILED.has(m.mixedGateUp) && mixedTail === "off";
  const mlpLayerNames = [...new Set(sets.flatMap(m =>
    [m.sameGateUp, ...(offMixed(m) ? [] : [m.mixedGateUp]), m.rowMajorDown, m.k3InterleavedDown]))];
  for (const name of mlpLayerNames) if (!LAYER_MODULES[name]) throw new Error(`${stem}: no module for layer ${name}`);
  const mlpImports = [...new Set(mlpLayerNames.map(name => LAYER_MODULES[name]!))].sort().map(module =>
    `import { ${mlpLayerNames.filter(name => LAYER_MODULES[name] === module).sort().join(", ")} } from "../../layers/${module}";`);
  const drafter = k.drafter;
  const affineLayers = [...new Set([...Object.values(k.projections).flatMap(p => p ? [p[3], p[4]] : []),
    ...(head ? Object.values(head).filter((layer): layer is string => layer !== null) : []),
    ...(drafter ? Object.values(drafter).filter((layer): layer is string => layer !== null) : []),
    ...(wider ? [wider.projections[3], wider.projections[4], wider.head] : [])])];
  /** The attention and DeltaNet projections are QuantizedLinear in every class and nothing else is built (no prompt
   *  blocks, no Neural Engine operation): mounted adapters apply everywhere, so no class changes with them. */
  const adapterSafe = NARROW.every(cls => k.projections[cls] === null) && head === null && ane === null && promptProjection === null;
  /** Each distinct [inputs, outputs] shape of the split projections, with the projections that have it (none where
   *  the device's table builds no Neural Engine operation). */
  const aneShapes = [...(ane?.projections ?? []).reduce((shapes, name) => {
    const shape = description.widths.projections[name];
    if (!shape) throw new Error(`${stem}: the description has no widths for ${name}`);
    const key = shape.join("→");
    shapes.set(key, [shape, [...(shapes.get(key)?.[1] ?? []), name]]);
    return shapes;
  }, new Map<string, [readonly [number, number], string[]]>()).values()];
  /** Blocks are built once per projection layer pair and shared by the classes that name it. */
  const blocksName = (p: { 3: string; 4: string }) => `${p[4][0]!.toLowerCase()}${p[4].slice(1).replace(/Linear$/, "")}Blocks`;
  const projectionSets = [...new Map([...NARROW.flatMap(cls => k.projections[cls] ? [[blocksName(k.projections[cls]!), k.projections[cls]!] as const] : []),
    ...(wider ? [[blocksName(wider.projections), wider.projections] as const] : [])]).entries()];
  const headName = (layer: string) => `${layer[0]!.toLowerCase()}${layer.slice(1).replace(/Linear$/, "")}Head`;
  const heads = head ? [...new Set([...Object.values(head).filter((layer): layer is string => layer !== null), ...(wider ? [wider.head] : [])])] : [];
  /** A layer's construction; a tailed layer takes the mixed-width tail (the split one where those layers are off). */
  const construct = (layer: string) => TAILED.has(layer)
    ? `new ${layer}(gate, up, ${JSON.stringify(mixedTail === "off" ? "split" : mixedTail)})` : `new ${layer}(gate, up)`;
  const mixedExpr = (m: MlpLayers) => construct(m.mixedGateUp);
  const gateUpExpr = (m: MlpLayers) => m.sameGateUp === m.mixedGateUp || offMixed(m) ? construct(m.sameGateUp)
    : `spec.gate === spec.up ? new ${m.sameGateUp}(gate, up) : ${mixedExpr(m)}`;
  const narrowSet = (cls: Narrow, indent: string) => {
    const m = k.mlp[cls], p = k.projections[cls];
    if (!m) return `${indent}this.#${cls} = this.#generic;`;
    return [
      `this.#${cls} = generic.map((layer, i) => {`,
      `  const spec = LAYERS[i]!, { gate, up, down } = mlpOf(i);`,
      ...(offMixed(m) ? [`  if (spec.gate !== spec.up) return genericStep(layer);`] : []),
      `  const gateUp = ${gateUpExpr(m)};`,
      `  const project = spec.downLayout === "k3Interleaved" ? new ${m.k3InterleavedDown}(down) : new ${m.rowMajorDown}(down);`,
      `  return layerStep(layer, mixerOf(${p ? `${blocksName(p)}[i]!` : "(layer.linearAttn ?? layer.selfAttn)!"}, spec.kind), post => {`,
      `    using mid = gateUp.forward(post);`,
      `    return project.forward(mid);`,
      `  });`,
      `});`,
    ].map(line => indent + line).join("\n");
  };
  const blockSets = projectionSets.map(([name, p]) =>
    `      const ${name} = LAYERS.map((_, i) => blockOf(i, ${p[3] === p[4] ? `linear => new ${p[3]}(linear)`
      : `(linear, _path, width) => width === 3 ? new ${p[3]}(linear) : new ${p[4]}(linear)`}));`).join("\n");
  /** The wider layers' block set (and, where a mixed forward packs its MLP rows, each layer's MLP alone). */
  const packedWider = !!wider && k.mixed === "packed";
  const widerSet = wider ? [
    `const widerMlp = generic.map((_, i) => {`,
    `  const spec = LAYERS[i]!, { gate, up, down } = mlpOf(i);`,
    `  const gateUp = ${gateUpExpr(wider.mlp)};`,
    `  const project = spec.downLayout === "k3Interleaved" ? new ${wider.mlp.k3InterleavedDown}(down) : new ${wider.mlp.rowMajorDown}(down);`,
    `  return (post: MlxArray) => {`,
    `    using mid = gateUp.forward(post);`,
    `    return project.forward(mid);`,
    `  };`,
    `});`,
    `this.#wider = generic.map((layer, i) => layerStep(layer, mixerOf(${blocksName(wider.projections)}[i]!, LAYERS[i]!.kind), widerMlp[i]!));`,
    ...(packedWider ? [`this.#widerMlp = widerMlp;`] : []),
  ].map(line => `      ${line}`).join("\n") : "";
  const narrowBlock = adapterSafe ? `    // Every class's attention and DeltaNet projections are the generic blocks' QuantizedLinear, which apply
    // mounted adapters: no class changes with them.
${NARROW.map(cls => narrowSet(cls, "    ")).join("\n")}` : `    if (composition.adapters) {
      // Every class runs the generic blocks: their QuantizedLinear projections
      // apply the mounted adapters (the one-kernel layers do not).
      this.#one = this.#twoToThree = this.#four = this.#fiveToEight = ${wider ? "this.#wider = " : ""}this.#generic;${packedWider ? `
      this.#widerMlp = generic.map((_, i) => (post: MlxArray) => mlpOf(i).forward(post, "window", "wider"));` : ""}
    } else {
${blockSets}
${NARROW.map(cls => narrowSet(cls, "      ")).join("\n")}${wider ? `\n${widerSet}` : ""}
    }`;
  /** One phase's operation of a class: its own block set (submitted every `asyncEvery` layers, or one lazy graph),
   *  or for a generic class the generic blocks with `genericLoop`. */
  const entry = (cls: Narrow, phase: ReadPhaseName, genericLoop: "bounded" | "lazy") =>
    `      ${cls}: (ids: MlxArray, state: Cache[], capture: MlxLayerCapture | null) =>
        ${generic(cls) ? `this.#${genericLoop}(wide, embed.encode(ids), state, ${JSON.stringify(phase)}, capture, null, ${JSON.stringify(cls)})`
      : asyncEvery !== null ? `this.#submitted(${cls}, embed.encode(ids), state, ${JSON.stringify(phase)}, capture, ${JSON.stringify(cls)})`
      : `this.#lazy(${cls}, embed.encode(ids), state, ${JSON.stringify(phase)}, capture, null, ${JSON.stringify(cls)})`},`;
  /** An operation of the wider layers for \`phase\`, evaluated as the narrow classes' own block sets are. */
  const widerRun = (phase: ReadPhaseName) => asyncEvery !== null
    ? `this.#submitted(wider, embed.encode(ids), state, ${JSON.stringify(phase)}, capture, "wider")`
    : `this.#lazy(wider, embed.encode(ids), state, ${JSON.stringify(phase)}, capture, null, "wider")`;
  /** Above the widest own block set (and at a generic class), a prompt window evaluates each layer before the next. */
  const boundedWidths = [...NARROW.filter(generic), "wider"].map(cls => `width === ${JSON.stringify(cls)}`).join(" || ");

  const row = (operation: string, mlp: string, projections: string) =>
    `//   ${operation.padEnd(31)}${mlp.padEnd(44)}${projections}`.trimEnd();
  const classRows = NARROW.map(cls => {
    const m = k.mlp[cls], p = k.projections[cls];
    return row(cls === "one" ? "decode.one, verify.one" : `decode/verify.${cls}`, m ? `${m.sameGateUp} + ${m.rowMajorDown}/K3i`
      : "the generic blocks (TrellisMlp)", p ? (p[3] === p[4] ? p[3] : `${p[3]}/${p[4]}`) : "QuantizedLinear");
  }).join("\n");
  /** Each run of consecutive classes whose mixed-width layer takes the composition's tail or is not their same-width
   *  one (the table row names that), with its layer and rows. */
  const mixedRuns: { layer: string; rows: [number, number] }[] = [];
  for (const cls of NARROW) {
    const m = k.mlp[cls], last = mixedRuns.at(-1);
    if (!m || (m.mixedGateUp === m.sameGateUp && !TAILED.has(m.mixedGateUp))) continue;
    if (last?.layer === m.mixedGateUp && last.rows[1] + 1 === CLASS_ROWS[cls][0]) last.rows[1] = CLASS_ROWS[cls][1];
    else mixedRuns.push({ layer: m.mixedGateUp, rows: [...CLASS_ROWS[cls]] });
  }
  /** Each run as a verb and its object; a run whose verb is the previous run's omits it. */
  const mixedPhrases = mixedRuns.map(({ layer, rows }) => !TAILED.has(layer) ? ["takes", `${layer} at ${rowsText(rows)}`]
    : mixedTail === "off" ? ["keeps", `the generic blocks (two projections, a separate SwiGLU) at ${rowsText(rows)}`]
    : ["takes", `${layer} at ${rowsText(rows)}, ${mixedTail === "split"
      ? "split tail (MLX's compiled SwiGLU bit for bit)" : "fused tail (the same-width kernel's float32 sigmoid)"}`]);
  const mixedSentence = `Mixed-width gate/up ${mixedPhrases.map(([verb, object], j) =>
    j > 0 && mixedPhrases[j - 1]![0] === verb ? object : `${verb} ${object}`).join(", and ")}.`;
  /** The most rows the graph's own block sets take (the generic blocks above). */
  const widest = generic("fiveToEight") ? "four" : "eight";
  const summary = c.kv !== "every" && head && asyncEvery !== null ? `// ${wrapComment(mixedSentence)} Every attention block reads
// through the cache (${kvLego}: bf16 through its first append, then the fused 4-bit decode
// kernel and the folded-GQA window kernel; the bf16 cache: the stock SDPA). The head is part of each phase: logits.twoToThree ${head.twoToThree}, logits.four
// ${head.four}, logits.fiveToEight ${head.fiveToEight}, QuantizedLinear otherwise. Decode and verify up to 8 rows submit
// the graph every ${asyncEvery} layers; prompt chunks, tails and wider verify windows evaluate each layer
// and its recurrent state.` : `// ${wrapComment([
    mixedSentence,
    ...(legoKv ? [`Every attention block reads through the cache the composition's KV scheme names: affine ${legoKv.bits}-bit`,
      `group ${legoKv.groupSize} composes ${kvLego} (bf16 through its first append, then the fused 4-bit decode kernel${
        k.kvLego?.decodeBlock === undefined ? "" : ` at ${k.kvLego.decodeBlock} tokens per threadgroup`} and the`,
      `folded-GQA window kernel), every other scheme its own lego (composedFullAttentionCache: the plain cache with ${plainRead},`,
      "bf16-first affine or per-layer affine with the composition's read kernels, or bf16-first TurboQuant with the",
      "dense read)."] : [
      "Every attention block reads through the cache the composition's KV scheme names (composedFullAttentionCache: the",
      `plain cache with ${plainRead}, bf16-first affine or per-layer affine with the composition's read kernels, or`,
      "bf16-first TurboQuant with the dense read)."]),
    ...(wider ? [`Every operation of nine to ${wider.maxRows} rows (decode and verify windows, prompt tails, known-token appends, a mixed`,
      "forward's packed MLP rows) runs the wider layers in the table; more rows run the generic blocks."] : []),
    head ? `The head is part of each phase: ${NARROW.filter(cls => cls !== "one").map(cls =>
      `logits.${cls} ${head[cls as Exclude<Narrow, "one">] ?? "QuantizedLinear"}`).join(", ")}${wider
      ? `, logits.wider ${wider.head} up to ${wider.maxRows} rows` : ""}, QuantizedLinear otherwise.`
      : "The head is QuantizedLinear at every width.",
    ...(drafter ? [`A bound drafter's own 4-bit projections take ${NARROW.filter(cls => drafter[cls] !== null).map(cls =>
      `${drafter[cls]} at ${cls}`).join(", ")}, QuantizedLinear otherwise (TargetView.draftProjections).`] : []),
    asyncEvery !== null ? `Decode, verify, prompt tails and embeddings up to ${widest} rows${wider
      ? ", and decode, verify windows and prompt tails on the wider layers," : ""} submit the graph every ${asyncEvery} layers; wider decode${
      k.widerVerify === "lazy" ? " and verify windows" : ""}${k.knownSpans ? "" : " and known-token appends"} are ${wider ? "otherwise " : ""}one lazy graph;`
      : `Decode${k.widerVerify === "lazy" ? ", verify" : ` and verify up to ${widest} rows`}${k.knownSpans ? "" : " and known-token appends"
      } are one lazy graph${k.widerVerify === "lazy" ? " at every width" : ""};`,
    wider ? `prompt chunks, prompt tails above ${wider.maxRows} rows and embeddings above ${widest} rows${k.widerVerify === "bounded" ? ", and wider verify windows" : ""},`
      : `prompt chunks, and prompt tails and embeddings above ${widest} rows${k.widerVerify === "bounded" ? ", and wider verify windows" : ""},`,
    "evaluate each layer and its recurrent state.",
  ].join(" "))}`;
  const kvText = c.kv !== "every" ? `bf16, or affine ${c.kv.bits}-bit group ${c.kv.groupSize} (composes ${kvLego})`
    : legoKv ? `every scheme: affine ${legoKv.bits}-bit group ${legoKv.groupSize} composing ${kvLego}, each other its own cache lego
//                (composedFullAttentionCache)`
    : "every scheme, each composing its own cache lego (composedFullAttentionCache)";
  const composition = c.kv === "every" ? `/** The composition this file was generated and gated for, its entry in the loader's table: the artifact's graph
 *  fingerprint on this GPU. KV scheme, rows per forward, draft depth, prefill chunk, adapters and the Neural Engine
 *  bridge constrain nothing: the constructor builds them from the record it is handed. */
export const COMPOSITION: ServedComposition = Object.freeze({
  fingerprint: ${JSON.stringify(c.fingerprint)}, device: ${JSON.stringify(c.device)},
});` : `/** The composition this file was generated and gated for, its entry in the loader's table: the artifact's graph
 *  fingerprint on this GPU, the bf16 cache or the affine ${c.kv.bits}-bit group-${c.kv.groupSize} lego, at most ${c.maxRows} rows per
 *  forward. Draft depth, prefill chunk, adapters and the Neural Engine bridge constrain nothing: the constructor
 *  builds them from the record it is handed. */
export const COMPOSITION: ServedComposition = Object.freeze({
  fingerprint: ${JSON.stringify(c.fingerprint)}, device: ${JSON.stringify(c.device)},
  kv: Object.freeze([Object.freeze({ kind: "bf16" }), Object.freeze({ kind: "affine", bits: ${c.kv.bits}, groupSize: ${c.kv.groupSize} })]),
  maxRows: ${c.maxRows},
});`;

  return `// GENERATED by packages/inference/scripts/gen-qwen38.ts from composition ${stem}; do not edit.
// Regenerate: bun packages/inference/scripts/gen-qwen38.ts <snapshot-or-description> ${stem}
// (tests/fixtures/qwen38-graphs holds the description it was generated from).
//
// Composition ${stem}:
//   artifact     ${description.source}, graph fingerprint ${c.fingerprint}
//   device       ${c.device} (${k.label})
//   KV           ${kvText}
//   draft depth  ${depthText(c.draftDepth)}
//   prefill      ${c.prefillChunkTokens}-token chunks
//   rows         ${c.maxRows === null ? "any number" : `up to ${c.maxRows}`} per forward
//   Neural Engine bridge ${c.aneBridge ? "present" : "absent"}${c.wiredWeights ? "\n//   weights      wired for each generation" : ""}
//
// An entire graph: built once at construction from the composition record it is
// handed, it inherits from no class and never calls another graph. Every phase and
// width class is one straight-line operation over blocks built for it from the
// quant table below (${deltanet} DeltaNet and ${layers.length - deltanet} attention layers; ${interleaved} down projections
// 3-bit block-interleaved, ${layers.length - interleaved} row-major; gate and up at different widths in layers
// ${mixed.join(", ")}) and the device's one-kernel layers:
//
${row("operation", "MLP gate/up + down (row-major/interleaved)", "affine projections (3/4-bit)")}
${classRows}
${wider ? `${row(`every operation 9..${wider.maxRows}`, `${wider.mlp.sameGateUp} + ${wider.mlp.rowMajorDown}/K3i`,
  wider.projections[3] === wider.projections[4] ? wider.projections[3] : `${wider.projections[3]}/${wider.projections[4]}`)}
` : ""}${promptProjection ? `${row(`decode/verify.wider${wider ? ` ${wider.maxRows + 1}+` : ""}`, "the generic blocks (TrellisMlp)", "QuantizedLinear")}
${row(wider ? `prefillChunk.gpu/Tail ${wider.maxRows + 1}+` : "prefillChunk.gpu/Tail.wider", "the generic blocks (TrellisMlp)", promptProjection)}`
  : `${row("decode/verify.wider,", "the generic blocks (TrellisMlp)", "QuantizedLinear")}
${row("prefillChunk.gpu, prefillTail", "", "")}`}${ane ? `
${row(`prefillChunk.ane, ${ane.minRows}..${ane.rows.at(-1)}`, `${ane.rowMajorMlp}/K3i`,
  splitsProjections ? `${ane.projections.join(", ")} split` : promptProjection ?? "QuantizedLinear")}` : ""}
//
${summary}
//
// Request shapes this graph serves:
${ane ? `//   - prompt chunks of exactly the composition's chunk size (prefillChunk.gpu) and prompt tails (prefillTail),
//     and, when the composition has the Neural Engine bridge and mounts no adapters, one sequence's chunk or tail
//     of ${ane.minRows} to ${ane.rows.at(-1)} positions on the Neural Engine operation whose programs hold it (prefillChunk.ane:
//     ${aneRanges.map(([minRows, rows]) => `${minRows}..${rows}`).join(", ")}, programs of ${ane.rows.join(", ")} positions);`
  : "//   - prompt chunks of exactly the composition's chunk size (prefillChunk.gpu) and prompt tails (prefillTail);"}
//   - decode, one position per sequence, and speculative verify windows of sequences x (depth + 1) positions
//     with the drafter's layer capture, for any number of sequences${c.maxRows === null ? "" : " up to the composition's rows per forward"}
//     (decode[width], verify[width], logits[width]);
//   - prepared image and video embeddings with request mRoPE positions (prefillEmbeddingsAtPositions,
//     prefillAtPositions, decodeAtPositions), on the generic blocks, and for direct generation at the
//     caches' own positions (prefillEmbeddings);
${k.knownSpans ? "//   - known-token appends over the committed read (appendKnown, knownSpans) on the generic blocks;"
  : "//   - known-token appends as windows on the block set of their width (appendKnown; no committed spans);"}
${k.mixed === "perGroup" ? "//   - mixed groups of rows in one forward (mixed), each group on its own phase's operation for its width;"
  : `//   - mixed groups of rows in one forward (mixed) on the generic blocks: each group's attention and recurrence over
//     its own caches, the prompt and decode groups' MLP rows packed into one call (the composition's packedMixedMlp);`}
//   - the scoring forward over a whole sequence (sequenceHidden, prefillTail's operation for its width);
${adapterSafe ? `//   - mounted LoRA adapters: the attention and DeltaNet projections are the generic blocks' QuantizedLinear at
//     every width, which apply them (the Trellis MLP is no adapter target);
//   - paged KV (the paged legos' own reads in place of the plain full-attention storage).` : `//   - mounted LoRA adapters when the composition mounts them: every operation then runs the generic blocks,
//     whose QuantizedLinear projections apply them${ane ? ", and the Neural Engine operation is not built" : ""};
//   - paged KV (the paged legos' own reads in place of the plain full-attention storage).
// The planner refuses, from the declared capabilities and never by routing to another graph, adapter requests
// when the composition mounts none (adapter-batch-unsupported).`}${kvLego ? `
// ${kvLego}'s row layouts read with its KV4 kernels while they hold one unpadded row and with the
// composition's affine kernels while they hold more.` : ""}

import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { ModelConfig } from "../../artifacts/config";
import type { Weights } from "../../artifacts/weights";
import type { Cache, GatedDeltaCache } from "../../contracts/mlx/cache";
import type { ${drafter ? "DraftProjectionKernel, " : ""}ResidualBasis, TargetView } from "../../contracts/mlx/draft-target";
import type {
${k.knownSpans ? `  MlxDeclaredGraph, MlxHeadForward, MlxKnownSpans, MlxLayerCapture, MlxMixedForward, MlxPhaseForward,
  MlxPrefillChunk, MlxTokenPhases, MlxWidthClass, MlxWidthTable,` : `  MlxDeclaredGraph, MlxHeadForward, MlxLayerCapture, MlxMixedForward, MlxPhaseForward, MlxPrefillChunk,
  MlxTokenPhases, MlxWidthClass, MlxWidthTable,`}
} from "../../contracts/mlx/graph";
import type { MediaEncoders, MlxPromptInput, Vision } from "../../contracts/mlx/media";
import type { MropeForwardState } from "../../contracts/mlx/positions";
import type { Composition } from "../../contracts/portable/composition";
import type { GraphCapabilities } from "../../contracts/portable/graph";
${[
    ...(k.mixed === "packed" ? [`import { mapTokenGroups } from "../../input/token-groups";`] : []),
    ...(ane ? [`import type { AneProgramSpec } from "../../kernels/ane/linear";`] : []),
    ...(affineLayers.length ? [`import { ${affineLayers.sort().join(", ")} } from "../../layers/affine-verify-linear";`] : []),
    ...(ane ? [splitsProjections ? `import {
  ${[ane.affine!, ane.rowMajorMlp, ane.k3InterleavedMlp].sort().join(", ")}, aneAffineSplitProgram, aneMlpSplitProgram, type AneMlpSplit,
} from "../../layers/ane-prefill-split";` : `import {
  ${[ane.rowMajorMlp, ane.k3InterleavedMlp].sort().join(", ")}, aneMlpSplitProgram, type AneMlpSplit,
} from "../../layers/ane-prefill-split";`] : []),
    ...(promptProjection ? [`import { ${promptProjection} } from "../../layers/${PROMPT_PROJECTION_MODULES[promptProjection]}";`] : []),
    `import { disposing } from "../../layers/helpers";`,
    `import { LoraState } from "../../layers/lora";`,
    `import { RMSNorm } from "../../layers/normalization";`,
    `import { QuantizedEmbedding } from "../../layers/quantized-embedding";`,
    `import { QuantizedLinear } from "../../layers/quantized-linear";`,
    ...(k.knownSpans ? [`import { qwenAppendChunkSize } from "../../layers/qwen-append";`] : []),
    `import { mropeInvFreq } from "../../layers/qwen-mrope";`,
    ...(ane ? [`import { compiledSwiglu } from "../../layers/swiglu";`] : []),
    ...mlpImports,
    ...(kvLego ? [`import { composedAffineKernels } from "../../state/affine-attention";`] : []),
    `import { AttentionMasks${windowRead} } from "../../state/attention-read";`,
    `import { ${kvLego ? `${kvLego}, ` : ""}composedFullAttentionCache } from "../../state/bf16-first-kv";`,
  ].join("\n")}
import { SSMCache } from "../../state/ssm";
import { denseTurboQuantKernels } from "../../state/turboquant-attention";
import { declareGraph } from "../capabilities";
import { atEveryWidth, byWidth, type ReadPhase } from "../graph";
import type { ServedComposition } from "../implementation";
${projectionSets.length || ane || promptProjection ? `import {
  GatedDeltaNet, Qwen3Attention, Qwen3Layer, type AttentionLinear, type AttentionLinearLoader, type TrellisMlp,
} from "./blocks";` : `import { Qwen3Layer, type AttentionLinear, type GatedDeltaNet, type Qwen3Attention, type TrellisMlp } from "./blocks";`}
import { qwen35WeightsView } from "./checkpoint";
import { bindQwenMediaInput, qwenDraftTarget, qwenMediaEncoders, qwenResidualBasis } from "./media-input";

export const ${constName(c)} = ${JSON.stringify(c.graph)};

${composition}

/** One decoder layer of the artifact: its block, the affine width of each projection (DeltaNet: ${DELTANET_PROJECTIONS.join(", ")};
 *  attention: ${ATTENTION_PROJECTIONS.join(", ")}), the Trellis gate and up widths, and the down projection's width and code layout. */
export interface Qwen38Layer {
  readonly kind: "deltanet" | "attention";
  readonly projections: readonly number[];
  readonly gate: number;
  readonly up: number;
  readonly down: number;
  readonly downLayout: "rowMajor" | "k3Interleaved";
}

/** The artifact's quant table and stored layouts, from ${description.source}. */
export const LAYERS: readonly Qwen38Layer[] = Object.freeze([
${table}
]);

const PREFIX = "language_model.model";
const DELTANET_PROJECTIONS = ${list(DELTANET_PROJECTIONS)} as const;
const ATTENTION_PROJECTIONS = ${list(ATTENTION_PROJECTIONS)} as const;
${splitsProjections ? `const ANE_PROJECTIONS: ReadonlySet<string> = new Set(${list(ane!.projections)});
` : ""}${asyncEvery !== null ? `/** Submit the graph every ${asyncEvery} layers at decode and verify widths up to 8 (measured best on this GPU). */
const ASYNC_EVERY = ${asyncEvery};
` : ""}${wider ? `/** Every operation above eight rows of at most WIDER_ROWS rows runs this GPU's wider layers, more rows the generic
 *  blocks: the boundary measured on the whole forward, from this machine's kernel table. */
export const WIDER_ROWS = ${wider.maxRows};
` : ""}${ane ? `/** The Neural Engine's channel fraction in each split. */
const ANE_FRACTION = ${ane.fraction};
/** The Neural Engine operations' row ranges, ascending: a one-sequence prompt piece, a chunk or a tail, whose positions
 *  one holds runs on it, over programs built for its \`rows\`. From this machine's kernel table. */
const ANE_RANGES: readonly { readonly minRows: number; readonly rows: number }[] = Object.freeze([
  ${aneRanges.map(([minRows, rows]) => `{ minRows: ${minRows}, rows: ${rows} }`).join(", ")},
]);

${splitsProjections ? `/** The widths a graph's Neural Engine programs are built at: the MLP's hidden and intermediate widths, and the input
 *  and output widths of each distinct split projection shape. */
export interface AneWidths {
  readonly hidden: number;
  readonly intermediate: number;
  readonly projections: readonly (readonly [number, number])[];
}

/** The artifact's widths, from its description: ${aneShapes.map(([shape, names]) => `${names.join(", ")} ${shape[0]}→${shape[1]}`).join("; ")}. */
export const ANE_WIDTHS: AneWidths = Object.freeze({
  hidden: ${description.widths.hidden}, intermediate: ${description.widths.intermediate},
  projections: Object.freeze([${aneShapes.map(([shape]) => `[${shape[0]}, ${shape[1]}]`).join(", ")}] as const),
});

/** The distinct Neural Engine programs this graph loads for \`composition\`, at the artifact's widths unless
 *  \`widths\` names others (a synthetic artifact laid out like it): none without the bridge or with adapters mounted,
 *  otherwise, for each range in \`ANE_RANGES\`, the whole-MLP slice every layer shares and one program per split
 *  projection shape, each for that range's rows. The install step compiles them into the system cache
 *  (\`compileAnePrograms\`, models/ane-install); the constructor only loads them, so one missing there fails
 *  construction with \`AneBridgeError\`. */
export function anePrograms(composition: Composition, widths: AneWidths = ANE_WIDTHS): AneProgramSpec[] {
  if (!composition.aneBridge || composition.adapters) return [];
  return ANE_RANGES.flatMap(({ rows }) => [aneMlpSplitProgram(widths.hidden, widths.intermediate, ANE_FRACTION, rows),
    ...widths.projections.map(([inputs, outputs]) => aneAffineSplitProgram(inputs, outputs, ANE_FRACTION, rows))]);
}
` : `/** The widths a graph's Neural Engine programs are built at: the MLP's hidden and intermediate widths (no projection
 *  is split). */
export interface AneWidths {
  readonly hidden: number;
  readonly intermediate: number;
}

/** The artifact's widths, from its description. */
export const ANE_WIDTHS: AneWidths = Object.freeze({ hidden: ${description.widths.hidden}, intermediate: ${description.widths.intermediate} });

/** The distinct Neural Engine programs this graph loads for \`composition\`, at the artifact's widths unless
 *  \`widths\` names others (a synthetic artifact laid out like it): none without the bridge or with adapters mounted,
 *  otherwise the whole-MLP slice every layer shares, for each range in \`ANE_RANGES\` at that range's rows. The
 *  install step compiles them into the system cache (\`compileAnePrograms\`, models/ane-install); the constructor
 *  only loads them, so one missing there fails construction with \`AneBridgeError\`. */
export function anePrograms(composition: Composition, widths: AneWidths = ANE_WIDTHS): AneProgramSpec[] {
  if (!composition.aneBridge || composition.adapters) return [];
  return ANE_RANGES.map(({ rows }) => aneMlpSplitProgram(widths.hidden, widths.intermediate, ANE_FRACTION, rows));
}
`}` : ""}
/** One layer: input norm, mixer, residual, post-attention norm, MLP, residual. Consumes nothing; returns an owned output.
 *  \`width\` is the class of \`x\`'s rows the calling operation names (the generic blocks' MLP reads it). */
type Step = (x: MlxArray, cache: Cache, phase: ReadPhase, mrope: MropeForwardState | null, width: MlxWidthClass) => MlxArray;
type Mixer = (normed: MlxArray, cache: Cache, phase: ReadPhase) => MlxArray;
/** A phase operation of this graph: each runs synchronously (\`MlxCapturingForward\` without the promise). */
export type RowsForward = (ids: MlxArray, state: Cache[], capture: MlxLayerCapture | null) => MlxArray;

/** A layer of one width class: \`mixer\` then \`mlp\` (post-attention-normed hidden to the MLP output, owned). */
function layerStep(layer: Qwen3Layer, mixer: Mixer, mlp: (post: MlxArray) => MlxArray): Step {
  const { inputNorm, postAttnNorm } = layer;
  return (x, cache, phase) => {
    using normed = inputNorm.forward(x);
    using mixed = mixer(normed, cache, phase);
    using hidden = ops.add(x, mixed);
    using post = postAttnNorm.forward(hidden);
    using out = mlp(post);
    return ops.add(hidden, out);
  };
}

${packedWider ? `/** A layer's MLP block over \`hidden\` (borrowed): the post-attention norm, \`mlp\`, the residual. Returns an owned output. */
function mlpBlock(layer: Qwen3Layer, hidden: MlxArray, mlp: (post: MlxArray) => MlxArray): MlxArray {
  using post = layer.postAttnNorm.forward(hidden);
  using out = mlp(post);
  return ops.add(hidden, out);
}

` : ""}/** The mixer of a layer's DeltaNet or attention block. */
function mixerOf(block: GatedDeltaNet<AttentionLinear> | Qwen3Attention<AttentionLinear>, kind: Qwen38Layer["kind"]): Mixer {
  if (kind === "deltanet") {
    const delta = block as GatedDeltaNet<AttentionLinear>;
    return (normed, cache, phase) => delta.forward(normed, cache as GatedDeltaCache, phase);
  }
  const attention = block as Qwen3Attention<AttentionLinear>;
  return (normed, cache, phase) => attention.forward(normed, cache, phase, null);
}

export class ${c.className} implements MlxDeclaredGraph, MlxTokenPhases {
  readonly graph = ${constName(c)};
  readonly config: ModelConfig;
  readonly composition: Composition;
  readonly weightsBytes: number;
${c.wiredWeights ? `  /** Its weights are wired for each generation, whatever their share of the working set (measured for this
   *  composition). */
  readonly wiredWeights = true;
` : ""}  /** Base path for LoRA target keys (weights carry the language_model prefix). */
  readonly prefixBase = PREFIX;
  readonly loraState = new LoraState();
  /** The full-attention layer whose cache supplies logical positions. */
  readonly faIdx: number;
  readonly embed: QuantizedEmbedding;
  readonly finalNorm: RMSNorm;
  readonly lmHead: QuantizedLinear;
  /** Prepared image and video embeddings with request mRoPE positions, layer
   *  taps for drafts (verify's capture), bounded prefill workspace; adapters ${adapterSafe ? "at\n   *  every width" : "only\n   *  when the composition mounts them"}. */
  readonly graphCapabilities: GraphCapabilities;
${c.verifyRoundCosts ? `  /** Verify forward cost by rows relative to one decode step (${c.verifyRoundCosts.host}). */
  readonly verifyRoundCosts: ReadonlyMap<number, number> = new Map([
    ${c.verifyRoundCosts.costs.map(([rows, cost]) => `[${rows}, ${cost}]`).join(", ")},
  ]);
` : ""}
  readonly prefillChunk: MlxPrefillChunk;
  readonly prefillTail: MlxWidthTable<RowsForward>;
  readonly decode: MlxWidthTable<RowsForward>;
  readonly verify: MlxWidthTable<RowsForward>;
  readonly appendKnown: MlxWidthTable<MlxPhaseForward>;
  readonly logits: MlxWidthTable<MlxHeadForward>;
${k.knownSpans ? `  /** Committed known-token spans: the append keeps the native affine arithmetic while Trellis shares up to four rows. */
  readonly knownSpans: MlxKnownSpans = Object.freeze({
    maxChunkSize: (state: readonly Cache[]) => qwenAppendChunkSize(state[0]!.offset),
  });
` : ""}
  /** The generic blocks: ${k.knownSpans ? "prompt chunks and tails on the GPU, wide windows, media positions, committed appends"
    : promptProjection ? `every operation above ${wider ? wider.maxRows : widest} rows (prompt pieces with the prompt blocks' projections), media positions${k.mixed === "packed" ? ", mixed rows" : ""}`
    : `prompt chunks, every operation above ${widest} rows, media positions${k.mixed === "packed" ? ", mixed rows" : ""}`}. */
  readonly #generic: Step[];
${k.mixed === "packed" ? `  /** The generic layers, which a mixed forward drives block by block. */
  readonly #layers: readonly Qwen3Layer[];
  /** The composition's \`packedMixedMlp\`: the groups of a mixed forward share one MLP call. */
  readonly #packMixedMlp: boolean;
` : ""}  readonly #one: Step[];
  readonly #twoToThree: Step[];
  readonly #four: Step[];
  readonly #fiveToEight: Step[];
${wider ? `  /** Operations above eight rows of at most WIDER_ROWS rows: every layer on the wider layers (the generic blocks when
   *  the composition mounts adapters). */
  readonly #wider: Step[];
${packedWider ? `  /** Each layer's MLP on the wider layers (the post-attention-normed hidden to its output, owned), for a mixed
   *  forward's packed MLP rows. */
  readonly #widerMlp: ((post: MlxArray) => MlxArray)[];
` : ""}` : ""}  /** The block sets of the classes up to eight rows, by class. */
  readonly #narrow: Readonly<Record<Exclude<MlxWidthClass, "wider">, Step[]>>;
  readonly #loraTargets: ReadonlyMap<string, QuantizedLinear>;
  readonly #mrope: { readonly invFreq: MlxArray; readonly rotaryDims: number };
  /** The artifact's R1 basis, from its fold record (read once, here). */
  readonly #residualBasis: ResidualBasis | undefined;
${drafter ? `  /** The kernels a bound drafter's own 4-bit projections run, per width class of their rows. */
  readonly #draftProjections: MlxWidthTable<DraftProjectionKernel>;
` : ""}  /** What this graph created and frees in \`dispose\`: its constants${ane ? " and the Neural Engine layers' programs" : ""}. */
  readonly #owned: { dispose(): void }[] = [];
  /** The attention cache of layer \`layer\` the composition's KV scheme names, over one \`makeCache\` call's masks. */
  readonly #attentionCache: (masks: AttentionMasks, layer: number) => Cache;

  constructor(weights: Weights, config: ModelConfig, composition: Composition) {
    // The 5.8-family export's names and stored conventions, mapped onto the
    // canonical space before the first tensor is read.
    const view = qwen35WeightsView(weights);
    if (view) weights.setView(view);
    const t = config.text;
    this.config = config;
    this.composition = composition;
    this.#residualBasis = qwenResidualBasis(this.graph, config.modelDir);
${drafter ? `    // A bound drafter's own 4-bit, group-64 projections, over the weights it quantized (borrowed).
    const draftAffine = (w: MlxArray, scales: MlxArray, biases: MlxArray) =>
      new QuantizedLinear(w, scales, biases, { bits: 4, groupSize: 64, mode: "affine" });
    this.#draftProjections = Object.freeze<MlxWidthTable<DraftProjectionKernel>>({
${[...NARROW, "wider" as const].map(cls => {
  const layer = cls === "wider" ? null : drafter[cls];
  return `      ${cls}: (w, scales, biases) => { const kernel = ${layer ? `new ${layer}(draftAffine(w, scales, biases))` : "draftAffine(w, scales, biases)"}; return x => kernel.forward(x); },`;
}).join("\n")}
    });
` : ""}    this.weightsBytes = [...weights.shards.files.values()].reduce((a, f) => a + f.mmap.size, 0);
    this.faIdx = t.fullAttentionInterval - 1;
    this.embed = QuantizedEmbedding.load(weights, \`\${PREFIX}.embed_tokens\`, config);
    const generic = LAYERS.map((_, i) => new Qwen3Layer(weights, config, i));
    this.finalNorm = new RMSNorm(weights.tensor(\`\${PREFIX}.norm.weight\`), t.rmsNormEps);
    this.lmHead = QuantizedLinear.load(weights, "language_model.lm_head", config);
    const ropeDims = Math.trunc(t.headDim * t.partialRotaryFactor);
    this.#mrope = { invFreq: mropeInvFreq(ropeDims, t.ropeParameters.full_attention?.ropeTheta ?? 10000), rotaryDims: ropeDims };

    // DeltaNet's q/k scale as the weight of the norm before it: bit-identical
    // (see GatedDeltaNet.qkScale), 96 fewer kernels per forward.
    const invScale = Math.pow(t.linearKeyHeadDim, -0.5);
    const qkScale = { q: ops.filledBf16(invScale * invScale, t.linearKeyHeadDim), k: ops.filledBf16(invScale, t.linearKeyHeadDim) };
    for (const layer of generic) if (layer.linearAttn) layer.linearAttn.qkScale = qkScale;
    this.#owned.push(qkScale.q, qkScale.k, this.#mrope.invFreq);
${projectionSets.length || ane || promptProjection ? `
    /** The block of layer \`i\` whose projections \`project\` builds from each one's stored width. */
    const blockOf = (i: number, project: (linear: QuantizedLinear, path: string, width: number) => AttentionLinear) => {
      const spec = LAYERS[i]!, block = \`\${PREFIX}.layers.\${i}.\${spec.kind === "deltanet" ? "linear_attn" : "self_attn"}\`;
      const names = spec.kind === "deltanet" ? DELTANET_PROJECTIONS : ATTENTION_PROJECTIONS;
      const widths = new Map<string, number>(names.map((name, j) => [\`\${block}.\${name}\`, spec.projections[j]!]));
      const load: AttentionLinearLoader<AttentionLinear> = (w, path, c) => project(QuantizedLinear.load(w, path, c), path, widths.get(path)!);
      if (spec.kind === "attention") return new Qwen3Attention(weights, config, block, load);
      const delta = new GatedDeltaNet(weights, config, block, load);
      delta.qkScale = qkScale;
      return delta;
    };` : ""}
    const mlpOf = (i: number) => generic[i]!.mlp as TrellisMlp;
    const genericStep = (layer: Qwen3Layer): Step => (x, cache, phase, mrope, width) => layer.forward(x, cache, phase, mrope, width);
    this.#generic = generic.map(genericStep);
${k.mixed === "packed" ? `    this.#layers = generic;
    this.#packMixedMlp = composition.packedMixedMlp;
` : ""}
${narrowBlock}
${promptProjection ? `
    // The attention and DeltaNet blocks of the prompt operations: every projection on ${promptProjection}, unless
    // the composition mounts adapters (then the generic blocks', whose QuantizedLinear applies them).
    const promptBlocks = composition.adapters ? null : LAYERS.map((_, i) => blockOf(i, linear => new ${promptProjection}(linear)));
` : ""}
${ane ? `${splitsProjections ? `    // The Neural Engine operations, one per range in ANE_RANGES, built largest
    // first. Layers run one at a time and the scheduler runs one operation at a
    // time, so each MLP split shares the previous split's program (identical
    // within a range) or its buffers (the largest range's), and projections of
    // one shape share theirs: one loaded program per shape and range, the
    // programs \`anePrograms\` lists, loaded from the compile cache the install
    // step filled (nothing compiles here).` : `    // The Neural Engine operations, one per range in ANE_RANGES, built largest
    // first. Layers run one at a time and the scheduler runs one operation at a
    // time, so each MLP split shares the previous split's program (identical
    // within a range) or its buffers (the largest range's): one loaded program
    // per range, the programs \`anePrograms\` lists, loaded from the compile
    // cache the install step filled (nothing compiles here).`}
    let ane: { readonly minRows: number; readonly rows: number; readonly steps: Step[] }[] | null = null;
    if (composition.aneBridge && !composition.adapters) {
      const started = performance.now();
${splitsProjections ? `      const affineShares = new Map<string, ${ane.affine}>();
` : ""}      let mlpShare: AneMlpSplit | undefined;
      ane = [...ANE_RANGES].reverse().map(({ minRows, rows }) => ({ minRows, rows, steps: generic.map((layer, i) => {
        const spec = LAYERS[i]!, { gate, up, down } = mlpOf(i);
        const split = mlpShare = (spec.downLayout === "k3Interleaved" ? ${ane.k3InterleavedMlp} : ${ane.rowMajorMlp})
          .build(gate, up, down, ANE_FRACTION, rows, mlpShare);
        this.#owned.push(split);
${splitsProjections ? `        const block = blockOf(i, (linear, path) => {
          if (!ANE_PROJECTIONS.has(path.slice(path.lastIndexOf(".") + 1))) return ${promptProjection ? `new ${promptProjection}(linear)` : "linear"};
          const shape = \`\${linear.inFeatures}:\${linear.outFeatures}\`;
          const projection = ${ane.affine}.build(linear, ANE_FRACTION, rows, affineShares.get(shape));
          affineShares.set(shape, projection);
          this.#owned.push(projection);
          return projection;
        });` : `        const block = ${promptProjection ? "promptBlocks![i]!" : "(layer.linearAttn ?? layer.selfAttn)!"};`}
        return layerStep(layer, mixerOf(block, spec.kind), post => split.forward(post, compiledSwiglu));
      }) })).reverse();
      console.error(\`[ane] prefill programs ready in \${Math.round(performance.now() - started)} ms\`);
    }

` : ""}    const targets = new Map<string, QuantizedLinear>();
    ${adapterSafe ? "" : "if (composition.adapters) "}generic.forEach((layer, i) => {
      const p = \`\${PREFIX}.layers.\${i}\`;
      if (layer.selfAttn) for (const name of ATTENTION_PROJECTIONS) targets.set(\`\${p}.self_attn.\${name}\`, layer.selfAttn[projectionMember[name]]);
      if (layer.linearAttn) for (const name of DELTANET_PROJECTIONS) targets.set(\`\${p}.linear_attn.\${name}\`, layer.linearAttn[projectionMember[name]]);
    });
    this.#loraTargets = targets;

    this.graphCapabilities = declareGraph({
      media: { input: "embeddings+positions", video: true }, hiddenLayerTaps: true,
      prefill: { boundedWorkspace: true },
${adapterSafe ? "" : "      adapters: composition.adapters ? {} : { batched: false, mountable: false },\n"}    });

${legoKv ? `    // The attention cache the composition's KV scheme names, chosen once: affine ${legoKv.bits}-bit group ${legoKv.groupSize}
    // composes ${kvLego}; ${c.kv === "every" ? `every other scheme its own lego, with the dense TurboQuant read
    // (\`composedFullAttentionCache\`).` : `bf16, the table's other scheme for this graph, the plain cache with the dense
    // TurboQuant read (\`composedFullAttentionCache\`).`}
    const kv = composition.kv;
    this.#attentionCache = kv.kind === "affine" && kv.bits === ${legoKv.bits} && kv.groupSize === ${legoKv.groupSize}
      ? (masks: AttentionMasks) => new ${kvLego}(kv.quantizedKvStart ?? 0, composedAffineKernels(kv.kernels), masks${decodeBlock})
      : (masks: AttentionMasks, layer: number) => composedFullAttentionCache(kv, layer, masks, denseTurboQuantKernels${windowRead});`
  : `    // The attention cache the composition's KV scheme names: that scheme's own lego, with the dense TurboQuant read
    // (\`composedFullAttentionCache\`).
    const kv = composition.kv;
    this.#attentionCache = (masks: AttentionMasks, layer: number) => composedFullAttentionCache(kv, layer, masks, denseTurboQuantKernels${windowRead});`}

    this.#narrow = Object.freeze({ ${NARROW.map(cls => `${cls}: this.#${cls}`).join(", ")} });
    const ${NARROW.filter(cls => !generic(cls)).map(cls => `${cls} = this.#${cls}`).join(", ")}, wide = this.#generic${wider ? ", wider = this.#wider" : ""};
    const embed = this.embed${ane ? ", split = ane" : ""};
${promptProjection ? `    // Prompt pieces above ${wider ? "WIDER_ROWS" : "eight"} rows: the prompt blocks, then the generic blocks' MLP.
    const prompt = promptBlocks ? generic.map((layer, i) => layerStep(layer, mixerOf(promptBlocks[i]!, LAYERS[i]!.kind),
      post => mlpOf(i).forward(post, "window", "wider"))) : wide;
    const gpu: RowsForward = (ids, state, capture) =>${wider ? ` ids.size <= WIDER_ROWS
      ? ${widerRun("window")}
      : ` : "\n      "}this.#bounded(prompt, embed.encode(ids), state, "window", capture, null, "wider");`
  : `    const gpu: RowsForward = (ids, state, capture) =>${wider ? ` ids.size <= WIDER_ROWS
      ? ${widerRun("window")}
      : ` : "\n      "}this.#bounded(wide, embed.encode(ids), state, "window", capture, null, "wider");`}
${ane ? `    this.prefillChunk = Object.freeze(split
      ? { gpu, ane: Object.freeze(split.map(({ minRows, rows, steps }) => Object.freeze({ minRows, rows,
        forward: (ids: MlxArray, state: Cache[], capture: MlxLayerCapture | null) =>
          this.#bounded(steps, embed.encode(ids), state, "window", capture, null, "wider") }))) }
      : { gpu });` : "    this.prefillChunk = Object.freeze({ gpu });"}
    this.prefillTail = Object.freeze({
${NARROW.map(cls => entry(cls, "window", "bounded")).join("\n")}
      wider: gpu,
    });
    this.decode = Object.freeze({
${NARROW.map(cls => entry(cls, "decode", "lazy")).join("\n")}
      wider: (ids: MlxArray, state: Cache[], capture: MlxLayerCapture | null) =>${wider ? ` ids.size <= WIDER_ROWS
        ? ${widerRun("decode")}
        :` : ""} this.#lazy(wide, embed.encode(ids), state, "decode", capture, null, "wider"),
    });
    this.verify = Object.freeze({
${NARROW.map(cls => entry(cls, "window", k.widerVerify)).join("\n")}
      wider: (ids: MlxArray, state: Cache[], capture: MlxLayerCapture | null) =>${wider ? ` ids.size <= WIDER_ROWS
        ? ${widerRun("window")}
        :` : ""} this.#${k.widerVerify}(wide, embed.encode(ids), state, "window", capture, null, "wider"),
    });
${k.knownSpans ? `    // Known tokens read committed on the generic blocks, evaluated per layer above four rows.
    this.appendKnown = byWidth<MlxPhaseForward>(width => width === "fiveToEight" || width === "wider"
      ? (ids, state) => this.#bounded(wide, embed.encode(ids), state, "committed", null, null, width)
      : (ids, state) => this.#lazy(wide, embed.encode(ids), state, "committed", null, null, width));`
  : wider ? `    // Known tokens append as windows (no committed spans) on the block set of their width (above eight rows the
    // wider layers' up to WIDER_ROWS rows, the generic blocks above), one lazy graph.
    this.appendKnown = byWidth<MlxPhaseForward>(width => {
      if (width === "wider")
        return (ids, state) => this.#lazy(ids.size <= WIDER_ROWS ? wider : wide, embed.encode(ids), state, "window", null, null, width);
      const steps = this.#narrow[width];
      return (ids, state) => this.#lazy(steps, embed.encode(ids), state, "window", null, null, width);
    });` : `    // Known tokens append as windows (no committed spans) on the block set of their width, one lazy graph.
    this.appendKnown = byWidth<MlxPhaseForward>(width => {
      const steps = width === "wider" ? wide : this.#narrow[width];
      return (ids, state) => this.#lazy(steps, embed.encode(ids), state, "window", null, null, width);
    });`}
    const head = this.lmHead, plain: MlxHeadForward = hidden => head.forward(hidden);
${head ? `    if (composition.adapters) this.logits = atEveryWidth(plain);
    else {
      const ${heads.map(layer => `${headName(layer)} = new ${layer}(head)`).join(", ")};
      this.logits = Object.freeze<MlxWidthTable<MlxHeadForward>>({
        one: plain,
${NARROW.filter(cls => cls !== "one").map(cls => {
  const layer = head[cls as Exclude<Narrow, "one">];
  return `        ${cls}: ${layer ? `hidden => ${headName(layer)}.forward(hidden)` : "plain"},`;
}).join("\n")}
        wider: ${wider ? `hidden => hidden.size / hidden.shape.at(-1)! <= WIDER_ROWS ? ${headName(wider.head)}.forward(hidden) : plain(hidden)` : "plain"},
      });
    }` : "    this.logits = atEveryWidth(plain);"}
    console.error(\`[graph] \${this.graph} \${COMPOSITION.fingerprint}${ane ? ` ane-prefill=\${ane ? "on" : "off"}` : ""} adapters=\${composition.adapters ? "on" : "off"}\`);
  }

  /** Frees what the graph created: its constants${ane ? " and the Neural Engine programs" : ""}. The weights stay the loader's. Call once,
   *  after the last forward. */
  dispose(): void { for (const resource of this.#owned.splice(0)) resource.dispose(); }

${legoKv ? `  /** One cache per layer, the attention caches sharing one mask memo: the cache the composition's KV scheme names
   *  (affine ${legoKv.bits}-bit group ${legoKv.groupSize}: ${kvLego}, bf16 through its first append, then the KV4 decode
   *  and folded window reads; ${c.kv === "every" ? `every other scheme: \`composedFullAttentionCache\`, the plain cache,
   *  ${k.bf16WindowRead === null ? "" : `its windows ${k.bf16WindowRead}, `}bf16-first affine or per-layer affine with the composition's read kernels,
   *  or bf16-first TurboQuant with the dense read). */` : `bf16: \`composedFullAttentionCache\`'s plain cache with the dense TurboQuant read). */`}`
  : `  /** One cache per layer, the attention caches sharing one mask memo: the cache the composition's KV scheme names
   *  (\`composedFullAttentionCache\`: the plain cache, bf16-first affine or per-layer affine with the composition's read
   *  kernels, or bf16-first TurboQuant with the dense read). */`}
  makeCache(): Cache[] {
    const masks = new AttentionMasks();
    return LAYERS.map((layer, i) => layer.kind === "deltanet" ? new SSMCache() : this.#attentionCache(masks, i));
  }

  /** The LoRA-applying projections${adapterSafe ? ": every attention and DeltaNet projection (the Trellis MLP is no adapter target)"
    : ", when the composition mounts adapters; none otherwise"}. */
  loraTargets(): Map<string, QuantizedLinear> { return new Map(this.#loraTargets); }

  bindMediaInput(input: Vision): MlxPromptInput { return bindQwenMediaInput(this, input.embeddings, input.mrope!); }
  mediaEncoders(modelDir: string): Promise<MediaEncoders> { return qwenMediaEncoders(this, modelDir); }
  /** The drafter's ports: its hidden-layer taps (verify's capture), embedding and head, ${drafter ? "the" : "and the"} residual basis
   *  (R1 seed, final-norm gain) the artifact's fold record declares${drafter ? ", and the kernels of its own projections" : ""}. */
  draftTarget(_caches: Cache[]): TargetView { return qwenDraftTarget(this, this.#residualBasis${drafter ? ", this.#draftProjections" : ""}); }

  // ── The layer loops. Each consumes h0 and returns the post-final-norm hidden,
  // owned by the caller; \`capture\` sees every layer's output, then the final hidden.
${asyncEvery !== null ? `
  /** Every layer, the graph submitted every ASYNC_EVERY layers. */
  #submitted(steps: Step[], h0: MlxArray, cache: Cache[], phase: ReadPhase, capture: MlxLayerCapture | null,
    width: MlxWidthClass): MlxArray {
    let h: MlxArray | null = h0;
    try {
      for (let i = 0; i < steps.length; i++) {
        const next = steps[i]!(h, cache[i]!, phase, null, width);
        h.dispose();
        h = next;
        if (i % ASYNC_EVERY === ASYNC_EVERY - 1 && i + 1 < steps.length) ops.asyncEvalAll([h]);
        capture?.(i, h);
      }
      const out = this.finalNorm.forward(h);
      h.dispose();
      h = null;
      capture?.(steps.length, out);
      return out;
    } finally { h?.dispose(); }
  }
` : ""}
  /** Every layer, each layer's output and recurrent state evaluated before the next. */
  #bounded(steps: Step[], h0: MlxArray, cache: Cache[], phase: ReadPhase, capture: MlxLayerCapture | null,
    mrope: MropeForwardState | null, width: MlxWidthClass): MlxArray {
    let h: MlxArray | null = h0;
    try {
      for (let i = 0; i < steps.length; i++) {
        const next = steps[i]!(h, cache[i]!, phase, mrope, width);
        h.dispose();
        h = next;
        const state = cache[i]!.state();
        try { ops.evalAll([h, ...state]); }
        finally { if (cache[i]!.stateNeedsDispose) for (const a of state) a.dispose(); }
        capture?.(i, h);
      }
      const out = disposing(h, this.finalNorm.forward(h));
      h = null;
      capture?.(steps.length, out);
      return out;
    } finally { h?.dispose(); }
  }

  /** Every layer, nothing evaluated. */
  #lazy(steps: Step[], h0: MlxArray, cache: Cache[], phase: ReadPhase, capture: MlxLayerCapture | null,
    mrope: MropeForwardState | null, width: MlxWidthClass): MlxArray {
    let h: MlxArray | null = h0;
    try {
      for (let i = 0; i < steps.length; i++) {
        const next = steps[i]!(h, cache[i]!, phase, mrope, width);
        h.dispose();
        h = next;
        capture?.(i, h);
      }
      const out = disposing(h, this.finalNorm.forward(h));
      h = null;
      capture?.(steps.length, out);
      return out;
    } finally { h?.dispose(); }
  }

${k.mixed === "perGroup" ? `  /** Mixed rows in one forward: each group runs its own phase's operation for its width, with its capture. The
   *  groups share no call: Trellis already takes its row kernels per group. */
  readonly mixed: MlxMixedForward = work => {
    const results: MlxArray[] = [];
    try {
      for (const group of work) {
        const table = group.phase === "decode" ? this.decode : group.phase === "verify" ? this.verify : this.prefillTail;
        results.push(table[group.width](group.ids, group.state, group.capture));
      }
      return results;
    } catch (error) { for (const result of results) result.dispose(); throw error; }
  };` : `  /** Mixed rows in one forward on the generic blocks: each group's attention and DeltaNet read its own caches in its
   *  own phase (a verify group reads a window); the MLP rows of the groups that may share it run as one call at
   *  \`width\`, the class of the groups' rows, when the composition packs mixed MLP, a verify group keeping its own
   *  geometry (\`mapTokenGroups\`)${packedWider ? `; a call above eight rows of at most WIDER_ROWS rows runs the wider
   *  layers' MLP` : ""}. A prompt group above four rows evaluates each layer and the groups' recurrent state
   *  before the next, as the prompt tail does. */
  readonly mixed: MlxMixedForward = (groups, width) => {
    const hidden: MlxArray[] = [];
    const results: MlxArray[] = [];
    const bounded = groups.some(group => group.phase === "window" && (group.width === "fiveToEight" || group.width === "wider"));
    try {
      for (const group of groups) hidden.push(this.embed.encode(group.ids));
      for (const [i, layer] of this.#layers.entries()) {
        const mids: MlxArray[] = [];
        try {
          for (const [row, group] of groups.entries())
            mids.push(layer.forwardAttn(hidden[row]!, group.state[i]!, group.phase === "decode" ? "decode" : "window"));
${packedWider ? `          const mlp = this.#widerMlp[i]!;
          const outputs = mapTokenGroups(groups, mids, (h, w) => w === "wider" && h.size / h.shape.at(-1)! <= WIDER_ROWS
            ? mlpBlock(layer, h, mlp) : layer.forwardMlp(h, "window", w), width, this.#packMixedMlp);`
  : `          const outputs = mapTokenGroups(groups, mids, (h, w) => layer.forwardMlp(h, "window", w), width, this.#packMixedMlp);`}
          for (const row of groups.keys()) { hidden[row]!.dispose(); hidden[row] = outputs[row]!; }
          // Materialize every layer's recurrent tail as well as the residual: it is not a dependency of the hidden
          // and otherwise pins prefill buffers.
          if (bounded) {
            const caches = groups.map(group => group.state[i]!);
            const state = caches.map(cache => cache.state());
            try { ops.evalAll([...hidden, ...state.flat()]); }
            finally { for (const [row, cache] of caches.entries())
              if (cache.stateNeedsDispose) for (const value of state[row]!) value.dispose(); }
          }
          for (const [row, group] of groups.entries()) group.capture?.(i, hidden[row]!);
        } finally { for (const mid of mids) mid.dispose(); }
      }
      for (const [row, group] of groups.entries()) {
        const out = this.finalNorm.forward(hidden[row]!); results.push(out);
        group.capture?.(this.#layers.length, out);
      }
      return results;
    } catch (error) { for (const result of results) result.dispose(); throw error; }
    finally { for (const h of hidden) h.dispose(); }
  };`}

  /** The scoring forward (\`MlxSequenceForward\`): \`prefillTail\`'s operation for \`width\`, without a capture. */
  sequenceHidden(ids: MlxArray, cache: Cache[], width: MlxWidthClass): MlxArray {
    return this.prefillTail[width](ids, cache, null);
  }

  /** The vision prefill phase: spliced embeddings (borrowed) at request positions, on the generic blocks. */
  prefillEmbeddingsAtPositions(embeds: MlxArray, cache: Cache[], positions: MlxArray, width: MlxWidthClass): MlxArray {
    return this.#positioned(ops.contiguous(embeds), cache, "window", positions, width);
  }

  /** A media prompt's token ids after a reused prefix, at request positions, on the generic blocks. */
  prefillAtPositions(ids: MlxArray, cache: Cache[], positions: MlxArray, width: MlxWidthClass): MlxArray {
    return this.#positioned(this.embed.encode(ids), cache, "window", positions, width);
  }

  /** One new token per row at request positions, including different positions for each row, on the generic blocks. */
  decodeAtPositions(ids: MlxArray, cache: Cache[], positions: MlxArray, width: MlxWidthClass): MlxArray {
    return this.#positioned(this.embed.encode(ids), cache, "decode", positions, width);
  }

  /** The vision prefill phase for direct generation: spliced embeddings (borrowed) at the cache's own positions, on
   *  the block set of their width (${asyncEvery !== null ? `the one-sequence row sets up to ${widest} rows, the generic blocks above`
    : `the one-sequence row sets up to ${widest} rows, one lazy graph; the generic blocks\n   *  above, each layer evaluated before the next`}). */
  prefillEmbeddings(embeds: MlxArray, cache: Cache[], _bidir: MlxArray | null, _ids: MlxArray | null,
    _multimodal: MlxArray | null, width: MlxWidthClass): MlxArray {
    const h0 = ops.contiguous(embeds);
    return ${boundedWidths} ? this.#bounded(this.#generic, h0, cache, "window", null, null, width)
      : ${asyncEvery !== null ? `this.#submitted(this.#narrow[width], h0, cache, "window", null, width)`
        : `this.#lazy(this.#narrow[width], h0, cache, "window", null, null, width)`};
  }

  /** Every layer over \`h0\` (consumed) on the generic blocks at request positions: a window above four rows
   *  evaluates each layer before the next; decode and narrower windows stay one lazy graph. */
  #positioned(h0: MlxArray, cache: Cache[], phase: ReadPhase, positions: MlxArray, width: MlxWidthClass): MlxArray {
    const mrope = { posIds: positions, ...this.#mrope };
    return phase === "window" && (width === "fiveToEight" || width === "wider")
      ? this.#bounded(this.#generic, h0, cache, phase, null, mrope, width)
      : this.#lazy(this.#generic, h0, cache, phase, null, mrope, width);
  }

  /** The plain head at any rows, for the scoring forward; the phases index \`logits\` by the rows they project. */
  logitsFromHidden(h: MlxArray): MlxArray { return this.lmHead.forward(h); }
}

/** Each LoRA target's member on its block. */
const projectionMember = {
  q_proj: "qProj", k_proj: "kProj", v_proj: "vProj", o_proj: "oProj",
  in_proj_qkv: "inProjQkv", in_proj_z: "inProjZ", in_proj_a: "inProjA", in_proj_b: "inProjB", out_proj: "outProj",
} as const;
`;
}

function constName(c: Qwen38Composition): string { return c.graphConstant; }

if (import.meta.main) {
  const { values, positionals } = parseArgs({
    args: Bun.argv.slice(2),
    options: { out: { type: "string" }, describe: { type: "boolean" }, help: { type: "boolean", short: "h" } },
    allowPositionals: true,
  });
  const stems = Object.keys(QWEN38_COMPOSITIONS).join("|");
  const usage = `Usage: bun packages/inference/scripts/gen-qwen38.ts <source> <${stems}> [--out <dir>]
       bun packages/inference/scripts/gen-qwen38.ts --describe <snapshot-dir>
Compile a Qwen3.8-27B Trellis composition into its entire graph module. <source> is the artifact's
snapshot directory (config.json, model.safetensors.index.json and the shard headers; no tensor bytes)
or the description JSON --describe prints for one (tests/fixtures/qwen38-graphs). The source's graph
fingerprint must be the composition's. The default output directory is src/models/qwen.
Compositions: ${Object.values(QWEN38_COMPOSITIONS).map(c => `${c.stem} (${c.artifact} on ${c.device}, ` +
  `KV ${c.kv === "every" ? "every scheme" : `affine ${c.kv.bits}/${c.kv.groupSize}`}, depth ${depthText(c.draftDepth)}, ` +
  `chunk ${c.prefillChunkTokens}, ${c.maxRows ?? "any"} rows, ANE ${c.aneBridge ? "yes" : "no"})`).join("; ")}.`;
  if (values.help) { console.log(usage); process.exit(0); }
  if (values.describe) {
    if (positionals.length !== 1) { console.error(usage); process.exit(1); }
    const { source, fingerprint, widths, layers } = await describeArtifact(positionals[0]!);
    console.log(`{\n "source": ${JSON.stringify(source)},\n "fingerprint": ${JSON.stringify(fingerprint)},\n` +
      ` "widths": ${JSON.stringify(widths)},\n "layers": [\n` +
      layers.map(layer => `  ${JSON.stringify(layer)}`).join(",\n") + "\n ]\n}");
    process.exit(0);
  }
  const [source, stem] = positionals;
  if (!source || !stem || positionals.length !== 2 || !(stem in QWEN38_COMPOSITIONS)) {
    console.error(usage);
    process.exit(1);
  }
  const outPath = `${values.out ?? GENERATED_DIR}/${stem}.ts`.replace(/\/\/+/g, "/");
  await Bun.write(outPath, emitQwen38Graph(await readDescription(source), stem));
  console.log(`wrote ${outPath}`);
}
