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
// committed file differs. Whether a regenerated graph computes the same bits is
// tests/models/qwen38-m4pro-identity.test.ts.

import { parseArgs } from "node:util";
import { loadModelConfig, quantFor } from "../src/artifacts/config";

/** The facts that fix one generated graph. The fingerprint and device choose the
 * layer and kernel tables written into the file; `kv` names the scheme whose
 * cache lego `makeCache` composes; `aneBridge` writes the Neural Engine chunk
 * operation. `draftDepth`, `prefillChunkTokens` and `maxRows` record what the
 * file was generated and gated for. Its exported `COMPOSITION`, the loader's
 * table entry, constrains the fingerprint, the device, the KV schemes (bf16 and
 * `kv`) and the row bound; the constructor builds from the record it is handed. */
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
  readonly kv: { readonly kind: "affine"; readonly bits: number; readonly groupSize: number };
  readonly draftDepth: { readonly adaptive: true; readonly max: number } | number;
  readonly prefillChunkTokens: number;
  readonly maxRows: number;
  readonly aneBridge: boolean;
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

export interface Qwen38Description {
  /** Where the description was read: repo@revision. */
  readonly source: string;
  readonly fingerprint: string;
  readonly layers: readonly Qwen38LayerSpec[];
}

export const DELTANET_PROJECTIONS = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"] as const;
export const ATTENTION_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"] as const;

/** The width classes up to eight rows (`MlxWidthTable` without `wider`, which is the generic blocks). */
const NARROW = ["one", "twoToThree", "four", "fiveToEight"] as const;
type Narrow = (typeof NARROW)[number];

/** Each device's kernels: the one-kernel layer of every width class. `null`
 * projections keep the generic blocks' QuantizedLinear (MLX's own qmv/qmm). */
interface DeviceKernels {
  readonly label: string;
  readonly mlp: Readonly<Record<Narrow, {
    readonly sameGateUp: string; readonly mixedGateUp: string; readonly rowMajorDown: string; readonly k3InterleavedDown: string;
  }>>;
  readonly projections: Readonly<Record<Narrow, { readonly 3: string; readonly 4: string } | null>>;
  /** The head of each class; `one` and `wider` keep QuantizedLinear. */
  readonly head: Readonly<Record<Exclude<Narrow, "one">, string>>;
  /** Submit the graph every N layers at decode and verify widths up to 8. */
  readonly asyncEvery: number;
  readonly kvLego: string;
  readonly ane: { readonly fraction: number; readonly rowMajorMlp: string; readonly k3InterleavedMlp: string;
    readonly affine: string; readonly projections: readonly string[] };
}

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
    kvLego: "Bf16FirstKv4Head256Cache",
    ane: { fraction: 0.55, rowMajorMlp: "AneTrellisMlpSplit", k3InterleavedMlp: "AneTrellisMlpSplitK3i", affine: "AneAffineSplit",
      projections: ["in_proj_z", "out_proj", "o_proj"] },
  },
});

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
  // The fingerprint function lives with the graph that introduced it and loads the native library.
  const { qwen38TrellisTqFingerprint } = await import("../src/models/qwen/qwen38-27b-trellis-tq");
  const t = config.text;
  const downNames = Array.from({ length: t.numHiddenLayers }, (_, i) => [
    `language_model.model.layers.${i}.mlp.down_proj.weight`, `model.language_model.layers.${i}.mlp.down_proj.weight`]).flat();
  const shapes = await storedShapes(dir, downNames);
  const layers = Array.from({ length: t.numHiddenLayers }, (_, i): Qwen38LayerSpec => {
    const prefix = `language_model.model.layers.${i}`;
    const kind = (i + 1) % t.fullAttentionInterval !== 0 ? "deltanet" : "attention";
    const bits = (path: string) => {
      const spec = quantFor(config.quantization, path);
      if (!spec) throw new Error(`${path}: no quantization entry`);
      return spec.bits;
    };
    const projections = kind === "deltanet"
      ? DELTANET_PROJECTIONS.map(name => bits(`${prefix}.linear_attn.${name}`))
      : ATTENTION_PROJECTIONS.map(name => bits(`${prefix}.self_attn.${name}`));
    const shape = shapes.get(downNames[2 * i]!) ?? shapes.get(downNames[2 * i + 1]!);
    if (!shape) throw new Error(`layer ${i}: no stored down_proj weight`);
    return { kind, projections, gate: bits(`${prefix}.mlp.gate_proj`), up: bits(`${prefix}.mlp.up_proj`),
      down: bits(`${prefix}.mlp.down_proj`), downLayout: shape.length === 3 ? "k3Interleaved" : "rowMajor" };
  });
  return { source: sourceOf(dir), fingerprint: qwen38TrellisTqFingerprint(config), layers };
}

export async function readDescription(source: string): Promise<Qwen38Description> {
  return source.endsWith(".json") ? await Bun.file(source).json() as Qwen38Description : describeArtifact(source);
}

// ── Template ─────────────────────────────────────────────────────────────────

const list = (names: readonly string[]) => `[${names.map(name => JSON.stringify(name)).join(", ")}]`;
const depthText = (depth: Qwen38Composition["draftDepth"]) => typeof depth === "number" ? `${depth}` : `adaptive up to ${depth.max}`;

/** The generated module's text. */
export function emitQwen38Graph(description: Qwen38Description, stem: string): string {
  const c = QWEN38_COMPOSITIONS[stem];
  if (!c) throw new Error(`no composition ${stem} (known: ${Object.keys(QWEN38_COMPOSITIONS).join(", ")})`);
  const k = KERNELS[c.device];
  if (!k) throw new Error(`${stem}: no kernel table for device ${c.device}`);
  if (description.fingerprint !== c.fingerprint)
    throw new Error(`${stem}: the source's graph fingerprint ${description.fingerprint} is not the composition's ${c.fingerprint}`);
  const layers = description.layers;
  const interleaved = layers.filter(layer => layer.downLayout === "k3Interleaved").length;
  const mixed = layers.map((layer, i) => layer.gate !== layer.up ? i : -1).filter(i => i >= 0);
  const deltanet = layers.filter(layer => layer.kind === "deltanet").length;
  const table = layers.map((layer, i) =>
    `  /* ${String(i).padStart(2)} */ { kind: ${JSON.stringify(layer.kind)}, projections: [${layer.projections.join(", ")}], ` +
    `gate: ${layer.gate}, up: ${layer.up}, down: ${layer.down}, downLayout: ${JSON.stringify(layer.downLayout)} },`).join("\n");
  const mlpLayers = [...new Set(Object.values(k.mlp).flatMap(m => [m.sameGateUp, m.mixedGateUp]))];
  const downLayers = [...new Set(Object.values(k.mlp).flatMap(m => [m.rowMajorDown, m.k3InterleavedDown]))];
  const affineLayers = [...new Set([...Object.values(k.projections).flatMap(p => p ? [p[3], p[4]] : []), ...Object.values(k.head)])];
  const one = k.mlp.one;
  const aneProjections = k.ane.projections;
  /** Blocks are built once per projection layer pair and shared by the classes that name it. */
  const blocksName = (p: { 3: string; 4: string }) => `${p[4][0]!.toLowerCase()}${p[4].slice(1).replace(/Linear$/, "")}Blocks`;
  const projectionSets = [...new Map(NARROW.flatMap(cls => k.projections[cls] ? [[blocksName(k.projections[cls]!), k.projections[cls]!] as const] : [])).entries()];
  const headName = (layer: string) => `${layer[0]!.toLowerCase()}${layer.slice(1).replace(/Linear$/, "")}Head`;
  const heads = [...new Set(Object.values(k.head))];
  const gateUpExpr = (m: DeviceKernels["mlp"][Narrow]) => m.sameGateUp === m.mixedGateUp ? `new ${m.sameGateUp}(gate, up)`
    : `spec.gate === spec.up ? new ${m.sameGateUp}(gate, up) : new ${m.mixedGateUp}(gate, up)`;
  const narrowSets = NARROW.map(cls => {
    const m = k.mlp[cls], p = k.projections[cls];
    return `      this.#${cls} = generic.map((layer, i) => {
        const spec = LAYERS[i]!, { gate, up, down } = mlpOf(i);
        const gateUp = ${gateUpExpr(m)};
        const project = spec.downLayout === "k3Interleaved" ? new ${m.k3InterleavedDown}(down) : new ${m.rowMajorDown}(down);
        return layerStep(layer, mixerOf(${p ? `${blocksName(p)}[i]!` : "(layer.linearAttn ?? layer.selfAttn)!"}, spec.kind), post => {
          using mid = gateUp.forward(post);
          return project.forward(mid);
        });
      });`;
  }).join("\n");
  const blockSets = projectionSets.map(([name, p]) =>
    `      const ${name} = LAYERS.map((_, i) => blockOf(i, (linear, _path, width) => width === 3 ? new ${p[3]}(linear) : new ${p[4]}(linear)));`).join("\n");

  const row = (operation: string, mlp: string, projections: string) =>
    `//   ${operation.padEnd(31)}${mlp.padEnd(44)}${projections}`.trimEnd();
  const classRows = NARROW.map(cls => {
    const m = k.mlp[cls], p = k.projections[cls];
    return row(cls === "one" ? "decode.one, verify.one" : `decode/verify.${cls}`, `${m.sameGateUp} + ${m.rowMajorDown}/K3i`,
      p ? `${p[3]}/${p[4]}` : "QuantizedLinear");
  }).join("\n");
  return `// GENERATED by packages/inference/scripts/gen-qwen38.ts from composition ${stem}; do not edit.
// Regenerate: bun packages/inference/scripts/gen-qwen38.ts <snapshot-or-description> ${stem}
// (tests/fixtures/qwen38-graphs holds the description it was generated from).
//
// Composition ${stem}:
//   artifact     ${description.source}, graph fingerprint ${c.fingerprint}
//   device       ${c.device} (${k.label})
//   KV           bf16, or affine ${c.kv.bits}-bit group ${c.kv.groupSize} (composes ${k.kvLego})
//   draft depth  ${depthText(c.draftDepth)}
//   prefill      ${c.prefillChunkTokens}-token chunks
//   rows         up to ${c.maxRows} per forward
//   Neural Engine bridge ${c.aneBridge ? "present" : "absent"}
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
${row("decode/verify.wider,", "the generic blocks (TrellisMlp)", "QuantizedLinear")}
${row("prefillChunk.gpu, prefillTail", "", "")}
${row("prefillChunk.ane", `${k.ane.rowMajorMlp}/K3i`, `${aneProjections.join(", ")} split`)}
//
// Mixed-width gate/up takes ${one.mixedGateUp} at one to four rows. Every attention block reads
// through the cache (${k.kvLego}: bf16 through its first append, then the fused 4-bit decode
// kernel and the folded-GQA window kernel; the bf16 cache: the stock SDPA). The head is part of each phase: logits.twoToThree ${k.head.twoToThree}, logits.four
// ${k.head.four}, logits.fiveToEight ${k.head.fiveToEight}, QuantizedLinear otherwise. Decode and verify up to 8 rows submit
// the graph every ${k.asyncEvery} layers; prompt chunks, tails and wider verify windows evaluate each layer
// and its recurrent state.
//
// Request shapes this graph serves:
//   - prompt chunks of exactly the composition's chunk size (prefillChunk.gpu; prefillChunk.ane when the
//     composition has the Neural Engine bridge and mounts no adapters) and prompt tails (prefillTail);
//   - decode, one position per sequence, and speculative verify windows of sequences x (depth + 1) positions
//     with the drafter's layer capture, for any number of sequences up to the composition's rows per forward
//     (decode[width], verify[width], logits[width]);
//   - prepared image and video embeddings with request mRoPE positions (forwardEmbeddingsAtPositions,
//     forwardHiddenAtPositions), on the generic blocks;
//   - committed-token appends (createAppend) on the generic blocks;
//   - mounted LoRA adapters when the composition mounts them: every operation then runs the generic blocks,
//     whose QuantizedLinear projections apply them, and the Neural Engine operation is not built.
// The loader refuses, from the declared capabilities and never by routing to another graph: adapter requests
// when the composition mounts none (adapter-batch-unsupported) and paged KV (paged-kv-batch-unsupported).
// ${k.kvLego}'s row layouts read with its KV4 kernels while they hold one unpadded row and with the
// composition's affine kernels while they hold more.

import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { ModelConfig } from "../../artifacts/config";
import type { Weights } from "../../artifacts/weights";
import type { Cache, GatedDeltaCache } from "../../contracts/mlx/cache";
import type { ResidualBasis, TargetView } from "../../contracts/mlx/draft-target";
import type {
  MlxDeclaredGraph, MlxHeadForward, MlxLayerCapture, MlxPhaseForward, MlxPrefillChunk, MlxTokenAppend, MlxTokenPhases,
  MlxVerifyForward, MlxWidthTable,
} from "../../contracts/mlx/graph";
import type { MediaEncoders, MlxPromptInput, Vision } from "../../contracts/mlx/media";
import type { MropeForwardState } from "../../contracts/mlx/positions";
import type { TokenGroup } from "../../contracts/mlx/token-work";
import type { Composition } from "../../contracts/portable/composition";
import type { GraphCapabilities } from "../../contracts/portable/graph";
import { argmaxLastPosition } from "../../kernels/logits";
import { ${affineLayers.sort().join(", ")} } from "../../layers/affine-verify-linear";
import { ${[k.ane.affine, k.ane.rowMajorMlp, k.ane.k3InterleavedMlp].sort().join(", ")}, type AneMlpSplit } from "../../layers/ane-prefill-split";
import { disposing } from "../../layers/helpers";
import { LoraState } from "../../layers/lora";
import { RMSNorm } from "../../layers/normalization";
import { QuantizedEmbedding } from "../../layers/quantized-embedding";
import { QuantizedLinear } from "../../layers/quantized-linear";
import { qwenAppendChunkSize } from "../../layers/qwen-append";
import { mropeInvFreq } from "../../layers/qwen-mrope";
import { compiledSwiglu } from "../../layers/swiglu";
import { ${downLayers.sort().join(", ")} } from "../../layers/trellis-down";
import { ${mlpLayers.sort().join(", ")} } from "../../layers/trellis-gate-up";
import { TRELLIS_MATVEC_MAX_M } from "../../layers/trellis-linear";
import { composedAffineKernels } from "../../state/affine-attention";
import { AttentionMasks } from "../../state/attention-read";
import { ${k.kvLego}, composedFullAttentionCache } from "../../state/bf16-first-kv";
import { SSMCache } from "../../state/ssm";
import { denseTurboQuantKernels } from "../../state/turboquant-attention";
import { declareGraph } from "../capabilities";
import { atEveryWidth, deprecatedPhase, type ReadPhase } from "../graph";
import type { ServedComposition } from "../implementation";
import {
  GatedDeltaNet, Qwen3Attention, Qwen3Layer, type AttentionLinear, type AttentionLinearLoader, type TrellisMlp,
} from "./blocks";
import { qwen35WeightsView } from "./checkpoint";
import { bindQwenMediaInput, qwenDraftTarget, qwenMediaEncoders, qwenResidualBasis } from "./media-input";

export const ${constName(c)} = ${JSON.stringify(c.graph)};

/** The composition this file was generated and gated for, its entry in the loader's table: the artifact's graph
 *  fingerprint on this GPU, the bf16 cache or the affine ${c.kv.bits}-bit group-${c.kv.groupSize} lego, at most ${c.maxRows} rows per
 *  forward. Draft depth, prefill chunk, adapters and the Neural Engine bridge constrain nothing: the constructor
 *  builds them from the record it is handed. */
export const COMPOSITION: ServedComposition = Object.freeze({
  fingerprint: ${JSON.stringify(c.fingerprint)}, device: ${JSON.stringify(c.device)},
  kv: Object.freeze([Object.freeze({ kind: "bf16" }), Object.freeze({ kind: "affine", bits: ${c.kv.bits}, groupSize: ${c.kv.groupSize} })]),
  maxRows: ${c.maxRows},
});

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
const ANE_PROJECTIONS: ReadonlySet<string> = new Set(${list(aneProjections)});
/** Submit the graph every ${k.asyncEvery} layers at decode and verify widths up to 8 (measured best on this GPU). */
const ASYNC_EVERY = ${k.asyncEvery};
/** The Neural Engine's channel fraction in each split. */
const ANE_FRACTION = ${k.ane.fraction};

/** One layer: input norm, mixer, residual, post-attention norm, MLP, residual. Consumes nothing; returns an owned output. */
type Step = (x: MlxArray, cache: Cache, phase: ReadPhase, mrope: MropeForwardState | null) => MlxArray;
type Mixer = (normed: MlxArray, cache: Cache, phase: ReadPhase) => MlxArray;

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

/** The mixer of a layer's DeltaNet or attention block. */
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
  /** Base path for LoRA target keys (weights carry the language_model prefix). */
  readonly prefixBase = PREFIX;
  readonly loraState = new LoraState();
  /** Layers whose attention reads plain keys and values: none; it attends the storage its caches hold. */
  readonly requiredDenseKvLayers: readonly number[] = Object.freeze([]);
  /** The full-attention layer whose cache supplies logical positions. */
  readonly faIdx: number;
  readonly embed: QuantizedEmbedding;
  readonly finalNorm: RMSNorm;
  readonly lmHead: QuantizedLinear;
  /** Prepared image and video embeddings with request mRoPE positions, layer
   *  taps for drafts (verify's capture), delayed affine rows, bounded prefill
   *  workspace; adapters only when the composition mounts them. Speculation over
   *  affine KV is the composition's \`speculativeAffineKv\`. */
  readonly graphCapabilities: GraphCapabilities;
  /** Verify forward cost by rows relative to one decode step (M4 Pro 24 GB). */
  readonly verifyRoundCosts: ReadonlyMap<number, number> = new Map([
    [1, 1], [2, 1.3], [3, 1.46], [4, 1.68], [5, 1.82], [6, 1.92], [7, 1.93], [8, 1.94],
  ]);

  readonly prefillChunk: MlxPrefillChunk;
  readonly decode: MlxWidthTable<MlxPhaseForward>;
  readonly verify: MlxWidthTable<MlxVerifyForward>;
  readonly logits: MlxWidthTable<MlxHeadForward>;

  /** @deprecated E2 deletes it with \`legacyForwardWithTaps\`: the layer tap the
   *  speculative binding sets around a deprecated forward. Only the deprecated
   *  entry points read it, turning it into the operations' capture. */
  hiddenTap: { layers: Set<number>; pos?: number; captured: Map<number, MlxArray> } | null = null;

  /** The generic blocks: prompt chunks and tails on the GPU, wide windows, media positions, committed appends. */
  readonly #generic: Step[];
  readonly #one: Step[];
  readonly #twoToThree: Step[];
  readonly #four: Step[];
  readonly #fiveToEight: Step[];
  readonly #loraTargets: ReadonlyMap<string, QuantizedLinear>;
  readonly #mrope: { readonly invFreq: MlxArray; readonly rotaryDims: number };
  /** The artifact's R1 basis, from its fold record (read once, here). */
  readonly #residualBasis: ResidualBasis | undefined;
  /** What this graph created and frees in \`dispose\`: its constants and the Neural Engine layers' programs. */
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
    this.weightsBytes = [...weights.shards.files.values()].reduce((a, f) => a + f.mmap.size, 0);
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
    };
    const mlpOf = (i: number) => generic[i]!.mlp as TrellisMlp;
    const genericStep = (layer: Qwen3Layer): Step => (x, cache, phase, mrope) => layer.forward(x, cache, phase, mrope);
    this.#generic = generic.map(genericStep);

    if (composition.adapters) {
      // Every class runs the generic blocks: their QuantizedLinear projections
      // apply the mounted adapters (the one-kernel layers do not).
      this.#one = this.#twoToThree = this.#four = this.#fiveToEight = this.#generic;
    } else {
${blockSets}
${narrowSets}
    }

    // The Neural Engine chunk operation, for exactly the composition's chunk
    // size. Layers run one at a time, so each MLP split shares the previous
    // layer's program and projections of one shape share theirs: one loaded
    // program per shape.
    let ane: Step[] | null = null;
    if (composition.aneBridge && !composition.adapters) {
      const started = performance.now(), chunk = composition.prefillChunkTokens;
      const affineShares = new Map<string, ${k.ane.affine}>();
      let mlpShare: AneMlpSplit | undefined;
      ane = generic.map((layer, i) => {
        const spec = LAYERS[i]!, { gate, up, down } = mlpOf(i);
        const split = mlpShare = (spec.downLayout === "k3Interleaved" ? ${k.ane.k3InterleavedMlp} : ${k.ane.rowMajorMlp})
          .build(gate, up, down, ANE_FRACTION, chunk, mlpShare);
        this.#owned.push(split);
        const block = blockOf(i, (linear, path) => {
          if (!ANE_PROJECTIONS.has(path.slice(path.lastIndexOf(".") + 1))) return linear;
          const shape = \`\${linear.inFeatures}:\${linear.outFeatures}\`;
          const projection = ${k.ane.affine}.build(linear, ANE_FRACTION, chunk, affineShares.get(shape));
          affineShares.set(shape, projection);
          this.#owned.push(projection);
          return projection;
        });
        return layerStep(layer, mixerOf(block, spec.kind), post => split.forward(post, compiledSwiglu));
      });
      console.error(\`[ane] prefill programs ready in \${Math.round(performance.now() - started)} ms\`);
    }

    const targets = new Map<string, QuantizedLinear>();
    if (composition.adapters) generic.forEach((layer, i) => {
      const p = \`\${PREFIX}.layers.\${i}\`;
      if (layer.selfAttn) for (const name of ATTENTION_PROJECTIONS) targets.set(\`\${p}.self_attn.\${name}\`, layer.selfAttn[projectionMember[name]]);
      if (layer.linearAttn) for (const name of DELTANET_PROJECTIONS) targets.set(\`\${p}.linear_attn.\${name}\`, layer.linearAttn[projectionMember[name]]);
    });
    this.#loraTargets = targets;

    this.graphCapabilities = declareGraph({
      media: { input: "embeddings+positions", video: true }, hiddenLayerTaps: true,
      kv: { delayedAffine: "all" }, prefill: { boundedWorkspace: true },
      adapters: composition.adapters ? {} : { batched: false, mountable: false },
      speculation: { affineKv: composition.speculativeAffineKv, immediateAffine4: true },
    });

    // The attention cache the composition's KV scheme names, chosen once: affine ${c.kv.bits}-bit group ${c.kv.groupSize}
    // composes ${k.kvLego}; bf16, the table's other scheme for this graph, the plain cache with the dense
    // TurboQuant read (\`composedFullAttentionCache\`).
    const kv = composition.kv;
    this.#attentionCache = kv.kind === "affine" && kv.bits === ${c.kv.bits} && kv.groupSize === ${c.kv.groupSize}
      ? (masks: AttentionMasks) => new ${k.kvLego}(kv.quantizedKvStart ?? 0, composedAffineKernels(kv.kernels), masks)
      : (masks: AttentionMasks, layer: number) => composedFullAttentionCache(kv, layer, masks, denseTurboQuantKernels);

    const ${NARROW.map(cls => `${cls} = this.#${cls}`).join(", ")}, wide = this.#generic;
    const embed = this.embed, split = ane;
    const gpu: MlxPhaseForward = (ids, state) => this.#bounded(wide, embed.encode(ids), state, "window", null, null);
    this.prefillChunk = Object.freeze(split
      ? { gpu, ane: (ids: MlxArray, state: Cache[]) => this.#bounded(split, embed.encode(ids), state, "window", null, null) }
      : { gpu });
    this.decode = Object.freeze({
${NARROW.map(cls => `      ${cls}: (ids: MlxArray, state: Cache[]) => this.#submitted(${cls}, embed.encode(ids), state, "decode", null),`).join("\n")}
      wider: (ids: MlxArray, state: Cache[]) => this.#lazy(wide, embed.encode(ids), state, "decode", null, null),
    });
    this.verify = Object.freeze({
${NARROW.map(cls => `      ${cls}: (ids: MlxArray, state: Cache[], capture: MlxLayerCapture | null) =>
        this.#submitted(${cls}, embed.encode(ids), state, "window", capture),`).join("\n")}
      wider: (ids: MlxArray, state: Cache[], capture: MlxLayerCapture | null) => this.#bounded(wide, embed.encode(ids), state, "window", capture, null),
    });
    const head = this.lmHead, plain: MlxHeadForward = hidden => head.forward(hidden);
    if (composition.adapters) this.logits = atEveryWidth(plain);
    else {
      const ${heads.map(layer => `${headName(layer)} = new ${layer}(head)`).join(", ")};
      this.logits = Object.freeze<MlxWidthTable<MlxHeadForward>>({
        one: plain,
${NARROW.filter(cls => cls !== "one").map(cls => `        ${cls}: hidden => ${headName(k.head[cls as Exclude<Narrow, "one">])}.forward(hidden),`).join("\n")}
        wider: plain,
      });
    }
    console.error(\`[graph] \${this.graph} \${COMPOSITION.fingerprint} ane-prefill=\${ane ? "on" : "off"} adapters=\${composition.adapters ? "on" : "off"}\`);
  }

  /** Frees what the graph created: its constants and the Neural Engine programs. The weights stay the loader's. Call once,
   *  after the last forward. */
  dispose(): void { for (const resource of this.#owned.splice(0)) resource.dispose(); }

  /** One cache per layer, the attention caches sharing one mask memo: the cache the composition's KV scheme names
   *  (affine ${c.kv.bits}-bit group ${c.kv.groupSize}: ${k.kvLego}, bf16 through its first append, then the KV4 decode
   *  and folded window reads; bf16: \`composedFullAttentionCache\`'s plain cache with the dense TurboQuant read). */
  makeCache(): Cache[] {
    const masks = new AttentionMasks();
    return LAYERS.map((layer, i) => layer.kind === "deltanet" ? new SSMCache() : this.#attentionCache(masks, i));
  }

  /** The LoRA-applying projections, when the composition mounts adapters; none otherwise. */
  loraTargets(): Map<string, QuantizedLinear> { return new Map(this.#loraTargets); }

  bindMediaInput(input: Vision): MlxPromptInput { return bindQwenMediaInput(this, input.embeddings, input.mrope!); }
  mediaEncoders(modelDir: string): Promise<MediaEncoders> { return qwenMediaEncoders(this, modelDir); }
  /** The drafter's ports: its hidden-layer taps (verify's capture), embedding and head, and the residual basis
   *  (R1 seed, final-norm gain) the artifact's fold record declares. */
  draftTarget(_caches: Cache[]): TargetView { return qwenDraftTarget(this, this.#residualBasis); }

  /** The positions left at the end of a prompt after its whole chunks, on the GPU. */
  prefillTail(ids: MlxArray, state: Cache[]): MlxArray {
    return this.#bounded(this.#generic, this.embed.encode(ids), state, "window", null, null);
  }

  // ── The layer loops. Each consumes h0 and returns the post-final-norm hidden,
  // owned by the caller; \`capture\` sees every layer's output, then the final hidden.

  /** Every layer, the graph submitted every ASYNC_EVERY layers. */
  #submitted(steps: Step[], h0: MlxArray, cache: Cache[], phase: ReadPhase, capture: MlxLayerCapture | null): MlxArray {
    let h: MlxArray | null = h0;
    try {
      for (let i = 0; i < steps.length; i++) {
        const next = steps[i]!(h, cache[i]!, phase, null);
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

  /** Every layer, each layer's output and recurrent state evaluated before the next. */
  #bounded(steps: Step[], h0: MlxArray, cache: Cache[], phase: ReadPhase, capture: MlxLayerCapture | null,
    mrope: MropeForwardState | null): MlxArray {
    let h: MlxArray | null = h0;
    try {
      for (let i = 0; i < steps.length; i++) {
        const next = steps[i]!(h, cache[i]!, phase, mrope);
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
    mrope: MropeForwardState | null): MlxArray {
    let h: MlxArray | null = h0;
    try {
      for (let i = 0; i < steps.length; i++) {
        const next = steps[i]!(h, cache[i]!, phase, mrope);
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

  // ── Deprecated entry points: E2 moves every caller onto the phases above and
  // deletes these. They reach the same blocks through #deprecated, the one
  // place this graph reads a shape (with deprecatedPhase and logitsFromHidden).

  /** @deprecated E2. One sequence takes the block set its positions name (one,
   *  two or three, four, five to eight, else the GPU prompt blocks); several sequences,
   *  media positions and committed spans take the generic blocks, evaluated per
   *  layer above four positions per sequence. The Neural Engine operation is
   *  not entered here. */
  #deprecated(h0: MlxArray, cache: Cache[], phase: ReadPhase, positions: MlxArray | null, capture: MlxLayerCapture | null): MlxArray {
    const [rows, tokens] = h0.shape as [number, number];
    if (positions || phase === "committed" || rows !== 1) {
      const mrope = positions ? { posIds: positions, ...this.#mrope } : null;
      return tokens > TRELLIS_MATVEC_MAX_M ? this.#bounded(this.#generic, h0, cache, phase, capture, mrope)
        : this.#lazy(this.#generic, h0, cache, phase, capture, mrope);
    }
    if (tokens === 1) return this.#submitted(this.#one, h0, cache, phase, capture);
    if (tokens <= 3) return this.#submitted(this.#twoToThree, h0, cache, phase, capture);
    if (tokens === 4) return this.#submitted(this.#four, h0, cache, phase, capture);
    if (tokens <= 8) return this.#submitted(this.#fiveToEight, h0, cache, phase, capture);
    return this.#bounded(this.#generic, h0, cache, phase, capture, null);
  }

  /** @deprecated E2: the speculative binding's layer tap, as the operations' capture, then \`capture\`. */
  #tapCapture(capture: MlxLayerCapture | null): MlxLayerCapture | null {
    const tap = this.hiddenTap;
    if (!tap) return capture;
    return (layer, h) => {
      if (layer < LAYERS.length && tap.layers.has(layer)) {
        const H = h.shape[2]!;
        const src = tap.pos !== undefined ? h.slice([0, tap.pos, 0], [1, tap.pos + 1, H]) : h;
        const copy = ops.contiguous(src);
        if (src !== h) src.dispose();
        tap.captured.set(layer, copy);
      }
      capture?.(layer, h);
    };
  }

  /** @deprecated E2. */
  forwardHidden(ids: MlxArray, cache: Cache[]): MlxArray {
    return this.#deprecated(this.embed.encode(ids), cache, deprecatedPhase(ids), null, this.#tapCapture(null));
  }

  /** @deprecated E2 decides mixed rows. Each group runs its own operation, with its capture. */
  forwardHiddenMixed(work: readonly TokenGroup[]): MlxArray[] {
    const results: MlxArray[] = [];
    try {
      for (const group of work)
        results.push(this.#deprecated(this.embed.encode(group.ids), group.cache, deprecatedPhase(group.ids), null,
          this.#tapCapture(group.captureLayer ?? null)));
      return results;
    } catch (error) { for (const result of results) result.dispose(); throw error; }
  }

  /** @deprecated E2. Borrowed request positions, including different positions for each row. */
  forwardHiddenAtPositions(ids: MlxArray, cache: Cache[], positions: MlxArray): MlxArray {
    return this.#deprecated(this.embed.encode(ids), cache, deprecatedPhase(ids), positions, this.#tapCapture(null));
  }

  /** @deprecated E2. Vision prefill: spliced embeddings (borrowed) with request positions. */
  forwardEmbeddingsAtPositions(embeds: MlxArray, cache: Cache[], positions: MlxArray): MlxArray {
    return this.#deprecated(ops.contiguous(embeds), cache, deprecatedPhase(embeds), positions, this.#tapCapture(null));
  }

  /** @deprecated E2. Spliced embeddings (borrowed) at the cache's own positions. */
  forwardEmbeddings(embeds: MlxArray, cache: Cache[], _bidir: MlxArray | null, _ids: MlxArray | null = null,
    _multimodal: MlxArray | null = null): MlxArray {
    return this.#deprecated(ops.contiguous(embeds), cache, deprecatedPhase(embeds), null, this.#tapCapture(null));
  }

  /** @deprecated E2's appendKnown. Committed tokens on the generic blocks: up
   *  to four positions at one row, or one position per row. */
  createAppend(policy: { hasAdapters: boolean; pagedKv: boolean }): MlxTokenAppend | null {
    if (policy.hasAdapters || policy.pagedKv || this.loraState.active.length) return null;
    return {
      affineKvBits: [4, 8],
      turboQuantFormats: [{ kBits: 8, vBits: 3 }],
      maxChunkSize: (state: readonly Cache[], rows = 1) => rows === 1 ? qwenAppendChunkSize(state[0]!.offset) : 1,
      forwardHidden: (ids: MlxArray, cache: Cache[]): MlxArray => {
        const [rows, tokens] = ids.shape as [number, number];
        if (ids.shape.length !== 2 || !(rows === 1 ? tokens <= 4 : rows > 1 && tokens === 1))
          throw new Error("Qwen committed-token append supports one position per shared row or up to four positions at B=1");
        return this.#deprecated(this.embed.encode(ids), cache, deprecatedPhase(ids, true), null, null);
      },
    };
  }

  /** @deprecated E2 indexes \`logits\` by the rows it projects; this picks the entry by the hidden's rows. */
  logitsFromHidden(h: MlxArray): MlxArray {
    const rows = h.size / h.shape.at(-1)!;
    const head = rows === 1 ? this.logits.one : rows <= 3 ? this.logits.twoToThree : rows === 4 ? this.logits.four
      : rows <= 8 ? this.logits.fiveToEight : this.logits.wider;
    return head(h);
  }

  /** @deprecated E2. Logits for \`tokens\` appended to \`cache\`. */
  forward(tokens: number[] | MlxArray, cache: Cache[]): MlxArray {
    const ids = Array.isArray(tokens) ? ops.fromInt32(tokens, [1, tokens.length]) : tokens;
    const h = this.forwardHidden(ids, cache);
    if (Array.isArray(tokens)) ids.dispose();
    const logits = this.logitsFromHidden(h);
    h.dispose();
    return logits;
  }

  /** Greedy generation on fresh caches (diagnostics and scripts). */
  generate(promptTokens: number[], maxTokens: number, eosIds: number[] = []): number[] {
    const cache = this.makeCache();
    const out: number[] = [];
    try {
      let tokens = promptTokens;
      for (let step = 0; step < maxTokens; step++) {
        const logits = this.forward(tokens, cache);
        const next = argmaxLastPosition(logits);
        logits.dispose();
        if (eosIds.includes(next)) break;
        out.push(next);
        tokens = [next];
      }
    } finally { for (const c of cache) c.dispose(); }
    return out;
  }
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
Compositions: ${Object.values(QWEN38_COMPOSITIONS).map(c => `${c.stem} (${c.artifact} on ${c.device}, KV affine ${c.kv.bits}/${c.kv.groupSize}, ` +
  `depth ${depthText(c.draftDepth)}, chunk ${c.prefillChunkTokens}, ${c.maxRows} rows, ANE ${c.aneBridge ? "yes" : "no"})`).join("; ")}.`;
  if (values.help) { console.log(usage); process.exit(0); }
  if (values.describe) {
    if (positionals.length !== 1) { console.error(usage); process.exit(1); }
    const { source, fingerprint, layers } = await describeArtifact(positionals[0]!);
    console.log(`{\n "source": ${JSON.stringify(source)},\n "fingerprint": ${JSON.stringify(fingerprint)},\n "layers": [\n` +
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
