// Gemma 4 graph generator: compiles one composition (an artifact's config
// fingerprint and the KV scheme its caches serve, the artifact's kv_config) into
// an entire graph file: the layer table read from the artifact (attention type,
// the cache each layer reads, its affine width, keys equal to values, the layer
// scalar), written out as data, and the class that composes each layer from the
// blocks in ../src/models/gemma4/blocks.ts for that table, with the named phases
// and the entry points its consumers call. The graph extends nothing and calls no
// other graph.
//
//   bun packages/inference/scripts/gen-gemma4.ts <model-dir> <composition> [--out <dir>]
//   e.g. bun packages/inference/scripts/gen-gemma4.ts ~/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-OptiQ-4bit/snapshots/<hash> gemma4-12b
//
// <model-dir> is an artifact snapshot (config.json, kv_config.json and the
// safetensors index's names; no weights). A change to the blocks, layers or this
// template is a regeneration, never a hand edit:
// tests/models/gemma4-generated-sync.test.ts regenerates every composition from
// tests/fixtures/gemma4-graphs and fails when a committed file differs. Whether a
// regenerated graph computes the same bits is tests/models/gemma4-generated-identity.test.ts
// (synthetic weights) and tests/parity/gemma4-generated.test.ts (real weights).

import { parseArgs } from "node:util";
import { loadModelConfig } from "../src/artifacts/config";
import type { ModelConfig } from "../src/artifacts/config";
import { configFingerprint } from "../src/artifacts/fingerprint";
import { GENERATED_GEMMA_FINGERPRINTS } from "../src/models/gemma4/family";

/** The facts that fix one generated graph: the artifact (its config fingerprint,
 * which covers the kv_config) and so the KV scheme its caches serve, the
 * artifact's per-layer kv_config (`--kv-quant config`). The file exports them as
 * `COMPOSITION`, which the loader's selection table keys on. */
export interface Gemma4Composition {
  /** Output file stem under src/models/gemma4/generated. */
  readonly stem: string;
  /** Key in GENERATED_GEMMA_FINGERPRINTS (models/gemma4/family.ts). */
  readonly key: keyof typeof GENERATED_GEMMA_FINGERPRINTS;
  /** The artifact the composition was generated for, for the usage text. */
  readonly artifact: string;
}

export const GEMMA4_COMPOSITIONS: Readonly<Record<string, Gemma4Composition>> = Object.freeze({
  "gemma4-12b": { stem: "gemma4-12b", key: "12b", artifact: "mlx-community/gemma-4-12B-it-OptiQ-4bit" },
  "gemma4-e4b": { stem: "gemma4-e4b", key: "e4b", artifact: "mlx-community/gemma-4-e4b-it-OptiQ-4bit" },
  "gemma4-26b": { stem: "gemma4-26b", key: "26b", artifact: "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit" },
});
export const GENERATED_DIR = new URL("../src/models/gemma4/generated/", import.meta.url).pathname;

export interface GraphSource {
  readonly config: ModelConfig;
  /** Tensor names from model.safetensors.index.json (layer_scalar and the prefix are weights-layout facts). */
  readonly tensorNames: ReadonlySet<string>;
}

export async function readGraphSource(modelDir: string): Promise<GraphSource> {
  const config = await loadModelConfig(modelDir);
  const indexFile = Bun.file(`${modelDir}/model.safetensors.index.json`);
  // Single-shard models carry no index: assume no layer_scalar.
  const tensorNames = await indexFile.exists()
    ? new Set(Object.keys(((await indexFile.json()) as { weight_map: Record<string, string> }).weight_map))
    : new Set<string>();
  return { config, tensorNames };
}

/** The repo@revision of a hub-layout snapshot path, else the path itself. */
function sourceOf(dir: string): string {
  const match = dir.replace(/\/+$/, "").match(/models--([^/]+)--([^/]+)\/snapshots\/([^/]+)$/);
  return match ? `${match[1]}/${match[2]}@${match[3]}` : dir;
}

export async function generateGemma4Graph(modelDir: string, stem: string): Promise<string> {
  return emitGemma4Graph(await readGraphSource(modelDir), stem, sourceOf(modelDir));
}

/** One decoder layer as the composition fixes it. */
interface LayerSpec {
  readonly sliding: boolean;
  /** The layer whose cache it reads: itself, or the last donor of its type. */
  readonly cache: number;
  /** That cache's affine width and group under the artifact's kv_config; null keeps it bf16. */
  readonly kv: { readonly bits: number; readonly groupSize: number } | null;
  readonly kEqV: boolean;
  readonly layerScalar: boolean;
}

const plural = (count: number, word: string) => `${count} ${word}${count === 1 ? "" : "s"}`;

/** `text` as `//` comment lines of at most `width` columns. */
function wrapComment(text: string, width = 84): string {
  const lines: string[] = [];
  let line = "//";
  for (const word of text.split(" ")) {
    if (line.length + 1 + word.length > width && line !== "//") { lines.push(line); line = "//"; }
    line += ` ${word}`;
  }
  return [...lines, line].join("\n");
}

/** The generated module's text for `source`; `label` is the provenance line. */
export function emitGemma4Graph({ config, tensorNames }: GraphSource, stem: string, label: string): string {
  const c = GEMMA4_COMPOSITIONS[stem];
  if (!c) throw new Error(`no composition ${stem} (known: ${Object.keys(GEMMA4_COMPOSITIONS).join(", ")})`);
  const t = config.text;
  const fingerprint = configFingerprint(config);
  if (fingerprint !== GENERATED_GEMMA_FINGERPRINTS[c.key])
    throw new Error(`${stem}: config fingerprint ${fingerprint} is not the registered ${GENERATED_GEMMA_FINGERPRINTS[c.key]} (models/gemma4/family.ts)`);
  const prefix = tensorNames.has("language_model.model.embed_tokens.weight") ? "language_model.model" : "model";
  const kvByLayer = new Map((config.kvQuant ?? []).map(e => [e.layerIdx, { bits: e.bits, groupSize: e.groupSize }]));

  const layerCount = t.numHiddenLayers;
  const caches = layerCount - t.numKvSharedLayers;
  const sharing = caches < layerCount;
  const moe = t.enableMoeBlock;
  const perLayer = t.hiddenSizePerLayerInput > 0;
  const softcap = t.finalLogitSoftcapping;
  const compiledDecode = !moe;
  const lastByType: Record<string, number> = {};
  for (let i = 0; i < caches; i++) lastByType[t.layerTypes[i]!] = i;
  const layers: LayerSpec[] = t.layerTypes.map((type, i) => {
    const cache = i < caches ? i : lastByType[type];
    if (cache === undefined) throw new Error(`${stem}: no donor layer of type ${type} for layer ${i}`);
    const sliding = type === "sliding_attention";
    return { sliding, cache, kv: kvByLayer.get(cache) ?? null, kEqV: t.attentionKEqV && !sliding,
      layerScalar: tensorNames.has(`${prefix}.layers.${i}.layer_scalar`) };
  });
  const shared = [...new Set(layers.filter((layer, i) => layer.cache !== i).map(layer => layer.cache))].sort((a, b) => a - b);
  const scalars = layers.filter(layer => layer.layerScalar).length;

  // ── Header facts ──
  const owned = layers.slice(0, caches);
  const widths = new Map<string, number>();
  for (const layer of owned) {
    const key = layer.kv ? `${layer.kv.bits}-bit group ${layer.kv.groupSize}` : "bf16";
    widths.set(key, (widths.get(key) ?? 0) + 1);
  }
  const kvLine = [...widths.entries()].map(([key, count]) => `${key} on ${plural(count, "cache")}`).join(", ");
  const slidingCount = layers.filter(layer => layer.sliding).length;
  const sharingLine = sharing
    ? `${caches} own a cache, ${layerCount - caches} attend the read of the last donor of their type (${shared.map(i =>
      `${i} ${layers[i]!.sliding ? "sliding" : "full"}`).join(", ")})`
    : `every layer owns a cache`;
  const pieces = [perLayer ? "per-layer inputs" : "", moe ? "the dense MLP and routed experts in parallel" : "dense feed-forward",
    scalars === layerCount ? "a layer scalar on every layer" : scalars === 0 ? "no layer scalar" : `a layer scalar on ${scalars} layers`,
    t.attentionKEqV ? "keys equal to values on the full layers" : ""].filter(Boolean).join("; ");

  // ── Table ──
  const table = layers.map((layer, i) => `  /* ${String(i).padStart(2)} */ { attention: ${JSON.stringify(layer.sliding ? "sliding" : "full")}, ` +
    `cache: ${layer.cache}, kv: ${layer.kv ? `{ bits: ${layer.kv.bits}, groupSize: ${layer.kv.groupSize} }` : "null"}, ` +
    `kEqV: ${layer.kEqV}, layerScalar: ${layer.layerScalar} },`).join("\n");

  // ── The feed-forward each layer composes ──
  const ffBody = (scaled: boolean) => {
    let expr = moe ? "layer.moeFeedforward(hMid)" : "layer.denseFeedforward(hMid)";
    if (perLayer) expr = `layer.perLayerGated(${expr}, perLayerInput)`;
    if (scaled) expr = `layer.scaled(${expr})`;
    return expr;
  };
  const ffParams = perLayer ? "(hMid, perLayerInput)" : "hMid";
  const ffDoc = `${moe ? "the dense MLP and the routed experts in parallel" : "the dense feed-forward"}${perLayer ? ", the per-layer input gate" : ""}`;
  const ffDocCap = `${ffDoc[0]!.toUpperCase()}${ffDoc.slice(1)}`;
  const ffBuilders = scalars === layerCount
    ? `/** ${ffDocCap}, then the layer scalar. */
const feedforwardOf = (layer: DecoderLayer): Feedforward => ${ffParams} =>
  ${ffBody(true)};`
    : scalars === 0
      ? `/** ${ffDocCap}. */
const feedforwardOf = (layer: DecoderLayer): Feedforward => ${ffParams} =>
  ${ffBody(false)};`
      : `/** ${ffDocCap}, then the layer scalar where the layer carries one. */
const feedforwardOf = (layer: DecoderLayer, scaled: boolean): Feedforward => scaled
  ? ${ffParams} => ${ffBody(true)}
  : ${ffParams} => ${ffBody(false)};`;
  const ffBuild = scalars === layerCount || scalars === 0 ? "this.layers.map(layer => feedforwardOf(layer))"
    : "this.layers.map((layer, i) => feedforwardOf(layer, LAYERS[i]!.layerScalar))";

  // ── The attention each layer composes ──
  const attendBuilders = sharing ? `/** Owns its cache and no later layer attends its read, which goes at once. */
const ownCache = (layer: DecoderLayer, i: number): Attend => (x, caches, phase) => {
  const { h, donor } = layer.forwardAttn(x, caches[i]!, phase, null);
  donor!.dispose();
  return h;
};
/** Owns its cache and later layers attend its read, kept in \`reads\` until the forward ends. */
const donorCache = (layer: DecoderLayer, i: number): Attend => (x, caches, phase, reads) => {
  const { h, donor } = layer.forwardAttn(x, caches[i]!, phase, null);
  reads[i] = donor;
  return h;
};
/** Attends the read its donor layer made in this forward. */
const donorRead = (layer: DecoderLayer, donor: number): Attend => (x, _caches, phase, reads) =>
  layer.forwardAttn(x, null, phase, reads[donor]!).h;` : `/** Owns its cache; no other layer attends its read, which goes at once. */
const ownCache = (layer: DecoderLayer, i: number): Attend => (x, caches, phase) => {
  const { h, donor } = layer.forwardAttn(x, caches[i]!, phase, null);
  donor!.dispose();
  return h;
};`;
  const attendBuild = sharing
    ? `this.layers.map((layer, i) => LAYERS[i]!.cache !== i ? donorRead(layer, LAYERS[i]!.cache)
      : SHARED.has(i) ? donorCache(layer, i) : ownCache(layer, i))`
    : "this.layers.map((layer, i) => ownCache(layer, i))";

  const readsDecl = sharing ? "\n    const reads: (DonorRead | null)[] = Array(CACHES).fill(null);" : "";
  const readsArg = sharing ? ", reads" : "";
  const readsFree = sharing ? "\n      for (const read of reads) read?.dispose();" : "";
  const idsParam = perLayer ? ", ids: MlxArray" : "";
  const idsArg = (ids: string) => perLayer ? `, ${ids}` : "";
  const layerBody = perLayer ? `        const mid = this.#attend[i]!(h, caches, phase${readsArg});
        let next: MlxArray;
        try {
          using input = perLayerInputs.layer(perLayer, i, B, L);
          next = this.#feedforward[i]!(mid, input);
        } finally { mid.dispose(); }` : `        const mid = this.#attend[i]!(h, caches, phase${readsArg});
        let next: MlxArray;
        try { next = this.#feedforward[i]!(mid); } finally { mid.dispose(); }`;

  const softcapHead = softcap === null ? `    this.#head = hidden => embed.asLinear(hidden);`
    : `    // The compiled softcap matches mlx-lm's @mx.compile logit_softcap; a compiled-decode trace records the plain ops.
    const compiled = compiledGegluActive();
    this.#head = hidden => {
      const logits = embed.asLinear(hidden);
      return disposing(logits, compiled && !isCompiledTrace()
        ? compiledLogitSoftcap(logits, FINAL_LOGIT_SOFTCAP) : logitSoftcap(logits, FINAL_LOGIT_SOFTCAP));
    };`;

  const forwardEmbeddings = perLayer ? `  /** @deprecated E2. Pre-merged (unscaled) input embeddings (borrowed) to hidden states: \`bidir\` (bool [L]) marks
   *  image tokens, which attend each other both ways in a prompt's first window; \`ids\` ([1, L], the spliced token ids)
   *  feed the per-layer inputs, with every multimodal position (\`multimodal\`, else \`bidir\`) taking token 0's. */
  forwardEmbeddings(embeds: MlxArray, cache: Cache[], bidir: MlxArray | null, ids: MlxArray,
    multimodal: MlxArray | null = null): MlxArray {
    const h = ops.mulScalar(embeds, this.embedScale);
    const zeroing = multimodal ?? bidir;
    let plIds = ids;
    if (zeroing) {
      using cond = ops.reshape(zeroing, ids.shape);
      using zero = ops.zeros(ids.shape, ids.dtype);
      plIds = ops.where(cond, zero, ids);
    }
    try { return this.#run(h, cache, deprecatedRead(h, bidir), plIds, this.#tapCapture(null)); }
    finally { if (plIds !== ids) plIds.dispose(); }
  }` : `  /** @deprecated E2. Pre-merged (unscaled) input embeddings (borrowed) to hidden states: \`bidir\` (bool [L]) marks
   *  image tokens, which attend each other both ways in a prompt's first window. */
  forwardEmbeddings(embeds: MlxArray, cache: Cache[], bidir: MlxArray | null, _ids: MlxArray,
    _multimodal: MlxArray | null = null): MlxArray {
    const h = ops.mulScalar(embeds, this.embedScale);
    return this.#run(h, cache, deprecatedRead(h, bidir), this.#tapCapture(null));
  }`;

  const mixedInputs = perLayer ? `
            const [B, L] = group.ids.shape as [number, number];
            inputs.push(perLayerInputs.layer(group.perLayer!, i, B, L));` : "";
  const mixedPacked = perLayer ? `
          // Per-layer inputs use the same token order as the hidden states.
          let packedInput: MlxArray | null = null;
          if (packedMlp && inputs.length) {
            const flat = inputs.map(input => ops.reshape(input, [1, -1, perLayerInputs.width]));
            try { packedInput = ops.concatAxis(flat, 1); }
            finally { for (const input of flat) input.dispose(); }
          }
          using pli = packedInput;` : "";
  const mixedFf = perLayer
    ? { packed: "feedforward(packed, pli!)", rows: "for (const [row, mid] of mids.entries()) outputs.push(feedforward(mid, inputs[row]!))" }
    : { packed: "feedforward(packed)", rows: "for (const mid of mids) outputs.push(feedforward(mid))" };

  const compiledMembers = compiledDecode ? `
  /** The compiled decode step: it records this graph's own decode forward (${sharing || perLayer
    ? "whole-graph, since KV sharing and per-layer inputs\n   *  plumb across layers" : "segmented at the growing caches'\n   *  layers, which run uncompiled"}), so it shares the graph's weights and constants. */
  compiledDecodeStep(): MlxCompiledDecodeStep { return CompiledDecode.for(this); }
  releaseCompiledDecode(): void { CompiledDecode.release(this); }
` : "";

  const kvHeaderLine = owned.every(layer => layer.kv) ? `the artifact's kv_config, affine per cache: ${kvLine}` : `the artifact's kv_config: ${kvLine}`;

  return `// GENERATED by packages/inference/scripts/gen-gemma4.ts from composition ${stem}; do not edit.
// Regenerate: bun packages/inference/scripts/gen-gemma4.ts <snapshot> ${stem}
// (tests/fixtures/gemma4-graphs holds the inputs it was generated from).
//
// Composition ${stem}:
//   artifact     ${label}, config fingerprint ${fingerprint}
//   KV           ${kvHeaderLine} (LAYERS)
//   layers       ${layerCount}: ${slidingCount} sliding, ${layerCount - slidingCount} full; ${sharingLine}
//   blocks       ${pieces}
//
${wrapComment(`An entire graph: built once at construction for this composition, it inherits from no class and never calls another graph. Each layer is composed from the blocks (../blocks.ts) its row of LAYERS names: its attention half (it owns a cache${sharing ? ", owns one whose read later layers attend, or attends its donor's read" : ""}) and its row-wise half (${ffDoc}${scalars ? ", the layer scalar" : ""}). Every attention read goes through the cache, for the phase the caller names; the cache attends with the kernel it was composed with. Gemma has no width-specialized kernels yet, so decode, verify and logits serve every width class with one operation each.`)}
//
// Request shapes this graph serves:
//   - prompt chunks and tails of token ids (prefillChunk.gpu, prefillTail);
//   - decode, one position per sequence, and speculative verify windows with the
//     drafter's layer capture, at every width (decode, verify, logits);
//   - prepared image and audio embeddings, image tokens attending each other both
//     ways (bindMediaInput, through forwardEmbeddings);${compiledDecode ? `
//   - the compiled decode step (compiledDecodeStep);` : ""}
//   - the assistant drafter's donor reads and the hidden-layer taps (draftTarget);
//   - mounted LoRA adapters (loraTargets: the blocks' QuantizedLinear projections apply them);
//   - paged KV (the paged caches' own reads), and mixed groups of rows in one
//     forward (forwardHiddenMixed).
// Its caches are the composition's (makeCache): under the scheme it serves, every
// cache the kv_config lists is the bf16-first affine lego, read bf16 through its
// first append and through its quantized storage after, with the read kernels the
// composition names (--fused-sdpa); the other caches stay bf16.
// The loader refuses, or gives the generic graph (Gemma4Model), from the declared
// capabilities and its selection table, never by routing inside this graph:
//   - any KV scheme other than the one COMPOSITION serves (bf16, uniform --kv-quant N,
//     TurboQuant): the table builds the generic graph;
//   - training: this graph declares no training operations (segmented backward,
//     prefix sharing, gradient checkpointing, the fused head and the training
//     forward over training caches are the generic graph's)${compiledDecode ? "" : `;
//   - the compiled decode step: shapeless replay retraces the experts' growing windows`}.

import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { ModelConfig } from "../../../artifacts/config";
import type { Weights } from "../../../artifacts/weights";
import type { Cache } from "../../../contracts/mlx/cache";
import type { TargetView } from "../../../contracts/mlx/draft-target";
import type {
  MlxCapturingForward, ${compiledDecode ? "MlxCompiledDecodeStep, " : ""}MlxDeclaredGraph, MlxHeadForward, MlxLayerCapture, MlxPhaseForward, MlxPrefillChunk,
  MlxTokenPhases,
  MlxVerifyForward, MlxWidthTable,
} from "../../../contracts/mlx/graph";
import type { MediaEncoders, MediaSidecarProbes, MlxPromptInput, Vision } from "../../../contracts/mlx/media";
import type { TokenGroup } from "../../../contracts/mlx/token-work";
import type { Composition } from "../../../contracts/portable/composition";
import type { GraphCapabilities } from "../../../contracts/portable/graph";
import { mapPackedTokens } from "../../../input/token-groups";
import { argmaxLastPosition${softcap === null ? "" : ", logitSoftcap"} } from "../../../kernels/logits";
${softcap === null ? "" : `import { compiledLogitSoftcap } from "../../../kernels/softcap";
import { compiledGegluActive } from "../../../layers/geglu";
`}import { disposing } from "../../../layers/helpers";
import { LoraState } from "../../../layers/lora";
import { RMSNorm } from "../../../layers/normalization";
import { QuantizedEmbedding } from "../../../layers/quantized-embedding";
import type { QuantizedLinear } from "../../../layers/quantized-linear";
${softcap === null ? "" : `import { isCompiledTrace } from "../../../runtime/compiled-trace";
`}import { disposeResources } from "../../../runtime/resources";
import { AttentionMasks } from "../../../state/attention-read";
import { composedFullAttentionCache, composedSlidingAttentionCache } from "../../../state/bf16-first-kv";
import { deferredTurboQuantKernels } from "../../../state/turboquant-attention";
import { declareGraph } from "../../capabilities";
import { atEveryWidth, deprecatedPhase } from "../../graph";
import type { ServedComposition } from "../../implementation";
import { DecoderLayer, ${perLayer ? "PerLayerInputs, " : ""}${sharing ? "type DonorRead, " : ""}type Gemma4Phase } from "../blocks";
${compiledDecode ? `import { CompiledDecode } from "../compiled-decode";
` : ""}import { GENERATED_GEMMA_FINGERPRINTS } from "../family";
import { bindGemma4MediaInput, gemma4DraftTarget, gemma4MediaEncoders } from "../media";

export const FINGERPRINT = GENERATED_GEMMA_FINGERPRINTS[${JSON.stringify(c.key)}];

/** One decoder layer as the composition fixes it: its attention type; the layer
 *  whose cache it reads (itself when it owns one, else the last donor of its type,
 *  whose read it attends in the same forward); that cache's affine width and group
 *  under the composition's KV scheme (null: bf16); whether its keys are its values;
 *  whether it carries a layer scalar. */
export interface Gemma4Layer {
  readonly attention: "sliding" | "full";
  readonly cache: number;
  readonly kv: { readonly bits: number; readonly groupSize: number } | null;
  readonly kEqV: boolean;
  readonly layerScalar: boolean;
}

/** The artifact's layers and kv_config, from ${label}. */
export const LAYERS: readonly Gemma4Layer[] = Object.freeze([
${table}
]);

/** Layers [0, CACHES) own a cache. */
const CACHES = ${caches};
${sharing ? `/** The donors whose read later layers attend. */
const SHARED: ReadonlySet<number> = new Set([${shared.join(", ")}]);
` : ""}const PREFIX = ${JSON.stringify(prefix)};
${softcap === null ? "" : `const FINAL_LOGIT_SOFTCAP = ${softcap};
`}
/** The composition this file was generated for, its entry in the loader's table:
 *  the artifact's config fingerprint and the KV scheme of the caches it serves (the
 *  artifact's kv_config, \`--kv-quant config\`). The scheme's read kernels and
 *  conversion start constrain nothing: \`makeCache\` composes them from the record
 *  the constructor is handed. */
export const COMPOSITION: ServedComposition = Object.freeze({
  fingerprint: FINGERPRINT,
  kv: Object.freeze([Object.freeze({
    kind: "affine-layers" as const,
    layers: Object.freeze(LAYERS.slice(0, CACHES).flatMap((layer, i) => layer.kv ? [Object.freeze({ layer: i, ...layer.kv })] : [])),
  })]),
});

/** A layer's attention half over \`x\` (borrowed) for one forward, to the post-attention residual (owned). */
type Attend = (x: MlxArray, caches: Cache[], phase: Gemma4Phase${sharing ? ", reads: (DonorRead | null)[]" : ""}) => MlxArray;
/** A layer's row-wise half over the post-attention residual \`hMid\` (borrowed)${perLayer ? " and the layer's per-layer input (borrowed)" : ""},
 *  to the layer output (owned). Rows are rows: it never sees how they group into sequences. */
type Feedforward = (hMid: MlxArray${perLayer ? ", perLayerInput: MlxArray" : ""}) => MlxArray;

${attendBuilders}
${ffBuilders}

/** @deprecated E2: the phase of a forward reaching a deprecated entry point, from
 *  its input's shape (\`deprecatedPhase\`); a window over an image mask (\`bidir\`) is
 *  the bidirectional read. */
function deprecatedRead(h0: MlxArray, bidir: MlxArray | null): Gemma4Phase {
  const read = deprecatedPhase(h0);
  return bidir && read === "window" ? { bidirectional: bidir } : read;
}

export class GeneratedGemma4 implements MlxDeclaredGraph, MlxTokenPhases {
  readonly config: ModelConfig;
  /** The served configuration this graph was built for: its KV scheme is what \`makeCache\` composes. */
  readonly composition: Composition;
  /** Total weight-shard bytes (for the conditional wired-limit scope). */
  readonly weightsBytes: number;
  /** Weight-name prefix of the LoRA targets. */
  readonly prefixBase = PREFIX;
  /** Active LoRA adapters for the current generation. */
  readonly loraState = new LoraState();
  /** Layers whose attention reads plain keys and values: none; it attends the storage its caches hold. */
  readonly requiredDenseKvLayers: readonly number[] = Object.freeze([]);
  /** Prepared image and audio embeddings, paged attention, layer taps for drafts${compiledDecode ? ", a compiled decode step" : ""};
   *  its caches convert delayed affine rows. No training operations. */
  readonly graphCapabilities: GraphCapabilities;
  readonly embed: QuantizedEmbedding;
  readonly layers: readonly DecoderLayer[];
  readonly finalNorm: RMSNorm;
  readonly embedScale: number;
  readonly windowSize: number;
  /** Layers [0, numDonors) own caches. */
  readonly numDonors = CACHES;
  /** The per-layer input width (0: none). */
  readonly perLayerWidth: number;

  readonly prefillChunk: MlxPrefillChunk;
  readonly prefillTail: MlxWidthTable<MlxCapturingForward>;
  readonly decode: MlxWidthTable<MlxPhaseForward>;
  readonly verify: MlxWidthTable<MlxVerifyForward>;
  /** Known tokens: this graph declares no committed spans, so they read as a window. */
  readonly appendKnown: MlxWidthTable<MlxPhaseForward>;
  readonly logits: MlxWidthTable<MlxHeadForward>;

  /** @deprecated E2 deletes it with \`legacyForwardWithTaps\`: the layer tap the
   *  speculative binding sets around a deprecated forward. Only the deprecated
   *  entry points read it, turning it into the forward's capture. */
  hiddenTap: { layers: Set<number>; pos?: number; captured: Map<number, MlxArray> } | null = null;

  readonly #attend: readonly Attend[];
  readonly #feedforward: readonly Feedforward[];
${perLayer ? "  readonly #perLayerInputs: PerLayerInputs;\n" : ""}  readonly #head: MlxHeadForward;
  /** The composition's \`packedMixedMlp\`: the groups of a mixed forward share one feed-forward call. */
  readonly #packMixedMlp: boolean;
  #disposed = false;

  constructor(weights: Weights, config: ModelConfig, composition: Composition) {
    const t = config.text;
    this.config = config;
    this.composition = composition;
    this.weightsBytes = [...weights.shards.files.values()].reduce((a, f) => a + f.mmap.size, 0);
    this.embed = QuantizedEmbedding.load(weights, \`\${PREFIX}.embed_tokens\`, config);
    this.layers = Object.freeze(LAYERS.map((_, i) => new DecoderLayer(weights, config, PREFIX, i, i < CACHES)));
    this.finalNorm = new RMSNorm(weights.tensor(\`\${PREFIX}.norm.weight\`), t.rmsNormEps);
    this.embedScale = Math.sqrt(t.hiddenSize);
    this.windowSize = t.slidingWindow;
${perLayer ? `    this.#perLayerInputs = new PerLayerInputs(weights, config, PREFIX);
    this.perLayerWidth = this.#perLayerInputs.width;
` : `    this.perLayerWidth = 0;
`}    this.#attend = Object.freeze(${attendBuild});
    this.#feedforward = Object.freeze(${ffBuild});
    const embed = this.embed;
${softcapHead}
    this.#packMixedMlp = composition.packedMixedMlp;
    this.graphCapabilities = declareGraph({
      media: { input: "embeddings", video: false }, pagedAttention: true, hiddenLayerTaps: true, compiledDecode: ${compiledDecode},
      kv: { delayedAffine: "all" },
    });

    const head = this.#head;
    const window: MlxCapturingForward = (ids, caches, capture) =>
      this.#run(this.#embedTokens(ids), caches, "window"${idsArg("ids")}, capture);
    this.prefillChunk = Object.freeze({ gpu: window });
    this.prefillTail = atEveryWidth(window);
    this.decode = atEveryWidth<MlxPhaseForward>((ids, caches) => this.#run(this.#embedTokens(ids), caches, "decode"${idsArg("ids")}, null));
    this.verify = atEveryWidth(window);
    this.appendKnown = atEveryWidth<MlxPhaseForward>((ids, caches) => window(ids, caches, null));
    this.logits = atEveryWidth<MlxHeadForward>(hidden => head(hidden));
  }

  /** One cache per layer that owns one, all sharing one mask memo, so each
   *  forward builds each mask once: the cache the composition's KV scheme names
   *  for the layer, a ring of the sliding window for sliding layers
   *  (\`composedSlidingAttentionCache\`) and for full layers a cache whose
   *  TurboQuant read is deferred (\`composedFullAttentionCache\`,
   *  \`deferredTurboQuantKernels\`). Under the scheme COMPOSITION serves, each
   *  cache LAYERS gives a width is the bf16-first affine lego. */
  makeCache(): Cache[] {
    const masks = new AttentionMasks(), kv = this.composition.kv;
    return LAYERS.slice(0, CACHES).map((layer, i) => layer.attention === "sliding"
      ? composedSlidingAttentionCache(kv, i, this.windowSize, masks) : composedFullAttentionCache(kv, i, masks, deferredTurboQuantKernels));
  }

  /** LoRA-mountable linears, keyed by weight-file module path: the attention and
   *  MLP projections${perLayer ? " and the per-layer-input projections" : ""}. Expert pools are not targets. */
  loraTargets(): Map<string, QuantizedLinear> {
    const out = new Map<string, QuantizedLinear>();
    this.layers.forEach((l, i) => {
      const p = \`\${PREFIX}.layers.\${i}\`;
      const add = (suffix: string, lin: QuantizedLinear | null) => { if (lin) out.set(\`\${p}.\${suffix}\`, lin); };
      add("self_attn.q_proj", l.attn.qProj);
      add("self_attn.k_proj", l.attn.kProj);
      add("self_attn.v_proj", l.attn.vProj);
      add("self_attn.o_proj", l.attn.oProj);
      add("mlp.gate_proj", l.mlp.gate);
      add("mlp.up_proj", l.mlp.up);
      add("mlp.down_proj", l.mlp.down);
      add("per_layer_input_gate", l.perLayerGate);
      add("per_layer_projection", l.perLayerProjection);
    });
    return out;
  }

  bindMediaInput(input: Vision): MlxPromptInput { return bindGemma4MediaInput(this, input); }
  mediaEncoders(modelDir: string, probes: MediaSidecarProbes): Promise<MediaEncoders> {
    return gemma4MediaEncoders(this, modelDir, probes);
  }
  /** The drafter's ports over the live caches: the assistant drafter's donor reads
   *  (the last sliding and full caches) and the hidden-layer taps. */
  draftTarget(caches: Cache[]): TargetView { return gemma4DraftTarget(this, caches); }
${compiledMembers}
  /** Release the graph's constants after all executions and compiled runners have
   *  stopped. Weight tensors and hidden taps are borrowed. */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    disposeResources(this.layers.flatMap(layer => layer.attn.ropeFreqs ? [layer.attn.ropeFreqs] : []));
  }

  /** Computed constants keep their constructor arithmetic when a restored request
   *  enters compiled decode without a prefill first. */
  materializeGraphConstants(): void {
    const constants = this.layers.flatMap(layer => layer.attn.ropeFreqs ? [layer.attn.ropeFreqs] : []);
    if (constants.length) ops.evalAll(constants);
  }

  /** The scaled token embeddings [B, L, hidden] of \`ids\` (owned). */
  #embedTokens(ids: MlxArray): MlxArray {
    const h = this.embed.encode(ids);
    return disposing(h, ops.mulScalar(h, this.embedScale));
  }

  /** Every layer over \`h0\` [B, L, hidden] (consumed), each reading for \`phase\`, to
   *  the post-final-norm hidden (owned). \`capture\` sees every layer's output, then
   *  the final hidden.${perLayer ? " `ids` [B, L] feed the per-layer inputs." : ""} */
  #run(h0: MlxArray, caches: Cache[], phase: Gemma4Phase${idsParam}, capture: MlxLayerCapture | null): MlxArray {
${perLayer ? `    const [B, L] = h0.shape as [number, number];
    const perLayerInputs = this.#perLayerInputs;
` : ""}    let h: MlxArray | null = h0;${readsDecl}${perLayer ? `
    let perLayer: MlxArray | null = null;` : ""}
    try {${perLayer ? `
      perLayer = perLayerInputs.compute(ids, h0);` : ""}
      for (let i = 0; i < LAYERS.length; i++) {
${layerBody}
        h.dispose();
        h = next;
        capture?.(i, h);
      }
      const out = this.finalNorm.forward(h);
      h.dispose();
      h = null;
      capture?.(LAYERS.length, out);
      return out;
    } finally {
      h?.dispose();${perLayer ? `
      perLayer?.dispose();` : ""}${readsFree}
    }
  }

  // ── Deprecated entry points: E2 moves every caller onto the phases above and
  // deletes these. They reach the same layers through #run, reading the phase
  // from the input's shape (deprecatedRead) and the speculative binding's tap.

  /** @deprecated E2: the speculative binding's layer tap (\`hiddenTap\`) as the
   *  forward's capture, then \`capture\`: each tapped layer's output and the final
   *  hidden copied as bf16, one position when \`pos\` is set. */
  #tapCapture(capture: MlxLayerCapture | null): MlxLayerCapture | null {
    const tap = this.hiddenTap;
    if (!tap) return capture;
    const dtype = this.embed.scales.dtype;
    return (layer, h) => {
      if (tap.layers.has(layer)) {
        const H = h.shape[2]!;
        const src = tap.pos !== undefined ? h.slice([0, tap.pos, 0], [1, tap.pos + 1, H]) : h;
        const copy = src.astype(dtype);
        if (src !== h) src.dispose();
        tap.captured.set(layer, copy);
      }
      capture?.(layer, h);
    };
  }

  /** @deprecated E2. ids [B, L] to the final-norm hidden [B, L, hidden]. */
  forwardHidden(ids: MlxArray, cache: Cache[]): MlxArray {
    const h = this.#embedTokens(ids);
    return this.#run(h, cache, deprecatedRead(h, null)${idsArg("ids")}, this.#tapCapture(null));
  }

${forwardEmbeddings}

  /** @deprecated E2 decides mixed rows. Each group's attention reads its own
   *  caches; the feed-forward runs once over the groups' real tokens packed
   *  together (unless a group keeps its token geometry or the composition's
   *  \`packedMixedMlp\` is off). A single group without a capture is \`forwardHidden\`. */
  forwardHiddenMixed(work: readonly TokenGroup[]): MlxArray[] {
    if (work.length === 1 && !work[0]!.captureLayer) return [this.forwardHidden(work[0]!.ids, work[0]!.cache)];
    const groups: Array<{ ids: MlxArray; cache: Cache[]; h: MlxArray; phase: Gemma4Phase${perLayer ? ";\n      perLayer: MlxArray | null" : ""}${sharing ? `;${perLayer ? " " : "\n      "}reads: (DonorRead | null)[]` : ""} }> = [];
    const results: MlxArray[] = [];
    const packedMlp = this.#packMixedMlp && !work.some(group => group.preserveTokenGeometry);
${perLayer ? "    const perLayerInputs = this.#perLayerInputs;\n" : ""}    try {
      for (const { ids, cache } of work) {
        const h = this.#embedTokens(ids);
        const group = { ids, cache, h, phase: deprecatedRead(h, null)${perLayer ? ",\n          perLayer: null as MlxArray | null" : ""}${sharing ? `,${perLayer ? " " : "\n          "}reads: Array<DonorRead | null>(CACHES).fill(null)` : ""} };
        groups.push(group);${perLayer ? `
        group.perLayer = perLayerInputs.compute(ids, group.h);` : ""}
      }
      for (let i = 0; i < LAYERS.length; i++) {
        const attend = this.#attend[i]!, feedforward = this.#feedforward[i]!;
        const mids: MlxArray[] = []${perLayer ? ", inputs: MlxArray[] = []" : ""};
        try {
          for (const group of groups) {
            mids.push(attend(group.h, group.cache, group.phase${sharing ? ", group.reads" : ""}));${mixedInputs}
          }${mixedPacked}
          const outputs: MlxArray[] = [];
          try {
            if (packedMlp) outputs.push(...mapPackedTokens(mids, packed => ${mixedFf.packed}));
            else ${mixedFf.rows};
          } catch (error) { for (const output of outputs) output.dispose(); throw error; }
          for (const [index, group] of groups.entries()) { group.h.dispose(); group.h = outputs[index]!; }
          for (const [index, group] of groups.entries()) work[index]!.captureLayer?.(i, group.h);
        } finally {
          for (const mid of mids) mid.dispose();${perLayer ? `
          for (const input of inputs) input.dispose();` : ""}
        }
      }
      for (const [index, group] of groups.entries()) {
        const hidden = this.finalNorm.forward(group.h);
        results.push(hidden);
        work[index]!.captureLayer?.(LAYERS.length, hidden);
      }
      return results;
    } catch (error) { for (const result of results) result.dispose(); throw error; }
    finally {
      for (const group of groups) {
        group.h.dispose();${perLayer ? `
        group.perLayer?.dispose();` : ""}${sharing ? `
        for (const read of group.reads) read?.dispose();` : ""}
      }
    }
  }

  /** @deprecated E2 indexes \`logits\` by the rows it projects; one head serves every width. */
  logitsFromHidden(h: MlxArray): MlxArray { return this.#head(h); }

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
`;
}

if (import.meta.main) {
  const { values, positionals } = parseArgs({
    args: Bun.argv.slice(2),
    options: { out: { type: "string" }, help: { type: "boolean", short: "h" } },
    allowPositionals: true,
  });
  const stems = Object.keys(GEMMA4_COMPOSITIONS).join("|");
  const usage = `Usage: bun packages/inference/scripts/gen-gemma4.ts <model-dir> <${stems}> [--out <dir>]
Compile a Gemma 4 composition into its entire graph module. <model-dir> is the artifact's
snapshot (config.json, kv_config.json, model.safetensors.index.json; no weights, no native
library). Its config fingerprint must be the one registered for the composition in
src/models/gemma4/family.ts. The default output directory is src/models/gemma4/generated.
Compositions: ${Object.values(GEMMA4_COMPOSITIONS).map(c => `${c.stem} (${c.artifact}, its kv_config)`).join("; ")}.`;
  if (values.help) { console.log(usage); process.exit(0); }
  const [modelDir, stem] = positionals;
  if (!modelDir || !stem || positionals.length !== 2 || !(stem in GEMMA4_COMPOSITIONS)) {
    console.error(usage);
    process.exit(1);
  }
  const outPath = `${values.out ?? GENERATED_DIR}/${stem}.ts`.replace(/\/\/+/g, "/");
  await Bun.write(outPath, await generateGemma4Graph(modelDir, stem));
  console.log(`wrote ${outPath}`);
}
