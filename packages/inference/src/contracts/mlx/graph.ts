import type { GraphCapabilities } from "../portable/graph";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { Cache } from "./cache";
import type { TargetView } from "./draft-target";
import type { NativeMtpHead } from "./drafter";
import type { TrainableGraph } from "./trainable";
import type { MediaEncoders, MediaSidecarProbes, MlxPromptInput, PixelInput, TextEmbeddingModel, Vision } from "./media";
import type { MlxRowTable } from "./row-table";
export type { MlxRowTable } from "./row-table";

/** Weight and expert-residency facts the wired-limit scope reads. */
export interface MlxModelMemory {
  readonly weightsBytes: number;
  readonly expertRuntime?: {
    readonly plan: { readonly plannedBytes: number };
    flushUsage?: () => void;
    finishUsage?: () => Promise<void>;
  } | null;
}

/** A verify forward's hidden-layer capture. The graph calls it once per
 * decoder layer, in layer order, with that layer's residual output, then once
 * with index `layerCount` (the decoder's layer count) and the post-final-norm
 * output, all before the forward returns. `hidden` [sequences, tokens per
 * sequence, hidden-width] is borrowed and possibly unevaluated: it is valid
 * only during the call. The callback copies the layers it keeps (for example
 * with `ops.contiguous`) and owns and disposes those copies. */
export type MlxLayerCapture = (layer: number, hidden: MlxArray) => void;

/** A prefill or decode forward. `ids` is integer token ids [sequences, tokens
 * per sequence]. The caller owns `ids` and may dispose them once the call
 * returns (or its promise settles); the result does not depend on the caller's
 * handle. The graph borrows `state` and appends every position of `ids` to
 * each layer's cache (keys and values, or recurrent state), advancing each
 * sequence by its token count; it never rolls back. The returned post-final-norm
 * hidden [sequences, tokens per sequence, hidden-width] is owned by the caller,
 * who disposes it, and may be unevaluated, as may the appended cache state.
 * A forward that throws may leave a partial append, which the caller discards
 * or rolls back. */
export type MlxPhaseForward = (ids: MlxArray, state: Cache[]) => MlxArray | Promise<MlxArray>;

/** A forward that also hands the drafter the hidden layers it taps:
 * `MlxPhaseForward`'s ownership and append, and `capture` receives each
 * decoder layer's residual output and then the post-final-norm output (see
 * `MlxLayerCapture`). `capture` is null when the composition's drafter reads
 * none; only a graph that declares `hiddenLayerTaps` receives a non-null
 * capture. Prompt chunks, prompt tails and verify windows take it. */
export type MlxCapturingForward = (ids: MlxArray, state: Cache[], capture: MlxLayerCapture | null) =>
  MlxArray | Promise<MlxArray>;

/** A speculative verify forward (`MlxCapturingForward`). */
export type MlxVerifyForward = MlxCapturingForward;

/** The output head over a post-final-norm hidden [.., hidden-width] (borrowed)
 * to logits [.., vocabulary] the caller owns, possibly unevaluated. */
export type MlxHeadForward = (hidden: MlxArray) => MlxArray;
export interface MlxHeadRange { readonly from: number; readonly to: number; readonly operation: MlxHeadForward }

/** The scoring and training forward, not a serving phase: every position of
 * `ids` [sequences, tokens] as one causal window over the caller's caches (the
 * stateless training caches, which own the mask of an unpadded or right-padded
 * batch and the training attention kernel), on the kernels of `width`, the
 * class of its rows. Returns the post-final-norm hidden the caller owns. */
export type MlxSequenceForward = (ids: MlxArray, state: Cache[], width: number) => MlxArray;

/** One contiguous range of positions in a single sequence's prompt piece.
 * The scheduler selects the operation; the graph never inspects its shape. */
export interface MlxPromptRange {
  readonly from: number;
  readonly to: number;
  readonly operation: MlxCapturingForward;
  readonly unit?: "gpu" | "ane";
}

/** A whole accepted span uses one declared operation; callers must not split
 * it to fit a verification table because that can change attention arithmetic. */
export interface MlxKnownRange { readonly from: number; readonly to: number; readonly operation: MlxPhaseForward }
export type MlxKnownOperations = MlxRowTable<MlxPhaseForward> | readonly MlxKnownRange[];

/** Operations selected by exact row count or prompt position range. */
export interface MlxTokenPhases {
  readonly prompt: readonly MlxPromptRange[];
  readonly multiSequencePrompt: readonly MlxPromptRange[];
  readonly decode: MlxRowTable<MlxCapturingForward>;
  readonly verify: MlxRowTable<MlxVerifyForward>;
  readonly appendKnown: MlxKnownOperations;
  readonly logits: MlxRowTable<MlxHeadForward>;
}

/** How a graph whose composition qualifies committed known-token spans takes
 * them (`MlxTokenPhases.appendKnown` over its committed read): how many
 * positions one call takes. Every cache lego's committed read keeps the
 * one-position arithmetic (the cache contract holds it to the same positions
 * read one at a time), whatever the composition's KV scheme. A graph without
 * it takes known tokens one position at a time. */
export interface MlxKnownSpans {
  /** Positions one sequence's `appendKnown` call takes at the caches' current
   * length. Recheck after each call: native arithmetic can change at a
   * cache-length boundary. */
  maxChunkSize(state: readonly Cache[]): number;
}

/** One group of a forward that serves several groups of rows at once, each in
 * its own phase over its own caches. */
export interface MlxPhaseGroup {
  /** `decode` (one position per sequence), `window` (a prompt piece) or
   * `verify` (a speculative window, which keeps its own projection geometry). */
  readonly phase: "decode" | "window" | "verify";
  /** The class of this group's rows, the caller's to name. */
  readonly rows: number;
  readonly ids: MlxArray;
  readonly state: Cache[];
  readonly capture: MlxLayerCapture | null;
}

/** Declared by a composition that serves mixed rows: decode, prompt and verify
 * groups in one forward, each group's attention and recurrence over its own
 * caches for its own phase, the row-wise layers over the groups' rows packed
 * together unless a verify group keeps its geometry. `width` is the class of
 * the packed rows, the caller's to name. Borrows inputs and state; returns one
 * owned post-final-norm hidden per group. */
export type MlxMixedForward = (groups: readonly MlxPhaseGroup[], width: number) => MlxArray[];

/** A graph's compiled single-token decode step: the whole forward replayed from
 * one recorded graph instead of rebuilt per token. The scheduler chooses it
 * once, for a row whose caches all declare the compiled-decode capability and
 * hold no batched layout; it has no fallback. */
export interface MlxCompiledDecodeStep {
  /** Consume one pending uint32 [1] token (unevaluated is fine) and advance
   * every cache one position. The returned logits [1,1,V] belong to the
   * caller; `evalWith` holds cache updates that must ride the same async
   * evaluation. A step that throws leaves `state` exactly as it was. */
  step(token: MlxArray, state: Cache[]): { logits: MlxArray; evalWith: MlxArray[] };
}

/** method `denoising`: what the canvas denoiser drives over a graph's weights, with the prefill state
 * an array of caches. A graph that declares the method implements these; binding refuses one that does not. */
export interface MlxDenoisingPrefillRange {
  readonly from: number;
  readonly to: number;
  readonly operation: (promptIds: number[], pixels: MlxArray | null) => Cache[];
}

export interface MlxDenoisingDecodeRange {
  readonly from: number;
  readonly to: number;
  readonly operation: (canvas: MlxArray, state: Cache[], feedback: MlxArray | null) => MlxArray;
}
export interface MlxDenoisingAppendRange {
  readonly from: number;
  readonly to: number;
  readonly operation: (tokens: MlxArray, state: Cache[]) => void;
}

export interface MlxDenoisingOperations {
  readonly denoisingDecode?: readonly MlxDenoisingDecodeRange[];
  readonly denoisingAppend?: readonly MlxDenoisingAppendRange[];
  readonly denoisingPrefill?: readonly MlxDenoisingPrefillRange[];
  readonly config: { readonly text: { readonly vocabSize: number } };
  readonly canvasLength: number;
  readonly embedScale: number;
  prefill?(promptIds: number[]): Cache[];
  /** Image-conditioned prefill from channel-first pixels. */
  prefillVision?(promptIds: number[], pixels: MlxArray): Cache[];
  extendPrefill(tokens: MlxArray, state: Cache[]): void;
  decoderLogits(canvas: MlxArray, state: Cache[], feedback: MlxArray | null): MlxArray;
  dequantEmbedWeight(): MlxArray;
  softEmbeddings(logits: MlxArray, weight: MlxArray): MlxArray;
}

/** The operations behind a graph's declared capabilities. A graph implements
 * the ones its `graphCapabilities` promise; composition checks the pairing once
 * when it binds (`declaredGraph`), so execution never probes for them. */
export interface MlxDeclaredGraph {
  readonly graphCapabilities: GraphCapabilities;
  /** Release what the graph built at construction (computed constants, Neural
   *  Engine programs, its own weight source where it opened one) once every
   *  execution and compiled runner has stopped. A graph that built nothing
   *  releases nothing. Its owner (the host, a draft provider) calls it before
   *  releasing the weights it supplied. */
  dispose(): void;
  /** Unscaled token embeddings, for media prompts that splice embeddings. */
  readonly embed?: TextEmbeddingModel["embed"];
  /** media `embeddings`/`embeddings+positions`: bind one request's prepared
   * embeddings into an input whose prefill is indivisible. */
  bindMediaInput?(input: Vision): MlxPromptInput;
  /** embeddings: the pooled, L2-normalized sentence vector of one sequence,
   * ids [1, L] to [1, hidden]. The caller appends the pooling token the
   * profile declares and owns the result. */
  embedPooled?(ids: MlxArray): MlxArray;
  /** media: the lazy encoders this graph loads from the checkpoint at `modelDir`. */
  mediaEncoders?(modelDir: string, probes: MediaSidecarProbes): Promise<MediaEncoders>;
  /** media `pixels`: the pixel input a denoising method consumes; null when the checkpoint has none. */
  pixelInput?(): PixelInput | null;
  /** Ports over this graph's live caches that draft sources may consume. */
  draftTarget?(caches: Cache[]): TargetView;
  /** `nativeDraft`: the checkpoint's own multi-token-prediction head, refused when its tier is not loaded. */
  nativeDraftHead?(): NativeMtpHead;

  /** What the trainer consumes: the head, segmented backward, prefix sharing,
   * gradient checkpointing, flash-attention constraint and denoising objective
   * this graph implements. Undeclared parts make the trainer refuse. */
  readonly trainable?: TrainableGraph;
  /** `compiledDecode`: the graph's compiled decode step, created on first use
   * and owned by the graph until `releaseCompiledDecode`. */
  compiledDecodeStep?(): MlxCompiledDecodeStep;
  /** Retire the compiled step before the graph unloads: its recorded graph
   * borrows the graph's weights and constants. Safe to call when none exists. */
  releaseCompiledDecode?(): void;
  /** Snapshot of the streamed expert residency, for diagnostics; null when the graph streams none. */
  expertResidency?(): Record<string, unknown> | null;
  /** Measured cost of a verify forward by its row count, relative to one
   *  single-row decode step (rows 1 = 1). Declared only by a graph measured on
   *  the hardware it serves; speculative depth scheduling reads it. */
  readonly verifyRoundCosts?: ReadonlyMap<number, number>;
  /** Committed known-token spans, declared when the composition qualifies them
   *  (`appendKnown` then reads committed). */
  readonly knownSpans?: MlxKnownSpans;
  /** Mixed rows in one forward, declared by a composition that serves them. */
  readonly mixed?: readonly MlxMixedRange[];
  /** Mixed groups that retain their own projection geometry. The scheduler
   * routes each group independently when a verify group is present. */
  readonly mixedSeparate?: MlxSeparateMixed;
  /** Full-sequence scoring operations, indexed by total token rows. */
  readonly sequence?: readonly MlxSequenceRange[];
  readonly sequenceLogits?: readonly MlxHeadRange[];
  /** Prepared embeddings without explicit positions, indexed by total rows. */
  readonly embeddingPrompt?: readonly import("./media").MlxEmbeddingsRange[];
}

/** Cache layout written by the graph, independent of request storage policy. */
export type MlxCacheLayer =
  | { readonly kind: "full"; readonly turboQuant: "dense" | "deferred"; readonly kv4Head256?: boolean }
  | { readonly kind: "sliding"; readonly window: number }
  | { readonly kind: "recurrent" }
  | { readonly kind: "latent"; readonly geometry: {
      readonly kvLoraRank: number; readonly ropeHeadDim: number; readonly maxTokens?: number; readonly indexHeadDim?: number;
    } };

/** Fixed mixed operation for a contiguous range of packed token rows. */
export interface MlxMixedRange {
  readonly from: number;
  readonly to: number;
  readonly operation: MlxMixedForward;
}


/** A full-sequence forward fixed for a contiguous range of token rows. */
export interface MlxSequenceRange {
  readonly from: number;
  readonly to: number;
  readonly operation: (ids: MlxArray, state: Cache[]) => MlxArray;
}

/** One mixed group using its own fixed projection geometry. */
export interface MlxMixedGroupRange {
  readonly from: number;
  readonly to: number;
  readonly operation: (group: MlxPhaseGroup) => MlxArray;
}

export interface MlxSeparateMixed {
  readonly decode: readonly MlxMixedGroupRange[];
  readonly window: readonly MlxMixedGroupRange[];
  readonly verify: readonly MlxMixedGroupRange[];
}
