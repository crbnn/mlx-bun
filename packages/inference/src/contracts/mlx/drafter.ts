// Drafter models as the draft sources see them. A drafter is a model: it is built
// and loaded in `models/` (`models/drafters.ts`), and the sources in
// `generation/speculative/` consume only these ports, never a concrete drafter.

import type { MlxArray } from "@mlx-bun/mlx/array";
import type { AssistantDonors } from "./attention";
import type { Cache } from "./cache";
import type { DraftProjection } from "./draft-projection";
import type { DraftProjectionKernels, ResidualBasis } from "./draft-target";
import type { MlxRowTable } from "./row-table";

export interface DrafterOperationRange<Operation> { readonly from: number; readonly to: number; readonly operation: Operation }
export type DrafterOperations<Operation> = MlxRowTable<Operation> | readonly DrafterOperationRange<Operation>[];

/** One drafted step of a KV-borrowing assistant: owned token ids and the post-projected hidden. */
export interface DrafterRowsStep { tokens: MlxArray; nextHidden: MlxArray }

/** One assistant draft step over B rows: the last token's embedding and the hidden it follows [B, 1, H]
 *  (borrowed), the donors, and the rows' position (one shared, or [B]); owned tokens and next hidden out. */
export type AssistantDraftStep = (lastTokenEmb: MlxArray, targetHidden: MlxArray, donors: AssistantDonors,
  position: number | MlxArray) => DrafterRowsStep;

/** The optiq KV-borrowing assistant: no cache of its own, it reads the target's donor attention. */
export interface AssistantDrafterModel {
  /** The draft step per row count or range of the rows it drafts for, each with the token selection its head
   *  bound for that class at construction; the caller indexes it by its rows (actual positive row count). */
  readonly forwardRows: MlxRowTable<AssistantDraftStep>;
  dispose(): void;
}

/** What a projected-context drafter (DSpark, DeepSpec) attends over the target's tapped hidden states. */
export interface ContextKV {
  k: MlxArray; // [1, nKvHeads, ctxLen, headDim]
  v: MlxArray; // [1, nKvHeads, ctxLen, headDim]
}

/** A projected-context drafter's context projection over B × L rows: the target's tapped hiddens
 *  [B, L, taps × H] (borrowed) to context rows [B, L, H] (owned). */
export type DrafterContextProjection = (targetHiddens: MlxArray) => MlxArray;

/** Each drafter layer's context K/V [B, kvHeads, L, headDim] (owned) of B × L context rows [B, L, H] (borrowed),
 *  at absolute positions from `position` (one shared, or [B] when the rows start apart). */
export type DrafterContextKV = (context: MlxArray, position: number | MlxArray) => ContextKV[];

/** Confidence-scheduled draft-length pruning and logits collection of one DSpark block. */
export interface DsparkDraftOptions {
  thresholds?: number[];
  minConf?: number;
  /** Concatenate the per-position draft logits into the result; default true. */
  collectLogits?: boolean;
  collectConfidence?: boolean;
}

/** Storage owns context validity; the DSpark graph supplies only Q and the block's K/V. */
export interface DsparkContextAttention {
  attend(layer: number, query: MlxArray, keys: MlxArray, values: MlxArray, scale: number): MlxArray;
}

/** Our trained DSpark module: block drafting over the target's multi-layer hidden context. */
export interface DsparkDrafterModel {
  readonly cfg: {
    readonly gamma: number;
    readonly nLayers: number;
    readonly tapLayers: number[];
    readonly sts?: { readonly thresholds: number[] };
  };
  projectContextRows(hidden: MlxArray): { k: MlxArray; v: MlxArray }[];
  forwardRows(model: DraftProjection, hCtx: MlxArray | null, anchors: readonly number[], gamma: number,
    opts?: DsparkDraftOptions, context?: DsparkContextAttention): { tokens: number[][] };
  dispose(): void;
}

/** Storage owns context validity; the DeepSpec graph supplies only Q and new block KV. */
export interface DeepspecContextAttention {
  attend(layer: number, query: MlxArray, blockKeys: MlxArray, blockValues: MlxArray): MlxArray;
}

export interface DeepspecDraftRows {
  tokens: number[][];
  conf: number[][];
  baseLogits?: MlxArray;
}

/** DeepSeek's released DSpark drafter (DeepSpec): a block forward over cached, already projected context K/V. */
export interface DeepspecDrafterModel {
  readonly cfg: { readonly num_target_layers: number; readonly num_hidden_layers: number };
  /** The target layers whose hiddens the drafter conditions on. */
  readonly tapLayers: number[];
  /** The trained block width. */
  readonly gamma: number;
  readonly hidden: number;
  /** The context projection per row count or range of the rows projected (B × L), indexed by the caller
   *  (`B × L`): MLX's quantized matmul (matmul for a BF16 checkpoint) at every class. */
  readonly projectContext: MlxRowTable<DrafterContextProjection>;
  /** Each layer's context K/V per row count or range of the context rows (B × L), indexed like `projectContext`:
   *  the same kernel at every class. */
  readonly projectContextKVRows: MlxRowTable<DrafterContextKV>;
  draftRows(context: DeepspecContextAttention, anchors: readonly number[], position: number | MlxArray,
    collectLogits?: boolean): DeepspecDraftRows;
  dispose(): void;
}

/** One recurrent multi-token-prediction forward: token embeddings [B, S, H] paired with hiddens
 * [B, S, H] (both borrowed) in, the module output (post final norm) [B, S, H] out, owned; appends S
 * positions per row to `cache`, which supplies positions and storage. */
export type RecurrentMtpForward = (tokenEmbeds: MlxArray, hiddens: MlxArray, cache: Cache) => MlxArray;

/** The recurrent multi-token-prediction block split from a Qwen release, its phases named by the
 * caller like a graph's: the draft step indexed by the rows it drafts for. */
export interface RecurrentMtpModule {
  /** One position per row (a draft step, the round's bridge or full-accept row), by the rows' row count. */
  readonly decode: MlxRowTable<RecurrentMtpForward>;
  /** Several positions per row (prompt rows, a verified continuation). */
  readonly window: RecurrentMtpForward;
}

/** The checkpoint-native multi-token-prediction row of a graph that declares `nativeDraft`, over
 * compressed (MLA) attention state. The graph owns the numerics and the head's weights; the draft
 * source owns sampling and state retention. */
export interface NativeMtpHead {
  readonly hiddenSize: number;
  /** Geometry of the compressed cache the head appends to. */
  readonly cache: { readonly kvLoraRank: number; readonly ropeHeadDim: number; readonly maxTokens: number };
  /** One draft step: `ids` [B, 1] (one token per sequence) over each row's anchor
   * `hidden` [B, 1, hidden], appended to `cache`; the caller owns the output
   * [B, 1, hidden]. */
  decode(ids: MlxArray, hidden: MlxArray, cache: Cache): Promise<MlxArray>;
  /** A window of two or more tokens per sequence, `ids` [B, L] over `hidden`
   * [B, L, hidden], appended to `cache`; the caller owns the output. The caller
   * names the phase from the window it built: one token per sequence is `decode`. */
  window(ids: MlxArray, hidden: MlxArray, cache: Cache): Promise<MlxArray>;
  /** Logits [B, positions, V] of the head's output. */
  project(hidden: MlxArray): MlxArray;
}

/** DFlash 2's projected-context attention: block keys follow the context rows. */
export interface Dflash2ContextAttention {
  attend(layer: number, query: MlxArray, blockKeys: MlxArray, blockValues: MlxArray, scale?: number): MlxArray;
}

/** The DFlash 2 block drafter: target taps → projected context K/V; one greedy block per anchor.
 *  It loads in the basis it was trained on and learns the target's when a draft group binds. */
export interface Dflash2DrafterModel {
  readonly cfg: { readonly layers: number; readonly numTargetLayers: number };
  readonly tapLayers: number[];
  /** Draft tokens per block (block size minus the anchor). */
  readonly gamma: number;
  /** Apply the target's declarations before any projection: its residual basis (null: the trained basis) and
   *  the kernels its composition names for the drafter's 4-bit projections (null: the drafter's own rule). The
   *  first call applies them; every later call must pass the same declarations. */
  bindTarget(basis: ResidualBasis | null, projections: DraftProjectionKernels | null): void;
  /** The context projection (fc over the target's taps, then the hidden norm) per row count or range of the rows
   *  projected (B × L: a prompt's rows at prefill, a commit's accepted rows), indexed by the caller
   *  (`B × L`). A 4-bit fc runs the kernel the target declares for the class or, by the drafter's
   *  own rule, the simdgroup matrix unit (`Affine4MmaLinear`, where the weight's shape fits it) at 5..8
   *  and MLX's quantized matmul (`QuantizedLinear`) at every other class; a BF16 fc runs one matmul at every
   *  class. No entry runs before `bindTarget`. */
  readonly projectContext: DrafterOperations<DrafterContextProjection>;
  /** Each layer's k and v over the context rows (k norm, RoPE), per row count or range of those rows (B × L), indexed
   *  like `projectContext` and on the kernel it names for the class. */
  readonly projectContextKVRows: DrafterOperations<DrafterContextKV>;
  /** The columns of a block drafting `depth` tokens: its anchor and up to `depth` mask positions, at most the
   *  trained block size. */
  blockColumns(depth: number): number;
  /** The draft block per row count or range of its rows (anchors times columns), indexed by the scheduler: every 4-bit
   *  projection on the kernel `projectContext` runs for the class; BF16 projections run one matmul at every
   *  class. */
  readonly draft: DrafterOperations<Dflash2DraftBlock>;
  dispose(): void;
}

/** One DFlash 2 draft block: one row per anchor, each the anchor and `columns - 1` mask positions, over the
 *  projected context; the greedy proposals left on the device, [anchors, columns - 1] uint32 row-major. Every
 *  projection in the block runs the kernel its row count names. */
export type Dflash2DraftBlock = (context: Dflash2ContextAttention, projection: DraftProjection, anchors: readonly number[],
  position: number | MlxArray, columns: number) => MlxArray;
