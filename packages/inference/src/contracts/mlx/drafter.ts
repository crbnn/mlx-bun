// Drafter models as the draft sources see them. A drafter is a model: it is built
// and loaded in `models/` (`models/drafters.ts`), and the sources in
// `generation/speculative/` consume only these ports, never a concrete drafter.

import type { MlxArray } from "@mlx-bun/mlx/array";
import type { AssistantDonors } from "./attention";
import type { Cache } from "./cache";
import type { DraftProjection } from "./draft-projection";
import type { ResidualBasis } from "./draft-target";
import type { MlxWidthTable } from "./width";

/** One drafted step of a KV-borrowing assistant: owned token ids and the post-projected hidden. */
export interface DrafterRowsStep { tokens: MlxArray; nextHidden: MlxArray }

/** One assistant draft step over B rows: the last token's embedding and the hidden it follows [B, 1, H]
 *  (borrowed), the donors, and the rows' position (one shared, or [B]); owned tokens and next hidden out. */
export type AssistantDraftStep = (lastTokenEmb: MlxArray, targetHidden: MlxArray, donors: AssistantDonors,
  position: number | MlxArray) => DrafterRowsStep;

/** The optiq KV-borrowing assistant: no cache of its own, it reads the target's donor attention. */
export interface AssistantDrafterModel {
  /** The draft step per width class of the rows it drafts for, each with the token selection its head
   *  bound for that class at construction; the caller indexes it by its rows (`widthClass`). */
  readonly forwardRows: MlxWidthTable<AssistantDraftStep>;
  dispose(): void;
}

/** What a projected-context drafter (DSpark, DeepSpec) attends over the target's tapped hidden states. */
export interface ContextKV {
  k: MlxArray; // [1, nKvHeads, ctxLen, headDim]
  v: MlxArray; // [1, nKvHeads, ctxLen, headDim]
}

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
  forwardInfer(model: DraftProjection, hCtx: MlxArray, anchor: number, gamma: number,
    opts?: DsparkDraftOptions): { tokens: number[] };
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

export interface DeepspecDraftBlock {
  /** Sequentially sampled draft tokens, length 0..gamma (0 iff confidence truncation fires first). */
  tokens: number[];
  /** Per-position sigmoid confidence, aligned with `tokens`. */
  conf: number[];
  /** Base logits of the full block [1, gamma, vocab]; the caller disposes. */
  baseLogits: MlxArray;
}

/** DeepSeek's released DSpark drafter (DeepSpec): a block forward over cached, already projected context K/V. */
export interface DeepspecDrafterModel {
  readonly cfg: { readonly num_target_layers: number; readonly num_hidden_layers: number };
  /** The target layers whose hiddens the drafter conditions on. */
  readonly tapLayers: number[];
  /** The trained block width. */
  readonly gamma: number;
  readonly hidden: number;
  projectContext(targetHiddens: MlxArray): MlxArray;
  projectContextKV(contextHidden: MlxArray, positions: number[]): ContextKV[];
  projectContextKVRows(contextHidden: MlxArray, position: number | MlxArray): ContextKV[];
  draftBlock(ctxKV: ContextKV[], anchorTok: number, anchorPos: number): DeepspecDraftBlock;
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
  /** One position per row (a draft step, the round's bridge or full-accept row), by the rows' width class. */
  readonly decode: MlxWidthTable<RecurrentMtpForward>;
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
  forward(ids: MlxArray, hidden: MlxArray, cache: Cache): Promise<MlxArray>;
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
  /** Apply the target's declared residual basis (null: the trained basis) before any projection.
   *  The first call applies it; every later call must pass the same declaration. */
  bindResidualBasis(basis: ResidualBasis | null): void;
  projectContext(taps: MlxArray): MlxArray;
  projectContextKVRows(context: MlxArray, position: number | MlxArray): { k: MlxArray; v: MlxArray }[];
  /** The columns of a block drafting `depth` tokens: its anchor and up to `depth` mask positions, at most the
   *  trained block size. */
  blockColumns(depth: number): number;
  /** The draft block per width class of its rows (anchors times columns), indexed by the scheduler. */
  readonly draft: MlxWidthTable<Dflash2DraftBlock>;
  dispose(): void;
}

/** One DFlash 2 draft block: one row per anchor, each the anchor and `columns - 1` mask positions, over the
 *  projected context; the greedy proposals left on the device, [anchors, columns - 1] uint32 row-major. Every
 *  projection in the block runs the kernel its width class names. */
export type Dflash2DraftBlock = (context: Dflash2ContextAttention, projection: DraftProjection, anchors: readonly number[],
  position: number | MlxArray, columns: number) => MlxArray;
