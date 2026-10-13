import { MlxArray } from "@mlx-bun/mlx/array";
import type { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { type Cache } from "../contracts/mlx/cache";
import { type CheckpointAttachment } from "../contracts/mlx/checkpoint";

export type CacheKind =
  | "kv"
  | "paged"
  | "rotating"
  | "qkv"
  | "rotating-qkv"
  | "ssm"
  | "turboquant"
  | "mla"
  | "mla-dsa"
  | "mtp-mla";

/** Header kinds of the latent attention legos (`state/latent-legos.ts`), which
 * their own codecs (`LATENT_LEGO_CODECS`) save and restore. */
export type LatentLegoKind = "latent-encoded" | "latent-paged";

export interface TensorSlot {
  blocks?: Array<{ hash: string; bytes: number }>;
  off: number;
  bytes: number;
  shape: number[];
  dtype: number;
  hash: string;
}

export interface CacheHeaderEntry {
  paged?: { capacityTokens: number; blockSize: number; blockTable: number[]; numBlocks: number;
    headDim: number; vHeadDim: number; dtype: Dtype; direct: boolean };
  kind: CacheKind | LatentLegoKind;
  offset: number;
  minimumReusableOffset?: number;
  /** rotating variants: ring write index */
  idx?: number;
  /** rotating variants: window size */
  maxSize?: number;
  /** quantized variants (qkv/rotating-qkv: mlx affine scheme) */
  groupSize?: number;
  bits?: number;
  /** turboquant: per-side bit widths (reuses the `bits` field's slot class
   *  but needs both — asymmetric key/value bit widths, unlike qkv's single
   *  `bits`). `groupSize` is unused for turboquant (fixed at 32, the
   *  BLOCK_SIZE constant — not a configurable field like qkv's). */
  kBits?: number;
  vBits?: number;
  /** turboquant: head_dim, needed to unpack kIdx/vPacked on restore
   *  (packed byte width alone doesn't recover the original element count). */
  headDim?: number;
  /** GLM checkpoint-native compressed-cache geometry. */
  kvLoraRank?: number;
  ropeHeadDim?: number;
  dsaHeadDim?: number;
  maxTokens?: number;
  /** latent-encoded: the conversion start, and whether the saved row still
   *  reads f32 (its f32 values follow its encoded planes). */
  quantizedKvStart?: number;
  latentShadow?: boolean;
  /** latent-paged: the row's page layout. */
  latentPages?: { blockSize: number; blockTable: number[]; numBlocks: number };
  /** kv/rotating: [k, v] · qkv/rotating-qkv: [kPacked, kScales, kBiases,
   *  vPacked, vScales, vBiases] · ssm: [conv, recurrent] · turboquant:
   *  [kIdx, kScales, kZeros, vPacked, vScales] */
  tensors: TensorSlot[];
}

export interface KvSaveMeta {
  attachments?: CheckpointAttachment[];
  modelId?: string;
  /** Caller-supplied store identity, including the model and cache scheme. */
  configFingerprint?: string;
  /** Adapter namespace (PromptCache ns — adapters joined with "+"). */
  ns?: string;
  /** sha256 of tokenizer.json: identical ids must mean identical text. */
  tokenizerHash?: string;
  /** In-flight generation state. The serialized caches cover `tokens`;
   *  pendingToken was sampled from that state but has not been emitted. */
  generationCheckpoint?: {
    key: string;
    cacheNs: string;
    originalPromptTokens: number;
    generatedTokens: number;
    pendingToken: number;
    /** Actual sampler seed, including a server-generated default. Required to
     *  continue the random stream exactly after a restart. */
    seed: number;
    /** True when `seed` was part of the client request rather than generated
     *  by the server. */
    seedWasExplicit: boolean;
  };
}

export interface KvFileHeader extends Omit<KvSaveMeta, "attachments"> {
  attachments?: Array<Omit<CheckpointAttachment, "tensors"> & { tensors: TensorSlot[] }>;
  /** Absent in earlier v3 files, which used mlx-cache-v3. */
  codecProvider?: string;
  formatVersion: 3 | 4 | 5;
  createdAt: number;
  tokens: number[];
  caches: CacheHeaderEntry[];
}

export interface SnapshotContext {
  slots: TensorSlot[];
  push(a: MlxArray, disposeAfter: boolean): void;
  liveSlice(a: MlxArray, upTo: number): MlxArray;
  liveMlaSlice(a: MlxArray, upTo: number): MlxArray;
  pushTriple(t: ops.QuantizedTensor, upTo: number | null): void;
}

export interface CloneContext {
  view(a: MlxArray): MlxArray;
  liveView(a: MlxArray, upTo: number): MlxArray;
  mlaView(a: MlxArray, upTo: number): MlxArray;
  tripleView(t: ops.QuantizedTensor, upTo: number | null): ops.QuantizedTensor;
}

export interface LoadContext {
  path: string;
  /** The restoring graph's own empty cache for this entry's layer (from its
   * `makeCache()`), borrowed for the load: what the graph composed into its
   * caches (the TurboQuant read a `KVCache` carries) reaches the restored one. */
  prototype: Cache;
  arr(slot: TensorSlot): MlxArray;
  grownArr(slot: TensorSlot, offset: number): MlxArray;
  triple(slots: TensorSlot[], at: number): ops.QuantizedTensor;
}

export interface CacheCodec {
  matches(cache: Cache): boolean;
  snapshot(cache: Cache, context: SnapshotContext): CacheHeaderEntry;
  clone(cache: Cache, context: CloneContext): Cache;
  load(entry: CacheHeaderEntry, context: LoadContext): Cache;
  headerTrimmable(entry: CacheHeaderEntry): boolean;
}

/** Codecs belong to the loaded backend binding. The persisted ID is checked
 * before allocating state; codecs are engine code, never loaded from a file. */
export interface CacheCodecProvider {
  readonly id: string;
  forCache(cache: Cache): CacheCodec;
  forHeader(entry: CacheHeaderEntry): CacheCodec;
}

/** Queued immutable snapshot; the queue releases its shared views on settle. */
export interface SpillItem {
  attachments?: CheckpointAttachment[];
  tokens: number[];
  caches: Cache[];
  ns: string;
}

export interface LoadedKvCache {
  attachments?: CheckpointAttachment[];
  tokens: number[];
  header: KvFileHeader;
  caches: Cache[];
}

export interface KvLoadExpect {
  /** Reject on metadata mismatch (pass what the server is running). */
  modelId?: string;
  configFingerprint?: string;
  tokenizerHash?: string;
  ns?: string;
  /** Verify every tensor hash before copying it in (off by default,
   *  `--ssd-cache-verify` — the hash pass roughly doubles restore reads). */
  verify?: boolean;
}
