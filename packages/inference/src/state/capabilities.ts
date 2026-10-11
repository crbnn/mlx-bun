import { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { BatchableCache,Cache,CompiledDecodeCache,RowBatchCache } from "../contracts/mlx/cache";
import { KVCache } from "./kv";
import { QuantizedKVCache } from "./quantized-kv";
import { RotatingKVCache } from "./rotating-kv";
import { RotatingQuantizedKVCache } from "./rotating-quantized-kv";


/** Retain an independent array handle for positions used across cache writes.
 * The contiguous view preserves the integer values and shares their storage;
 * callers release it with the fetched attention state. */
export function captureRopeOffsets(cache: Pick<Cache, "ropeOffsetArr">): MlxArray | undefined {
  const offsets = cache.ropeOffsetArr;
  return offsets ? ops.contiguous(offsets) : undefined;
}

export function cacheSignature(cache: Cache | undefined): string {
  return cache ? cache.signature() : "unknown";
}

/** The storage a cache holds: a composed lego's (`Cache.composed`), else the
 * cache itself. What a binding probes to learn which layers a fresh cache list
 * can lay out and convert. */
export function composedStorageOf(cache: Cache): Cache {
  return cache.composed ? cache.composed.storage() : cache;
}

/** `stored` (restored or cloned storage) as the cache of a layer whose graph
 * cache is `layer`: composed again by that layer's lego (`ComposedStorage.over`),
 * so it converts as the lego that stored it would have. A cache that already is
 * a lego stays, and so do paged pages, which the paged request policy holds in
 * the lego's place (`state/request-policy.ts`). */
export function composeAtLayer(layer: Cache, stored: Cache): Cache {
  const composed = layer.composed;
  return !composed || stored.composed || cacheSignature(stored) === "kv:paged" ? stored : composed.over(stored);
}

export function isRowBatchCache(cache: Cache): cache is RowBatchCache {
  const candidate = cache as Partial<RowBatchCache>;
  return typeof candidate.batchSize === "number" &&
    typeof candidate.filterRows === "function" &&
    typeof candidate.extractRow === "function";
}

/** Recurrent state (gated DeltaNet): fixed-size, never trimmed. */
export function isRecurrentCache(cache: Cache | undefined): boolean {
  return cacheSignature(cache) === "ssm";
}

export function isPlainKvCache(cache: Cache | undefined): cache is KVCache {
  return cacheSignature(cache) === "kv:plain";
}

export function isQuantizedKvCache(cache: Cache | undefined): cache is QuantizedKVCache {
  return cacheSignature(cache).startsWith("kv:quant:");
}

export function isRotatingPlainCache(cache: Cache | undefined): cache is RotatingKVCache {
  return cacheSignature(cache) === "kv:rotating-plain";
}

export function isTurboQuantKvCache(cache: Cache | undefined): boolean {
  return cacheSignature(cache).startsWith("kv:turboquant:");
}

export function isRotatingQuantizedCache(
  cache: Cache | undefined,
): cache is RotatingQuantizedKVCache {
  return cacheSignature(cache).startsWith("kv:rotating-quant:");
}

/** A cache the compiled decode step can drive (see CompiledDecodeCache). A batched
 * layout that inherits it is a row cache too: callers exclude those separately. */
export function isCompiledDecodeCache(cache: Cache): cache is CompiledDecodeCache {
  const candidate = cache as Partial<CompiledDecodeCache>;
  return typeof candidate.prepareDecodeStep === "function" &&
    typeof candidate.decodePhase === "function" &&
    typeof candidate.decodeSlot === "function" &&
    typeof candidate.decodeInputs === "function" &&
    typeof candidate.commitDecodeStep === "function";
}

export function isBatchableCache(cache: Cache): cache is BatchableCache {
  const candidate = cache as Partial<BatchableCache>;
  return (
    typeof candidate.makeEmptyBatch === "function" &&
    typeof candidate.mergeRows === "function" &&
    typeof candidate.extractRow === "function" &&
    typeof candidate.filterRows === "function" &&
    typeof candidate.projectedBytes === "function" &&
    Array.isArray(candidate.rowOffsets) &&
    Array.isArray(candidate.leftPad)
  );
}
