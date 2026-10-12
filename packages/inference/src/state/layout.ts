import type { BatchableCache, Cache, SpeculativeRowLayout } from "../contracts/mlx/cache";
import { isBatchableCache } from "./capabilities";
import { unfusedAffineKernels } from "./affine-attention";
import { PaddedKVRows } from "./batched-mask";
import { PaddedQuantKVRows } from "./batched-quant";
import { BatchedKVCache } from "./batched-kv";
import { BatchedQuantizedKVCache } from "./batched-quantized-kv";
import { BatchedRotatingCache } from "./batched-rotating";
import { BatchedRotatingQuantCache } from "./batched-rotating-quant";
import { BatchedSSMCache } from "./batched-ssm";
import { BatchedTurboQuantKVCache } from "./batched-turboquant-kv";
import { DelayedQuantizedKVCache } from "./delayed-quantized-kv";
import { DelayedRotatingQuantizedKVCache } from "./delayed-rotating-quantized-kv";
import { DelayedTurboQuantKVCache } from "./delayed-turboquant-kv";
import { MLACache } from "./glm52-cache";
import { KVCache } from "./kv";
import { PagedKVCache } from "./paged/cache";
import { PagedKvRows } from "./paged/rows";
import { QuantizedKVCache } from "./quantized-kv";
import { RotatingKVCache } from "./rotating-kv";
import { RotatingAffineLayout, SpeculativeRotatingAffineLayout } from "./rotating-kv-layout";
import { RotatingQuantizedKVCache } from "./rotating-quantized-kv";
import { SpeculativeRotatingKVCache } from "./speculative-rotating-kv";
import { SSMCache } from "./ssm";
import { TurboQuantKVCache } from "./turboquant-kv";

/** The row layouts one storage joins, declared together. Each returns a new
 * empty layout whose `mergeRows` takes this storage (and the layouts the same
 * call makes):
 * - `decode`: the layout a running batch keeps the rows in for continuous
 *   decoding;
 * - `prefill`: the layout a prefill cohort keeps them in;
 * - `verify`: the verification rows, which hold the speculative transaction. */
interface RowLayouts<C extends Cache> {
  decode(cache: C): BatchableCache;
  prefill(cache: C): BatchableCache;
  verify(cache: C): SpeculativeRowLayout;
}

/** A storage class: what `instanceof` tests (its constructor may be private). */
type StorageClass<C extends Cache> = Function & { readonly prototype: C };
interface RowLayoutEntry<C extends Cache> extends RowLayouts<C> {
  readonly storage: StorageClass<C>;
}
const entry = <C extends Cache>(storage: StorageClass<C>, layouts: RowLayouts<NoInfer<C>>): RowLayoutEntry<C> =>
  ({ storage, ...layouts });
/** A row layout's own declaration: its decode and prefill siblings are its
 * empty batch, and so are its verification rows when it holds the transaction. */
const ownEmptyBatch = <C extends SpeculativeRowLayout>(storage: StorageClass<C>): RowLayoutEntry<C> =>
  entry(storage, { decode: cache => cache.makeEmptyBatch(), prefill: cache => cache.makeEmptyBatch(), verify: cache => cache.makeEmptyBatch() });

const pagedRows = (cache: PagedKVCache) => new PagedKvRows(cache.capacityTokens, cache.blockSize, cache.direct, cache.quantization);
const rotatingAffineVerify = (cache: { readonly maxSize: number; readonly groupSize: number; readonly bits: number;
  readonly kernels: RotatingAffineLayout["kernels"]; readonly masks: RotatingAffineLayout["masks"] }) =>
  new SpeculativeRotatingAffineLayout(cache.maxSize, cache.groupSize, cache.bits, cache.kernels, cache.masks);

/**
 * Every storage a cache lego holds and the row layouts it joins: the row-layout
 * port's one table, as the codec table in `state/persistence.ts` is the SSD
 * form's. A lego's storage (its graph cache, the storage a bf16-first lego
 * holds, the delayed rows kv maintenance makes, the paged blocks the request
 * policy pages a layer with) declares all three layouts here; a storage with no
 * entry joins no rows and is not a lego. First match wins: a subclass precedes
 * its base. A row layout reached through an entry is not looked up again: its
 * sibling is its own empty batch (the `BatchableCache` port).
 */
const ROW_LAYOUTS: readonly RowLayoutEntry<Cache>[] = [
  entry(PagedKVCache, { decode: pagedRows, prefill: pagedRows, verify: pagedRows }),
  ownEmptyBatch(PagedKvRows),
  ownEmptyBatch(MLACache),
  entry(DelayedRotatingQuantizedKVCache, {
    decode: cache => cache.makeEmptyBatch(), prefill: cache => cache.makeEmptyBatch(),
    verify: cache => new DelayedRotatingQuantizedKVCache(cache.maxSize, cache.groupSize, cache.bits, cache.start,
      cache.kernelSet, undefined, true, cache.compose),
  }),
  entry(RotatingAffineLayout, { decode: cache => cache.makeEmptyBatch(), prefill: cache => cache.makeEmptyBatch(),
    verify: rotatingAffineVerify }),
  entry(BatchedRotatingQuantCache, { decode: cache => cache.makeEmptyBatch(), prefill: cache => cache.makeEmptyBatch(),
    verify: rotatingAffineVerify }),
  entry(RotatingQuantizedKVCache, {
    decode: cache => BatchedRotatingQuantCache.empty(cache.maxSize, cache.groupSize, cache.bits, cache.kernels, [], cache.masks),
    prefill: cache => new RotatingAffineLayout(cache.maxSize, cache.groupSize, cache.bits, cache.kernels, cache.masks),
    verify: rotatingAffineVerify,
  }),
  ownEmptyBatch(SpeculativeRotatingKVCache),
  entry(RotatingKVCache, {
    decode: cache => new BatchedRotatingCache(cache.maxSize, [], cache.masks),
    // Plain rotating rows never convert here (a start of Infinity), so no read
    // takes the layout's kernel set; their dense reads say so.
    prefill: cache => new DelayedRotatingQuantizedKVCache(cache.maxSize, 64, 4, Infinity, unfusedAffineKernels),
    verify: cache => new SpeculativeRotatingKVCache(cache.maxSize, cache.masks),
  }),
  ownEmptyBatch(DelayedTurboQuantKVCache),
  ownEmptyBatch(DelayedQuantizedKVCache),
  ownEmptyBatch(BatchedTurboQuantKVCache),
  ownEmptyBatch(BatchedKVCache),
  ownEmptyBatch(BatchedQuantizedKVCache),
  ownEmptyBatch(BatchedSSMCache),
  entry(TurboQuantKVCache, {
    decode: cache => new BatchedTurboQuantKVCache(cache.kBits, cache.vBits, cache.turboQuantKernels, cache.fusedDecode, cache.masks),
    prefill: cache => new BatchedTurboQuantKVCache(cache.kBits, cache.vBits, cache.turboQuantKernels, cache.fusedDecode, cache.masks),
    verify: cache => new BatchedTurboQuantKVCache(cache.kBits, cache.vBits, cache.turboQuantKernels, cache.fusedDecode, cache.masks),
  }),
  entry(SSMCache, { decode: () => new BatchedSSMCache(), prefill: () => new BatchedSSMCache(), verify: () => new BatchedSSMCache() }),
  entry(QuantizedKVCache, {
    decode: cache => new PaddedQuantKVRows(cache.groupSize, cache.bits, cache.kernels, cache.masks),
    prefill: cache => new BatchedQuantizedKVCache(cache.groupSize, cache.bits, cache.kernels, cache.masks),
    verify: cache => new BatchedQuantizedKVCache(cache.groupSize, cache.bits, cache.kernels, cache.masks),
  }),
  entry(KVCache, {
    decode: cache => new PaddedKVRows(cache.masks),
    prefill: cache => new BatchedKVCache(cache.masks),
    verify: cache => new BatchedKVCache(cache.masks),
  }),
];

/** The table's entry for `cache`'s storage class; undefined for a storage that
 * joins no rows. */
function declaredRowLayouts(cache: Cache): RowLayouts<Cache> | undefined {
  return ROW_LAYOUTS.find(item => cache instanceof item.storage);
}

/** The row layout a running batch keeps this cache's rows in, for ordinary
 * continuous decoding: an empty layout whose `mergeRows` accepts the cache (and
 * the layouts it made). Every storage a graph's caches, or their precision
 * maintenance, produce has one; undefined for a cache that cannot batch. Answers
 * without allocating, so a binding can probe support. */
export function ownedCacheLayoutFactory(cache: Cache): (() => BatchableCache) | undefined {
  const layouts = declaredRowLayouts(cache);
  if (layouts) return () => layouts.decode(cache);
  return isBatchableCache(cache) ? () => cache.makeEmptyBatch() : undefined;
}

/** Bind encoded state to its row layout. Execution owners only move rows. */
export function ownedCacheLayout(cache: Cache): BatchableCache | undefined {
  return ownedCacheLayoutFactory(cache)?.();
}

/** The layout a prefill cohort keeps this cache's rows in. */
export function prefillCacheLayout(cache: Cache): BatchableCache {
  const layouts = declaredRowLayouts(cache);
  if (layouts) return layouts.prefill(cache);
  if (isBatchableCache(cache)) return cache.makeEmptyBatch();
  throw new Error(`No prefill row layout for ${cache.signature()}`);
}

/** The verification rows of this cache's storage, an empty layout holding the
 * speculative transaction, built over the cache's own numerical codec;
 * undefined for a storage that joins no rows. Answers without allocating. */
export function targetRowLayoutFactory(cache: Cache): (() => SpeculativeRowLayout) | undefined {
  const layouts = declaredRowLayouts(cache);
  return layouts ? () => layouts.verify(cache) : undefined;
}

/** The verification rows of this cache's storage (see `targetRowLayoutFactory`). */
export function targetCacheLayout(cache: Cache): SpeculativeRowLayout {
  const make = targetRowLayoutFactory(cache);
  if (!make) throw new Error(`No target row layout for ${cache.signature()}`);
  return make();
}
