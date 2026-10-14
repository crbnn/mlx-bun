import type { BatchableCache, Cache, KvMaintenance, SpeculativeRowLayout } from "../contracts/mlx/cache";
import type { KvStorageScheme } from "../contracts/portable/composition";
import { isBatchableCache } from "./capabilities";
import { composedAffineKernels, unfusedAffineKernels } from "./affine-attention";
import { turboQuantFusedDecode } from "./turboquant-codec";
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
import { EncodedLatentCache, PagedLatentCache } from "./latent-legos";
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

const pagedRows = (cache: PagedKVCache) => new PagedKvRows(cache.capacityTokens, cache.blockSize, cache.direct, cache.quantization, cache.turbo);
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
  ownEmptyBatch(EncodedLatentCache),
  ownEmptyBatch(PagedLatentCache),
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
    prefill: cache => new BatchedKVCache(cache.masks, cache.windowRead),
    verify: cache => new BatchedKVCache(cache.masks, cache.windowRead),
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

/** A one-row layout's row in its serial form (owned; the layout keeps its own
 * state), or null for a cache that is already serial: what persistence takes
 * for a row that never joined another. */
export function extractSoleRow(cache: Cache): Cache | null {
  return isBatchableCache(cache) && cache.batchSize === 1 ? cache.extractRow(0) : null;
}

/** Row `row` of a running layout in its serial form (owned), or null when the
 * layout cannot publish that row covering exactly `tokens` positions (a
 * recurrence that cannot be trimmed to them). */
export function publishRow(layout: Cache, row: number, tokens: number): Cache | null {
  return isBatchableCache(layout) && (layout.canPublishRow?.(row, tokens) ?? true) ? layout.extractRow(row) : null;
}

/** Keep only the rows `keep` (sorted ascending) of a running layout. Serial
 * state is not row-filtered: a lone adopted row leaves with the whole batch. */
export function filterLayoutRows(layout: Cache, keep: readonly number[]): void {
  if (!isBatchableCache(layout)) throw new Error(`applyFilter: adopted ${layout.signature()} state cannot be row-filtered`);
  layout.filterRows(keep);
}

/** What a graph's fresh caches say about admitting a row when each one's
 * storage owns its dynamic-row batching (a layout such as GLM-5.2's compressed
 * state): its projected KV bytes at a length, and the most positions it holds.
 * Null otherwise: admission bills the composition's KV scheme by the config.
 * The projections read geometry only, so the caches may be released after. */
export function rowAdmission(caches: readonly Cache[]): { bytes(tokens: number): number; readonly maxTokens: number } | null {
  if (!caches.every(isBatchableCache)) return null;
  const layouts = caches as readonly BatchableCache[];
  const projectors = layouts.map(cache => (tokens: number) => cache.projectedBytes(tokens));
  return Object.freeze({
    bytes: (tokens: number) => projectors.reduce((sum, project) => sum + project(tokens), 0),
    maxTokens: Math.min(...layouts.map(cache => cache.maxTokens ?? Infinity)),
  });
}

// Batch preparation: where a request's serial caches join row layouts under the
// composition's KV scheme. The graph's `makeCache` composes the scheme's legos
// (`state/bf16-first-kv.ts`), which convert themselves inside their own
// appends, so nothing here converts a serial cache. A lego's storage moves into
// the layout, and the rows of a scheme that converts later than the first
// append (or inside a prefill cohort) take the delayed row layouts, which
// convert each row at its own planned boundary.

export type { KvMaintenance };

/** Maintenance of the bf16 scheme: no layer converts. */
export const unchangedKv: KvMaintenance = Object.freeze({});

/** One layer's serial cache made ready for a row layout: a lego hands over its
 * storage (and how it composes storage back for serial use); other storage is
 * itself. */
function released(cache: Cache): { row: Cache; compose?: (storage: Cache) => Cache } {
  const composed = cache.composed;
  return composed ? { row: composed.release(), compose: composed.over } : { row: cache };
}

/** The positions a committed append may add before any layer's transition. */
function maxAppendTokens(cache: readonly Cache[]): number {
  return Math.min(Infinity, ...cache.map(c => c.maxAppendTokens?.() ?? Infinity));
}

/** The row-layout maintenance of the composition's KV scheme `kv`.
 * `preparePrefill` (a prefill cohort forms) routes every converting layer into
 * its delayed row layout, built with the scheme's kernel set; `prepareBatch`
 * (a prefilled row joins decode) routes a lego's storage to the batch layouts
 * and, when the scheme converts after a start, into the same delayed layouts.
 * A layout built from a lego hands back the lego's composition for the rows it
 * extracts. */
export function createKvMaintenance(kv: KvStorageScheme): KvMaintenance {
  if (kv.kind === "bf16") return unchangedKv;
  const start = kv.quantizedKvStart ?? 0;
  if (kv.kind === "turbo") {
    const fusedDecode = turboQuantFusedDecode();
    const preparePrefill = (cache: Cache[]): void => {
      for (let layer = 0; layer < cache.length; layer++) {
        const { row, compose } = released(cache[layer]!);
        cache[layer] = row instanceof KVCache || row instanceof TurboQuantKVCache
          ? new DelayedTurboQuantKVCache(kv.kBits, kv.vBits, start, fusedDecode, row.turboQuantKernels, row, compose) : row;
      }
    };
    return Object.freeze(start > 0 ? { preparePrefill, prepareBatch: preparePrefill, maxAppendTokens }
      : { preparePrefill, prepareBatch: routeStorage });
  }
  const kernelSet = composedAffineKernels(kv.kernels);
  // Resolve layer policy once when composing execution, never per token.
  const spec = kv.kind === "affine" ? () => ({ bits: kv.bits, groupSize: kv.groupSize })
    : (() => { const layers = new Map(kv.layers.map(entry => [entry.layer, entry])); return (layer: number) => layers.get(layer); })();
  const preparePrefill = (cache: Cache[]): void => {
    for (let layer = 0; layer < cache.length; layer++) {
      const { row, compose } = released(cache[layer]!);
      const scheme = spec(layer);
      if (!scheme || !(row instanceof KVCache || row instanceof QuantizedKVCache || row instanceof RotatingKVCache ||
          row instanceof RotatingQuantizedKVCache)) { cache[layer] = row; continue; }
      cache[layer] = row instanceof RotatingKVCache || row instanceof RotatingQuantizedKVCache
        ? new DelayedRotatingQuantizedKVCache(row.maxSize, scheme.groupSize, scheme.bits, start, kernelSet, row, false, compose)
        : new DelayedQuantizedKVCache(scheme.groupSize, scheme.bits, start, kernelSet, row, compose);
    }
  };
  return Object.freeze(start > 0 ? { preparePrefill, prepareBatch: preparePrefill, maxAppendTokens }
    : { preparePrefill, prepareBatch: routeStorage });
}

/** A prefilled row joins decode under a scheme that converts after the first
 * append: every lego has converted (or holds a layer the scheme leaves bf16),
 * so its storage joins the batch layouts as it is. */
function routeStorage(cache: Cache[]): void {
  for (let layer = 0; layer < cache.length; layer++) cache[layer] = released(cache[layer]!).row;
}
