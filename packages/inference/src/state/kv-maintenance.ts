// The row-layout half of a composition's KV scheme. The graph's `makeCache`
// composes the scheme's legos (`state/bf16-first-kv.ts`), which convert
// themselves inside their own appends, so nothing here converts a serial cache.
// What remains is where a request's serial caches join row layouts: a lego's
// storage moves into the layout, and the rows of a scheme that converts later
// than the first append (or inside a prefill cohort) take the delayed row
// layouts, which convert each row at its own planned boundary (B2 moves this
// into `state/layout.ts`).
import type { KvStorageScheme } from "../contracts/portable/composition";
import { type Cache, type KvMaintenance } from "../contracts/mlx/cache";
import { composedAffineKernels } from "./affine-attention";
import { KVCache } from "./kv";
import { QuantizedKVCache } from "./quantized-kv";
import { RotatingKVCache } from "./rotating-kv";
import { RotatingQuantizedKVCache } from "./rotating-quantized-kv";
import { TurboQuantKVCache } from "./turboquant-kv";
import { turboQuantFusedDecode } from "./turboquant-codec";
import { DelayedRotatingQuantizedKVCache } from "./delayed-rotating-quantized-kv";
import { DelayedQuantizedKVCache } from "./delayed-quantized-kv";
import { DelayedTurboQuantKVCache } from "./delayed-turboquant-kv";

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

/** Whether every serial cache a decoding row of a composition holds is one the
 * compiled decode step drives (`CompiledDecodeCache`): bf16 storage, and affine
 * storage converting at the first append. TurboQuant storage, the delayed row
 * layouts a later start routes rows into and paged blocks have no compiled form
 * yet. The loader resolves `Composition.compiledDecode` with this, once; nothing
 * at run time asks a cache. */
export function decodesCompiled(kv: KvStorageScheme, paged: boolean): boolean {
  return !paged && (kv.kind === "bf16" || (kv.kind !== "turbo" && (kv.quantizedKvStart ?? 0) === 0));
}

/** A prefilled row joins decode under a scheme that converts after the first
 * append: every lego has converted (or holds a layer the scheme leaves bf16),
 * so its storage joins the batch layouts as it is. */
function routeStorage(cache: Cache[]): void {
  for (let layer = 0; layer < cache.length; layer++) cache[layer] = released(cache[layer]!).row;
}
