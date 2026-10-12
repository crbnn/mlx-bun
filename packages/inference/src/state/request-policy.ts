import { namespacedCache } from "./namespaced-cache";
import { runtimeConfig, runtimeFlag, withRuntimeConfig, type RuntimeConfig } from "../runtime/config";
import type { KvStorageScheme } from "../contracts/portable/composition";
import { type Cache } from "../contracts/mlx/cache";
import { PagedKVCache, type PagedQuantization } from "./paged/cache";
import type { MlxPrefixCache as RowPromptCache } from "./checkpoint";
import { disposeResources } from "../runtime/resources";
import { composedStorageOf, isPlainKvCache } from "./capabilities";

/** The composition's paging (`Composition.pagedKv`): pages of `blockSize` positions. */
export interface PagedKv { readonly blockSize: number }
export interface CacheFactory { makeCache(): Cache[] }

/** Request state construction and reusable-prefix policy. Equal keys declare
 * compatible state layouts; the executor need not inspect their settings. */
export interface MlxRequestStatePolicy {
  readonly key: string;
  readonly promptCache?: RowPromptCache;
  create(): Cache[];
}

/** The page quantization of the composition's KV scheme `kv`: a uniform 4- or
 * 8-bit affine scheme pages quantized; bf16 pages bf16. */
function pageQuantization(kv: KvStorageScheme): PagedQuantization | undefined {
  return kv.kind === "affine" && (kv.bits === 4 || kv.bits === 8) ? { bits: kv.bits, groupSize: kv.groupSize } : undefined;
}

/** Whether the paged lego stores what `kv` names: bf16 pages, or affine pages of
 * a uniform 4- or 8-bit scheme. Per-layer and TurboQuant pages are not
 * implemented yet; a composition naming them with paging is refused where it is
 * resolved (`resolveComposition`). */
export function pagesStore(kv: KvStorageScheme): boolean {
  return kv.kind === "bf16" || pageQuantization(kv) !== undefined;
}

/** Replace fresh full-attention storage before prefill with pages quantized
 * as the composition's scheme `kv` names, sized for `capacityTokens` (the pool
 * grows past it). Sliding layers retain their layout; pre-warmed caches
 * retain their existing storage. Nothing changes without paging. */
export function maybePageKv(cache: Cache[], kv: KvStorageScheme, paged: PagedKv | null, capacityTokens: number): void {
  if (!paged) return;
  if (cache.some((c) => c.offset > 0)) return;
  for (let i = 0; i < cache.length; i++) {
    if (isPlainKvCache(composedStorageOf(cache[i]!))) {
      cache[i]!.dispose(); // fresh (offset 0) — nothing stored yet
      cache[i] = new PagedKVCache(capacityTokens, paged.blockSize, undefined, pageQuantization(kv));
    }
  }
}

/** The prefix namespace of paged state over a method's namespace `base`. */
export function pagedPrefixNamespace(kv: KvStorageScheme, paged: PagedKv, base: string,
  direct = runtimeFlag("MLX_BUN_PAGED_ATTN", false)): string {
  const pages = pageQuantization(kv);
  return JSON.stringify(["paged-v1", paged.blockSize, pages?.bits ?? 0, pages?.groupSize ?? 64, direct, base]);
}

/** The key of a request whose prefix is not reusable (unidentified prepared media). */
const UNCACHED_PREPARED = "uncached-prepared-input";

/** How every request of one composition holds its state, bound once with the
 * graph: the composition's paging over the caches the graph composes, and the
 * prefix namespace it adds. */
export interface RequestState {
  /** The prefix namespace of a request whose method's namespace is `base`. */
  namespace(base: string): string;
  /** The state of one request that may grow to `capacityTokens`. `promptCache`
   * is the cache it reuses prefixes through; `prepared` marks a request whose
   * prefix is not reusable, which takes fresh state and no cache. Undefined:
   * the execution group's own caches and cache. */
  policy(capacityTokens: number, promptCache: RowPromptCache | undefined, prepared: boolean): MlxRequestStatePolicy | undefined;
}

/** The request state of the composition whose KV scheme is `kv` and paging is
 * `paged`, over `model`'s caches. `runtime` scopes cache construction. */
export function bindRequestState(model: CacheFactory, kv: KvStorageScheme, paged: PagedKv | null,
  runtime: RuntimeConfig = runtimeConfig()): RequestState {
  if (!paged) return Object.freeze({
    namespace: (base: string) => base,
    policy: (_capacity: number, _cache: RowPromptCache | undefined, prepared: boolean) =>
      prepared ? { key: UNCACHED_PREPARED, create: () => model.makeCache() } : undefined,
  });
  const direct = runtime.flag("MLX_BUN_PAGED_ATTN", false);
  const namespace = (base: string) => pagedPrefixNamespace(kv, paged, base, direct);
  return Object.freeze({
    namespace,
    policy(capacityTokens: number, cache: RowPromptCache | undefined, prepared: boolean): MlxRequestStatePolicy {
      const create = () => withRuntimeConfig(runtime, () => {
        const caches = model.makeCache();
        try { maybePageKv(caches, kv, paged, capacityTokens); return caches; }
        catch (error) { disposeResources(caches); throw error; }
      });
      return prepared ? { key: `${UNCACHED_PREPARED}:${namespace("")}`, create }
        : { key: namespace(""), promptCache: cache ? namespacedCache(cache, namespace) : undefined, create };
    },
  });
}
