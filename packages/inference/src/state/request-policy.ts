import { namespacedCache } from "./namespaced-cache";
import { runtimeConfig, withRuntimeConfig, type RuntimeConfig } from "../runtime/config";
import type { KvStorageScheme, PagedKvComposition } from "../contracts/portable/composition";
import { type Cache } from "../contracts/mlx/cache";
import { PagedKVCache, type PagedQuantization, type PagedTurboQuant } from "./paged/cache";
import type { KVCache } from "./kv";
import { turboQuantFusedDecode } from "./turboquant-codec";
import type { MlxPrefixCache as RowPromptCache } from "./checkpoint";
import { disposeResources } from "../runtime/resources";
import { composedStorageOf, isPlainKvCache } from "./capabilities";

export interface CacheFactory { makeCache(): Cache[] }

/** Request state construction and reusable-prefix policy. Equal keys declare
 * compatible state layouts; the executor need not inspect their settings. */
export interface MlxRequestStatePolicy {
  readonly key: string;
  readonly promptCache?: RowPromptCache;
  create(): Cache[];
}

/** The affine pages the composition's KV scheme `kv` names for the layer at
 * cache index `layer`: every layer of a uniform scheme, the listed layers of a
 * per-layer one; none (bf16 or TurboQuant pages) otherwise. */
function pageQuantization(kv: KvStorageScheme, layer: number): PagedQuantization | undefined {
  if (kv.kind === "affine") return { bits: kv.bits, groupSize: kv.groupSize };
  if (kv.kind !== "affine-layers") return undefined;
  const entry = kv.layers.find(item => item.layer === layer);
  return entry ? { bits: entry.bits, groupSize: entry.groupSize } : undefined;
}

/** Replace fresh full-attention storage before prefill with pages storing what
 * the composition's scheme `kv` names for each layer (bf16, affine at the
 * layer's width, TurboQuant read with the layer's composed read), sized for
 * `capacityTokens` (the pool grows past it). Pages store their format from the
 * first position. Sliding layers retain their layout; pre-warmed caches retain
 * their existing storage. Nothing changes without paging. */
export function maybePageKv(cache: Cache[], kv: KvStorageScheme, paged: PagedKvComposition | null, capacityTokens: number): void {
  if (!paged) return;
  if (cache.some((c) => c.offset > 0)) return;
  for (let i = 0; i < cache.length; i++) {
    const storage = composedStorageOf(cache[i]!);
    if (!isPlainKvCache(storage)) continue;
    const turbo: PagedTurboQuant | undefined = kv.kind === "turbo" ? Object.freeze({ kBits: kv.kBits, vBits: kv.vBits,
      fusedDecode: turboQuantFusedDecode(), kernels: (storage as KVCache).turboQuantKernels }) : undefined;
    cache[i]!.dispose(); // fresh (offset 0) — nothing stored yet
    cache[i] = new PagedKVCache(capacityTokens, paged.blockSize, paged.direct, pageQuantization(kv, i), undefined, turbo);
  }
}

/** The prefix namespace of paged state over a method's namespace `base`: the
 * block size, what the pages store and their read (bf16 and uniform affine
 * pages keep the namespace they had before per-layer and TurboQuant pages
 * existed). */
export function pagedPrefixNamespace(kv: KvStorageScheme, paged: PagedKvComposition, base: string): string {
  const direct = paged.direct;
  if (kv.kind === "affine-layers")
    return JSON.stringify(["paged-v1", paged.blockSize, "layers", kv.layers.map(entry => [entry.layer, entry.bits, entry.groupSize]), direct, base]);
  if (kv.kind === "turbo") return JSON.stringify(["paged-v1", paged.blockSize, "turbo", kv.kBits, kv.vBits, direct, base]);
  const pages = pageQuantization(kv, 0);
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
export function bindRequestState(model: CacheFactory, kv: KvStorageScheme, paged: PagedKvComposition | null,
  runtime: RuntimeConfig = runtimeConfig()): RequestState {
  if (!paged) return Object.freeze({
    namespace: (base: string) => base,
    policy: (_capacity: number, _cache: RowPromptCache | undefined, prepared: boolean) =>
      prepared ? { key: UNCACHED_PREPARED, create: () => model.makeCache() } : undefined,
  });
  const namespace = (base: string) => pagedPrefixNamespace(kv, paged, base);
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
