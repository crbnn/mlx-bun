import { namespacedCache } from "./namespaced-cache";
import { runtimeConfig, runtimeFlag, withRuntimeConfig, type RuntimeConfig } from "../runtime/config";
import type { KvStorageScheme } from "../contracts/portable/composition";

export interface PagedRequestOptions { pagedKv?: { blockSize?: number } }
export interface CacheFactory { makeCache(): Cache[] }

import { type Cache } from "../contracts/mlx/cache";
import { PagedKVCache, type PagedQuantization } from "./paged/cache";
import type { MlxPrefixCache as RowPromptCache } from "./checkpoint";
import { disposeResources } from "../runtime/resources";
import { composedStorageOf, isPlainKvCache } from "./capabilities";

/** Request state construction and reusable-prefix policy. Equal keys declare
 * compatible state layouts; the executor need not inspect their settings. */
export interface MlxRequestStatePolicy {
  readonly key: string;
  readonly promptCache?: RowPromptCache;
  create(): Cache[];
}

/** The page quantization of the composition's KV scheme `kv`: a uniform 4- or
 * 8-bit affine scheme pages quantized; every other scheme pages bf16. */
function pageQuantization(kv: KvStorageScheme): PagedQuantization | undefined {
  return kv.kind === "affine" && (kv.bits === 4 || kv.bits === 8) ? { bits: kv.bits, groupSize: kv.groupSize } : undefined;
}

/** Replace fresh full-attention storage before prefill with pages quantized
 * as the composition's scheme `kv` names. Sliding layers retain their layout;
 * pre-warmed library caches retain their existing storage. */
export function maybePageKv(
  cache: Cache[], kv: KvStorageScheme, options: PagedRequestOptions, capacityTokens: number,
): void {
  if (!options.pagedKv) return;
  if (cache.some((c) => c.offset > 0)) return;
  const blockSize = options.pagedKv.blockSize ?? PagedKVCache.DEFAULT_BLOCK_SIZE;
  for (let i = 0; i < cache.length; i++) {
    if (isPlainKvCache(composedStorageOf(cache[i]!))) {
      cache[i]!.dispose(); // fresh (offset 0) — nothing stored yet
      cache[i] = new PagedKVCache(capacityTokens, blockSize, undefined, pageQuantization(kv));
    }
  }
}

export function pagedPrefixNamespace(kv: KvStorageScheme, options: PagedRequestOptions, base: string,
  direct = runtimeFlag("MLX_BUN_PAGED_ATTN", false)): string {
  const pages = pageQuantization(kv);
  return JSON.stringify(["paged-v1", options.pagedKv?.blockSize ?? PagedKVCache.DEFAULT_BLOCK_SIZE,
    pages?.bits ?? 0, pages?.groupSize ?? 64, direct, base]);
}

export function bindPagedRequestState(model: CacheFactory, kv: KvStorageScheme, options: PagedRequestOptions,
  capacityTokens: number, cache?: RowPromptCache, runtime: RuntimeConfig = runtimeConfig()): MlxRequestStatePolicy | undefined {
  if (!options.pagedKv) return undefined;
  const blockSize = options.pagedKv.blockSize ?? PagedKVCache.DEFAULT_BLOCK_SIZE;
  const stateOptions = { pagedKv: { blockSize } };
  const direct = runtime.flag("MLX_BUN_PAGED_ATTN", false);
  const namespace = (base: string) => pagedPrefixNamespace(kv, stateOptions, base, direct);
  return { key: namespace(""),
    promptCache: cache ? namespacedCache(cache, namespace) : undefined, create: () => withRuntimeConfig(runtime, () => {
    const caches = model.makeCache();
    try { maybePageKv(caches, kv, stateOptions, capacityTokens); return caches; }
    catch (error) { disposeResources(caches); throw error; }
  }) };
}
