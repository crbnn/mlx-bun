import type { KvStorageScheme } from "../contracts/portable/composition";
import { type Cache } from "../contracts/mlx/cache";
import { PagedKVCache, type PagedQuantization } from "./paged/cache";
import { composedStorageOf, isPlainKvCache } from "./capabilities";

/** The composition's paging (`Composition.pagedKv`): pages of `blockSize` positions. */
export interface PagedKv { readonly blockSize: number }

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

/** The prefix namespace of paged state over a method's namespace `base`;
 * `direct` is the paged read the composition runs (`MLX_BUN_PAGED_ATTN`). */
export function pagedPrefixNamespace(kv: KvStorageScheme, paged: PagedKv, base: string, direct: boolean): string {
  const pages = pageQuantization(kv);
  return JSON.stringify(["paged-v1", paged.blockSize, pages?.bits ?? 0, pages?.groupSize ?? 64, direct, base]);
}
