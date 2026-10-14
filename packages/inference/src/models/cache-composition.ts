import type { Cache } from "../contracts/mlx/cache";
import type { MlxCacheLayer } from "../contracts/mlx/graph";
import type { Composition } from "../contracts/portable/composition";
import { AttentionMasks } from "../state/attention-read";
import { Bf16FirstKv4Head256Cache, composedFullAttentionCache, composedSlidingAttentionCache } from "../state/bf16-first-kv";
import { composedAffineKernels } from "../state/affine-attention";
import { SSMCache } from "../state/ssm";
import { composedLatentCache } from "../state/latent-legos";
import { maybePageKv } from "../state/request-policy";
import { disposeResources } from "../runtime/resources";
import { denseTurboQuantKernels, deferredTurboQuantKernels } from "../state/turboquant-attention";

/** The loader composes a request's cache legos from the file's declaration. */
export function composeGraphCaches(graph: { readonly cacheLayers: readonly MlxCacheLayer[]; readonly cachePaging?: boolean },
  composition: {readonly kv: Composition["kv"]; readonly pagedKv: {readonly blockSize: number; readonly direct?: boolean} | null}): Cache[] {
  const masks = new AttentionMasks();
  const { kv, pagedKv } = composition;
  const caches = graph.cacheLayers.map((layer, index) => {
    switch (layer.kind) {
      case "recurrent": return new SSMCache();
      case "sliding": return composedSlidingAttentionCache(kv, index, layer.window, masks);
      case "latent": return composedLatentCache(kv, pagedKv, index, layer.geometry);
      case "full":
        if (layer.kv4Head256 && kv.kind === "affine" && kv.bits === 4 && kv.groupSize === 64)
          return new Bf16FirstKv4Head256Cache(kv.quantizedKvStart ?? 0, composedAffineKernels(kv.kernels), masks);
        return composedFullAttentionCache(kv, index, masks,
          layer.turboQuant === "deferred" ? deferredTurboQuantKernels : denseTurboQuantKernels);
    }
  });
  if (graph.cachePaging) {
    try { maybePageKv(caches, kv, pagedKv ? {...pagedKv,direct:pagedKv.direct??false} : null, pagedKv?.blockSize ?? 0); }
    catch (error) { disposeResources(caches); throw error; }
  }
  return caches;
}

/** Bind request-state construction at load; the graph only declares layout. */
export function bindGraphCaches<Graph extends { readonly cacheLayers: readonly MlxCacheLayer[]; readonly cachePaging?: boolean }>(graph: Graph,
  composition: {readonly kv: Composition["kv"]; readonly pagedKv: {readonly blockSize: number; readonly direct?: boolean} | null}): Graph & { makeCache(): Cache[] } {
  return Object.assign(graph, { makeCache: () => composeGraphCaches(graph, composition) });
}
