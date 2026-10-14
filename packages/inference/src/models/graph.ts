import type { ModelConfig } from "../artifacts/config";
import type { Cache } from "../contracts/mlx/cache";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxDeclaredGraph, MlxModelMemory, MlxTokenPhases, MlxRowTable, MlxCacheLayer } from "../contracts/mlx/graph";
import { DEFAULT_MAX_ROWS, PREFILL_CHUNK_TOKENS } from "./composition";
import type { Composition } from "../contracts/portable/composition";

/** What the shared token executors (the batch group, its preparation cohorts,
 * the autoregressive binding) consume from a resident graph, whatever family
 * it is: caches, the named phases (`tokenPhasesOf`), weights for the
 * wired-limit scope, and the ports its declaration promises. A graph outside
 * the model registry satisfies it structurally. */
export interface MlxTokenGraph extends MlxDeclaredGraph, Partial<MlxTokenPhases>, MlxModelMemory {
  readonly config: ModelConfig;
  /** The served configuration the graph was built for; absent for a graph
   * that composes no KV scheme (its caches are its own). */
  readonly composition?: Composition;
  /** The mounted-adapter selection; absent when the graph mounts none. */
  readonly loraState?: { active: string[] };
  /** One cache per cache layer: the caches the composition's KV scheme names
   * (`composedFullAttentionCache`, `composedSlidingAttentionCache`). */
  readonly cacheLayers?: readonly MlxCacheLayer[];
  makeCache(): Cache[];
  /** The vision prefill phase over a prompt's prepared embeddings (borrowed), for
   *  direct generation: image tokens attending each other both ways where
   *  `imageMask` marks them, `ids` feeding per-layer inputs; `width` is the class
   *  of the prompt's rows. */
  prefillEmbeddings?(embeddings: MlxArray, state: Cache[], imageMask: MlxArray | null,
    ids: MlxArray, multimodalMask: MlxArray | null, width: number): MlxArray;
}

/** Materialize the exact row keys the caller's scheduler can supply. */
export function rowTable<Operation>(operation: Operation, rows: number = DEFAULT_MAX_ROWS * PREFILL_CHUNK_TOKENS): MlxRowTable<Operation> {
  return Object.freeze(Object.fromEntries(Array.from({ length: rows }, (_, i) => [i + 1, operation])));
}

/** Generic graphs bind their row-specific operations once, before execution. */
export function rowTableFrom<Operation>(build: (rows: number) => Operation, rows: number = DEFAULT_MAX_ROWS * PREFILL_CHUNK_TOKENS): MlxRowTable<Operation> {
  return Object.freeze(Object.fromEntries(Array.from({ length: rows }, (_, i) => [i + 1, build(i + 1)])));
}

/** Maximum token rows the scheduler can present under this configuration. */
export function phaseRowLimit(composition: Pick<Composition, "maxRows" | "prefillChunkTokens" | "draftDepth">): number {
  const depth = typeof composition.draftDepth === "number" ? composition.draftDepth : composition.draftDepth.max;
  return composition.maxRows * Math.max(composition.prefillChunkTokens, depth + 1);
}

/** The phase a forward runs, named by its caller: one new position per row
 * (`decode`), a causal window of positions (`window`: prefill chunks and tails,
 * speculative verify windows), or a span of already decided positions
 * (`committed`: token fill). The names are the cache's own reads
 * (`AttentionCache.appendDecode`, `appendWindow`,
 * `CommittedAttentionCache.appendCommitted`). */
export type ReadPhase = "decode" | "window" | "committed";
