import type { ModelConfig } from "../artifacts/config";
import type { Cache } from "../contracts/mlx/cache";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxDeclaredGraph, MlxModelMemory, MlxTokenPhases, MlxWidthClass, MlxWidthTable } from "../contracts/mlx/graph";
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
  /** Attention layers whose reads must be plain keys and values (see `kv.denseReads`). */
  readonly requiredDenseKvLayers: readonly number[];
  /** One cache per cache layer: the caches the composition's KV scheme names
   * (`composedFullAttentionCache`, `composedSlidingAttentionCache`). */
  makeCache(): Cache[];
  /** The vision prefill phase over a prompt's prepared embeddings (borrowed), for
   *  direct generation: image tokens attending each other both ways where
   *  `imageMask` marks them, `ids` feeding per-layer inputs; `width` is the class
   *  of the prompt's rows. */
  prefillEmbeddings?(embeddings: MlxArray, state: Cache[], imageMask: MlxArray | null,
    ids: MlxArray, multimodalMask: MlxArray | null, width: MlxWidthClass): MlxArray;
}

/** The width table of a graph whose one operation serves every width class. */
export function atEveryWidth<Operation>(operation: Operation): MlxWidthTable<Operation> {
  return Object.freeze({ one: operation, twoToThree: operation, four: operation, fiveToEight: operation, wider: operation });
}

/** A width table whose entry for each class `build` makes once, at construction. */
export function byWidth<Operation>(build: (width: MlxWidthClass) => Operation): MlxWidthTable<Operation> {
  return Object.freeze({ one: build("one"), twoToThree: build("twoToThree"), four: build("four"),
    fiveToEight: build("fiveToEight"), wider: build("wider") });
}

/** The phase a forward runs, named by its caller: one new position per row
 * (`decode`), a causal window of positions (`window`: prefill chunks and tails,
 * speculative verify windows), or a span of already decided positions
 * (`committed`: token fill). The names are the cache's own reads
 * (`AttentionCache.appendDecode`, `appendWindow`,
 * `CommittedAttentionCache.appendCommitted`). */
export type ReadPhase = "decode" | "window" | "committed";
