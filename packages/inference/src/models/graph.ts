import type { AutoregressiveGraph, GraphDescriptor, LogitSelection } from "../contracts/portable/graph";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { ModelConfig } from "../artifacts/config";
import type { Cache } from "../contracts/mlx/cache";
import type { MlxDeclaredGraph, MlxModelMemory, MlxTokenAppend, MlxWidthClass, MlxWidthTable } from "../contracts/mlx/graph";
import type { MixedTokenModel } from "../contracts/mlx/token-work";
import type { Composition } from "../contracts/portable/composition";

/** Structural input: generated and hand-written graphs need no union membership. */
export interface MlxGraphOperations<State> {
  forwardHidden(ids: MlxArray, state: State): MlxArray;
  forwardHiddenAsync?: (ids: MlxArray, state: State) => Promise<MlxArray>;
  logitsFromHidden(hidden: MlxArray): MlxArray;
}

/** What the shared token executors (the batch group, its preparation cohorts,
 * the autoregressive binding) consume from a resident graph, whatever family
 * it is: caches, forward and projection, weights for the wired-limit scope,
 * and the ports its declaration promises. A graph outside the model registry
 * satisfies it structurally. */
export interface MlxTokenGraph extends MlxGraphOperations<Cache[]>, MlxDeclaredGraph, MlxModelMemory,
    Partial<MixedTokenModel> {
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
  /** `hiddenLayerTaps`: the per-forward hidden-state capture the graph fills
   * while it is set (the caller owns and disposes what it captured). */
  hiddenTap?: { layers: Set<number>; pos?: number; captured: Map<number, MlxArray> } | null;
  /** Hold the verify kernel's geometry fixed across a speculative run. */
  setSpecKernelPinned?(enabled: boolean): void;
  forwardEmbeddings?(embeddings: MlxArray, state: Cache[], imageMask: MlxArray | null,
    ids: MlxArray, multimodalMask: MlxArray | null): MlxArray;
  createAppend?(policy: { hasAdapters: boolean; pagedKv: boolean }): MlxTokenAppend | null;
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

/** The phase of a forward reaching one of a graph's deprecated entry points
 * (`forwardHidden`, `forwardHiddenMixed`, `forwardHiddenAtPositions`,
 * `forwardEmbeddingsAtPositions`, `forwardEmbeddings`, `createAppend`'s
 * `forwardHidden`, the Qwen MTP companion's `forward`, the generated Gemma 4
 * graphs' `forwardLayers` fallback), derived from the shape of the input it was
 * handed, [rows, tokens, ...]. A committed-append forward (`committedAppend`) of
 * one row is a committed span; otherwise one position per row is a decode and
 * more is a window. A padded forward of one position per row would read as a
 * decode, which ignores prefill padding; no caller pads a one-position forward.
 * @deprecated E2 deletes this adapter when the scheduler calls the named phases
 * (`prefillChunk`, `prefillTail`, `decode`, `verify` and the committed append).
 * Nothing else in a graph or its blocks derives the phase from a shape. */
export function deprecatedPhase(input: MlxArray): "decode" | "window";
export function deprecatedPhase(input: MlxArray, committedAppend: boolean): ReadPhase;
export function deprecatedPhase(input: MlxArray, committedAppend = false): ReadPhase {
  const [rows, tokens] = input.shape as [number, number];
  if (committedAppend && rows === 1) return "committed";
  return tokens === 1 ? "decode" : "window";
}

/** Bind once. No array wrapper, await, readback, or synchronization is added
 * to synchronous forward calls. Existing quant-specific kernels stay inside
 * the supplied operations. This adapter does not take ownership of weights. */
export function bindMlxGraph<State>(
  operations: MlxGraphOperations<State>,
  descriptor: Omit<GraphDescriptor, "backend" | "graphAbi">,
): AutoregressiveGraph<MlxArray, State, MlxArray> {
  const forward = operations.forwardHiddenAsync
    ? operations.forwardHiddenAsync.bind(operations)
    : operations.forwardHidden.bind(operations);
  const project = operations.logitsFromHidden.bind(operations);
  return {
    descriptor: Object.freeze({ ...descriptor, backend: "mlx", graphAbi: "mlx-hidden-bsh-v1" }),
    forwardHidden: forward,
    projectLogits(hidden: MlxArray, selection: LogitSelection) {
      if (selection.type === "all") return project(hidden);
      const shape = hidden.shape;
      if (shape.length !== 3) throw new Error("MLX hidden state must have shape [batch, positions, width]");
      const [batch, positions, width] = shape as [number, number, number];
      const start = selection.type === "last" ? positions - 1 : selection.start;
      const end = selection.type === "last" ? positions : selection.end;
      if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > positions)
        throw new Error("invalid logits position selection");
      const selected = hidden.slice([0, start, 0], [batch, end, width]);
      try { return project(selected); } finally { selected.dispose(); }
    },
  };
}
