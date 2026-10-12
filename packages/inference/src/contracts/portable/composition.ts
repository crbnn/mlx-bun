/** The read kernels of a composition's affine-quantized caches. `tiled-causal`
 * is OptiQ's rule: a window through the N-tiled fused port while the cache's own
 * mask is plain causal (where the port takes the configuration), every other
 * read through the unfused port. `unfused` is mlx-lm's port for every read.
 * `--fused-sdpa on` composes the first and `off` the second; unset, a per-layer
 * scheme (`--kv-quant config`) composes `tiled-causal` and a uniform one
 * `unfused`, which keeps each scheme's oracle parity. */
export type AffineKernelChoice = "tiled-causal" | "unfused";

/** How a composition's attention caches store keys and values. A quantized
 * scheme carries `quantizedKvStart` only when it is nonzero: storage stays
 * bf16 until a sequence holds that many tokens, then converts in place. */
export type KvStorageScheme =
  | { readonly kind: "bf16" }
  /** Every attention layer, affine-quantized with one bit width and group size. */
  | { readonly kind: "affine"; readonly bits: number; readonly groupSize: number; readonly quantizedKvStart?: number;
      readonly kernels: AffineKernelChoice }
  /** The layers listed (cache index), each with its own bit width and group size; the others stay bf16. */
  | { readonly kind: "affine-layers"; readonly layers: readonly AffineKvLayer[]; readonly quantizedKvStart?: number;
      readonly kernels: AffineKernelChoice }
  /** Full-attention layers TurboQuant-encoded; sliding-window layers stay bf16. */
  | { readonly kind: "turbo"; readonly kBits: number; readonly vBits: number; readonly quantizedKvStart?: number };

/** Full-attention keys and values held in pages (`--paged-kv`). */
export interface PagedKvComposition {
  /** Positions per page. */
  readonly blockSize: number;
  /** The pages' read of at most eight queries without an array mask (decode,
   * and prefill and verification windows that short): the direct Metal kernel
   * over the blocks (Lab numerics) where true, otherwise the gathered blocks
   * through the fused SDPA (the numerical control). Pages the kernel does not
   * read (TurboQuant, affine widths that straddle a 32-bit word) read gathered.
   * Pages the kernel reads have no compiled decode form (the kernel's
   * partitions follow the context length): their composition resolves
   * `compiledDecode` off. */
  readonly direct: boolean;
}

export interface AffineKvLayer {
  readonly layer: number;
  readonly bits: number;
  readonly groupSize: number;
}

/** The bf16 KV scheme: what a graph whose composition names no scheme stores. */
export const BF16_KV: KvStorageScheme = Object.freeze({ kind: "bf16" });

/** Drafted tokens per verify round: a fixed count (0 when the composition has
 * no drafter), or adaptive, where the scheduler picks each round's count up to
 * `max`. */
export type DraftDepth = number | { readonly adaptive: true; readonly max: number };

/** The facts that fix one served configuration, resolved once by the
 * application before the loader builds the graph and consumed as data. Every
 * field holds a resolved value; nothing here is "unset" or "auto". */
export interface Composition {
  /** The GPU's architecture identifier as MLX reports it (for example `applegpu_g16s`). */
  readonly device: string;
  /** The Neural Engine bridge is composed: a graph that builds a Neural Engine
   * operation builds it at load, and a bridge that fails there fails the load. */
  readonly aneBridge: boolean;
  /** The storage and reads of every attention cache the graph's `makeCache` composes. */
  readonly kv: KvStorageScheme;
  /** Full-attention keys and values held in pages of `blockSize` positions
   * (`--paged-kv`), or null. Every request's state takes the paged lego in place
   * of the plain full-attention storage the graph composes; its pages store
   * what `kv` names. */
  readonly pagedKv: PagedKvComposition | null;
  readonly draftDepth: DraftDepth;
  /** The prefill chunk size, in prompt tokens, the composition is built for.
   * The prefill planner may hand the graph fewer rows than this as a prompt's
   * tail. Its per-prompt halving for a graph with a bounded prefill workspace
   * is the planner's business and does not change what the composition built. */
  readonly prefillChunkTokens: number;
  /** LoRA adapters are mounted when the model loads. */
  readonly adapters: boolean;
  /** The most rows one forward carries (`--batch`, mlx-lm's `--batch-size`). */
  readonly maxRows: number;
  /** The groups of a mixed forward share one MLP pass (default on); off, each
   * group runs its own. */
  readonly packedMixedMlp: boolean;
  /** A decode step whose live rows are all plain greedy samples them with one
   * argmax over the batch instead of one sampler call per row (default on): the
   * same per-row arithmetic in one graph. Off, every row samples alone. */
  readonly vectorGreedySampling: boolean;
  /** A lone ordinary row decodes through the graph's compiled step where the
   * graph builds one (default on). The step is bit-identical to the ordinary
   * decode forward over every cache lego that has a compiled form; off, every
   * row decodes through the forward. Direct pages have none (`pagedKv.direct`),
   * so their composition resolves it off. */
  readonly compiledDecode: boolean;
  /** One forward carries a decode group and a prefill group where the graph
   * declares mixed rows (default off). */
  readonly mixedPrefill: boolean;
  /** Prompt tokens one mixed forward carries beside its decode rows (default 256). */
  readonly mixedTokenBudget: number;
  /** A prompt's last token prefills alone, as the first one-position step
   * (mlx-lm's convention, default on); off, the final chunk prefills whole and
   * token zero samples from it. */
  readonly prefillTailSplit: boolean;
  /** Grammar-forced continuations enter the KV without one forward per token
   * (jump-forward decoding, default off). */
  readonly grammarJump: boolean;
  /** Every speculative row drafts by argmax instead of the request sampler's
   * coupled draw (default off). The draft moves acceptance and draft cost, not
   * the output distribution (see `generation/speculative/draft-policy`). */
  readonly greedyDraft: boolean;
  /** The GeGLU activations (and a Gemma head's logit softcap) run the compiled
   * closures, the reference's `mx.compile`d geglu and softcap (default on); off,
   * the spelled-out ops, which compute the same bits. */
  readonly compiledGeglu: boolean;
}

/** The value of each composition fact a caller leaves unstated: what
 * `resolveComposition` resolves, and what a graph built without a record runs. */
export const COMPOSITION_DEFAULTS = Object.freeze({
  pagedKv: null,
  packedMixedMlp: true,
  vectorGreedySampling: true,
  compiledDecode: true,
  mixedPrefill: false,
  mixedTokenBudget: 256,
  prefillTailSplit: true,
  grammarJump: false,
  greedyDraft: false,
  compiledGeglu: true,
} satisfies Partial<Composition>);
