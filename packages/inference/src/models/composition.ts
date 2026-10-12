// The one resolver of a composition record. The application calls it once per
// load, before the graph is built; the convenience load API calls it for
// callers that supply no record.
import { deviceArchitecture } from "@mlx-bun/mlx/ffi";
import type { ModelConfig, TurboQuantScheme } from "../artifacts/config";
import { COMPOSITION_DEFAULTS, type Composition } from "../contracts/portable/composition";
import { runtimeConfig, type RuntimeConfig } from "../runtime/config";
import { resolveKvScheme, type KvQuantOverride } from "../state/kv-scheme";
import { PagedKVCache } from "../state/paged/cache";
import { pagesStore } from "../state/request-policy";
import { decodesCompiled } from "../state/kv-maintenance";

/** Rows per forward when the caller states no cap (the app's `--batch` default). */
export const DEFAULT_MAX_ROWS = 8;
/** The prefill chunk when `MLX_BUN_RD_PREFILL_CHUNK` is unset, the prefill
 * policy's ceiling (`generation/bindings/prefill-policy.ts`). */
export const PREFILL_CHUNK_TOKENS = 2048;

/** The chunk the prefill policy starts from: its `MLX_BUN_RD_PREFILL_CHUNK`
 * override, read through the runtime configuration exactly as it reads it
 * (a set, non-numeric value counts as 2048), else {@link PREFILL_CHUNK_TOKENS}. */
function resolveChunkTokens(runtime: RuntimeConfig): number {
  const explicit = runtime.value("MLX_BUN_RD_PREFILL_CHUNK") === undefined
    ? undefined : runtime.number("MLX_BUN_RD_PREFILL_CHUNK", 2048);
  return explicit === undefined ? PREFILL_CHUNK_TOKENS : Math.max(1, Math.floor(explicit));
}

/** What the caller decided; each absent value takes the default named on it. */
export interface CompositionRequest {
  /** The KV flags: `--kv-quant` (`override`), its TurboQuant spelling,
   * `--quantized-kv-start` and `--fused-sdpa` (the affine read kernels: on,
   * tiled-causal; off, unfused; absent, by the scheme's kind). Absent: bf16.
   * A `config` override reads the artifact's `kv_config.json` from the model
   * config. */
  readonly kv?: {
    readonly override?: KvQuantOverride;
    readonly turboQuant?: TurboQuantScheme;
    readonly quantizedKvStart?: number;
    readonly fusedSdpa?: boolean;
  };
  /** Paged KV (`--paged-kv`): full-attention storage in pages of `blockSize`
   * positions (default 256). Absent or null: unpaged. Pages store bf16 or a
   * uniform 4- or 8-bit affine scheme; another scheme with paging is refused. */
  readonly pagedKv?: { readonly blockSize?: number } | null;
  /** The loaded drafter's tokens per round, and whether the scheduler adapts
   * each round's count up to it. Absent or null: no drafter. */
  readonly draft?: { readonly numDraftTokens: number; readonly adaptive?: boolean } | null;
  /** Adapters are mounted at load. Default false. */
  readonly adapters?: boolean;
  /** The most rows one forward carries. Default {@link DEFAULT_MAX_ROWS}. */
  readonly maxRows?: number;
  /** The groups of a mixed forward share one MLP pass. Default true. */
  readonly packedMixedMlp?: boolean;
  /** One argmax over a decode step whose live rows are all plain greedy. Default true. */
  readonly vectorGreedySampling?: boolean;
  /** A lone ordinary row decodes through the graph's compiled step. Default true;
   * off whatever is asked for a KV scheme or paging with no compiled form. */
  readonly compiledDecode?: boolean;
  /** One forward carries a decode group and a prefill group. Default false. */
  readonly mixedPrefill?: boolean;
  /** Prompt tokens a mixed forward carries beside its decode rows. Default 256. */
  readonly mixedTokenBudget?: number;
  /** A prompt's last token prefills alone, as the first one-position step. Default true. */
  readonly prefillTailSplit?: boolean;
  /** Grammar-forced continuations enter the KV without one forward per token. Default false. */
  readonly grammarJump?: boolean;
  /** Compose the Neural Engine bridge where a graph builds a Neural Engine
   * operation. Default true: every Apple-silicon Mac has the unit. */
  readonly aneBridge?: boolean;
}

/** Resolve the record for `config` on this machine. Reads the GPU architecture
 * and the prefill chunk override from the current runtime configuration. The
 * result is frozen. */
export function resolveComposition(config: ModelConfig, request: CompositionRequest = {}): Composition {
  const maxRows = request.maxRows ?? DEFAULT_MAX_ROWS;
  if (!Number.isSafeInteger(maxRows) || maxRows < 1) throw new Error("composition maxRows must be a positive integer");
  const draft = request.draft;
  if (draft && (!Number.isSafeInteger(draft.numDraftTokens) || draft.numDraftTokens < 1))
    throw new Error("composition draft tokens must be a positive integer");
  const mixedTokenBudget = request.mixedTokenBudget ?? COMPOSITION_DEFAULTS.mixedTokenBudget;
  if (!Number.isSafeInteger(mixedTokenBudget) || mixedTokenBudget < 1)
    throw new Error("composition mixedTokenBudget must be a positive integer");
  const kv = resolveKvScheme({ ...request.kv, config: config.kvQuant }).storage;
  const pagedKv = request.pagedKv ? Object.freeze({ blockSize: request.pagedKv.blockSize ?? PagedKVCache.DEFAULT_BLOCK_SIZE }) : null;
  if (pagedKv && (!Number.isSafeInteger(pagedKv.blockSize) || pagedKv.blockSize < 1))
    throw new Error("composition paged KV block size must be a positive integer");
  if (pagedKv && !pagesStore(kv))
    throw new Error("paged KV stores bf16 or uniform 4- or 8-bit affine pages; per-layer and TurboQuant pages are not implemented");
  const runtime = runtimeConfig();
  return Object.freeze({
    device: deviceArchitecture(),
    aneBridge: request.aneBridge ?? true,
    kv,
    pagedKv,
    draftDepth: !draft ? 0 : draft.adaptive ? Object.freeze({ adaptive: true as const, max: draft.numDraftTokens }) : draft.numDraftTokens,
    prefillChunkTokens: resolveChunkTokens(runtime),
    adapters: request.adapters ?? false,
    maxRows,
    packedMixedMlp: request.packedMixedMlp ?? COMPOSITION_DEFAULTS.packedMixedMlp,
    vectorGreedySampling: request.vectorGreedySampling ?? COMPOSITION_DEFAULTS.vectorGreedySampling,
    // A cache lego without a compiled form decodes eagerly: decided here, once.
    compiledDecode: (request.compiledDecode ?? COMPOSITION_DEFAULTS.compiledDecode) && decodesCompiled(kv, pagedKv !== null),
    mixedPrefill: request.mixedPrefill ?? COMPOSITION_DEFAULTS.mixedPrefill,
    mixedTokenBudget,
    prefillTailSplit: request.prefillTailSplit ?? COMPOSITION_DEFAULTS.prefillTailSplit,
    grammarJump: request.grammarJump ?? COMPOSITION_DEFAULTS.grammarJump,
  });
}

/** Refuse `composition` for a graph that carries no composition record: such a
 * graph composes its own caches (DiffusionGemma's denoising state, GLM-5.2's
 * compressed MLA), and no lego of a quantized or paged KV scheme exists for
 * them yet, so the record could not be served as named. Serving it as bf16
 * would report an option as implemented when it is not. */
export function assertComposedCaches(model: object, composition: Composition): void {
  if ((model as { composition?: Composition }).composition) return;
  if (composition.kv.kind === "bf16" && !composition.pagedKv) return;
  const named = composition.pagedKv ? "paged KV" : `${composition.kv.kind} KV`;
  throw new Error(`this graph composes its own caches, which have no ${named} form yet (--kv-quant, --paged-kv)`);
}
