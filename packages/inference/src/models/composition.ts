// The one resolver of a composition record. The application calls it once per
// load, before the graph is built; the convenience load API calls it for
// callers that supply no record.
import { deviceArchitecture } from "@mlx-bun/mlx/ffi";
import type { ModelConfig, TurboQuantScheme } from "../artifacts/config";
import type { Composition } from "../contracts/portable/composition";
import { aneAvailable } from "../kernels/ane/linear";
import { runtimeConfig, type RuntimeConfig } from "../runtime/config";
import { resolveKvScheme, type KvQuantOverride } from "../state/kv-scheme";

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
  /** The loaded drafter's tokens per round, and whether the scheduler adapts
   * each round's count up to it. Absent or null: no drafter. */
  readonly draft?: { readonly numDraftTokens: number; readonly adaptive?: boolean } | null;
  /** Adapters are mounted at load. Default false. */
  readonly adapters?: boolean;
  /** The most rows one forward carries. Default {@link DEFAULT_MAX_ROWS}. */
  readonly maxRows?: number;
  /** Speculative verification over affine-quantized KV where a graph's
   * verifier qualifies it. Default true. */
  readonly speculativeAffineKv?: boolean;
  /** The groups of a mixed forward share one MLP pass. Default true. */
  readonly packedMixedMlp?: boolean;
}

/** Resolve the record for `config` on this machine. Reads the GPU architecture,
 * asks the Neural Engine bridge whether it compiles programs (the first call in
 * a process compiles one small program) and reads the prefill chunk override
 * from the current runtime configuration. The result is frozen. */
export function resolveComposition(config: ModelConfig, request: CompositionRequest = {}): Composition {
  const maxRows = request.maxRows ?? DEFAULT_MAX_ROWS;
  if (!Number.isSafeInteger(maxRows) || maxRows < 1) throw new Error("composition maxRows must be a positive integer");
  const draft = request.draft;
  if (draft && (!Number.isSafeInteger(draft.numDraftTokens) || draft.numDraftTokens < 1))
    throw new Error("composition draft tokens must be a positive integer");
  const runtime = runtimeConfig();
  return Object.freeze({
    device: deviceArchitecture(),
    aneBridge: aneAvailable(),
    kv: resolveKvScheme({ ...request.kv, config: config.kvQuant }).storage,
    draftDepth: !draft ? 0 : draft.adaptive ? Object.freeze({ adaptive: true as const, max: draft.numDraftTokens }) : draft.numDraftTokens,
    prefillChunkTokens: resolveChunkTokens(runtime),
    adapters: request.adapters ?? false,
    maxRows,
    speculativeAffineKv: request.speculativeAffineKv ?? true,
    packedMixedMlp: request.packedMixedMlp ?? true,
  });
}
