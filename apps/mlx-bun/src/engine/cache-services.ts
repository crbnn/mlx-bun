import { readFileSync } from "node:fs";
import { modelIdentity } from "@mlx-bun/hub/registry";
import { kvSchemeOf } from "@mlx-bun/inference/state/kv-scheme";
import type { CacheCodecProvider, KvCacheService, KvCacheServiceOptions, CacheServiceGraph } from "@mlx-bun/inference/state";
import type { LoadedModelContext } from "./model-host";
import type { ModelBinding } from "./model-binding";
import { createKvBudget, type KvBudget } from "./kv-budget";

/** Cache policy: the serve flags the composition's cache service is built
 * with. The KV scheme and paging are the loaded composition's
 * (`LoadedModelContext.composition`), resolved with the model; nothing here
 * resolves them again. */
export interface CacheServiceOptions {
  promptCacheBytes?: number;
  /** Aggregate projected KV bytes of admitted requests (`--kv-budget`); unset means no budget. */
  kvBudgetBytes?: number;
  /** The saved-state root; each model's store lives in its own directory under it. */
  ssdCacheDir?: string;
  /** Bytes the whole root may hold, across every model's directory; unset means unlimited. */
  ssdCacheMaxBytes?: number;
  /** One budget shared by every store in the process; without it this call makes one over `ssdCacheDir`. */
  ssdBudget?: KvBudget;
  ssdCacheVerify?: boolean;
  ssdDemoteIdleSec?: number;
  generationCheckpointTokens?: number;
  /** An explicit allocator cap also limits optional RAM cache residency. */
  allocatorLimitBytes?: number;
}

/** Construction ports let CPU tests and replacement runtimes observe and
 * replace what the app supplies to the library's cache service. */
export interface CacheServiceDependencies {
  defaultStateCodecs: CacheCodecProvider;
  createService(graph: CacheServiceGraph, options: KvCacheServiceOptions): KvCacheService;
  costSizeRetention(): NonNullable<KvCacheServiceOptions["retention"]>;
  configFingerprint(config: LoadedModelContext["model"]["config"]): string;
  modelIdentity(modelDir: string): string;
  activeMemory(): number;
  maxWorkingSet(): number;
  scheduleDemotion(run: () => void, intervalMs: number): () => void;
}

async function defaultDependencies(): Promise<CacheServiceDependencies> {
  const [state, artifacts, memory] = await Promise.all([
    import("@mlx-bun/inference/state"), import("@mlx-bun/inference/artifacts"), import("@mlx-bun/mlx/ffi"),
  ]);
  return {
    defaultStateCodecs: state.legacyCacheCodecs,
    createService: (graph, options) => state.createKvCacheService(graph, options),
    costSizeRetention: () => new state.CostSizeRetention(), configFingerprint: artifacts.configFingerprint,
    modelIdentity,
    activeMemory: memory.activeMemory, maxWorkingSet: memory.maxRecommendedWorkingSetSize,
    scheduleDemotion(run, interval) { const timer = setInterval(run, interval); timer.unref(); return () => clearInterval(timer); },
  };
}

/** The loaded model's cache service (`state/cache-service`), built from the
 * serve flags without starting execution: the scheduler, the server and the
 * model host all reach request state, prefix reuse, saved state and its budget
 * through it. The caller borrows it until execution has drained, then closes
 * it before disposing the model. SSD is opt-in; plain KV and 8 GB RAM retain
 * main's defaults. Native library imports are deferred until construction. */
export async function createCacheServices(context: LoadedModelContext, binding: ModelBinding,
  options: CacheServiceOptions = {}, supplied?: CacheServiceDependencies) {
  if (options.generationCheckpointTokens !== undefined) {
    if (!Number.isInteger(options.generationCheckpointTokens) || options.generationCheckpointTokens < 1)
      throw new Error("generationCheckpointTokens must be a positive integer");
    if (!options.ssdCacheDir) throw new Error("generation checkpoints require an SSD cache directory");
  }
  const cap = options.promptCacheBytes ?? 8e9;
  if (options.ssdCacheDir && cap <= 0) throw new Error("SSD cache requires a nonzero RAM prompt cache");
  if (options.ssdCacheDir && !binding.stateCompatibility?.length)
    throw new Error("SSD cache requires the model binding's stateCompatibility identity");
  // The composition's KV scheme: what the graph's caches compose, keys the saved
  // state and describes `/stats`. A context an engine-owned implementation built
  // composes none, and stores bf16.
  const resolvedKvScheme = kvSchemeOf(context.composition?.kv ?? { kind: "bf16" });
  const deps = supplied ?? await defaultDependencies();
  const runtime = binding.gateway.runtime;
  const stateCodecs = context.stateCodecs ?? deps.defaultStateCodecs;
  let saved: KvCacheServiceOptions["saved"];
  if (options.ssdCacheDir) {
    const tokenizer = readFileSync(`${context.model.config.modelDir}/tokenizer.json`);
    // The hub owns model identity. Snapshot revisions have distinct paths;
    // ordinary restarts reuse the same directory without reading weight bytes.
    const architecture = deps.configFingerprint(context.model.config);
    const artifact = deps.modelIdentity(context.model.config.modelDir);
    const rawQueueGb = Number(runtime.value("MLX_BUN_SSD_SPILL_QUEUE_GB"));
    saved = { dir: options.ssdCacheDir,
      identity: { modelId: context.modelId,
        configFingerprint: `${architecture}-${resolvedKvScheme.cacheKey}-${Bun.hash(binding.stateCompatibility).toString(16)}-${Bun.hash(artifact).toString(16)}`,
        tokenizerHash: Bun.hash(tokenizer).toString(16) },
      budget: options.ssdBudget ?? createKvBudget(options.ssdCacheDir, options.ssdCacheMaxBytes ?? Infinity),
      verify: options.ssdCacheVerify,
      storage: { layout: runtime.value("MLX_BUN_SSD_LAYOUT") === "blocks" ? "blocks" : "whole",
        segmented: runtime.value("MLX_BUN_SSD_SEGMENTED") !== "0" },
      writeBehind: runtime.value("MLX_BUN_SSD_WRITEBEHIND") !== "0",
      prefetch: runtime.value("MLX_BUN_SSD_PREFETCH") !== "0",
      demoteIdleMs: (options.ssdDemoteIdleSec ?? 300) * 1000, schedule: deps.scheduleDemotion,
      ...(options.generationCheckpointTokens ? { checkpointEveryTokens: options.generationCheckpointTokens,
        checkpointQueueBytes: (Number.isFinite(rawQueueGb) && rawQueueGb >= 0 ? rawQueueGb : 2) * 1024 ** 3 } : {}) };
  }
  const workingSet = Math.min(deps.maxWorkingSet(), options.allocatorLimitBytes ?? Infinity);
  const cache = deps.createService(binding.gateway.graph, {
    promptCacheBytes: cap, codecs: stateCodecs, runtime,
    ...(options.kvBudgetBytes !== undefined ? { kvBudgetBytes: options.kvBudgetBytes } : {}),
    ...(runtime.value("MLX_BUN_CACHE_RETENTION") === "cost-size" ? { retention: deps.costSizeRetention() } : {}),
    pressure: held => Math.max(deps.activeMemory(), context.model.weightsBytes + held) > workingSet * 0.85,
    identity: { artifact: context.profile.artifact, implementation: context.profile.profile.execution,
      stateAbi: "legacy-cache-array-v1", codecs: stateCodecs.id },
    ...(saved ? { saved } : {}),
  });
  return { cache, resolvedKvScheme, adapterNamespace: (adapters: string[]) => context.adapters.cacheNamespace(adapters) };
}
