import { expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createCacheServices, type CacheServiceDependencies } from "../../src/engine/cache-services";
import { createAppEngine } from "../../src/engine";
import type { LoadedModelContext } from "../../src/engine/model-host";
import type { ModelBinding } from "../../src/engine/model-binding";
import { KVCache, createKvCacheService, legacyCacheCodecs, type CacheServiceGraph, type KvCacheService,
  type KvCacheServiceOptions } from "@mlx-bun/inference/state";
import { modelIdentity } from "@mlx-bun/hub/registry";
import { MlxArray } from "@mlx-bun/mlx/array";
import { disposeResources } from "@mlx-bun/inference/runtime/resources";
import { createRuntimeConfig } from "@mlx-bun/inference/runtime/config";
import type { KvStorageScheme } from "@mlx-bun/inference/contracts/portable";

/** `setup`'s loaded model, composed with `kv` (bf16 unless named). The app hands
 * the library's cache service the options it resolved from the serve flags; a
 * fake service records them and its lifecycle. */
function setup(values: Record<string, string> = {}, kv: KvStorageScheme = { kind: "bf16" }) {
  const events: string[] = [];
  let interval = 0, demote: (() => void) | undefined, active = 0;
  let resolved: KvCacheServiceOptions | undefined, graph: CacheServiceGraph | undefined;
  const stateCodecs = { id: "replacement-state" };
  const context = { modelId: "test", model: { config: { modelDir: "/unused" }, weightsBytes: 20 },
    kvConfig: [{ layerIdx: 0, bits: 4, groupSize: 64 }], stateCodecs, composition: { kv },
    profile: { artifact: "artifact", profile: { execution: "implementation" } },
    adapters: { cacheNamespace: (names: string[]) => names.join(":") }, dispose() { events.push("model dispose"); },
  } as unknown as LoadedModelContext;
  const boundGraph = { makeCache: () => [], config: context.model.config } as unknown as CacheServiceGraph;
  const binding = { stateCompatibility: "binding-v1", gateway: { runtime: createRuntimeConfig(values), graph: boundGraph },
  } as unknown as ModelBinding;
  const service = { async close() { events.push("cache close"); return { durable: true }; } } as unknown as KvCacheService;
  const deps = {
    defaultStateCodecs: { id: "default" },
    createService(created: CacheServiceGraph, options: KvCacheServiceOptions) {
      graph = created; resolved = options; events.push("service"); return service;
    },
    costSizeRetention: () => ({ name: "cost-size" }), configFingerprint: () => "config", modelIdentity: () => "test-model",
    activeMemory: () => active, maxWorkingSet: () => 100,
    scheduleDemotion(run: () => void, ms: number) { demote = run; interval = ms; return () => events.push("timer stop"); },
  } as unknown as CacheServiceDependencies;
  return { context, binding, boundGraph, deps, events, service,
    setActive(n: number) { active = n; }, get options() { return resolved!; }, get graph() { return graph; },
    schedule(run: () => void, ms: number) { return resolved!.saved!.schedule!(run, ms); },
    get interval() { return interval; }, demote() { demote?.(); } };
}

async function withTokenizer(run: (directory: string) => Promise<void>) {
  const directory = mkdtempSync(join(tmpdir(), "mlx-app-cache-"));
  try { writeFileSync(join(directory, "tokenizer.json"), "{}"); await run(directory); }
  finally { rmSync(directory, { recursive: true, force: true }); }
}

test("default cache is 8 GB RAM, plain KV, and no SSD even when model quantization config exists", async () => {
  const f = setup(); const caches = await createCacheServices(f.context, f.binding, {}, f.deps);
  expect(caches.cache).toBe(f.service);
  // The service is built for the bound graph, with the loaded backend's codecs and the binding's runtime.
  expect(f.graph).toBe(f.boundGraph);
  expect(f.options).toMatchObject({ promptCacheBytes: 8e9, codecs: f.context.stateCodecs, runtime: f.binding.gateway.runtime });
  expect(f.options.saved).toBeUndefined(); expect(f.options.kvBudgetBytes).toBeUndefined(); expect(f.options.retention).toBeUndefined();
  expect(caches.resolvedKvScheme.kind).toBe("bf16");
  expect(caches.adapterNamespace(["a", "b"])).toBe("a:b");
  expect(f.options.identity).toEqual({ artifact: "artifact", implementation: "implementation",
    stateAbi: "legacy-cache-array-v1", codecs: "replacement-state" });
  // Optional RAM state yields above 85% of the working set, weights and held bytes counted.
  expect(f.options.pressure!(0)).toBe(false); f.setActive(86); expect(f.options.pressure!(0)).toBe(true);
  f.setActive(0); expect(f.options.pressure!(66)).toBe(true);
});

test("explicit RAM disabling retains its meaning; the KV scheme is the loaded composition's; --kv-budget reaches the service", async () => {
  const f = setup({}, { kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" });
  const caches = await createCacheServices(f.context, f.binding, { promptCacheBytes: 0, kvBudgetBytes: 3e9 }, f.deps);
  expect(f.options.promptCacheBytes).toBe(0); expect(f.options.kvBudgetBytes).toBe(3e9);
  expect(caches.resolvedKvScheme.options.kvBits).toBe(4);
  expect(caches.resolvedKvScheme.cacheKey).toBe("kv4");
  const layered = setup({}, { kind: "affine-layers", layers: [{ layer: 0, bits: 4, groupSize: 64 }], kernels: "tiled-causal" });
  const second = await createCacheServices(layered.context, layered.binding, {}, layered.deps);
  expect(second.resolvedKvScheme.options.kvConfig).toEqual(f.context.kvConfig!);
});

test("SSD defaults bind identity, codecs, storage, checkpoints, idle demotion and the root's budget", async () => {
  await withTokenizer(async directory => {
    const f = setup({ MLX_BUN_CACHE_RETENTION: "cost-size" }); f.context.model.config.modelDir = directory;
    const caches = await createCacheServices(f.context, f.binding, { ssdCacheDir: directory, generationCheckpointTokens: 32 }, f.deps);
    const saved = f.options.saved!;
    expect(saved).toMatchObject({ dir: directory, verify: undefined,
      identity: { modelId: "test", tokenizerHash: Bun.hash(Buffer.from("{}")).toString(16),
        configFingerprint: `config-${caches.resolvedKvScheme.cacheKey}-${Bun.hash("binding-v1").toString(16)}-${Bun.hash("test-model").toString(16)}` },
      storage: { layout: "whole", segmented: true }, writeBehind: true, prefetch: true, demoteIdleMs: 300_000,
      checkpointEveryTokens: 32, checkpointQueueBytes: 2 * 1024 ** 3 });
    // Without a shared budget the app makes one over the root, unlimited unless --ssd-cache-max says otherwise.
    expect(saved.budget.maxBytes).toBe(Infinity);
    expect(f.options.retention!.name).toBe("cost-size");
    // Idle demotion runs on the app's timer.
    let demoted = 0; f.schedule(() => { demoted++; }, 75_000); expect(f.interval).toBe(75_000);
    f.demote(); expect(demoted).toBe(1);
  });
});

test("every store shares the process's saved-state budget", async () => {
  await withTokenizer(async directory => {
    const f = setup(); f.context.model.config.modelDir = directory;
    const ssdBudget = { root: directory, maxBytes: 1000, attach: () => ({ limit: () => 700, detach() {} }),
      reclaim: () => ({ removedFiles: 0, removedBytes: 0 }), usage: () => ({ bytes: 0, idleBytes: 0, maxBytes: 1000 }) };
    await createCacheServices(f.context, f.binding, { ssdCacheDir: directory, ssdBudget }, f.deps);
    expect(f.options.saved!.budget).toBe(ssdBudget);
    const own = setup(); own.context.model.config.modelDir = directory;
    await createCacheServices(own.context, own.binding, { ssdCacheDir: directory, ssdCacheMaxBytes: 4096 }, own.deps);
    expect(own.options.saved!.budget.maxBytes).toBe(4096);
  });
});

test("optional SSD policy overrides reach the service unchanged", async () => {
  await withTokenizer(async directory => {
    const f = setup({ MLX_BUN_SSD_LAYOUT: "blocks", MLX_BUN_SSD_SEGMENTED: "0", MLX_BUN_SSD_PREFETCH: "0",
      MLX_BUN_SSD_WRITEBEHIND: "0", MLX_BUN_SSD_SPILL_QUEUE_GB: "0.5" }); f.context.model.config.modelDir = directory;
    await createCacheServices(f.context, f.binding, { ssdCacheDir: directory, ssdDemoteIdleSec: 0,
      allocatorLimitBytes: 50, generationCheckpointTokens: 4, ssdCacheMaxBytes: 1000, ssdCacheVerify: true }, f.deps);
    expect(f.options.saved).toMatchObject({ verify: true, storage: { layout: "blocks", segmented: false },
      writeBehind: false, prefetch: false, demoteIdleMs: 0, checkpointEveryTokens: 4, checkpointQueueBytes: 0.5 * 1024 ** 3 });
    expect(f.options.saved!.budget.maxBytes).toBe(1000);
    // An explicit allocator cap bounds optional RAM state.
    expect(f.options.pressure!(23)).toBe(true); expect(f.options.pressure!(22)).toBe(false);
  });
});

test("invalid checkpoint and SSD configurations fail before the service is built", async () => {
  const f = setup();
  for (const options of [{ generationCheckpointTokens: 0 }, { generationCheckpointTokens: 1 },
    { ssdCacheDir: "unused", promptCacheBytes: 0 }])
    await expect(createCacheServices(f.context, f.binding, options, f.deps)).rejects.toThrow();
  await expect(createCacheServices(f.context, { ...f.binding, stateCompatibility: "" }, { ssdCacheDir: "unused" }, f.deps)).rejects.toThrow("identity");
  expect(f.events).toEqual([]);
});

test("shutdown attempts the cache service's close and model disposal when the close fails", async () => {
  const f = setup();
  const failure = new Error("checkpoint disk failure");
  const { cache } = await createCacheServices(f.context, f.binding, {}, f.deps);
  cache.close = async () => { f.events.push("checkpoint failure"); throw failure; };
  const engine = await createAppEngine(f.context, { capacity: 1, binding: f.binding,
    beforeModelDispose: async () => { await cache.close(); } });
  await expect(engine.close()).rejects.toBe(failure); await expect(engine.close()).rejects.toBe(failure);
  expect(f.events.slice(-2)).toEqual(["checkpoint failure", "model dispose"]);
});

test("engine construction failure preserves the original error and attempts hook and model cleanup once", async () => {
  const f = setup(); const hookFailure = new Error("hook failure");
  try {
    await createAppEngine(f.context, { capacity: 0, binding: f.binding,
      beforeModelDispose() { f.events.push("cache close"); throw hookFailure; } });
    throw new Error("expected failure");
  } catch (error) {
    expect(error).toBeInstanceOf(AggregateError);
    const errors = (error as AggregateError).errors;
    expect(errors[0].message).toContain("positive integer"); expect(errors[1]).toBe(hookFailure);
  }
  expect(f.events).toEqual(["cache close", "model dispose"]);
});

test("a TurboQuant scheme the codec cannot encode for this model is refused before any cache is composed, as main's startup did", async () => {
  const turbo: KvStorageScheme = { kind: "turbo", kBits: 8, vBits: 3 };
  const withHeadDim = (globalHeadDim: number, kv: KvStorageScheme) => {
    const f = setup({}, kv); (f.context.model.config as { text?: unknown }).text = { globalHeadDim }; return f;
  };
  for (const globalHeadDim of [96, 72]) {
    const f = withHeadDim(globalHeadDim, turbo);
    await expect(createCacheServices(f.context, f.binding, {}, f.deps)).rejects.toThrow(
      `--kv-quant turbo: this model's full-attention head_dim is ${globalHeadDim}; TurboQuant supports {64,128,256,512}`);
    expect(f.events).toEqual([]);   // no cache service was built
  }
  for (const globalHeadDim of [128, 256]) {
    const f = withHeadDim(globalHeadDim, turbo);
    const caches = await createCacheServices(f.context, f.binding, {}, f.deps);
    expect(caches.resolvedKvScheme.kind).toBe("turbo");
  }
  // Other schemes do not depend on the TurboQuant codec's dimensions.
  const f = withHeadDim(96, { kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" });
  const caches = await createCacheServices(f.context, f.binding, {}, f.deps);
  expect(caches.resolvedKvScheme.options.kvBits).toBe(4);
});

test("model directories have separate saved-prefix stores; restarts reuse them without reading weights", async () => {
  const root = mkdtempSync(join(tmpdir(), "mlx-app-ssd-id-")), ssd = join(root, "ssd");
  const model = (weights: string) => {
    const dir = mkdtempSync(join(root, "model-")); writeFileSync(join(dir, "tokenizer.json"), "{}");
    writeFileSync(join(dir, "config.json"), "{}"); writeFileSync(join(dir, "model.safetensors"), weights); return dir;
  };
  // The real library service over a one-layer graph, with the app's identity for each model directory.
  const open = async (modelDir: string) => {
    const f = setup(); f.context.model.config.modelDir = modelDir;
    f.context.stateCodecs = legacyCacheCodecs; f.deps.modelIdentity = modelIdentity;
    (f.boundGraph as { makeCache(): unknown[] }).makeCache = () => [new KVCache()];
    f.deps.createService = createKvCacheService;
    return (await createCacheServices(f.context, f.binding, { ssdCacheDir: ssd, promptCacheBytes: 1 << 20 }, f.deps)).cache;
  };
  const state = () => {
    const cache = new KVCache();
    cache.restoreState(MlxArray.fromFloat32(new Float32Array([1, 2, 3]), [1, 1, 3, 1]),
      MlxArray.fromFloat32(new Float32Array([4, 5, 6]), [1, 1, 3, 1]), 3);
    return cache;
  };
  /** Publish `tokens`' state and make it durable. */
  const save = async (cache: KvCacheService, tokens: number[]) => {
    const request = { promptIds: tokens, maxTokens: 1 };
    cache.publish(request, "", tokens, { caches: [state()] });
    expect((await cache.flush()).durable).toBe(true);
  };
  /** Tokens of `prompt` the service reuses (0: a miss). */
  const reused = (cache: KvCacheService, prompt: number[]) => {
    const request = { promptIds: prompt, maxTokens: 1 };
    const acquired = cache.acquire(request, "");
    try { return acquired.reusedTokens; } finally { disposeResources(acquired.caches); acquired.retain?.(); cache.release(request, false); }
  };
  try {
    const four = model("q4 weights"), eight = model("q8 weights");
    const first = await open(four); await save(first, [1, 2, 3]); await first.close();
    // Different registered directories keep separate state.
    const second = await open(eight);
    expect(second.report().saved!.entries).toBe(0); expect(reused(second, [1, 2, 3, 9])).toBe(0);
    await save(second, [7, 8, 9]); await second.close();
    // Each restart still sees exactly its own entry; nothing was unlinked.
    const again = await open(four);
    expect(again.report().saved!.entries).toBe(1);
    expect(reused(again, [1, 2, 3, 9])).toBe(3); expect(reused(again, [7, 8, 9, 0])).toBe(0);
    await again.close();
    const eightAgain = await open(eight);
    expect(eightAgain.report().saved!.entries).toBe(1); expect(reused(eightAgain, [7, 8, 9, 0])).toBe(3);
    await eightAgain.close();
    // Replacing weights in place leaves the identity alone; the caller can clear cached state.
    writeFileSync(join(four, "model.safetensors"), "q4 weights, retrained");
    const retrained = await open(four);
    expect(retrained.report().saved!.entries).toBe(1); expect(reused(retrained, [1, 2, 3, 9])).toBe(3);
    await retrained.close();
  } finally { rmSync(root, { recursive: true, force: true }); }
});
