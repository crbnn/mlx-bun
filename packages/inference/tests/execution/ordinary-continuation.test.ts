import { expect, test } from "bun:test";
import { createOrdinaryContinuationRequest } from "../../src/execution/continuation-request";
import { KvCacheService, type CacheServiceGraph } from "../../src/state/cache-service";
import { ContinuationPersistence, type ContinuationStore } from "../../src/state/continuation-persistence";
import type { Cache } from "../../src/contracts/mlx/cache";
import type { ResolvedExecution } from "../../src/contracts/portable/execution";

const execution = { method: "autoregressive", mechanism: "continuous",
  promptCache: true, checkpoint: true, fill: false, compiledDecode: false, grammarJump: false,
  reasons: [] } satisfies ResolvedExecution;
const graph = { makeCache: () => [], config: {} } as unknown as CacheServiceGraph;

/** A cache service whose generation checkpoints go to `store`, written as soon as they are queued. */
function service(store: ContinuationStore, identity: unknown) {
  const persistence = new ContinuationPersistence(store, { maxBytes: 1024, runStep: async step => step() });
  return { persistence, cache: new KvCacheService(graph, { continuations: { store, persistence, everyTokens: 1 } }, { identity }) };
}
const request = (options: Parameters<typeof createOrdinaryContinuationRequest>[0]["options"]) =>
  createOrdinaryContinuationRequest({ options, prompt: [1, 2], execution, interval: 1, onToken() {} });

test("a resumed generation recovers the implicit seed and keeps the namespace on save and completion", async () => {
  const captures: any[] = [], removed: string[] = [], lookups: any[] = [];
  const caches = [] as Cache[];
  const store = { findGenerationCheckpoint(...args: unknown[]) { lookups.push(args); return {}; },
    restore: () => ({ tokens: [1, 2, 3], caches, header: {
      generationCheckpoint: { originalPromptTokens: 2, generatedTokens: 1, pendingToken: 4, seed: 77 } } }),
    storeGenerationCheckpoint(...args: unknown[]) { captures.push(args); return Promise.resolve(true); },
    removeGenerationCheckpoints(key: string) { removed.push(key); } } as unknown as ContinuationStore;
  const { cache, persistence } = service(store, "fixture-v1");
  const item = request({ seed: 999, seedWasExplicit: false });
  try {
    const row = { promptIds: [1, 2], maxTokens: 4, continuation: item.continuation };
    const key = item.continuation.key("adapter:revision-a", "fixture-v1");
    const acquired = cache.acquire(row, "adapter:revision-a");
    expect(acquired.continuation).toMatchObject({ seed: 77, generatedTokens: 1, pendingToken: 4, cacheTokens: [1, 2, 3] });
    expect(acquired.reusedTokens).toBe(2);
    cache.checkpoint(row, { caches, cacheTokens: [1, 2, 3, 4], generatedTokens: 2, pendingToken: 5 });
    await persistence.flush();
    expect(captures[0][2]).toMatchObject({ key, seed: 77, seedWasExplicit: false, cacheNs: "adapter:revision-a" });
    expect(lookups[0]).toEqual([[1, 2], key, "adapter:revision-a"]);
    expect(item.continuation.key("adapter:revision-b", "fixture-v1")).not.toBe(key);
    expect(item.continuation.key("adapter:revision-a", "fixture-v2")).not.toBe(key);
    expect(removed).toEqual([]);
    cache.release(row, true); await persistence.flush(); expect(removed).toEqual([key]);
  } finally { item.dispose(); }
});

test("inconsistent restored coverage releases owned state before rejecting", () => {
  let disposed = 0;
  const store = { findGenerationCheckpoint() { return {}; }, restore: () => ({ tokens: [1, 2, 3],
    caches: [{ dispose() { disposed++; } }], header: { generationCheckpoint: {
      originalPromptTokens: 2, generatedTokens: 2, pendingToken: 4, seed: 0 } } }) } as unknown as ContinuationStore;
  const { cache } = service(store, "fixture");
  const item = request({});
  try {
    expect(() => cache.acquire({ promptIds: [1, 2], maxTokens: 4, continuation: item.continuation }, ""))
      .toThrow("inconsistent continuation metadata");
    expect(disposed).toBe(1);
  } finally { item.dispose(); }
});

test("restoring an implicit seed recovers the original sampler", async () => {
  const { makeStepSampler } = await import("../../src/sampling/index");
  const { MlxArray } = await import("@mlx-bun/mlx/array");
  const first = { temperature: 0.7, seed: 123, seedWasExplicit: false };
  const retry = { temperature: 0.7, seed: 456, seedWasExplicit: false };
  const stored: any[] = [];
  const store = { findGenerationCheckpoint: () => ({}),
    restore: () => ({ caches: [], tokens: [1, 2, 3], header: { generationCheckpoint: {
      originalPromptTokens: 2, generatedTokens: 1, pendingToken: 4, seed: first.seed } } }),
    async storeGenerationCheckpoint(...args: unknown[]) { stored.push(args); return true; },
    removeGenerationCheckpoints() {} } as unknown as ContinuationStore;
  const { cache, persistence } = service(store, "fixture");
  const original = request(first), item = request(retry);
  try { expect(original.continuation.key("", "fixture")).toBe(item.continuation.key("", "fixture")); }
  finally { original.dispose(); }
  const row = { promptIds: [1, 2], maxTokens: 4, continuation: item.continuation };
  const oracle = makeStepSampler(first, { tokenRepresentation: "device", grammarWait: "external", historyUpdate: "after-sample", initialHistory: [1, 2, 3, 4] });
  try {
    const state = cache.acquire(row, "").continuation!;
    item.continuation.resumeSampling(state);
    using logits = MlxArray.fromFloat32(new Float32Array([0, 1, 2, 3, 4]), [1, 5]);
    using actual = item.sample(logits, 2);
    using expected = oracle.sample(logits, 2).token;
    expect(actual.toIntTokens()).toEqual(expected.toIntTokens());
    cache.checkpoint(row, { caches: [], cacheTokens: [1, 2, 3, 4], generatedTokens: 2, pendingToken: 0 });
    await persistence.flush(); expect(stored[0][2].seed).toBe(first.seed);
    expect(stored[0][2].seedWasExplicit).toBe(false);
  } finally { item.dispose(); oracle.dispose(); }
  for (const [options, greedy] of [[{ temperature: 0 }, true], [{ temperature: 0, repetitionPenalty: 1.1 }, false]] as const) {
    const plain = request(options);
    try { expect(plain.plainGreedy).toBe(greedy); } finally { plain.dispose(); }
  }
});
