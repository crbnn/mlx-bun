// The composition's cache service: one per loaded model, constructed with its
// budget. The scheduler acquires each request's state (reused tokens, 0 for a
// miss) and releases the request; what the service holds is in its own report.
import { afterEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MlxArray } from "@mlx-bun/mlx/array";
import type { BatchableCache, Cache } from "../../src/contracts/mlx/cache";
import type { ModelConfig } from "../../src/artifacts/config";
import { createKvCacheService, KvCacheService, type CacheServiceGraph, type SavedStateBudget } from "../../src/state/cache-service";
import { KVCache } from "../../src/state/kv";
import { PromptCache } from "../../src/state/prefix-cache";
import { pagedPrefixNamespace } from "../../src/state/request-policy";
import { batchRowKvBytes } from "../../src/state/kv-budget";
import { kvSchemeOf } from "../../src/state/kv-scheme";
import { disposeResources } from "../../src/runtime/resources";
import { testComposition } from "../models/test-composition";

const config = { text: { numHiddenLayers: 2, numAttentionHeads: 8, numKeyValueHeads: 2,
  headDim: 64, globalHeadDim: 64, numGlobalKeyValueHeads: 2, attentionKEqV: false,
  layerTypes: ["full_attention", "sliding_attention"], slidingWindow: 1024,
  maxPositionEmbeddings: 32768, enableMoeBlock: false } } as unknown as ModelConfig;
/** One plain full-attention layer. */
const graph = (fields: Partial<CacheServiceGraph> = {}): CacheServiceGraph => ({ makeCache: () => [new KVCache()], config, ...fields });
/** A layer holding `tokens` positions. */
function state(tokens: number): KVCache {
  const cache = new KVCache(), values = Float32Array.from({ length: tokens }, (_, i) => i + 1);
  cache.restoreState(MlxArray.fromFloat32(values, [1, 1, tokens, 1]), MlxArray.fromFloat32(values, [1, 1, tokens, 1]), tokens);
  return cache;
}
const request = (promptIds: number[], maxTokens = 8) => ({ promptIds, maxTokens });
/** The tokens `cache` reuses for `prompt` (0: a miss); the state is released. */
function reused(cache: KvCacheService, prompt: number[], namespace = ""): number {
  const row = request(prompt), acquired = cache.acquire(row, namespace);
  try { return acquired.reusedTokens; } finally { disposeResources(acquired.caches); acquired.retain?.(); cache.release(row, false); }
}
const roots: string[] = [];
const root = () => { const dir = mkdtempSync(join(tmpdir(), "mlx-cache-service-")); roots.push(dir); return dir; };
afterEach(() => { for (const dir of roots.splice(0)) rmSync(dir, { recursive: true, force: true }); });

/** A budget over one saved-state root that lends `lent` and records what the service asks of it. */
function budget(lent = Infinity) {
  const calls: string[] = [];
  const value: SavedStateBudget & { calls: string[]; dir?: string } = { calls, maxBytes: Infinity,
    attach(member) { value.dir = member.dir; calls.push("attach"); return { limit: () => lent, detach: () => { calls.push("detach"); } }; },
    reclaim() { calls.push("reclaim"); return {}; } };
  return value;
}
const identity = { configFingerprint: "fixture-v1", modelId: "fixture", tokenizerHash: "fixture" };

test("a service with no parts reuses and keeps nothing, and gives every request fresh state", () => {
  const cache = new KvCacheService(graph());
  const row = request([1, 2, 3]);
  expect(cache.keeps(row)).toBe(false);
  const acquired = cache.acquire(row, "");
  try {
    expect(acquired.reusedTokens).toBe(0);
    expect(acquired.caches.map(layer => layer.signature())).toEqual(["kv:plain"]);
    expect(() => cache.publish(row, "", [1, 2, 3], { caches: acquired.caches })).toThrow("keeps no state");
  } finally { disposeResources(acquired.caches); }
  const owned = [state(2)];
  expect(() => cache.checkpoint(row, { caches: owned, cacheTokens: [1, 2], generatedTokens: 0, pendingToken: 3 }))
    .toThrow("persistence coordinator");
  cache.boundary(); cache.release(row, true); cache.release(row, true);
  expect(cache.report()).toEqual({ prefixes: { entries: 0, bytes: 0, maxBytes: 0, hits: 0, misses: 0, sessionHits: 0, sessionMisses: 0,
    prefixScans: 0, objectHits: 0, objectMisses: 0, objectRestores: 0, demotions: 0 }, saved: null,
    pending: { pendingSnapshots: 0, pendingSpills: 0, pendingSpillBytes: 0, droppedSpills: 0, failedSpills: 0 },
    kv: { projectedBytes: 0, budgetBytes: null } });
});

test("admission holds each request's projected KV under --kv-budget until it is released", () => {
  const need = batchRowKvBytes(config, 3, 8, kvSchemeOf({ kind: "bf16" }));
  const cache = new KvCacheService(graph(), {}, { kvBudgetBytes: need * 1.5 });
  const [a, b, c] = [request([1, 2, 3]), request([4, 5, 6]), request([7, 8, 9])];
  expect(cache.admit(a)).toBe(true);
  // One that fits alone but not beside the admitted rows waits; nothing was held for it.
  expect(cache.fits(b)).toBe(false); expect(cache.admit(b)).toBe(false);
  expect(cache.report().kv).toEqual({ projectedBytes: need, budgetBytes: need * 1.5 });
  cache.release(a, true);
  expect(cache.fits(b)).toBe(true); expect(cache.admit(b)).toBe(true);
  // A request over the budget alone waits while others hold state, and is refused once alone.
  const huge = request([1], 1 << 20);
  expect(cache.admit(huge)).toBe(false);
  cache.release(b, false);
  expect(() => cache.admit(huge)).toThrow(/^kv budget: request needs ~[\d.]+ GB KV \(prompt 1 \+ max_tokens 1048576\), over --kv-budget/);
  // Without a budget every request is admitted; the report still projects what they hold.
  const open = new KvCacheService(graph());
  expect(open.admit(c)).toBe(true); expect(open.fits(huge)).toBe(true);
  expect(open.report().kv).toEqual({ projectedBytes: need, budgetBytes: null });
});

test("storage that projects its own rows bills by them and refuses a request past its positions", () => {
  // A layout such as GLM-5.2's compressed state: it projects its own bytes and holds at most 16 positions.
  const layout = { makeEmptyBatch() {}, mergeRows() {}, extractRow() {}, filterRows() {}, rowOffsets: [], leftPad: [],
    projectedBytes: (tokens: number) => tokens * 10, maxTokens: 16, dispose() {} } as unknown as BatchableCache;
  const cache = new KvCacheService(graph({ makeCache: () => [layout as unknown as Cache] }), {}, { kvBudgetBytes: 200 });
  expect(cache.admit(request([1, 2], 8))).toBe(true);
  expect(cache.report().kv.projectedBytes).toBe(100);
  expect(() => cache.admit(request([1, 2, 3], 14))).toThrow("context limit: prompt 3 + max_tokens 14 exceeds 16");
});

test("RAM prefixes: a published prefix is reused as the graph's composed caches, and the report counts it", async () => {
  const cache = createKvCacheService(graph(), { promptCacheBytes: 1 << 20 });
  const row = request([1, 2, 3]);
  expect(cache.keeps(row)).toBe(true);
  cache.publish(row, "adapter", [1, 2, 3], { caches: [state(3)] });
  expect(reused(cache, [1, 2, 3, 9])).toBe(0);              // another namespace's prefix is never reused
  expect(reused(cache, [1, 2, 3, 9], "adapter")).toBe(3);
  expect(cache.peekPrefixLen([1, 2, 3, 9], "adapter")).toBe(3);
  // A prepared prefix with no identity takes fresh state and keeps none.
  expect(cache.keeps({ ...row, reusesPrefix: false })).toBe(false);
  const report = cache.report();
  expect(report.prefixes).toMatchObject({ entries: 1, maxBytes: 1 << 20, hits: 1, misses: 1 });
  expect(report.prefixes.bytes).toBeGreaterThan(0); expect(report.saved).toBeNull();
  expect((await cache.flush()).durable).toBe(true);
  const closed = cache.close();
  expect(cache.close()).toBe(closed);
  expect((await closed).durable).toBe(true);
  expect(cache.report().prefixes.entries).toBe(0);
});

test("paging is the composition's: fresh state pages, and prefixes are kept in the paged namespace", () => {
  const composition = testComposition(undefined, { pagedKv: { blockSize: 16, direct: false } });
  const prefixes = new PromptCache(1 << 20);
  const cache = new KvCacheService(graph({ composition }), { prefixes });
  const row = request([1, 2, 3]), acquired = cache.acquire(row, "upper");
  try { expect(acquired.caches.map(layer => layer.signature())).toEqual(["kv:paged"]); }
  finally { disposeResources(acquired.caches); cache.release(row, false); }
  cache.publish(row, "upper", [1, 2, 3], { caches: [state(3)] });
  expect(prefixes.findExact([1, 2, 3], pagedPrefixNamespace(composition.kv, { blockSize: 16, direct: false }, "upper"))).not.toBeNull();
  expect(prefixes.findExact([1, 2, 3], "upper")).toBeNull();
  prefixes.clear();
});

test("saved state: the store joins the root's budget, persists what RAM keeps, and a fresh service restores it", async () => {
  const dir = root(), first = budget(1 << 20);
  const cache = createKvCacheService(graph(), { promptCacheBytes: 1 << 20,
    saved: { dir, identity, budget: first, storage: { layout: "whole", segmented: true } } });
  expect(first.calls).toEqual(["attach", "reclaim"]);
  expect(first.dir).toBe(join(dir, identity.configFingerprint));
  // The store enforces the room the budget lends it.
  expect(cache.report().saved).toMatchObject({ entries: 0, maxBytes: 1 << 20 });
  cache.publish(request([1, 2, 3]), "", [1, 2, 3], { caches: [state(3)] });
  const flushed = await cache.flush();
  expect(flushed).toMatchObject({ durable: true, pendingSnapshots: 0, pendingSpills: 0 });
  expect(cache.report().saved).toMatchObject({ entries: 1, spills: 1, longestDurablePrefixTokens: 3 });
  expect([...cache.savedPrefixes()]).toEqual([[1, 2, 3]]);
  await cache.close();
  // Closed, its directory is idle: it counts against the root and may be evicted.
  expect(first.calls).toEqual(["attach", "reclaim", "detach", "reclaim"]);
  expect(existsSync(join(dir, identity.configFingerprint))).toBe(true);
  const restarted = createKvCacheService(graph(), { promptCacheBytes: 1 << 20, saved: { dir, identity, budget: budget() } });
  try {
    expect(restarted.report().saved!.entries).toBe(1);
    expect(reused(restarted, [1, 2, 3, 4])).toBe(3);
    expect(restarted.report().saved!.restores).toBe(1);
  } finally { await restarted.close(); }
});

test("a saved prefix is read ahead of admission unless prefetching is off", async () => {
  const dir = root();
  const seed = createKvCacheService(graph(), { promptCacheBytes: 1 << 20, saved: { dir, identity, budget: budget() } });
  seed.publish(request([1, 2, 3]), "", [1, 2, 3], { caches: [state(3)] });
  await seed.close();
  for (const prefetch of [true, false]) {
    const cache = createKvCacheService(graph(), { promptCacheBytes: 1 << 20, saved: { dir, identity, budget: budget(), prefetch } });
    try {
      const release = await cache.prefetch([1, 2, 3, 4], "");
      expect({ prefetch, restores: cache.report().saved!.restores }).toEqual({ prefetch, restores: prefetch ? 1 : 0 });
      release();
      expect(reused(cache, [1, 2, 3, 4])).toBe(3);
    } finally { await cache.close(); }
  }
});

test("idle demotion runs on the schedule the service was given and stops before the drain", async () => {
  const runs: number[] = [], stops: string[] = [];
  let tick: (() => void) | undefined;
  const cache = createKvCacheService(graph(), { promptCacheBytes: 1 << 20, saved: { dir: root(), identity, budget: budget(),
    demoteIdleMs: 400_000, schedule(run, interval) { runs.push(interval); tick = run; return () => stops.push("stop"); } } });
  expect(runs).toEqual([100_000]);
  tick!();
  cache.stopIdleDemotion(); cache.stopIdleDemotion();
  expect(stops).toEqual(["stop"]);
  await cache.close();
  expect(stops).toEqual(["stop"]);
  // No saved state, no demotion; and an interval of 0 never demotes.
  const none: number[] = [];
  createKvCacheService(graph(), { promptCacheBytes: 1 << 20, saved: { dir: root(), identity, budget: budget(),
    demoteIdleMs: 0, schedule(_run, interval) { none.push(interval); return () => {}; } } });
  expect(none).toEqual([]);
});

test("memory pressure releases optional RAM state, judged on what the prefix store holds", () => {
  const held: number[] = [];
  let over = false;
  const cache = createKvCacheService(graph(), { promptCacheBytes: 1 << 20, pressure: bytes => { held.push(bytes); return over; } });
  cache.publish(request([1, 2, 3]), "", [1, 2, 3], { caches: [state(3)] });
  cache.boundary();
  expect(cache.report().prefixes.entries).toBe(1);
  expect(held.at(-1)).toBe(cache.report().prefixes.bytes);
  over = true; cache.boundary();
  expect(cache.report().prefixes).toMatchObject({ entries: 0, demotions: 1 });
});
