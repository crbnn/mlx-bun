import type { TestGraph } from "../models/test-composition";
import type { Composition } from "../../src/contracts/portable/composition";
import { expect, test } from "bun:test";
import { bindMlxGateway } from "../../src/execution/gateway-binding";
import { Gemma4Model } from "../../src/models/gemma4/model";
import { UniversalDenseModel } from "../../src/models/universal/dense";
import { FillSession } from "../../src/generation/fill";
import type { GenerateOptions } from "../../src/generation";
import type { ResolvedExecution } from "../../src/contracts/portable/execution";
import { Qwen35Model } from "../../src/models/qwen/qwen3_5";
import { KVCache } from "../../src/state/kv";
import { KvCacheService } from "../../src/state/cache-service";
import { ContinuationPersistence, type ContinuationStore } from "../../src/state/continuation-persistence";
import { pagedPrefixNamespace } from "../../src/state/request-policy";
import { kvOf, testComposition, type KvOptions } from "../models/test-composition";

const shape = { hasVision: false, hasAdapters: false, hasRepetitionPenalty: false, userSeed: false,
  hasLogitsExtras: false, hasGrammar: false, wantsLogprobs: false, hasDraft: false };
/** The cache service keeps generation checkpoints, or keeps none. */
const schedule = { checkpoints: true }, unscheduled = { checkpoints: false };

for (const [name, prototype] of [["gemma4", Gemma4Model.prototype], ["qwen3_5", Qwen35Model.prototype]] as const) {
test(`ordinary shared continuation binds ${name} through its loaded backend`, () => {
  const make = (kv: KvOptions = {}) => Object.assign(Object.create(prototype), {
    config: { modelType: name, text: { enableMoeBlock: false }, eosTokenIds: [] },
    makeCache: () => [new KVCache()], composition: testComposition(kvOf(kv)),
    loraState: { active: [] },
  }) as TestGraph<Gemma4Model>;
  const model = make();
  const binding = bindMlxGateway(model);
  expect(binding.plan(shape, {}, unscheduled).checkpoint).toBe(false);
  expect(binding.plan(shape, {}, schedule)).toMatchObject({ mechanism: "continuous", checkpoint: true });
  const adapterShape = { ...shape, hasAdapters: true };
  expect(binding.plan(adapterShape, { adapters: ["upper"] }, schedule)).toMatchObject({
    method: "autoregressive", mechanism: "continuous", checkpoint: true,
  });
  expect(binding.plan(adapterShape, { adapters: ["upper"] }, unscheduled).checkpoint).toBe(false);
  for (const excluded of [{ hasDraft: true }, { hasGrammar: true },
    { wantsLogprobs: true }]) {
    const request = { ...shape, ...excluded };
    const baseline = binding.plan(request, {}, unscheduled);
    const plan = binding.plan(request, {}, schedule);
    expect(plan.checkpoint).toBe(false);
    expect(plan.method).toBe(baseline.method);
    expect(plan.mechanism).toBe(baseline.mechanism);
    expect(plan).toEqual(baseline);
  }
  for (const options of [{ kvBits: 4 }, { kvBits: 8, quantizedKvStart: 14 },
    { turboQuant: { kBits: 8, vBits: 3 }, quantizedKvStart: 14 },
    { kvConfig: [{ layerIdx: 0, bits: 4, groupSize: 64 }], quantizedKvStart: 14 }] as KvOptions[]) {
    const composed = bindMlxGateway(make(options));
    expect(composed.plan(shape, {}, unscheduled).checkpoint).toBe(false);
    expect(composed.plan(shape, {}, schedule)).toMatchObject({ mechanism: "continuous", checkpoint: true });
  }
});
}

test("Gemma2 softcap plans shared continuation for plain, affine and TurboQuant KV, keeping every other exclusion", () => {
  const make = (kv: KvOptions = {}) => Object.assign(Object.create(UniversalDenseModel.prototype), { encodedKvAttention: false,
    args: { modelType: "gemma2", maskArray: true, attnLogitSoftcap: 50, layerTypes: null },
    config: { modelType: "gemma2", text: { enableMoeBlock: false }, eosTokenIds: [] },
    makeCache: () => [new KVCache()], loraState: { active: [] },
    composition: testComposition(kvOf(kv)),
  }) as TestGraph<UniversalDenseModel>;
  const model = make();
  const binding = bindMlxGateway(model);
  expect(binding.plan(shape, {}, unscheduled)).toMatchObject({ mechanism: "continuous", checkpoint: false });
  // Greedy, seeded and history-dependent sampling, and adapter requests, which Gemma2 already batches.
  const planned: [typeof shape, GenerateOptions][] = [[shape, {}],
    [{ ...shape, userSeed: true, hasRepetitionPenalty: true }, { seed: 42, repetitionPenalty: 1.1, repetitionContextSize: 32 }],
    [{ ...shape, hasAdapters: true }, { adapters: ["upper"] }]];
  for (const [request, options] of planned) {
    expect(binding.plan(request, options, schedule)).toMatchObject({ method: "autoregressive", mechanism: "continuous",
      promptCache: true, checkpoint: true, fill: false });
    expect(binding.plan(request, options, unscheduled).checkpoint).toBe(false);
  }
  // Affine KV (its rows rejected at their transition) and TurboQuant (read
  // decoded) place ordinarily and take continuation checkpoints; their placement
  // is otherwise the one without checkpoints.
  for (const kv of [{ kvBits: 4 }, { turboQuant: { kBits: 8, vBits: 3 } }] as KvOptions[]) {
    const configured = bindMlxGateway(make(kv));
    const plan = configured.plan(shape, {}, schedule), baseline = configured.plan(shape, {}, unscheduled);
    const { checkpoint: planned, ...placement } = plan, { checkpoint: unplanned, ...unchanged } = baseline;
    expect({ kv, placement }).toEqual({ kv, placement: unchanged });
    expect({ kv, method: plan.method, mechanism: plan.mechanism, planned, unplanned })
      .toEqual({ kv, method: "autoregressive", mechanism: "continuous", planned: true, unplanned: false });
  }
  // Grammar, fill, drafts, media and logprobs keep their
  // placement and bind no continuation: placed requests carry no checkpoint,
  // and the rest stay unsupported for this graph.
  const fill = { fill: new FillSession({ rows: [], eos: [], echo: null }, [2, 651]) };
  for (const [request, options, mechanism] of [[{ ...shape, hasGrammar: true }, {}, "continuous"], [shape, fill, "continuous"],
    [{ ...shape, wantsLogprobs: true }, {}, "continuous"], [{ ...shape, hasDraft: true }, {}, "unsupported"],
    [{ ...shape, hasVision: true }, {}, "unsupported"]] as const) {
    const plan = binding.plan(request, options, schedule), baseline = binding.plan(request, options, unscheduled);
    const { checkpoint: _planned, ...placement } = plan, { checkpoint: _baseline, ...unchanged } = baseline;
    expect({ request, placement }).toEqual({ request, placement: unchanged });
    expect({ request, mechanism: plan.mechanism, checkpoint: plan.mechanism === "continuous" && plan.checkpoint })
      .toEqual({ request, mechanism, checkpoint: false });
    expect(binding.continuationRequest!(plan, options, [2, 651], () => {}, 4)).toBeUndefined();
  }
});

test("a paged composition pages plain, adapter and media requests alike and keys their checkpoints in its namespace", () => {
  const graph = <M extends object>(prototype: M, modelType: string, pagedKv: { blockSize: number; direct: boolean } | null) => Object.assign(Object.create(prototype), {
    config: { modelType, text: { enableMoeBlock: false }, eosTokenIds: [] },
    makeCache: () => [new KVCache()], loraState: { active: [] },
    composition: testComposition(undefined, { pagedKv }),
  }) as TestGraph<M>;
  const lookups: unknown[][] = [];
  const store = { findGenerationCheckpoint: (...args: unknown[]) => { lookups.push(args); return null; } } as unknown as ContinuationStore;
  const continuations = { store, persistence: new ContinuationPersistence(store, { maxBytes: 1024 }), everyTokens: 4 };
  const pagedGraph = graph(Gemma4Model.prototype, "gemma4", { blockSize: 16, direct: false }), unpagedGraph = graph(Gemma4Model.prototype, "gemma4", null);
  const binding = bindMlxGateway(pagedGraph);
  const unpaged = bindMlxGateway(unpagedGraph);
  // The cache service owns the request state: the composition's paging over the graph's caches.
  const pagedCache = new KvCacheService(binding.graph, { continuations }, { identity: "fixture" });
  const unpagedCache = new KvCacheService(unpaged.graph, { continuations }, { identity: "fixture" });
  const adapters = { adapters: ["upper"] }, adapterShape = { ...shape, hasAdapters: true };
  const prompt = [2, 651, 9];
  const fill = { fill: new FillSession({ rows: [], eos: [], echo: null }, prompt) };
  // Paging is the composition's state, never a planner input: every request is
  // planned exactly as on the unpaged composition, checkpoints and compiled
  // decode included.
  const placed = (plan: ResolvedExecution) => plan;
  for (const [request, options] of [[shape, {}], [adapterShape, adapters], [{ ...shape, hasVision: true }, {}],
    [{ ...adapterShape, hasGrammar: true }, adapters], [adapterShape, { ...adapters, ...fill }],
    [{ ...adapterShape, wantsLogprobs: true }, adapters]] as const)
    expect({ request, plan: placed(binding.plan(request, options, schedule)) })
      .toEqual({ request, plan: placed(unpaged.plan(request, options, schedule)) });
  expect(binding.plan(shape, {}, schedule).compiledDecode).toBe(true);
  expect(unpaged.plan(shape, {}, schedule).compiledDecode).toBe(true);
  // The adapter row pages; its prefixes and checkpoints live in the paged namespace over its own.
  const plan = binding.plan(adapterShape, adapters, schedule);
  expect(plan).toMatchObject({ method: "autoregressive", mechanism: "continuous", promptCache: true, checkpoint: true });
  // The method's namespace is the composition's alike; the cache service adds its paging.
  expect(binding.prefixNamespace!(plan, adapters, "upper@a")).toBe("upper@a");
  expect(unpaged.prefixNamespace!(plan, adapters, "upper@a")).toBe("upper@a");
  const namespace = pagedPrefixNamespace((pagedGraph as TestGraph<Gemma4Model> & { composition: Composition }).composition.kv, { blockSize: 16, direct: false }, "upper@a");
  expect(JSON.parse(namespace)[0]).toBe("paged-v1");
  expect(binding.reusesPrefix(plan)).toBe(true);
  const requests = [binding.continuationRequest!(plan, adapters, prompt, () => {}, 4)!,
    unpaged.continuationRequest!(plan, adapters, prompt, () => {}, 4)!];
  try {
    const [paged, plain] = [pagedCache, unpagedCache].map((cache, index) => {
      const row = { promptIds: prompt, maxTokens: 61, continuation: requests[index]!.continuation };
      const acquired = cache.acquire(row, "upper@a");
      cache.release(row, false);
      try { return acquired.caches.map(state => state.signature()); } finally { for (const state of acquired.caches) state.dispose(); }
    });
    expect(paged).toEqual(["kv:paged"]);
    expect(plain).toEqual(["kv:plain"]);
    expect(lookups.map(lookup => lookup[2])).toEqual([namespace, "upper@a"]);
  } finally { for (const request of requests) request.dispose(); }
  // An unidentified media prefix takes fresh paged state and reuses nothing.
  const media = binding.plan({ ...shape, hasVision: true }, {}, schedule);
  expect(binding.reusesPrefix(media)).toBe(false);
  const prepared = { promptIds: prompt, maxTokens: 61, reusesPrefix: false };
  expect(pagedCache.keeps(prepared)).toBe(false);
  const fresh = pagedCache.acquire(prepared, "");
  try { expect({ reused: fresh.reusedTokens, caches: fresh.caches.map(state => state.signature()) }).toEqual({ reused: 0, caches: ["kv:paged"] }); }
  finally { for (const state of fresh.caches) state.dispose(); pagedCache.release(prepared, false); }
  // Any graph pages: the composition pages Qwen 3.5's plain full-attention storage the same way.
  const qwen = bindMlxGateway(graph(Qwen35Model.prototype, "qwen3_5", { blockSize: 16, direct: false }));
  const qwenCache = new KvCacheService(qwen.graph);
  for (const request of [shape, adapterShape]) {
    const placed = qwen.plan(request, adapters, schedule);
    expect(placed).toMatchObject({ mechanism: "continuous", checkpoint: true });
    const row = { promptIds: prompt, maxTokens: 61, reusesPrefix: qwen.reusesPrefix(placed) };
    const acquired = qwenCache.acquire(row, "upper@a");
    try { expect(acquired.caches.map(state => state.signature())).toEqual(["kv:paged"]); }
    finally { for (const state of acquired.caches) state.dispose(); qwenCache.release(row, false); }
  }
});
