import { describe, expect, test } from "bun:test";
import { bindMlxGateway, type MlxGatewayBinding } from "../../src/execution/gateway-binding";
import { KvCacheService } from "../../src/state/cache-service";
import { declareGraph, ORDINARY_VERIFICATION } from "../../src/models/capabilities";
import type { RuntimeModel } from "../../src/models/factory";
import { createRuntimeConfig, withRuntimeConfig } from "../../src/runtime/config";
import { KVCache } from "../../src/state/kv";
import { rowTable } from "../../src/models/graph";
import { NgramProvider } from "../../src/generation/speculative/sources/ngram-source";
import type { GenerateOptions } from "../../src/generation/index";
import type { GraphCapabilities } from "../../src/contracts/portable/graph";
import { kvOf, testComposition, type KvOptions } from "../models/test-composition";

// A graph that is none of the model classes. Placement can only follow what it
// declares: its name, its model type, and its shape mean nothing to the scheduler.
const shape = { hasVision: false, hasAdapters: false, hasRepetitionPenalty: false, userSeed: false,
  hasLogitsExtras: false, hasGrammar: false, wantsLogprobs: false, hasDraft: false };
/** The graph's composition naming the KV scheme every request stores. */
const composed = (kv: KvOptions) => ({ composition: testComposition(kvOf(kv)) });
const schedule = { checkpoints: false };

/** Placement never forwards: every phase a stand-in declares throws. */
const never = () => { throw new Error("placement never forwards"); };
const phases = { prompt: [{from:1,to:Infinity,operation:never}], multiSequencePrompt: [{from:1,to:Infinity,operation:never}], decode: rowTable(never),
  verify: rowTable(never), appendKnown: rowTable(never), logits: rowTable(never) };
function replacement(declared: Parameters<typeof declareGraph>[0] = {}, extra: object = {}, modelType = "replacement"): RuntimeModel {
  return {
    config: { modelType, text: { enableMoeBlock: false, numHiddenLayers: 1, layerTypes: ["full_attention"] }, eosTokenIds: [] },
    makeCache: () => [new KVCache()], loraState: { active: [] }, logitsFromHidden() {}, ...phases,
    graphCapabilities: declareGraph(declared), ...extra,
  } as unknown as RuntimeModel;
}
const bind = (model: RuntimeModel, draft = false) => bindMlxGateway(model,
  draft ? { provider: new NgramProvider(), numDraftTokens: 3 } : undefined);
const refusals = (plan: { reasons: readonly string[] }) => plan.reasons.filter(reason => reason.endsWith("-unsupported"));
/** A plan without its compiled-decode note: the loader, not the planner, resolves
 * compiled decode off for storage with no compiled form (paged, TurboQuant, delayed). */
const placement = (plan: { reasons: readonly string[] }) =>
  ({ ...plan, reasons: plan.reasons.filter(reason => reason !== "compiled-decode-unavailable-for-request") });

describe("placement follows the graph's declarations", () => {
  test("the binding reports what the graph declared, once", () => {
    const declared = declareGraph({ hiddenLayerTaps: true });
    expect(bind(replacement({ hiddenLayerTaps: true })).capabilities).toEqual(declared);
  });

  test("a model type or class name declares nothing", () => {
    // Named like the graphs whose declarations grant these capabilities, this graph declares none.
    const binding = bind(replacement({}, {}, "gemma4"));
    expect(refusals(binding.plan({ ...shape, hasVision: true }, {}, schedule))).toEqual(["media-batch-unsupported"]);
    expect(binding.plan(shape, {}, schedule).compiledDecode).toBe(false);
  });

  test("a paged composition pages every request's state; adapters and compiled decode are placed from their declarations", () => {
    // Paging is the composition's, not the graph's: any graph's plain full-attention storage pages, and
    // requests are planned exactly as without it.
    const paged = bind(replacement({}, { composition: testComposition(undefined, { pagedKv: { blockSize: 16, direct: false } }) }));
    // The cache service builds every request's state from the composition: paged pages, unpaged plain storage.
    const fresh = (binding: MlxGatewayBinding, reusesPrefix: boolean) => {
      const cache = new KvCacheService(binding.graph), row = { promptIds: [1, 2], maxTokens: 62, reusesPrefix };
      const { caches } = cache.acquire(row, "");
      try { return caches.map(state => state.signature()); } finally { for (const state of caches) state.dispose(); cache.release(row, false); }
    };
    for (const request of [shape, { ...shape, hasAdapters: true }, { ...shape, hasGrammar: true }]) {
      const plan = paged.plan(request, {}, schedule);
      expect(placement(plan)).toEqual(placement(bind(replacement()).plan(request, {}, schedule)));
      expect(fresh(paged, paged.reusesPrefix(plan))).toEqual(["kv:paged"]);
    }
    expect(fresh(bind(replacement()), true)).toEqual(["kv:plain"]);
    const adapted = bind(replacement());
    expect(refusals(adapted.plan({ ...shape, hasAdapters: true }, {}, schedule))).toEqual([]);
    const selected = adapted.bindAdapterContext!(["first"], "first");
    const restore = selected.enter();
    expect((adapted.graph as unknown as {loraState:{active:string[]}}).loraState.active).toEqual(["first"]);
    restore();
    expect((adapted.graph as unknown as {loraState:{active:string[]}}).loraState.active).toEqual([]);
    const compiled = bind(replacement({ compiledDecode: true }));
    expect(compiled.plan(shape, {}, schedule).compiledDecode).toBe(true);

  });

  test("prepared media enters through the graph's own operation, declared and provided together", () => {
    const forward = () => { throw new Error("placement never forwards"); };
    const media = bind(replacement({ media: { input: "embeddings", video: false } }, { bindMediaInput: () => ({ prefill: forward }) }));
    expect(media.plan({ ...shape, hasVision: true }, {}, schedule)).toMatchObject({ mechanism: "continuous" });
    expect(media.mediaInput!({ embeddings: {} as never }).prefill).toBe(forward);
    // Denoising pixels are that method's own input, not the shared media path.
    expect(() => bind(replacement({ media: { input: "embeddings", video: false } }))).toThrow("provides no bindMediaInput");
    expect(() => bind({ config: {}, makeCache: () => [] } as unknown as RuntimeModel)).toThrow("must declare its capabilities");
  });

  test("every cache lego is planned alike: a drafted request speculates over delayed affine and TurboQuant KV", () => {
    const request = { ...shape, hasDraft: true };
    const reference = bind(replacement({}, composed({})), true).plan(request, {}, schedule);
    expect(reference).toMatchObject({ method: "speculative", mechanism: "continuous" });
    for (const kv of [{ kvBits: 4, quantizedKvStart: 64 }, { kvBits: 8 }, { turboQuant: { kBits: 8, vBits: 3 }, quantizedKvStart: 0 }])
      expect(placement(bind(replacement({}, composed(kv)), true).plan(request, {}, schedule)), JSON.stringify(kv)).toEqual(placement(reference));
  });

  test("a verifier that does not qualify a request shape leaves it ordinary", () => {
    const specs: [keyof GraphCapabilities["speculation"], Record<string, unknown>, KvOptions][] = [
      ["logprobs", { wantsLogprobs: true }, {}],
    ];
    for (const [flag, extra, kv] of specs) {
      const request = { ...shape, hasDraft: true, ...extra }, options: GenerateOptions = {};
      expect(bind(replacement({}, composed(kv)), true).plan(request, options, schedule), `${flag} qualified`).toMatchObject({ method: "speculative" });
      const unqualified = bind(replacement({ speculation: { [flag]: false } }, composed(kv)), true).plan(request, options, schedule);
      expect(unqualified, `${flag} unqualified`).toMatchObject({ method: "autoregressive" });
      expect(unqualified.reasons).toContain("draft-incompatible-with-request");
    }
    // Plain-KV verification qualifies none of them at once.
    const plain = bind(replacement({ speculation: ORDINARY_VERIFICATION }), true);
    expect(plain.plan({ ...shape, hasDraft: true }, {}, schedule)).toMatchObject({ method: "speculative", mechanism: "continuous" });
    expect(plain.plan({ ...shape, hasDraft: true, wantsLogprobs: true }, {}, schedule)).toMatchObject({ method: "autoregressive" });
  });
});
