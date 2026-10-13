import { describe, expect, test } from "bun:test";
import { bindMlxGateway, type MlxGatewayBinding } from "../../src/execution/gateway-binding";
import { createRuntimeConfig, withRuntimeConfig } from "../../src/runtime/config";
import { Gemma4Model } from "../../src/models/gemma4/model";
import { UniversalDenseModel } from "../../src/models/universal/dense";
import { universalCacheWindows, type UniversalArgs } from "../../src/models/universal/archs";
import { KVCache } from "../../src/state/kv";
import { RotatingKVCache } from "../../src/state/rotating-kv";
import { KvScheme } from "../../src/state/kv-scheme";
import type { GenerateOptions } from "../../src/generation/index";
import type { ResolvedExecution } from "../../src/contracts/portable/execution";
import type { RuntimeModel } from "../../src/models/factory";
import { NgramProvider } from "../../src/generation/speculative/sources/ngram-source";
import type { DraftProvider } from "../../src/generation/speculative/source";
import { TwoModelProvider } from "../../src/generation/speculative/sources/two-model";
import { FillSession } from "../../src/generation/fill/session";
import { DiffusionGemmaModel } from "../../src/models/diffusion-gemma/model";
import { Glm52Model } from "../../src/models/glm52/model";
import { MiniCPM5Model } from "../../src/models/minicpm5/model";
import { Qwen3Model } from "../../src/models/qwen/qwen3";
import { Qwen3MoeModel } from "../../src/models/qwen/qwen3-moe";
import { Qwen35Model } from "../../src/models/qwen/qwen3_5";
import { BF16_KV, COMPOSITION_DEFAULTS } from "../../src/contracts/portable/composition";
import type { Composition, KvStorageScheme } from "../../src/contracts/portable/composition";
import { stubPhases } from "../models/phase-forward";

// Placement only: bound graphs are prototype stand-ins and no tensors are created.
const shape = { hasVision: false, hasAdapters: false, hasRepetitionPenalty: false, userSeed: false,
  hasLogitsExtras: false, hasGrammar: false, wantsLogprobs: false, hasDraft: false };

/** The KV options a request named before E3, now the composition's KV scheme. */
interface KvOptions {
  readonly kvBits?: number;
  readonly kvConfig?: readonly { layerIdx: number; bits: number; groupSize: number }[];
  readonly quantizedKvStart?: number;
  readonly turboQuant?: { readonly kBits: number; readonly vBits: number };
}
/** The composition's scheme for those options, with their starts: a uniform
 * width converts at 5000 tokens unless a start is named, as before. */
const kvOf = (options: KvOptions): KvStorageScheme => options.turboQuant
  ? new KvScheme("turbo", { turboQuant: { ...options.turboQuant }, quantizedKvStart: options.quantizedKvStart ?? 0 }).storage
  : options.kvConfig?.length
    ? new KvScheme("affine-config", { kvConfig: options.kvConfig.map(entry => ({ ...entry })), quantizedKvStart: options.quantizedKvStart }).storage
    : options.kvBits !== undefined
      ? new KvScheme("affine-uniform", { kvBits: options.kvBits, quantizedKvStart: options.quantizedKvStart }).storage
      : BF16_KV;
/** The composition facts a graph reads: the KV scheme, its paging, and the
 * defaults of the fields that were runtime flags. */
const composition = (kv: KvStorageScheme = BF16_KV, pagedKv: { blockSize: number } | null = null) =>
  ({ kv, ...COMPOSITION_DEFAULTS, pagedKv }) as unknown as Composition;
/** `model`, composed with the KV scheme (and paging) every request it serves stores. */
const composedWith = <M extends object>(model: M, options: KvOptions, pagedKv: { blockSize: number } | null = null): M =>
  Object.assign(model, { composition: composition(kvOf(options), pagedKv) });
/** `model`, its composition (bf16 unless it names one) turning grammar jump on. */
const jumpGraph = <M extends object>(model: M): M =>
  Object.assign(model, { composition: { ...((model as { composition?: Composition }).composition ?? composition()), grammarJump: true } });
const AFFINE4 = { kvBits: 4 } as const, TURBO = { turboQuant: { kBits: 8, vBits: 3 } } as const;
const grammar = { ...shape, hasGrammar: true };
const schedule = { checkpoints: false };
const environments: Record<string, string>[] = [{}, { MLX_BUN_GRAMMAR_BATCH: "0" }, { MLX_BUN_GRAMMAR_BATCH: "1" }];

/** Placement never forwards: every phase a stand-in graph declares throws. */
const placementPhases = stubPhases(() => { throw new Error("placement never forwards"); },
  () => { throw new Error("placement never projects"); });

function gemma4(): Gemma4Model {
  return Object.assign(Object.create(Gemma4Model.prototype), {
    config: { modelType: "gemma4", text: { enableMoeBlock: false }, eosTokenIds: [] },
    makeCache: () => [new KVCache()], loraState: { active: [] }, ...placementPhases
  });
}

function softcapUniversal(): UniversalDenseModel {
  return Object.assign(Object.create(UniversalDenseModel.prototype), {
    args: { modelType: "gemma2", maskArray: true, attnLogitSoftcap: 50, layerTypes: null }, encodedKvAttention: false,
    config: { modelType: "gemma2", text: { enableMoeBlock: false }, eosTokenIds: [] },
    makeCache: () => [new KVCache()], loraState: { active: [] }, ...placementPhases
  });
}

const planUnder = (env: Record<string, string>, model: Gemma4Model | UniversalDenseModel, request = grammar) =>
  withRuntimeConfig(createRuntimeConfig(env), () => bindMlxGateway(model)).plan(request, {}, schedule);

const standIn = (prototype: object, modelType: string, extra: object = {}): RuntimeModel =>
  Object.assign(Object.create(prototype), {
    config: { modelType, text: { enableMoeBlock: false }, eosTokenIds: [] },
    makeCache: () => [new KVCache()], loraState: { active: [] }, ...placementPhases, composition: composition(), ...extra,
  });

// Every autoregressive class the binding distinguishes. Stand-ins share plain
// KV caches, so only each graph's declared verifier differs: Gemma2's softcap
// graph is the one class whose grammar jump commits forced spans instead of
// verifying proposals, and whose verifier qualifies ordinary rounds only.
const qualifiedFamilies: [string, () => RuntimeModel][] = [
  ["universal dense", () => standIn(UniversalDenseModel.prototype, "llama",
    { args: { modelType: "llama", maskArray: false, attnLogitSoftcap: null, layerTypes: null } })],
  ["gemma4", gemma4],
  ["qwen3", () => standIn(Qwen3Model.prototype, "qwen3", { encodedKvAttention: true })],
  ["qwen3-moe", () => standIn(Qwen3MoeModel.prototype, "qwen3_moe", { encodedKvAttention: true })],
  ["qwen3.5", () => standIn(Qwen35Model.prototype, "qwen3_5")],
  ["minicpm5", () => standIn(MiniCPM5Model.prototype, "minicpm5")],
  ["glm52", () => standIn(Glm52Model.prototype, "glm_moe_dsa")],
];
const families: [string, () => RuntimeModel][] = [["gemma2 softcap", softcapUniversal], ...qualifiedFamilies];

const place = (binding: MlxGatewayBinding, request: typeof shape) => binding.plan(request, {}, schedule);
const refusals = (plan: ResolvedExecution) => plan.reasons.filter(reason => reason.endsWith("-unsupported"));

// A strict or echo fill session; placement reads only its plan.
const fillOptions = (echo: boolean) => ({ fill: new FillSession({ rows: [], eos: [],
  echo: echo ? { k: 8, maxSpan: 8, maxCandidates: 24, indexMax: 1024 } : null }, [2, 651]) });

// The cache legos a composition can name. Every one implements the whole cache
// contract, so the planner never reads which one a graph composes.
const LEGOS: readonly [string, KvOptions, { blockSize: number } | null][] = [
  ["bf16", {}, null],
  ["affine 4", { kvBits: 4, quantizedKvStart: 0 }, null],
  ["affine 8, delayed", { kvBits: 8, quantizedKvStart: 64 }, null],
  ["affine per layer, delayed", { kvConfig: [{ layerIdx: 0, bits: 4, groupSize: 64 }], quantizedKvStart: 64 }, null],
  ["TurboQuant", { turboQuant: { kBits: 8, vBits: 3 } }, null],
  ["TurboQuant, delayed", { turboQuant: { kBits: 8, vBits: 3 }, quantizedKvStart: 64 }, null],
  ["paged bf16", {}, { blockSize: 16 }],
  ["paged 4", { kvBits: 4, quantizedKvStart: 0 }, { blockSize: 16 }],
];

test.each(families)("%s plans every request identically under every cache lego, drafted, jumping and filled alike", (_, model) => {
  const bind = (kv: KvOptions, pagedKv: { blockSize: number } | null, jump: boolean, drafted: boolean) => {
    const graph = composedWith(model(), kv, pagedKv);
    const binding = bindMlxGateway(jump ? jumpGraph(graph) : graph,
      drafted ? { provider: new NgramProvider(), numDraftTokens: 3 } : undefined);
    return binding;
  };
  const requests: [typeof shape, GenerateOptions][] = [
    [shape, {}], [{ ...shape, hasDraft: true }, {}], [grammar, {}], [{ ...grammar, hasDraft: true }, {}],
    [{ ...shape, hasAdapters: true }, { adapters: ["upper"] }], [{ ...shape, hasAdapters: true, hasDraft: true }, { adapters: ["upper"] }],
    [{ ...shape, wantsLogprobs: true }, {}], [{ ...shape, hasDraft: true, wantsLogprobs: true }, {}],
    [{ ...shape, hasVision: true }, {}], [{ ...shape, userSeed: true, hasRepetitionPenalty: true }, { seed: 3, repetitionPenalty: 1.1 }],
    [shape, fillOptions(false)], [{ ...shape, hasDraft: true }, fillOptions(true)],
  ];
  for (const jump of [false, true]) for (const drafted of [false, true]) {
    const reference = bind({}, null, jump, drafted);
    for (const [name, kv, pagedKv] of LEGOS) {
      const binding = bind(kv, pagedKv, jump, drafted);
      for (const [request, options] of requests) {
        const scheduling = { checkpoints: true };
        expect({ lego: name, jump, drafted, request, plan: binding.plan(request, options, scheduling) })
          .toEqual({ lego: name, jump, drafted, request, plan: reference.plan(request, options, scheduling) });
      }
    }
  }
});

test("the retired MLX_BUN_GRAMMAR_BATCH switch no longer changes grammar placement", () => {
  const [baseline, ...others] = environments.map(env => planUnder(env, gemma4()));
  expect(baseline).toMatchObject({ method: "autoregressive", mechanism: "continuous", grammarJump: false });
  expect(baseline!.reasons).toEqual([]);
  for (const plan of others) expect(plan).toEqual(baseline!);
});

test("Gemma2 softcap grammar and adapter requests plan continuous", () => {
  for (const env of environments)
    for (const request of [shape, grammar, { ...shape, hasAdapters: true }, { ...grammar, hasAdapters: true }])
      expect(planUnder(env, softcapUniversal(), request))
        .toMatchObject({ mechanism: "continuous", fill: false, checkpoint: false, grammarJump: false });
});

test.each(families)("%s places ordinary grammar exactly like the same request without grammar", (_, model) => {
  const binding = bindMlxGateway(model());
  const ordinary = place(binding, grammar);
  expect(ordinary).toMatchObject({ method: "autoregressive", mechanism: "continuous", grammarJump: false });
  expect(ordinary.reasons.filter(reason => reason.includes("grammar"))).toEqual([]);
  // Refused shapes keep their own reasons; grammar adds none.
  for (const kv of [{}, AFFINE4, TURBO]) {
    const composed = bindMlxGateway(composedWith(model(), kv));
    for (const extra of [{}, { hasAdapters: true }, { wantsLogprobs: true }, { userSeed: true },
      { hasRepetitionPenalty: true }, { hasLogitsExtras: true }, { hasVision: true }, { hasDraft: true }])
      expect({ kv, extra, plan: place(composed, { ...shape, ...extra, hasGrammar: true }) })
        .toEqual({ kv, extra, plan: place(composed, { ...shape, ...extra }) });
  }
});

test.each(qualifiedFamilies)("%s qualifies multi-token grammar jump when it is enabled", (_, model) => {
  const binding = bindMlxGateway(jumpGraph(model()));
  const plan = place(binding, grammar);
  expect(plan).toMatchObject({ method: "speculative", mechanism: "continuous", grammarJump: true });
  expect(refusals(plan)).toEqual([]);
  // Committed spans are bound only where the verifier does not qualify proposals.
  expect(binding.methodRequest!(plan, {})!.key).not.toBe("grammar-forced-span");
});

test("Gemma2 softcap selects committed grammar spans without changing the ordinary lane", () => {
  const binding = bindMlxGateway(jumpGraph(softcapUniversal()));
  const plan = place(binding, grammar);
  expect(plan).toMatchObject({ method: "autoregressive", mechanism: "continuous", grammarJump: true,
    checkpoint: false, promptCache: true, fill: false });
  expect(plan.reasons).not.toContain("grammar-jump-incompatible-with-request");
  expect(binding.methodRequest!(plan, {})!.key).toBe("grammar-forced-span");
  expect(refusals(plan)).toEqual([]);
  for (const request of [{ ...grammar, wantsLogprobs: true }, { ...grammar, hasVision: true }])
    expect(place(binding, request).grammarJump).toBe(false);
  // Spans commit over every cache lego; a row whose next append its storage
  // would not read plain is refused by the span method before that append.
  for (const kv of [TURBO, AFFINE4]) {
    const composed = bindMlxGateway(jumpGraph(composedWith(softcapUniversal(), kv)));
    expect(place(composed, grammar), JSON.stringify(kv)).toMatchObject({ mechanism: "continuous", grammarJump: true });
    expect(composed.methodRequest!(place(composed, grammar), {})!.key).toBe("grammar-forced-span");
  }
});

test.each(families)("%s places grammar with a bound grouped draft exactly as the draft alone", (_, model) => {
  const binding = bindMlxGateway(model(), { provider: new NgramProvider(), numDraftTokens: 3 });
  const draft = { ...shape, hasDraft: true };
  const plan = place(binding, { ...draft, hasGrammar: true });
  expect(plan).toEqual(place(binding, draft));
  expect(plan).toMatchObject({ method: "speculative", mechanism: "continuous", grammarJump: false });
});

describe("DiffusionGemma interleaved denoising binding", () => {
  const diffusion = () => standIn(DiffusionGemmaModel.prototype, "diffusion_gemma") as DiffusionGemmaModel;
  const planWith = (binding: MlxGatewayBinding, request: typeof shape, options: GenerateOptions = {}) =>
    binding.plan(request, options, schedule);

  test("plain, seeded and adapted requests place continuously without AR-only reuse", () => {
    const binding = bindMlxGateway(diffusion());
    for (const extra of [{}, { userSeed: true }, { hasAdapters: true }])
      expect(place(binding, { ...shape, ...extra })).toMatchObject({ method: "denoising", mechanism: "continuous",
        promptCache: false, checkpoint: false, fill: false, compiledDecode: false, grammarJump: false });
    expect(binding.methodRequest!(place(binding, shape), {})).toMatchObject({ key: JSON.stringify(["denoising", "legacy-diffusion-gemma"]) });
  });

  test.each([
    [{ hasGrammar: true }, {}, "grammar-batch-unsupported"],
    [{ hasDraft: true }, {}, "draft-method-unsupported"],
    [{ wantsLogprobs: true }, {}, "logprobs-method-unsupported"],
    [{ hasRepetitionPenalty: true }, {}, "repetition-penalty-method-unsupported"],
    [{ hasLogitsExtras: true }, {}, "logits-extras-method-unsupported"],
    [{}, { fill: { plan: {} } }, "fill-method-unsupported"],
  ] as const)("an unsupported denoising shape is refused before execution: %#", (extra, options, reason) => {
    for (const binding of [bindMlxGateway(diffusion()), bindMlxGateway(diffusion(), { provider: new NgramProvider(), numDraftTokens: 3 }),
      bindMlxGateway(jumpGraph(diffusion()))]) {
      const plan = planWith(binding, { ...shape, ...extra }, options as GenerateOptions);
      expect(plan).toMatchObject({ method: "denoising", mechanism: "unsupported", fill: false, grammarJump: false });
      expect(refusals(plan)).toEqual([reason]);
    }
  });

  test("an image request places continuously and its method borrows the request's pixels", () => {
    const binding = bindMlxGateway(diffusion());
    const pixels = { dispose() {} } as unknown as NonNullable<GenerateOptions["visionPixels"]>;
    const options: GenerateOptions = { seed: 1, visionPixels: pixels };
    // Denoising pixels are that method's prefill input, so hasVision stays false.
    const plan = planWith(binding, { ...shape, userSeed: true }, options);
    expect(plan).toMatchObject({ method: "denoising", mechanism: "continuous", promptCache: false, checkpoint: false });
    expect((binding.methodRequest!(plan, options)!.data as GenerateOptions).visionPixels).toBe(pixels);
    // Prepared autoregressive media still needs a media binding this graph lacks.
    expect(refusals(planWith(binding, { ...shape, hasVision: true }))).toEqual(["media-batch-unsupported"]);
    expect(place(bindMlxGateway(gemma4()), { ...shape, hasVision: true }))
      .toMatchObject({ method: "autoregressive", mechanism: "continuous" });
  });

  test("adapters are row state: the group context is neutral, and token methods never bind", () => {
    const model = diffusion();
    const binding = bindMlxGateway(model);
    const context = binding.bindAdapterContext!(["upper"], "adapters:[\"upper\"]");
    expect(context.key).toBe(binding.bindAdapterContext!(["lower"], "adapters:[\"lower\"]").key);
    const leave = context.enter();
    expect(model.loraState.active).toEqual([]);
    leave();
    // Token methods never bind to the denoising graph, even when configured.
    const drafted = bindMlxGateway(model, { provider: new NgramProvider(), numDraftTokens: 3 });
    const denoise = place(drafted, shape);
    for (const execution of [{ ...denoise, method: "speculative" }, { ...denoise, method: "autoregressive", fill: true }])
      expect(drafted.methodRequest!(execution, { fill: { plan: {} } } as unknown as GenerateOptions)).toBeUndefined();
  });
});

function minicpm5(layers = 4): MiniCPM5Model {
  return Object.assign(Object.create(MiniCPM5Model.prototype), {
    config: { modelType: "minicpm5", text: { enableMoeBlock: false, numHiddenLayers: layers, layerTypes: Array(layers).fill("full_attention") }, eosTokenIds: [] },
    makeCache: () => Array.from({ length: layers }, () => new KVCache()), loraState: { active: [] }, ...placementPhases
  });
}

/** A universal descriptor with the graph's own cache layout: caches follow the descriptor. */
function universal(args: Record<string, unknown> = {}, layers = 4): UniversalDenseModel {
  const descriptor = { modelType: "qwen2", maskArray: false, attnLogitSoftcap: null, layerTypes: null, slidingWindow: null,
    numHiddenLayers: layers, ...args };
  return Object.assign(Object.create(UniversalDenseModel.prototype), {
    args: descriptor,
    // The bound attention facts a constructed graph derives from this descriptor.
    encodedKvAttention: descriptor.attnLogitSoftcap === null,
    makeCache: () => universalCacheWindows(descriptor as unknown as UniversalArgs)
      .map(window => window ? new RotatingKVCache(window) : new KVCache()),
    config: { modelType: descriptor.modelType, text: { enableMoeBlock: false, numHiddenLayers: layers,
      layerTypes: (descriptor.layerTypes as string[] | null) ?? Array(layers).fill("full_attention") }, eosTokenIds: [] },
    loraState: { active: [] }, ...placementPhases,
  });
}

const SLIDING = ["full_attention", "sliding_attention", "full_attention", "sliding_attention"];

// Delayed affine KV used to keep these graphs ordinary: their drafts refused,
// their fill and adapter drafts ignored, their grammar committed as spans. The
// lego converts each row inside its own appends and holds the speculative
// transaction, so they now speculate, fill and verify grammar proposals as
// over bf16.
test.each([["MiniCPM5", () => minicpm5()], ["plain universal", () => universal()],
  ["mixed full/sliding universal", () => universal({ modelType: "llama", layerTypes: SLIDING, slidingWindow: 16 })],
  ["qwen3", () => standIn(Qwen3Model.prototype, "qwen3", { encodedKvAttention: true })]] as const)(
  "%s serves drafts, fill, adapter drafts and grammar proposals over delayed affine KV", (_, make) => {
    for (const kv of [{ kvBits: 4, quantizedKvStart: 8 }, { kvBits: 8 },
      { kvConfig: [{ layerIdx: 0, bits: 4, groupSize: 64 }, { layerIdx: 2, bits: 8, groupSize: 64 }], quantizedKvStart: 8 }]) {
      const binding = bindMlxGateway(jumpGraph(composedWith(make(), kv)), { provider: new NgramProvider(), numDraftTokens: 3 });
      const scheduling = { checkpoints: true };
      expect(binding.plan(shape, {}, scheduling)).toMatchObject({ method: "autoregressive", mechanism: "continuous", checkpoint: true });
      expect(binding.plan({ ...shape, hasDraft: true }, {}, scheduling)).toMatchObject({ method: "speculative", mechanism: "continuous" });
      const adapted = binding.plan({ ...shape, hasDraft: true, hasAdapters: true }, { adapters: ["upper"] }, scheduling);
      expect(adapted).toMatchObject({ method: "speculative", mechanism: "continuous" });
      const filled = binding.plan(shape, fillOptions(false), scheduling);
      expect(filled).toMatchObject({ method: "autoregressive", mechanism: "continuous", fill: true });
      expect(JSON.parse(binding.methodRequest!(filled, fillOptions(false))!.key)[0]).toBe("fill");
      expect(binding.plan(grammar, {}, scheduling)).toMatchObject({ method: "speculative", grammarJump: true });
    }
  });

test.each(families)("%s places plain-KV fill on the shared fill binding", (_, model) => {
  const binding = bindMlxGateway(model());
  for (const echo of [false, true]) {
    const options = fillOptions(echo);
    for (const extra of [{}, { userSeed: true }, { hasRepetitionPenalty: true }, { hasLogitsExtras: true }]) {
      const plan = binding.plan({ ...shape, ...extra }, options, schedule);
      expect({ extra, plan }).toMatchObject({ extra, plan: { method: "autoregressive", mechanism: "continuous", fill: true, checkpoint: false } });
      expect(refusals(plan)).toEqual([]);
      expect(JSON.parse(binding.methodRequest!(plan, options)!.key)[0]).toBe("fill");
    }
  }
});

/** A two-model provider for placement only: its grouped rows must never open here. */
function twoModelDraft(): TwoModelProvider {
  const unexpected = () => { throw new Error("placement opened draft rows"); };
  return Object.assign(Object.create(TwoModelProvider.prototype), { id: "gemma2-draft", weightsBytes: 0,
    grouped: { checkpointNamespace: () => "gemma2-draft", open: unexpected, openPrefill: unexpected } }) as TwoModelProvider;
}

/** The n-gram provider's operations behind a plain object: no class identity, same contracts. */
function delegatingNgram(): DraftProvider {
  const inner = new NgramProvider(), grouped = inner.grouped;
  return { id: "ngram-delegate", weightsBytes: 0, dispose: () => inner.dispose(),
    grouped: { checkpointNamespace: () => grouped.checkpointNamespace!(), supportsTargetAdapters: grouped.supportsTargetAdapters,
      ...(grouped.targetTapLayers ? { targetTapLayers: target => grouped.targetTapLayers!(target) } : {}),
      open: options => grouped.open(options), openPrefill: options => grouped.openPrefill(options) } };
}

test("each request binds one draft selection, resolved when the binding binds it and keyed into its group", () => {
  /** The n-gram provider's operations declaring a listed draft vocabulary; counts each declaration asked. */
  const listing = (asked: { count: number }): DraftProvider => {
    const inner = delegatingNgram();
    return { ...inner, grouped: { ...inner.grouped!, draftVocabulary: () => { asked.count++; return [1, 2, 3]; } } };
  };
  const selectionOf = (binding: MlxGatewayBinding, options: GenerateOptions) => {
    const plan = binding.plan({ ...shape, hasDraft: true }, options, schedule);
    expect(plan).toMatchObject({ method: "speculative", mechanism: "continuous" });
    return JSON.parse(binding.methodRequest!(plan, options)!.key).at(-1);
  };
  const asked = { count: 0 };
  const binding = bindMlxGateway(softcapUniversal(), { provider: listing(asked), numDraftTokens: 3 });
  // The vocabulary is the composition's: declared once, for the bound target.
  expect(asked.count).toBe(1);
  // A draft sampler that reproduces on the listed rows drafts over them; XTC does
  // not, so that request draws over the full head in a group of its own.
  expect(selectionOf(binding, {})).toBe("listed");
  expect(selectionOf(binding, { temperature: 0.7, topP: 0.9, seed: 3 })).toBe("listed");
  expect(selectionOf(binding, { temperature: 0.7, xtcProbability: 0.5, xtcThreshold: 0.1 })).toBe("sampled");
  expect(asked.count).toBe(1);
  // Without a declared vocabulary every request draws over the full head.
  expect(selectionOf(bindMlxGateway(softcapUniversal(), { provider: new NgramProvider(), numDraftTokens: 3 }), {})).toBe("sampled");
  // Argmax drafts (the composition's greedyDraft) never consult the vocabulary.
  const greedyAsked = { count: 0 };
  const greedy = bindMlxGateway(Object.assign(softcapUniversal(), { composition: { ...composition(), greedyDraft: true } }),
    { provider: listing(greedyAsked), numDraftTokens: 3 });
  expect([selectionOf(greedy, {}), greedyAsked.count]).toEqual(["greedy", 0]);
  // The runtime configuration does not reach the selection.
  const flagged = withRuntimeConfig(createRuntimeConfig({ MLX_BUN_SPEC_GREEDY_DRAFT: "1" }),
    () => bindMlxGateway(softcapUniversal(), { provider: listing({ count: 0 }), numDraftTokens: 3 }));
  expect(selectionOf(flagged, {})).toBe("listed");
});

test.each([["two-model", twoModelDraft, "gemma2-draft"], ["n-gram", () => new NgramProvider(), "ngram"],
  ["delegating n-gram", delegatingNgram, "ngram-delegate"]] as const)(
  "Gemma2 softcap speculates with the %s provider over every cache lego", (_, provider, id) => {
  const scheduling = { checkpoints: true };
  const draft = { ...shape, hasDraft: true };
  for (const kv of [{}, TURBO, { kvBits: 4, quantizedKvStart: 64 }] as KvOptions[]) {
    const binding = bindMlxGateway(composedWith(softcapUniversal(), kv), { provider: provider(), numDraftTokens: 3 });
    // Grammar is shared sampling: the target's mask rides the verifier's accept walk, as in main.
    for (const [request, options] of [[draft, {}], [{ ...draft, userSeed: true, hasRepetitionPenalty: true }, { seed: 3, repetitionPenalty: 1.1 }],
      [draft, { logitBias: { 5: -100 } }], [{ ...draft, hasGrammar: true }, {}]] as const) {
      const plan = binding.plan(request, options as GenerateOptions, scheduling);
      expect({ kv, request, plan }).toMatchObject({ kv, request, plan: { method: "speculative", mechanism: "continuous", fill: false, checkpoint: false } });
      expect(refusals(plan)).toEqual([]);
      expect(JSON.parse(binding.methodRequest!(plan, options as GenerateOptions)!.key).slice(0, 2)).toEqual(["speculative", id]);
    }
    // As main did, a logprobs request ignores its draft and decodes ordinarily.
    const logprobs = binding.plan({ ...draft, wantsLogprobs: true }, {}, scheduling);
    expect(logprobs).toMatchObject({ method: "autoregressive", mechanism: "continuous" });
    expect(logprobs.reasons).toContain("draft-incompatible-with-request");
    // Drafted media stays refused, including when fill is supplied.
    for (const supplied of [{}, fillOptions(true)] as GenerateOptions[]) {
      const plan = binding.plan({ ...draft, hasVision: true }, supplied, scheduling);
      expect(plan.mechanism).toBe("unsupported");
      expect(plan.reasons).toContain("media-batch-unsupported");
    }
    // Plain softcap requests without a draft keep their placement.
    expect(binding.plan(shape, {}, scheduling)).toMatchObject({ method: "autoregressive", mechanism: "continuous", checkpoint: true });
  }
  // Grammar proposals stay off this graph: its jump commits spans.
  const jump = bindMlxGateway(jumpGraph(softcapUniversal()), { provider: provider(), numDraftTokens: 3 });
  expect(jump.plan({ ...shape, hasGrammar: true }, {}, schedule)).toMatchObject({ method: "autoregressive", grammarJump: true });
  expect(jump.plan({ ...shape, hasGrammar: true, hasAdapters: true, hasDraft: true }, {}, schedule))
    .toMatchObject({ method: "autoregressive", mechanism: "continuous", grammarJump: true });
  expect(jump.plan({ ...shape, hasGrammar: true, hasDraft: true }, {}, schedule))
    .toMatchObject({ method: "speculative", mechanism: "continuous", grammarJump: false });
});

test.each([["two-model", twoModelDraft], ["n-gram", () => new NgramProvider()], ["delegating n-gram", delegatingNgram]] as const)(
  "Gemma2 softcap ignores fill with the %s draft, as main's serial verifier did", (_, provider) => {
  const binding = bindMlxGateway(softcapUniversal(), { provider: provider(), numDraftTokens: 3 });
  const strict = new FillSession({ rows: [{ trigger: [651], emit: [42, 43], kind: "scaffold" }], eos: [], echo: null }, [2, 651]);
  let verifyCalls = 0;
  const verify = new FillSession({ rows: [], eos: [], echo: null }, [2, 651], { sources: [{
    name: "verify", windowNeeded: 1,
    propose: () => { verifyCalls++; return { ids: [42, 43], policy: "verify", origin: "echo" }; },
  }] });
  for (const fill of [strict, verify, fillOptions(true).fill]) for (const wantsLogprobs of [false, true]) {
    const request = { ...shape, hasDraft: true, wantsLogprobs };
    const options: GenerateOptions = { fill, maxTokens: 16, temperature: 0,
      ...(wantsLogprobs ? { logprobs: true, topLogprobs: 3 } : {}) };
    const plan = binding.plan(request, options, schedule);
    expect(plan).toMatchObject({ method: wantsLogprobs ? "autoregressive" : "speculative",
      mechanism: "continuous", fill: false, checkpoint: false });
    expect(refusals(plan)).toEqual([]);
    expect(plan.reasons).toContain("fill-incompatible-with-request");
    const method = binding.methodRequest!(plan, options);
    if (wantsLogprobs) {
      expect(plan.reasons).toContain("draft-incompatible-with-request");
      expect(method).toBeUndefined();
    } else {
      const { fill: ignored, ...numericalOptions } = options;
      expect(method!.data).toEqual({ ...numericalOptions, fill: undefined });
      expect(method!.key).toBe(binding.methodRequest!(binding.plan(request, numericalOptions, schedule), numericalOptions)!.key);
      expect(options.fill).toBe(ignored); // stripping does not mutate the caller's session
    }
  }
  expect(verifyCalls).toBe(0);
});

test("softcap drafts ignore echo even if their provider advertises external tokens; other graphs retain echo", () => {
  const provider = new NgramProvider();
  Object.defineProperty(provider.grouped, "supportsExternalTokens", { value: true });
  const options = fillOptions(true), request = { ...shape, hasDraft: true };
  const softcap = bindMlxGateway(softcapUniversal(), { provider, numDraftTokens: 3 });
  const plan = softcap.plan(request, options, schedule);
  expect(plan).toMatchObject({ method: "speculative", mechanism: "continuous", fill: false });
  expect((softcap.methodRequest!(plan, options)!.data as GenerateOptions).fill).toBeUndefined();
  for (const [, model] of qualifiedFamilies) {
    const binding = bindMlxGateway(model(), { provider, numDraftTokens: 3 });
    const other = binding.plan(request, options, schedule);
    expect(other).toMatchObject({ method: "speculative", mechanism: "continuous", fill: true });
    expect((binding.methodRequest!(other, options)!.data as GenerateOptions).fill).toBe(options.fill);
  }
});

test.each([["two-model", twoModelDraft], ["n-gram", () => new NgramProvider()], ["delegating n-gram", delegatingNgram]] as const)(
  "Gemma2 softcap adapters ignore the %s draft and fill, preserving ordinary decoding over every cache lego", (_, provider) => {
  const scheduling = { checkpoints: true };
  const request = { ...shape, hasDraft: true, hasAdapters: true };
  for (const kv of [{}, AFFINE4, TURBO] as KvOptions[]) {
    const model = composedWith(softcapUniversal(), kv);
    const binding = bindMlxGateway(model, { provider: provider(), numDraftTokens: 3 });
    for (const fill of [undefined, fillOptions(false).fill, fillOptions(true).fill]) {
      for (const extra of [{}, { wantsLogprobs: true }, { userSeed: true }, { hasGrammar: true }]) {
        const options: GenerateOptions = { adapters: ["upper"], fill, ...(extra.userSeed ? { seed: 42 } : {}) };
        const plan = binding.plan({ ...request, ...extra }, options, scheduling);
        // The ignored draft leaves the ordinary checkpoint rules: no fill, grammar
        // or logprobs.
        expect(plan).toMatchObject({ method: "autoregressive", mechanism: "continuous", fill: false,
          checkpoint: !fill && !("wantsLogprobs" in extra) && !("hasGrammar" in extra), promptCache: true, grammarJump: false });
        expect(refusals(plan)).toEqual([]);
        expect(plan.reasons).toContain("draft-incompatible-with-request");
        if (fill) expect(plan.reasons).toContain("fill-incompatible-with-request");
        // Dispatch must consume neither provider nor fill, even if both were supplied.
        expect(binding.methodRequest!(plan, options)).toBeUndefined();
      }
    }
    const previous = model.loraState.active;
    const leave = binding.bindAdapterContext!(["upper"], "upper").enter();
    expect(model.loraState.active).toEqual(["upper"]);
    leave();
    expect(model.loraState.active).toBe(previous);
    const media = binding.plan({ ...request, hasVision: true }, { adapters: ["upper"] }, scheduling);
    expect(media.mechanism).toBe("unsupported");
    expect(media.reasons).toContain("media-batch-unsupported");
  }
});

test("an adapter request its draft cannot serve decodes ordinarily with ordinary checkpoints on every graph", () => {
  // Including sliding and explicit-mask universal graphs: the draft is ignored
  // and the request checkpoints like a draftless adapter request.
  const graphs: [string, () => RuntimeModel][] = [...families,
    ["sliding universal", () => universal({ modelType: "llama", layerTypes: SLIDING, slidingWindow: 16 })],
    ["explicit-mask sliding universal", () => universal({ modelType: "llama", maskArray: true, layerTypes: SLIDING, slidingWindow: 16 })]];
  const request = { ...shape, hasDraft: true, hasAdapters: true };
  const options: GenerateOptions = { adapters: ["upper"] };
  for (const [name, model] of graphs) {
    const binding = bindMlxGateway(model(), { provider: twoModelDraft(), numDraftTokens: 3 });
    const scheduling = { checkpoints: true };
    const plan = binding.plan(request, options, scheduling);
    expect({ name, plan }).toMatchObject({ name, plan: { method: "autoregressive", mechanism: "continuous", checkpoint: true } });
    expect(plan.reasons).toContain("draft-incompatible-with-request");
    expect(binding.methodRequest!(plan, options)).toBeUndefined();
    for (const extra of [{ hasGrammar: true }, { wantsLogprobs: true }])
      expect({ name, extra, checkpoint: binding.plan({ ...request, ...extra }, options, scheduling).checkpoint })
        .toEqual({ name, extra, checkpoint: false });
    // The same draft still serves the request without adapters, uncheckpointed.
    expect({ name, plan: binding.plan({ ...shape, hasDraft: true }, {}, scheduling) })
      .toMatchObject({ name, plan: { method: "speculative", mechanism: "continuous", checkpoint: false } });
    // A provider that serves target adapters speculates with them, except on the
    // softcap graph, which serves adapters ordinarily.
    const aware = bindMlxGateway(model(), { provider: new NgramProvider(), numDraftTokens: 3 });
    const softcap = name === "gemma2 softcap";
    expect({ name, plan: aware.plan(request, options, scheduling) }).toMatchObject({ name,
      plan: { method: softcap ? "autoregressive" : "speculative", mechanism: "continuous", checkpoint: softcap } });
  }
});

test.each(families)("%s places a delegating provider exactly like the provider it delegates to", (_, model) => {
  const direct = bindMlxGateway(model(), { provider: new NgramProvider(), numDraftTokens: 3 });
  const delegate = bindMlxGateway(model(), { provider: delegatingNgram(), numDraftTokens: 3 });
  for (const extra of [{}, { userSeed: true }, { hasGrammar: true }, { wantsLogprobs: true }, { hasAdapters: true },
    { hasRepetitionPenalty: true }] as const) {
    const request = { ...shape, hasDraft: true, ...extra };
    const options: GenerateOptions = "hasAdapters" in extra ? { adapters: ["upper"] } : {};
    const expected = direct.plan(request, options, schedule);
    const plan = delegate.plan(request, options, schedule);
    expect({ extra, plan }).toEqual({ extra, plan: expected });
    if (plan.method === "speculative" && plan.mechanism === "continuous")
      expect(JSON.parse(delegate.methodRequest!(plan, options)!.key).slice(0, 2)).toEqual(["speculative", "ngram-delegate"]);
  }
});

/** A provider whose rows tap target hidden layers; placement must never open its rows. */
function tapper(layers: readonly number[]) {
  const unexpected = () => { throw new Error("placement opened draft rows"); };
  const taps = Object.freeze([...layers]);
  return { id: "tapper", weightsBytes: 0, grouped: { checkpointNamespace: () => "tapper",
    targetTapLayers: () => taps, open: unexpected, openPrefill: unexpected } } as never;
}

test("a provider that taps target layers binds only where the forward captures them; otherwise placement refuses it", () => {
  const request = { ...shape, hasDraft: true };
  const refused = (binding: MlxGatewayBinding) => {
    const plan = binding.plan(request, {}, schedule);
    expect(plan).toMatchObject({ method: "speculative", mechanism: "unsupported" });
    expect(refusals(plan)).toEqual(["method-batch-unsupported"]);
    expect(plan.reasons).not.toContain("draft-incompatible-with-request");
  };
  // Graphs without a hidden-tap operation, softcap or not, refuse it rather than decode ordinarily.
  refused(bindMlxGateway(softcapUniversal(), { provider: tapper([2, 5]), numDraftTokens: 3 }));
  refused(bindMlxGateway(qualifiedFamilies[0]![1](), { provider: tapper([2, 5]), numDraftTokens: 3 }));
  // The Gemma4 graph fields its legacy target view reads. Its phases take the drafter's capture, so it
  // binds taps within its layer range, the post-final-norm sentinel included, and refuses layers beyond it.
  const graph = () => Object.assign(gemma4(), { numDonors: 8,
    layers: Array.from({ length: 8 }, () => ({ layerType: "full_attention" })),
    config: { modelType: "gemma4", text: { enableMoeBlock: false, numHiddenLayers: 8 }, eosTokenIds: [] } });
  expect(bindMlxGateway(graph(), { provider: tapper([2, 8]), numDraftTokens: 3 }).plan(request, {}, schedule))
    .toMatchObject({ method: "speculative", mechanism: "continuous" });
  refused(bindMlxGateway(graph(), { provider: tapper([9]), numDraftTokens: 3 }));
  // Declaring no taps needs no tap operation.
  for (const model of [softcapUniversal(), qualifiedFamilies[0]![1]()])
    expect(bindMlxGateway(model, { provider: tapper([]), numDraftTokens: 3 }).plan(request, {}, schedule))
      .toMatchObject({ method: "speculative", mechanism: "continuous" });
});

test("Gemma2 softcap fill runs with adapters and keeps unbound drafts unsupported", () => {
  const options = fillOptions(true);
  const plain = bindMlxGateway(softcapUniversal());
  const unbound = plain.plan({ ...shape, hasDraft: true }, options, schedule);
  expect(unbound).toMatchObject({ mechanism: "unsupported", fill: false });
  expect(refusals(unbound)).toEqual(["method-batch-unsupported"]);
  expect(unbound.reasons).toContain("fill-incompatible-with-request");
  // Fill with adapters is placed on the shared fill binding, as for every family;
  // the request's adapter context encloses the fill group.
  for (const binding of [plain, bindMlxGateway(qualifiedFamilies[0]![1]())]) {
    const plan = binding.plan({ ...shape, hasAdapters: true }, { ...options, adapters: ["upper"] }, schedule);
    expect(plan).toMatchObject({ method: "autoregressive", mechanism: "continuous", fill: true, checkpoint: false });
    expect(refusals(plan)).toEqual([]);
  }
});

test("gateway policy follows the graph as bound, not a later descriptor", () => {
  // Bound with softcap; the descriptor later drops it: its token-method policy is
  // still the softcap graph's, whatever the descriptor now says.
  const softcap = universal({ modelType: "gemma2", maskArray: true, attnLogitSoftcap: 50 });
  (softcap.args as { attnLogitSoftcap: number | null }).attnLogitSoftcap = null;
  const request = { ...shape, hasGrammar: true };
  expect(bindMlxGateway(softcap).plan(request, {}, schedule)).toEqual(bindMlxGateway(softcapUniversal()).plan(request, {}, schedule));
});

const LAYERS = ["full_attention", "sliding_attention", "full_attention", "sliding_attention"];

test("universal graphs batch and bind drafts, whatever their cache layout", () => {
  const graphs: [string, UniversalDenseModel][] = [
    ["plain", universal()],
    ["sliding, full first", universal({ modelType: "llama", layerTypes: LAYERS, slidingWindow: 16 })],
    ["sliding, sliding first", universal({ modelType: "llama", layerTypes: [...LAYERS].reverse(), slidingWindow: 16 })],
    ["explicit mask, mixed", universal({ modelType: "llama", maskArray: true, layerTypes: LAYERS, slidingWindow: 16 })],
    ["gemma2", universal({ modelType: "gemma2", maskArray: true, attnLogitSoftcap: 50 })],
  ];
  for (const [name, model] of graphs) {
    const binding = bindMlxGateway(model, { provider: new NgramProvider(), numDraftTokens: 4 });
    expect({ name, plan: place(binding, shape).mechanism, drafted: place(binding, { ...shape, hasDraft: true }).method })
      .toEqual({ name, plan: "continuous", drafted: "speculative" });
  }
});

test("planning never probes a graph's storage, on the span graph as on any other", () => {
  let made = 0;
  const encoded = universal({ modelType: "llama", layerTypes: LAYERS, slidingWindow: 16 });
  encoded.makeCache = () => { made++; return []; };
  const binding = bindMlxGateway(encoded, { provider: new NgramProvider(), numDraftTokens: 4 });
  for (let round = 0; round < 3; round++)
    for (const request of [shape, { ...shape, hasDraft: true }, { ...shape, hasGrammar: true }]) place(binding, request);
  // The softcap graph's spans read every storage dense: they bind no layers and probe nothing.
  const softcap = universal({ modelType: "gemma2", maskArray: true, attnLogitSoftcap: 50 });
  softcap.makeCache = () => { made++; return []; };
  const spans = bindMlxGateway(softcap);
  for (let round = 0; round < 3; round++) place(spans, grammar);
  expect(made).toBe(0);
});
