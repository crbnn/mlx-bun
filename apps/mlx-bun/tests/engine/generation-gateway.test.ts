import { expect, test } from "bun:test";
import { GenerationGateway } from "../../src/engine/generation-gateway";
import type { RequestShape, Vision } from "../../src/engine/completion";
import { declareGraph, graphCapabilitiesOf } from "@mlx-bun/inference/models";
import { UnsupportedExecutionError } from "../../src/engine/completion";
import { runtimeConfig } from "@mlx-bun/inference/runtime/config";
import type { MlxGatewayBinding, MlxBatchGroup } from "@mlx-bun/inference/execution";
import type { GenerateOptions } from "@mlx-bun/inference/generation";
import type { KvStorageScheme, ResolvedExecution } from "@mlx-bun/inference/contracts/portable";
import type { DeviceStepSampler, DeviceStepSamplerConfig } from "@mlx-bun/inference/sampling";

const shape = (): RequestShape => ({ hasVision: false, hasAdapters: false,
  hasRepetitionPenalty: false, userSeed: false,
  hasLogitsExtras: false, hasGrammar: false, wantsLogprobs: false, hasDraft: false });
const execution: ResolvedExecution = { method: "autoregressive", mechanism: "continuous",
  promptCache: true, checkpoint: false, fill: false,
  compiledDecode: false, grammarJump: false, reasons: [] };
const result = { promptTokens: 1, cachedTokens: 0, generatedTokens: 1,
  finishReason: "length" as const, prefillMs: 1, decodeMs: 1 };
/** The cache service a gateway borrows: here, its checkpoint interval and a saved-prefix read ahead. */
type GatewayCache = NonNullable<NonNullable<ConstructorParameters<typeof GenerationGateway>[2]>["cache"]>;
const cacheOf = (fields: { checkpointEveryTokens?: number; prefetch?: GatewayCache["prefetch"] }) =>
  ({ async prefetch() { return () => {}; }, ...fields }) as unknown as GatewayCache;
function fake(options: { submit?: MlxBatchGroup["submit"]; plan?: MlxGatewayBinding["plan"]; failSetup?: boolean } = {}) {
  let samplingDisposed = 0, created = 0, capacity = 0, closed = false;
  const group: MlxBatchGroup = { activeRows: 0, pendingRows: 0, kick() {}, async close() { closed = true; },
    submit: options.submit ?? (async request => { await request.onToken(7); return result; }),
  };
  const binding = { config: { modelType: "replacement", eosTokenIds: [2] } as MlxGatewayBinding["config"], runtime: runtimeConfig(),
    capabilities: graphCapabilitiesOf({ graphCapabilities: declareGraph(), dispose() {} }),
    plan: options.plan ?? (() => execution), reusesPrefix: () => true,
    graph: { makeCache: () => [], config: { modelType: "replacement", eosTokenIds: [2] } as MlxGatewayBinding["config"] },
    continuationRequest(_execution, _options, _prompt, onToken) {
      if (options.failSetup) throw new Error("sampling setup failed");
      return { continuation: { interval: 1, key: () => "fixture", seed: 0, seedWasExplicit: false, resumeSampling() {} },
        sample() { throw new Error("CPU fake never samples tensors"); }, plainGreedy: true,
        onToken, dispose() { samplingDisposed++; } };
    },
    createBatchGroup(opts) { created++; capacity = opts.maxBatch!; return group; },
    mediaInput() { return { prefill() { throw new Error("CPU fake never forwards"); } }; },
  } as MlxGatewayBinding;
  return { binding, group, get samplingDisposed() { return samplingDisposed; },
    get created() { return created; }, get capacity() { return capacity; }, get closed() { return closed; } };
}
function grammar() { let disposed = 0; return {
  value: { dispose() { disposed++; } } as GenerateOptions["grammar"],
  get disposed() { return disposed; },
}; }

for (const capacity of [1, 4]) test(`capacity ${capacity} uses the same continuous submission path`, async () => {
  const f = fake(); const gateway = new GenerationGateway(f.binding, capacity);
  const s = shape(); const placement = gateway.place(s);
  expect(placement.mechanism).toBe("continuous");
  const tokens: number[] = [];
  expect((await gateway.run([1], { maxTokens: 1 }, token => { tokens.push(token); }, undefined, s, placement)).generatedTokens).toBe(1);
  expect(tokens).toEqual([7]); expect(f.created).toBe(1); expect(f.capacity).toBe(capacity);
  expect(f.samplingDisposed).toBe(1); expect(gateway.submittedRows).toBe(1);
  await gateway.close(); expect(f.closed).toBe(true);
});

test("denoising image pixels stay borrowed through the row and are released once it settles", async () => {
  // Outcome of the fake row: resolve, reject, or never submitted (cancelled before start).
  for (const outcome of ["resolve", "reject", "unstarted"] as const) {
    let disposed = 0;
    const pixels = { dispose() { disposed++; } } as unknown as NonNullable<GenerateOptions["visionPixels"]>;
    let borrowed: unknown;
    const f = fake({ plan: () => ({ ...execution, method: "denoising", promptCache: false }),
      submit: async request => {
        // The row borrows the request's pixels; nothing releases them while it runs.
        borrowed = (request.method!.data as GenerateOptions).visionPixels;
        expect(disposed).toBe(0);
        if (outcome === "reject") throw new DOMException("client left", "AbortError");
        return result;
      } });
    f.binding.methodRequest = (_execution, options) => ({ key: "denoising", data: options,
      open: () => { throw new Error("CPU fake never opens a method"); } });
    const gateway = new GenerationGateway(f.binding, 1), s = shape(), abort = new AbortController();
    if (outcome === "unstarted") abort.abort(new DOMException("client left", "AbortError"));
    try {
      const run = gateway.run([1], { maxTokens: 8, visionPixels: pixels }, () => {}, undefined, s, gateway.place(s), abort.signal);
      if (outcome === "resolve") await run;
      else await expect(run).rejects.toHaveProperty("name", "AbortError");
      expect({ outcome, borrowed: borrowed === pixels, disposed })
        .toEqual({ outcome, borrowed: outcome !== "unstarted", disposed: 1 });
    } finally { await gateway.close(); }
  }
});

// Each path holds a reservation, then fails before the row reaches the group.
for (const [path, message] of [["prefetch failure", "prefetch failed"], ["wrong placement", "does not belong"],
  ["abort after reservation", "client left"]] as const) test(`denoising image pixels and grammar release once after ${path}`, async () => {
  let disposed = 0, submitted = 0;
  const pixels = { dispose() { disposed++; } } as unknown as NonNullable<GenerateOptions["visionPixels"]>;
  const controller = grammar(), abort = new AbortController();
  const f = fake({ plan: () => ({ ...execution, method: "denoising", promptCache: false }),
    submit: async () => { submitted++; return result; } });
  const gateway = new GenerationGateway(f.binding, 1, { cache: cacheOf({ async prefetch() {
    if (path === "prefetch failure") throw new Error("prefetch failed");
    if (path === "abort after reservation") abort.abort(new DOMException("client left", "AbortError"));
    return () => {};
  } }) });
  const s = shape(), placement = gateway.place(path === "wrong placement" ? shape() : s);
  try {
    await expect(gateway.run([1], { grammar: controller.value, visionPixels: pixels }, () => {}, undefined, s, placement, abort.signal))
      .rejects.toThrow(message);
    expect({ pixels: disposed, grammar: controller.disposed, submitted }).toEqual({ pixels: 1, grammar: 1, submitted: 0 });
  } finally { await gateway.close(); }
});

test("decode throughput counts every denoised token and keeps the autoregressive first-token exclusion", async () => {
  for (const [method, generatedTokens, decodeMs, decodeTps] of [
    ["denoising", 1, 50, 20], ["denoising", 5, 250, 20],
    ["autoregressive", 1, 50, 0], ["autoregressive", 5, 200, 20],
  ] as const) {
    const submitted: Parameters<MlxBatchGroup["submit"]>[0][] = [];
    const f = fake({ plan: () => ({ ...execution, method }),
      submit: async request => { submitted.push(request); return { ...result, generatedTokens, decodeMs }; } });
    if (method === "denoising")
      f.binding.methodRequest = () => ({ key: "denoising", data: {}, open: () => { throw new Error("CPU fake never opens a method"); } });
    const gateway = new GenerationGateway(f.binding, 1), s = shape();
    try {
      const stats = await gateway.run([1], { maxTokens: 8 }, () => {}, undefined, s, gateway.place(s));
      expect(stats).toMatchObject({ generatedTokens, decodeMs });
      expect(stats.decodeTps).toBeCloseTo(decodeTps, 9);
      // Denoising rows ride their grouped method; AR rows carry a sampler.
      expect(submitted[0]!.method !== undefined).toBe(method === "denoising");
    } finally { await gateway.close(); }
  }
});

test("unsupported methods report typed exclusion reasons without unrelated compilation diagnostics", () => {
  const f = fake({ plan: () => ({ ...execution, method: "denoising", mechanism: "unsupported",
    reasons: ["method-batch-unsupported", "compiled-decode-unavailable-for-request"] }) });
  const gateway = new GenerationGateway(f.binding, 1);
  let failure: unknown;
  try { gateway.place(shape()); } catch (error) { failure = error; }
  expect(failure).toBeInstanceOf(UnsupportedExecutionError);
  expect(failure).toMatchObject({ modelType: "replacement", method: "denoising", reasons: ["method-batch-unsupported"],
    message: "method denoising does not support shared execution: method-batch-unsupported" });
  expect(f.created).toBe(0);
});

test("two rows interleave at shared capacity and cancelling one preserves the other", async () => {
  type Request = Parameters<MlxBatchGroup["submit"]>[0];
  const rows = new Map<number, { request: Request; finish: () => void }>();
  const bothEntered = Promise.withResolvers<void>();
  const f = fake({ submit: request => {
    const task = Promise.withResolvers<typeof result>();
    const id = request.promptIds[0]!;
    const abort = () => task.reject(request.signal!.reason);
    request.signal?.addEventListener("abort", abort, { once: true });
    rows.set(id, { request, finish: () => task.resolve(result) });
    if (rows.size === 2) bothEntered.resolve();
    return task.promise.finally(() => { request.signal?.removeEventListener("abort", abort); rows.delete(id); });
  } });
  Object.defineProperty(f.group, "activeRows", { get: () => rows.size });
  const gateway = new GenerationGateway(f.binding, 2), abort = new AbortController();
  const firstShape = shape(), secondShape = shape(), firstGrammar = grammar(), secondGrammar = grammar();
  const emitted: string[] = [];
  const first = gateway.run([1], { grammar: firstGrammar.value }, token => { emitted.push(`a${token}`); },
    undefined, firstShape, gateway.place(firstShape), abort.signal);
  const second = gateway.run([2], { grammar: secondGrammar.value }, token => { emitted.push(`b${token}`); },
    undefined, secondShape, gateway.place(secondShape));
  await bothEntered.promise;
  expect(gateway.activeRows).toBe(2); expect(f.created).toBe(1); expect(f.capacity).toBe(2);
  expect(gateway.submittedRows).toBe(2);
  await rows.get(1)!.request.onToken(7);
  await rows.get(2)!.request.onToken(8);
  await rows.get(1)!.request.onToken(9);
  abort.abort(new Error("caller left mid-stream"));
  await expect(first).rejects.toThrow("caller left mid-stream");
  expect(gateway.activeRows).toBe(1); expect(gateway.busy).toBe(true);
  expect(firstGrammar.disposed).toBe(1); expect(secondGrammar.disposed).toBe(0); expect(f.samplingDisposed).toBe(1);
  await rows.get(2)!.request.onToken(10); rows.get(2)!.finish(); await second;
  expect(emitted).toEqual(["a7", "b8", "a9", "b10"]);
  expect(secondGrammar.disposed).toBe(1); expect(f.samplingDisposed).toBe(2);
  expect(gateway.activeRows).toBe(0); expect(gateway.pendingRows).toBe(0); expect(gateway.busy).toBe(false);
  await gateway.close(); expect(f.closed).toBe(true);
});

for (const failure of ["setup", "submit", "none"] as const) test(`request-owned media and grammar release after ${failure}`, async () => {
  let mediaDisposed = 0;
  const controller = grammar();
  const f = fake({ failSetup: failure === "setup", submit: async () => {
    expect(mediaDisposed).toBe(0); if (failure === "submit") throw new Error("submission failed"); return result;
  } });
  const gateway = new GenerationGateway(f.binding, 1); const s = { ...shape(), hasVision: true };
  const vision = { embeddings: { dispose() { mediaDisposed++; } } } as Vision;
  const run = gateway.run([1], { grammar: controller.value }, () => {}, vision, s, gateway.place(s));
  if (failure === "none") await run; else await expect(run).rejects.toThrow("failed");
  expect(mediaDisposed).toBe(1); expect(controller.disposed).toBe(1);
  expect(f.samplingDisposed).toBe(failure === "setup" ? 0 : 1); await gateway.close();
});

test("cancelled queued work frees its inputs without entering the group", async () => {
  const entered = Promise.withResolvers<void>(); const finish = Promise.withResolvers<void>(); let submitted = 0;
  const f = fake({ submit: async () => { submitted++; entered.resolve(); await finish.promise; return result; } });
  const gateway = new GenerationGateway(f.binding, 1); const firstShape = shape();
  const first = gateway.run([1], {}, () => {}, undefined, firstShape, gateway.place(firstShape));
  await entered.promise;
  const controller = grammar(); const abort = new AbortController(); const secondShape = shape();
  const second = gateway.run([1], { grammar: controller.value }, () => {}, undefined, secondShape, gateway.place(secondShape), abort.signal);
  abort.abort(new Error("caller left")); await expect(second).rejects.toThrow("caller left");
  expect(controller.disposed).toBe(1); expect(submitted).toBe(1);
  finish.resolve(); await first; await gateway.close();
});

test("a placement from another request cannot consume native input", async () => {
  const f = fake(); const gateway = new GenerationGateway(f.binding, 1); const controller = grammar();
  await expect(gateway.run([1], { grammar: controller.value }, () => {}, undefined, shape(), gateway.place(shape())))
    .rejects.toThrow("does not belong");
  expect(controller.disposed).toBe(1); expect(f.created).toBe(0); await gateway.close();
});

test("exclusive mutations share the scheduler lock and release aborted waiters", async () => {
  const f = fake(); const gateway = new GenerationGateway(f.binding, 1);
  const held = await gateway.acquireExecutionLease(); const abort = new AbortController();
  const waiter = gateway.acquireExecutionLease(abort.signal); abort.abort(new Error("cancel lease"));
  await expect(waiter).rejects.toThrow("cancel lease");
  expect(gateway.busy).toBe(true); held.dispose(); await gateway.onIdle();
  expect(gateway.busy).toBe(false); await gateway.close();
});


// Importing the real graph/binding loads MLX, even though placement creates no
// tensors. Keep this qualification check out of the native-blocked CPU suite.
test.skipIf(process.env.MLX_BUN_GEMMA2_NATIVE !== "1")("Gemma2 admits grammar, adapter, fill and n-gram draft requests under every KV scheme its composition names", async () => {
  const { UniversalDenseModel } = await import("@mlx-bun/inference/models/universal");
  const { bindMlxGateway } = await import("@mlx-bun/inference/execution");
  const { KVCache } = await import("@mlx-bun/inference/state");
  /** The graph composed with `kv`, the scheme every request it serves stores (bf16 unless named). */
  const graph = (kv: KvStorageScheme = { kind: "bf16" }) => Object.assign(Object.create(UniversalDenseModel.prototype), {
    args: { modelType: "gemma2", maskArray: true, attnLogitSoftcap: 50, layerTypes: null }, encodedKvAttention: false,
    // The text facts KV scheme policy reads; this one-layer graph uses full attention.
    config: { modelType: "gemma2", text: { enableMoeBlock: false, numHiddenLayers: 1, layerTypes: ["full_attention"] }, eosTokenIds: [] },
    makeCache: () => [new KVCache()], loraState: { active: [] }, composition: { kv },
    // The token phases a speculative binding reads; placement never forwards.
    ...Object.fromEntries(["prefillChunk", "prefillTail", "decode", "verify", "appendKnown", "logits"]
      .map(name => [name, () => { throw new Error("placement never forwards"); }])),
  });
  const model = graph();
  const binding = bindMlxGateway(model);
  const AFFINE: KvStorageScheme = { kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" };
  const TURBO: KvStorageScheme = { kind: "turbo", kBits: 8, vBits: 3 };
  let created = false;
  binding.createBatchGroup = () => { created = true; throw new Error("unexpected execution"); };
  const gateway = new GenerationGateway(binding, 4);
  try {
    expect(gateway.place(shape())).toMatchObject({ mechanism: "continuous",
      execution: { method: "autoregressive", compiledDecode: false, fill: false, checkpoint: false } });
    for (const request of [{ hasGrammar: true }, { hasAdapters: true }, { hasGrammar: true, hasAdapters: true }])
      expect(gateway.place({ ...shape(), ...request })).toMatchObject({ mechanism: "continuous",
        execution: { method: "autoregressive", compiledDecode: false, fill: false, checkpoint: false, grammarJump: false } });
    expect(binding.bindAdapterContext).toBeFunction();
    // A draft no provider can serve is refused; logprobs and adapter requests
    // ignore a configured draft on this graph as on every other, as main did.
    for (const request of [{ hasDraft: true }, { hasVision: true }, { hasGrammar: true, hasVision: true }])
      expect(() => gateway.place({ ...shape(), ...request })).toThrow(UnsupportedExecutionError);
    // The composition's KV scheme is the cache lego's: plain, grammar and fill
    // requests place the same way over affine and TurboQuant storage.
    for (const kv of [AFFINE, TURBO]) {
      const composed = bindMlxGateway(graph(kv));
      composed.createBatchGroup = binding.createBatchGroup;
      const storing = new GenerationGateway(composed, 4);
      try {
        for (const request of [{}, { hasGrammar: true }, { hasAdapters: true }])
          expect(storing.place({ ...shape(), ...request })).toMatchObject({ mechanism: "continuous",
            execution: { method: "autoregressive", fill: false } });
        expect(storing.place(shape(), { fill: { plan: { echo: null } } } as GenerateOptions)).toMatchObject({ mechanism: "continuous",
          execution: { method: "autoregressive", fill: true } });
      } finally { await storing.close(); }
    }
    for (const request of [{ hasDraft: true, wantsLogprobs: true }, { hasAdapters: true, hasDraft: true }])
      expect(gateway.place({ ...shape(), ...request })).toMatchObject({ mechanism: "continuous", execution: { method: "autoregressive" } });
    const fill = { fill: { plan: { echo: null } } } as GenerateOptions;
    expect(gateway.place(shape(), fill)).toMatchObject({ mechanism: "continuous",
      execution: { method: "autoregressive", compiledDecode: false, fill: true, checkpoint: false } });
    expect(gateway.place({ ...shape(), hasAdapters: true }, { ...fill, adapters: ["upper"] })).toMatchObject({ mechanism: "continuous",
      execution: { method: "autoregressive", compiledDecode: false, fill: true, checkpoint: false } });
    expect(() => gateway.place({ ...shape(), hasDraft: true }, fill)).toThrow(UnsupportedExecutionError);
    // A served n-gram draft speculates on the shared scheduler; as in main,
    // logprobs and adapter requests ignore it, and a drafted request ignores fill.
    const { NgramProvider } = await import("@mlx-bun/inference/generation/speculative");
    const drafted = bindMlxGateway(model, { provider: new NgramProvider(), numDraftTokens: 10 });
    drafted.createBatchGroup = binding.createBatchGroup;
    const speculating = new GenerationGateway(drafted, 4);
    try {
      expect(speculating.place({ ...shape(), hasDraft: true })).toMatchObject({ mechanism: "continuous",
        execution: { method: "speculative", fill: false, checkpoint: false } });
      expect(speculating.place({ ...shape(), hasDraft: true, hasGrammar: true })).toMatchObject({ mechanism: "continuous",
        execution: { method: "speculative", grammarJump: false } });
      expect(speculating.place({ ...shape(), hasDraft: true, wantsLogprobs: true })).toMatchObject({ mechanism: "continuous",
        execution: { method: "autoregressive" } });
      expect(speculating.place({ ...shape(), hasDraft: true, hasAdapters: true })).toMatchObject({ mechanism: "continuous",
        execution: { method: "autoregressive" } });
      expect(speculating.place({ ...shape(), hasDraft: true }, fill)).toMatchObject({ mechanism: "continuous",
        execution: { method: "speculative", fill: false } });
    } finally { await speculating.close(); }
    expect(created).toBe(false);
  } finally { await gateway.close(); }
});

test.skipIf(process.env.MLX_BUN_GEMMA2_NATIVE !== "1")("Gemma2 places shared generation checkpoints only when its cache service keeps them", async () => {
  const { UniversalDenseModel } = await import("@mlx-bun/inference/models/universal");
  const { bindMlxGateway } = await import("@mlx-bun/inference/execution");
  const { KVCache } = await import("@mlx-bun/inference/state");
  const model = Object.assign(Object.create(UniversalDenseModel.prototype), {
    args: { modelType: "gemma2", maskArray: true, attnLogitSoftcap: 50, layerTypes: null }, encodedKvAttention: false,
    config: { modelType: "gemma2", text: { enableMoeBlock: false }, eosTokenIds: [] },
    makeCache: () => [new KVCache()], loraState: { active: [] },
  });
  const binding = bindMlxGateway(model);
  binding.createBatchGroup = () => { throw new Error("unexpected execution"); };
  // serve's cache service keeps checkpoints only with an SSD store and --generation-checkpoint.
  const gateways = [new GenerationGateway(binding, 4, { cache: cacheOf({ checkpointEveryTokens: 4 }) }),
    new GenerationGateway(binding, 4, { cache: cacheOf({}) }), new GenerationGateway(binding, 4)];
  const [enabled, ramOnly, disabled] = gateways;
  try {
    expect(ramOnly!.place(shape()).execution).toMatchObject({ mechanism: "continuous", checkpoint: false });
    for (const request of [{}, { userSeed: true, hasRepetitionPenalty: true }])
      expect(enabled!.place({ ...shape(), ...request })).toMatchObject({ mechanism: "continuous",
        execution: { method: "autoregressive", promptCache: true, checkpoint: true, fill: false } });
    expect(disabled!.place(shape()).execution.checkpoint).toBe(false);
    const fill = { fill: { plan: { echo: null } } } as GenerateOptions;
    for (const [request, options] of [[{ hasGrammar: true }, {}], [{ wantsLogprobs: true }, {}], [{}, fill]] as const)
      expect(enabled!.place({ ...shape(), ...request }, options)).toMatchObject({ mechanism: "continuous",
        execution: { checkpoint: false } });
    for (const request of [{ hasDraft: true }, { hasVision: true }])
      expect(() => enabled!.place({ ...shape(), ...request })).toThrow(UnsupportedExecutionError);
    // A composed KV scheme checkpoints as bf16 does: its lego owns the state's codec.
    for (const kv of [{ kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" }, { kind: "turbo", kBits: 8, vBits: 3 }] as const) {
      const composed = bindMlxGateway(Object.assign(Object.create(UniversalDenseModel.prototype), { ...model, composition: { kv } }));
      composed.createBatchGroup = () => { throw new Error("unexpected execution"); };
      const storing = new GenerationGateway(composed, 4, { cache: cacheOf({ checkpointEveryTokens: 4 }) });
      try {
        expect(storing.place(shape())).toMatchObject({ mechanism: "continuous",
          execution: { method: "autoregressive", promptCache: true, checkpoint: true } });
      } finally { await storing.close(); }
    }
  } finally { for (const gateway of gateways) await gateway.close(); }
});

/** A binding with no method/continuation hooks, so ordinary rows sample through the step sampler. */
function ordinaryFake(submit: MlxBatchGroup["submit"]) {
  const f = fake({ submit });
  delete (f.binding as { continuationRequest?: unknown }).continuationRequest;
  return f;
}

test("a supplied sampler factory builds the ordinary row's sampler and is disposed with the row", async () => {
  const calls: { options: GenerateOptions; config: DeviceStepSamplerConfig }[] = [];
  let disposed = 0, sampled = 0, plainGreedy: boolean | undefined;
  const supplied = { isPlainGreedy: false, capturesLogprobs: false, needsHistory: false,
    sample() { sampled++; return { token: "supplied-token", extras: null }; },
    seedHistory() {}, commitDevice() {}, commitNumbers() {}, dispose() { disposed++; } } as unknown as DeviceStepSampler;
  const f = ordinaryFake(async request => {
    plainGreedy = request.plainGreedy;
    expect(request.sample!(null as never, 0)).toBe("supplied-token" as never);
    await request.onToken(7); return result;
  });
  const gateway = new GenerationGateway(f.binding, 1, { samplerFactory: (options, config) => {
    calls.push({ options, config }); return supplied;
  } });
  const s = shape(), options: GenerateOptions = { maxTokens: 1, temperature: 0, logprobs: true, topLogprobs: 2 };
  await gateway.run([5, 6], options, () => {}, undefined, s, gateway.place(s));
  expect(calls).toHaveLength(1); expect(calls[0]!.options).toBe(options);
  expect(calls[0]!.config).toMatchObject({ tokenRepresentation: "device", grammarWait: "external",
    historyUpdate: "after-sample", initialHistory: [5, 6], captureSelectedLogprob: true, captureTopLogprobs: 2 });
  expect(sampled).toBe(1); expect(plainGreedy).toBe(false); expect(disposed).toBe(1);
  await gateway.close();
});

test("without a sampler factory ordinary rows use makeStepSampler", async () => {
  let plainGreedy: boolean | undefined, sampler: unknown;
  const f = ordinaryFake(async request => { plainGreedy = request.plainGreedy; sampler = request.sample; return result; });
  const gateway = new GenerationGateway(f.binding, 1);
  const s = shape();
  await gateway.run([1], { maxTokens: 1, temperature: 0 }, () => {}, undefined, s, gateway.place(s));
  expect(typeof sampler).toBe("function");
  expect(plainGreedy).toBe(true); // makeStepSampler reports plain greedy for temperature 0 without processors
  await gateway.close();
});
