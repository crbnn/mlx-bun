import { describe, expect, test } from "bun:test";
import { resolveExecution } from "../../src/execution/plan";
import type { ExecutionCapabilities, ExecutionRequirements } from "../../src/contracts/portable/execution";

const request: ExecutionRequirements = {
  hasVision: false, hasAdapters: false, hasRepetitionPenalty: false, userSeed: false,
  hasLogitsExtras: false, hasGrammar: false,
  wantsLogprobs: false, hasDraft: false,
};
// The composition's KV scheme is an executor fact every request stores with.
const affineKv = { kvQuant: true } as const, turboKv = { turboQuant: true } as const;
const capabilities: ExecutionCapabilities = {
  method: "autoregressive", continuous: true, quantizedBatch: true,
  grammarBatch: true, checkpoints: true,
};

test("logprobs compose with continuous ordinary decoding", () => {
  const plan = resolveExecution({ ...request, wantsLogprobs: true }, capabilities);
  expect(plan).toMatchObject({ method: "autoregressive", mechanism: "continuous" });
  expect(plan.reasons).toEqual([]);
});

test("prepared media uses ordinary shared decode without token-only reuse or speculative proposals", () => {
  const supported = { ...capabilities, mediaBatch: true, sharedGrammarProposals: true,
    sharedFill: true, sharedCheckpoints: true, groupedMethods: ["autoregressive", "speculative"] };
  for (const options of [{}, { hasDraft: true }, { hasGrammar: true }, { wantsLogprobs: true }]) {
    const plan = resolveExecution({ ...request, hasVision: true, ...options }, supported,
      { fill: true, grammarJump: true });
    expect(plan).toMatchObject({ mechanism: "continuous", method: "autoregressive",
      promptCache: false, checkpoint: false, fill: false, grammarJump: false });
  }
  expect(resolveExecution({ ...request, hasVision: true }, capabilities).mechanism).toBe("unsupported");
});

test("prepared-prefix identity enables qualified shared reuse independently of media decode", () => {
  const media = { ...request, hasVision: true, hasPreparedPrefixIdentity: true };
  const supported = { ...capabilities, mediaBatch: true, mediaPrefixCache: true, sharedCheckpoints: true };
  expect(resolveExecution(media, supported)).toMatchObject({ mechanism: "continuous", promptCache: true, checkpoint: false });
  expect(resolveExecution({ ...media, hasPreparedPrefixIdentity: false }, supported).promptCache).toBe(false);
  expect(resolveExecution(media, { ...supported, mediaPrefixCache: false }).promptCache).toBe(false);
  expect(resolveExecution(media, { ...supported, continuous: false })).toMatchObject({ mechanism: "unsupported", promptCache: false });
});

test("an explicit seed composes with ordinary logprobs and grammar in continuous execution", () => {
  const plan = resolveExecution({ ...request, userSeed: true, wantsLogprobs: true, hasGrammar: true }, capabilities);
  expect(plan).toMatchObject({ method: "autoregressive", mechanism: "continuous" });
  expect(plan.reasons).toEqual([]);
});

test("with grammar coordination and no optional features, grammar never changes placement", () => {
  const variants: Partial<ExecutionCapabilities>[] = [{}, { continuous: false }, { quantizedBatch: false },
    { mediaBatch: true }, { adapterBatch: true }, { turboQuantBatch: true }, { method: "denoising" },
    { groupedMethods: ["autoregressive", "speculative"], speculativeLogprobs: true }];
  for (const variant of variants) for (const kv of [{}, affineKv, turboKv]) for (const extra of [{}, { hasVision: true },
    { hasAdapters: true }, { wantsLogprobs: true }, { userSeed: true }, { hasDraft: true }]) {
    const supported = { ...capabilities, ...variant, ...kv };
    expect({ variant, kv, extra, plan: resolveExecution({ ...request, ...extra, hasGrammar: true }, supported) })
      .toEqual({ variant, kv, extra, plan: resolveExecution({ ...request, ...extra }, supported) });
  }
});

test.each(["hasVision", "hasAdapters", "wantsLogprobs", "kvQuant", "turboQuant"] as const)(
  "%s uses AR and retains an explicit draft fallback reason", (key) => {
    const scheme = key === "kvQuant" || key === "turboQuant";
    const plan = resolveExecution({ ...request, hasDraft: true, ...(scheme ? {} : { [key]: true }) },
      scheme ? { ...capabilities, [key]: true } : capabilities);
    expect(plan.method).toBe("autoregressive");
    expect(plan.mechanism).toBe(key === "wantsLogprobs" || key === "kvQuant" ? "continuous" : "unsupported");
    expect(plan.reasons).toContain("draft-incompatible-with-request");
  },
);

test("grammar and an explicit seed select speculation, which needs a grouped verifier", () => {
  const plan = resolveExecution({ ...request, hasDraft: true, hasGrammar: true, userSeed: true }, capabilities);
  expect(plan).toMatchObject({ method: "speculative", mechanism: "unsupported", promptCache: false, checkpoint: false });
  expect(plan.reasons).toEqual(["method-batch-unsupported"]);
});

test("qualified speculative KV without a grouped verifier is unsupported; other guards keep AR", () => {
  const qualified = { ...capabilities, ...affineKv, speculativeKvQuant: true };
  const draft = { ...request, hasDraft: true };
  expect(resolveExecution(draft, qualified)).toMatchObject({
    method: "speculative", mechanism: "unsupported", promptCache: false, checkpoint: false,
  });
  for (const key of ["hasVision", "hasAdapters", "wantsLogprobs"] as const)
    expect(resolveExecution({ ...draft, [key]: true }, qualified).method).toBe("autoregressive");
  expect(resolveExecution(draft, { ...qualified, ...turboKv }).method).toBe("autoregressive");
});

test("supported affine KV batches; unavailable layouts and TurboQuant are unsupported", () => {
  expect(resolveExecution(request, { ...capabilities, ...affineKv }).mechanism).toBe("continuous");
  expect(resolveExecution(request, { ...capabilities, ...affineKv, quantizedBatch: false }).mechanism).toBe("unsupported");
  expect(resolveExecution(request, { ...capabilities, ...turboKv }).mechanism).toBe("unsupported");
});

test("fill requires a qualified shared binding and cannot run in another method or resumable checkpoint", () => {
  const features = { fill: true };
  expect(resolveExecution(request, capabilities, features).fill).toBe(false);
  for (const kv of [{}, affineKv, turboKv]) {
    const plan = resolveExecution(request, { ...capabilities, ...kv, continuous: false }, features);
    expect(plan).toMatchObject({ mechanism: "unsupported", fill: false, checkpoint: false });
    expect(plan.reasons).toContain("fill-incompatible-with-request");
  }
  expect(resolveExecution({ ...request, hasDraft: true }, capabilities, features).fill).toBe(false);
  const denoising = resolveExecution(request, { ...capabilities, method: "denoising" }, features);
  expect(denoising).toMatchObject({ method: "denoising", mechanism: "unsupported", fill: false, checkpoint: false });
  expect(Object.isFrozen(denoising)).toBe(true);
  expect(Object.isFrozen(denoising.reasons)).toBe(true);
});

test("qualified shared fill retains seeded sampling and excludes grammar and metadata", () => {
  const supported = { ...capabilities, sharedFill: true, turboQuantBatch: true, compiledDecode: true };
  const features = { fill: true, compiledDecode: true };
  for (const [shape, kv] of [[request, {}], [{ ...request, userSeed: true }, {}], [request, affineKv],
    [request, turboKv]] as const) {
    expect(resolveExecution(shape, { ...supported, ...kv }, features)).toMatchObject({
      method: "autoregressive", mechanism: "continuous", fill: true,
      promptCache: true, checkpoint: false, compiledDecode: false,
    });
  }
  for (const extra of [{ hasGrammar: true }, { wantsLogprobs: true }, { hasDraft: true }])
    expect(resolveExecution({ ...request, ...extra }, supported, features).fill).toBe(false);
  expect(resolveExecution({ ...request, userSeed: true }, { ...supported, continuous: false }, features).fill).toBe(false);
});

test("compiled replay permission is fixed by graph capability and request composition", () => {
  const features = { fill: false, compiledDecode: true };
  const supported = { ...capabilities, compiledDecode: true };
  expect(resolveExecution(request, supported, features).compiledDecode).toBe(true);
  expect(resolveExecution(request, capabilities, features).compiledDecode).toBe(false);
  expect(resolveExecution(request, supported).compiledDecode).toBe(false);
  for (const incompatible of [{ hasAdapters: true }, { hasDraft: true }]) {
    const plan = resolveExecution({ ...request, ...incompatible }, supported, features);
    expect(plan.compiledDecode).toBe(false);
    expect(plan.reasons).toContain("compiled-decode-unavailable-for-request");
  }
  // Media uses the graph's ordinary decode path, compiled where the composition allows it.
  expect(resolveExecution({ ...request, hasVision: true }, { ...supported, mediaBatch: true }, features).compiledDecode).toBe(true);
});

test("grammar jump requires a qualified shared method, never an unsupported plan", () => {
  const features = { fill: false, grammarJump: true };
  const grammar = { ...request, hasGrammar: true };
  const unavailable = { ...capabilities, continuous: false };
  const rejected = resolveExecution(grammar, unavailable, features);
  expect(rejected).toMatchObject({ mechanism: "unsupported", grammarJump: false });
  expect(rejected.reasons).toEqual(["continuous-unavailable", "grammar-jump-incompatible-with-request"]);
  expect(resolveExecution(grammar, capabilities, features).grammarJump).toBe(false);
  expect(resolveExecution(request, unavailable, features).grammarJump).toBe(false);
  expect(resolveExecution(grammar, { ...unavailable, method: "denoising" }, features).grammarJump).toBe(false);
});


test("adapter-capable groups retain ordinary sampling features", () => {
  const plan = resolveExecution({ ...request, hasAdapters: true, wantsLogprobs: true,
    userSeed: true }, { ...capabilities, adapterBatch: true }, { fill: false, compiledDecode: true });
  expect(plan).toMatchObject({ method: "autoregressive", mechanism: "continuous", compiledDecode: false });
});

test("registered grouped methods compose speculation with sampling and KV storage", () => {
  const composed = { ...capabilities, ...affineKv, groupedMethods: ["autoregressive", "speculative"],
    speculativeLogprobs: true, speculativeKvQuant: true };
  const plan = resolveExecution({ ...request, hasDraft: true, wantsLogprobs: true,
    userSeed: true, hasGrammar: true }, composed);
  expect(plan).toMatchObject({ method: "speculative", mechanism: "continuous" });
  expect(plan.reasons).not.toContain("method-batch-unsupported");
  expect(plan.reasons).not.toContain("draft-incompatible-with-request");
});


test("TurboQuant group capability composes with ordinary and speculative methods", () => {
  const turbo = request;
  const supported = { ...capabilities, ...turboKv, turboQuantBatch: true };
  expect(resolveExecution(turbo, supported).mechanism).toBe("continuous");
  const mtp = resolveExecution({ ...turbo, hasDraft: true }, { ...supported,
    speculativeTurboQuant: true, groupedMethods: ["autoregressive", "speculative"] });
  expect(mtp.method).toBe("speculative");
  expect(mtp.mechanism).toBe("continuous");
  expect(resolveExecution({ ...turbo, hasDraft: true }, supported).method).toBe("autoregressive");
});

test("qualified target-adapter speculation stays in the shared adapter context", () => {
  const adapted = { ...request, hasDraft: true, hasAdapters: true, wantsLogprobs: true, userSeed: true, hasGrammar: true };
  const supported = { ...capabilities, adapterBatch: true, sharedSpeculativeAdapters: true,
    speculativeLogprobs: true, groupedMethods: ["autoregressive", "speculative"] };
  expect(resolveExecution(adapted, supported)).toMatchObject({ method: "speculative", mechanism: "continuous" });
  expect(resolveExecution(adapted, { ...supported, sharedSpeculativeAdapters: false }).method).toBe("autoregressive");
  for (const disabled of [{ continuous: false }, { adapterBatch: false }, { grammarBatch: false }])
    expect(resolveExecution(adapted, { ...supported, ...disabled })).toMatchObject({ method: "autoregressive", mechanism: "unsupported" });
  expect(resolveExecution({ ...adapted, hasVision: true }, supported).method).toBe("autoregressive");
});

test("bound ordinary checkpoint capability qualifies shared requests without changing other methods", () => {
  const supported = { ...capabilities, checkpoints: true, sharedCheckpoints: true };
  expect(resolveExecution(request, supported)).toMatchObject({ mechanism: "continuous", checkpoint: true });
  expect(resolveExecution(request, { ...supported, sharedCheckpoints: false }).checkpoint).toBe(false);
  for (const excluded of [{ hasGrammar: true }, { wantsLogprobs: true }, { hasVision: true }])
    expect(resolveExecution({ ...request, ...excluded }, supported).checkpoint).toBe(false);
  expect(resolveExecution({ ...request, hasDraft: true }, { ...supported,
    speculativeLogprobs: true, groupedMethods: ["autoregressive", "speculative"] }).checkpoint).toBe(false);
  expect(resolveExecution(request, supported, { fill: true }).checkpoint).toBe(false);
});

test("grammar proposals select the shared verifier only when its request combination is supported", () => {
  const grammar = { ...request, hasGrammar: true };
  const features = { fill: false, grammarJump: true };
  const supported = { ...capabilities, sharedGrammarProposals: true,
    groupedMethods: ["autoregressive", "speculative"] };
  expect(resolveExecution(grammar, supported, features)).toMatchObject({
    method: "speculative", mechanism: "continuous", grammarJump: true,
  });
  expect(resolveExecution({ ...grammar, wantsLogprobs: true }, { ...supported, speculativeLogprobs: true }, features))
    .toMatchObject({ method: "speculative", mechanism: "continuous", grammarJump: true });
  expect(resolveExecution(grammar, supported).method).toBe("autoregressive");
  for (const option of [{ wantsLogprobs: true }, { hasAdapters: true }])
    expect(resolveExecution({ ...grammar, ...option }, supported, features).method).toBe("autoregressive");
  for (const kv of [affineKv, turboKv])
    expect(resolveExecution(grammar, { ...supported, ...kv }, features).method).toBe("autoregressive");
  expect(resolveExecution(grammar, { ...supported, ...affineKv, speculativeKvQuant: true }, features).grammarJump).toBe(true);
  expect(resolveExecution(grammar, { ...supported, ...turboKv,
    turboQuantBatch: true, speculativeTurboQuant: true }, features).grammarJump).toBe(true);
  expect(resolveExecution(grammar, { ...supported, groupedMethods: ["autoregressive"] }, features).method).toBe("autoregressive");
  expect(resolveExecution({ ...grammar, hasDraft: true }, supported, features)).toMatchObject({
    method: "speculative", grammarJump: false,
  });
});

test("a provider consuming external tokens composes echo with shared speculation", () => {
  const shape = { ...request, hasDraft: true, userSeed: true };
  const supported = { ...capabilities, ...affineKv, groupedMethods: ["autoregressive", "speculative"],
    speculativeKvQuant: true, sharedSpeculativeEcho: true };
  const features = { fill: true };
  expect(resolveExecution(shape, supported, features)).toMatchObject({
    method: "speculative", mechanism: "continuous", fill: true });
  for (const disabled of [{ sharedSpeculativeEcho: false }, { continuous: false }])
    expect(resolveExecution(shape, { ...supported, ...disabled }, features).fill).toBe(false);
  for (const extra of [{ hasVision: true }, { hasGrammar: true }, { wantsLogprobs: true }])
    expect(resolveExecution({ ...shape, ...extra }, supported, features).fill).toBe(false);
});

describe("interleaved denoising placement", () => {
  const denoising: ExecutionCapabilities = { method: "denoising", continuous: true, quantizedBatch: false,
    grammarBatch: false, checkpoints: true, adapterBatch: true, sharedCheckpoints: true,
    groupedMethods: ["denoising"] };
  const features = { fill: false, compiledDecode: true, grammarJump: false };

  test("plain, seeded and adapted requests run continuously with AR-only reuse disabled", () => {
    for (const extra of [{}, { userSeed: true }, { hasAdapters: true }, { userSeed: true, hasAdapters: true }])
      expect(resolveExecution({ ...request, ...extra }, denoising, features)).toEqual({
        method: "denoising", mechanism: "continuous", promptCache: false, checkpoint: false,
        fill: false, compiledDecode: false, grammarJump: false, reasons: ["compiled-decode-unavailable-for-request"] });
  });

  test.each([
    [{ hasDraft: true }, {}, "draft-method-unsupported", {}],
    [{ hasGrammar: true }, {}, "grammar-batch-unsupported", {}],
    [{ wantsLogprobs: true }, {}, "logprobs-method-unsupported", {}],
    [{ hasRepetitionPenalty: true }, {}, "repetition-penalty-method-unsupported", {}],
    [{ hasLogitsExtras: true }, {}, "logits-extras-method-unsupported", {}],
    [{}, { fill: true }, "fill-method-unsupported", {}],
    [{}, {}, "kv-scheme-batch-unsupported", affineKv],
    [{}, {}, "turbo-kv-batch-unsupported", turboKv],
    [{ hasVision: true }, {}, "media-batch-unsupported", {}],
  ] as const)("a token-level or AR-state request is refused, never discarded: %#", (shape, feature, reason, kv: Partial<ExecutionCapabilities>) => {
    const plan = resolveExecution({ ...request, ...shape }, { ...denoising, ...kv }, { ...features, ...feature });
    expect(plan).toMatchObject({ method: "denoising", mechanism: "unsupported", promptCache: false,
      checkpoint: false, fill: false, grammarJump: false });
    expect(plan.reasons.filter(entry => entry.endsWith("-unsupported"))).toEqual([reason]);
    expect(plan.reasons).not.toContain("draft-incompatible-with-request");
    expect(plan.reasons).not.toContain("fill-incompatible-with-request");
  });

  test("grammar jump cannot select speculation for a denoising request", () => {
    const plan = resolveExecution({ ...request, hasGrammar: true }, { ...denoising, sharedGrammarProposals: true,
      groupedMethods: ["denoising", "speculative"] }, { ...features, grammarJump: true });
    expect(plan).toMatchObject({ method: "denoising", mechanism: "unsupported", grammarJump: false });
    expect(plan.reasons).toContain("grammar-batch-unsupported");
  });
});

test("accepted plans keep their selected features", () => {
  const supported = { ...capabilities, sharedCheckpoints: true, sharedFill: true, compiledDecode: true,
    sharedGrammarProposals: true, groupedMethods: ["autoregressive", "speculative"] };
  expect(resolveExecution(request, supported, { fill: false, compiledDecode: true })).toEqual({
    method: "autoregressive", mechanism: "continuous", promptCache: true, checkpoint: true,
    fill: false, compiledDecode: true, grammarJump: false, reasons: [] });
  expect(resolveExecution({ ...request, userSeed: true }, supported, { fill: true, compiledDecode: true })).toEqual({
    method: "autoregressive", mechanism: "continuous", promptCache: true, checkpoint: false,
    fill: true, compiledDecode: false, grammarJump: false, reasons: ["compiled-decode-unavailable-for-request"] });
  expect(resolveExecution({ ...request, hasGrammar: true }, supported, { fill: false, grammarJump: true })).toEqual({
    method: "speculative", mechanism: "continuous", promptCache: false, checkpoint: false,
    fill: false, compiledDecode: false, grammarJump: true, reasons: [] });
});

test.each([
  [{}, { continuous: false }, {}, "continuous-unavailable"],
  [{ hasVision: true }, {}, {}, "media-batch-unsupported"],
  [{ hasAdapters: true }, {}, {}, "adapter-batch-unsupported"],
  [{}, { ...affineKv, quantizedBatch: false }, {}, "kv-scheme-batch-unsupported"],
  [{}, turboKv, {}, "turbo-kv-batch-unsupported"],
  [{ hasGrammar: true }, { grammarBatch: false }, {}, "grammar-batch-unsupported"],
  [{}, { method: "denoising" }, {}, "method-batch-unsupported"],
] as const)("an unsupported plan names its rejection reason: %#", (shape, capability, feature, reason) => {
  const plan = resolveExecution({ ...request, ...shape }, { ...capabilities, ...capability },
    { fill: false, ...feature });
  expect(plan).toMatchObject({ mechanism: "unsupported", fill: false, grammarJump: false });
  expect(plan.reasons[0]).toBe(reason);
  expect(plan.reasons.filter(entry => entry.endsWith("-unsupported") || entry === "continuous-unavailable")).toEqual([reason]);
});


test("committed grammar spans stay AR and do not replace verified grammar proposals", () => {
  const features = { fill: true, grammarJump: true };
  const grammar = { ...request, hasGrammar: true };
  const spans = { ...capabilities, adapterBatch: true, sharedGrammarJump: true };
  expect(resolveExecution(grammar, spans, features)).toMatchObject({
    method: "autoregressive", mechanism: "continuous", grammarJump: true,
    checkpoint: false, promptCache: true, fill: false,
  });
  // An ignored draft is not an additional grammar restriction: the final
  // selected ordinary method is what main's direct grammar policy consumed.
  expect(resolveExecution({ ...grammar, hasAdapters: true, hasDraft: true }, spans, features))
    .toMatchObject({ method: "autoregressive", mechanism: "continuous", grammarJump: true });
  for (const [req, cap] of [[{ ...grammar, wantsLogprobs: true }, spans],
    [grammar, { ...spans, continuous: false }], [request, spans],
    [grammar, { ...spans, sharedGrammarJump: false }]] as const)
    expect(resolveExecution(req, cap, features).grammarJump).toBe(false);
  const proposals = { ...spans, sharedGrammarProposals: true,
    groupedMethods: ["autoregressive", "speculative"] };
  expect(resolveExecution(grammar, proposals, features))
    .toMatchObject({ method: "speculative", mechanism: "continuous", grammarJump: true });
  expect(resolveExecution({ ...grammar, hasDraft: true }, proposals, features))
    .toMatchObject({ method: "speculative", mechanism: "continuous", grammarJump: false });
});
