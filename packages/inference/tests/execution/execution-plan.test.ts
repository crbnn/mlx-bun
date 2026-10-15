import { describe, expect, test } from "bun:test";
import { resolveExecution } from "../../src/execution/plan";
import type { ExecutionCapabilities, ExecutionRequirements } from "../../src/contracts/portable/execution";

// The planner reads requests and what the graph and its bound methods declare.
// How the composition's caches store keys and values is not an input: every
// cache lego serves every method.
const request: ExecutionRequirements = {
  hasVision: false, hasAdapters: false, hasRepetitionPenalty: false, userSeed: false,
  hasLogitsExtras: false, hasGrammar: false,
  wantsLogprobs: false, hasDraft: false,
};
const capabilities: ExecutionCapabilities = {
  method: "autoregressive", grammarBatch: true, checkpoints: true,
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
  expect(resolveExecution(media, { ...supported, mediaBatch: false })).toMatchObject({ mechanism: "unsupported", promptCache: false });
});

test("an explicit seed composes with ordinary logprobs and grammar in continuous execution", () => {
  const plan = resolveExecution({ ...request, userSeed: true, wantsLogprobs: true, hasGrammar: true }, capabilities);
  expect(plan).toMatchObject({ method: "autoregressive", mechanism: "continuous" });
  expect(plan.reasons).toEqual([]);
});

test("with grammar coordination and no optional features, grammar never changes placement", () => {
  const variants: Partial<ExecutionCapabilities>[] = [{}, { mediaBatch: true }, { method: "denoising" },
    { groupedMethods: ["autoregressive", "speculative"], speculativeLogprobs: true }];
  for (const variant of variants) for (const extra of [{}, { hasVision: true },
    { hasAdapters: true }, { wantsLogprobs: true }, { userSeed: true }, { hasDraft: true }]) {
    const supported = { ...capabilities, ...variant };
    expect({ variant, extra, plan: resolveExecution({ ...request, ...extra, hasGrammar: true }, supported) })
      .toEqual({ variant, extra, plan: resolveExecution({ ...request, ...extra }, supported) });
  }
});

test.each(["hasVision", "hasAdapters", "wantsLogprobs"] as const)(
  "%s uses AR and retains an explicit draft fallback reason", (key) => {
    const plan = resolveExecution({ ...request, hasDraft: true, [key]: true }, capabilities);
    expect(plan.method).toBe("autoregressive");
    expect(plan.mechanism).toBe(key === "hasVision" ? "unsupported" : "continuous");
    expect(plan.reasons).toContain("draft-incompatible-with-request");
  },
);

test("grammar and an explicit seed select speculation, which needs a grouped verifier", () => {
  const plan = resolveExecution({ ...request, hasDraft: true, hasGrammar: true, userSeed: true }, capabilities);
  expect(plan).toMatchObject({ method: "speculative", mechanism: "unsupported", promptCache: false, checkpoint: false });
  expect(plan.reasons).toEqual(["method-batch-unsupported"]);
});

test("a draft without a grouped verifier is refused, never ignored; media, adapters and logprobs keep AR", () => {
  const draft = { ...request, hasDraft: true };
  expect(resolveExecution(draft, capabilities)).toMatchObject({
    method: "speculative", mechanism: "unsupported", promptCache: false, checkpoint: false,
  });
  for (const key of ["hasVision", "hasAdapters", "wantsLogprobs"] as const)
    expect(resolveExecution({ ...draft, [key]: true }, capabilities).method).toBe("autoregressive");
  const grouped = { ...capabilities, groupedMethods: ["autoregressive", "speculative"] };
  expect(resolveExecution(draft, grouped)).toMatchObject({ method: "speculative", mechanism: "continuous" });
});

test("fill requires a qualified shared binding and cannot run in another method or resumable checkpoint", () => {
  const features = { fill: true };
  expect(resolveExecution(request, capabilities, features).fill).toBe(false);
  const unbound = resolveExecution(request, { ...capabilities }, features);
  expect(unbound).toMatchObject({ mechanism: "continuous", fill: false, checkpoint: false });
  expect(unbound.reasons).toContain("fill-incompatible-with-request");
  expect(resolveExecution({ ...request, hasDraft: true }, capabilities, features).fill).toBe(false);
  const denoising = resolveExecution(request, { ...capabilities, method: "denoising" }, features);
  expect(denoising).toMatchObject({ method: "denoising", mechanism: "unsupported", fill: false, checkpoint: false });
  expect(Object.isFrozen(denoising)).toBe(true);
  expect(Object.isFrozen(denoising.reasons)).toBe(true);
});

test("qualified shared fill retains seeded sampling and excludes grammar and metadata", () => {
  const supported = { ...capabilities, sharedFill: true, compiledDecode: true };
  const features = { fill: true, compiledDecode: true };
  for (const shape of [request, { ...request, userSeed: true }]) {
    expect(resolveExecution(shape, supported, features)).toMatchObject({
      method: "autoregressive", mechanism: "continuous", fill: true,
      promptCache: true, checkpoint: false, compiledDecode: false,
    });
  }
  for (const extra of [{ hasGrammar: true }, { wantsLogprobs: true }, { hasDraft: true }])
    expect(resolveExecution({ ...request, ...extra }, supported, features).fill).toBe(false);
  expect(resolveExecution({ ...request, userSeed: true }, { ...supported, method: "denoising" }, features).fill).toBe(false);
});

test("compiled replay permission is fixed by graph capability, the composition and the request", () => {
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
  const unavailable = { ...capabilities, grammarBatch: false };
  const rejected = resolveExecution(grammar, unavailable, features);
  expect(rejected).toMatchObject({ mechanism: "unsupported", grammarJump: false });
  expect(rejected.reasons).toEqual(["grammar-batch-unsupported", "grammar-jump-incompatible-with-request"]);
  expect(resolveExecution(grammar, capabilities, features).grammarJump).toBe(false);
  expect(resolveExecution(request, unavailable, features).grammarJump).toBe(false);
  expect(resolveExecution(grammar, { ...unavailable, method: "denoising" }, features).grammarJump).toBe(false);
});

test("adapter-capable groups retain ordinary sampling features", () => {
  const plan = resolveExecution({ ...request, hasAdapters: true, wantsLogprobs: true,
    userSeed: true }, { ...capabilities }, { fill: false, compiledDecode: true });
  expect(plan).toMatchObject({ method: "autoregressive", mechanism: "continuous", compiledDecode: false });
});

test("registered grouped methods compose speculation with sampling", () => {
  const composed = { ...capabilities, groupedMethods: ["autoregressive", "speculative"], speculativeLogprobs: true };
  const plan = resolveExecution({ ...request, hasDraft: true, wantsLogprobs: true,
    userSeed: true, hasGrammar: true }, composed);
  expect(plan).toMatchObject({ method: "speculative", mechanism: "continuous" });
  expect(plan.reasons).not.toContain("method-batch-unsupported");
  expect(plan.reasons).not.toContain("draft-incompatible-with-request");
});

test("qualified target-adapter speculation stays in the shared adapter context", () => {
  const adapted = { ...request, hasDraft: true, hasAdapters: true, wantsLogprobs: true, userSeed: true, hasGrammar: true };
  const supported = { ...capabilities, sharedSpeculativeAdapters: true,
    speculativeLogprobs: true, groupedMethods: ["autoregressive", "speculative"] };
  expect(resolveExecution(adapted, supported)).toMatchObject({ method: "speculative", mechanism: "continuous" });
  expect(resolveExecution(adapted, { ...supported, sharedSpeculativeAdapters: false }).method).toBe("autoregressive");
  for (const disabled of [{ grammarBatch: false }])
    expect(resolveExecution(adapted, { ...supported, ...disabled })).toMatchObject({ method: "autoregressive", mechanism: "unsupported" });
  expect(resolveExecution({ ...adapted, hasVision: true }, supported).method).toBe("autoregressive");
});

test("bound ordinary checkpoint capability qualifies shared requests without changing other methods", () => {
  const supported = { ...capabilities, checkpoints: true, sharedCheckpoints: true };
  expect(resolveExecution(request, supported)).toMatchObject({ mechanism: "continuous", checkpoint: true });
  expect(resolveExecution(request, { ...supported, sharedCheckpoints: false }).checkpoint).toBe(false);
  expect(resolveExecution(request, { ...supported, checkpoints: false }).checkpoint).toBe(false);
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
  expect(resolveExecution(grammar, { ...supported, groupedMethods: ["autoregressive"] }, features).method).toBe("autoregressive");
  expect(resolveExecution({ ...grammar, hasDraft: true }, supported, features)).toMatchObject({
    method: "speculative", grammarJump: false,
  });
});

test("a provider consuming external tokens composes echo with shared speculation", () => {
  const shape = { ...request, hasDraft: true, userSeed: true };
  const supported = { ...capabilities, groupedMethods: ["autoregressive", "speculative"], sharedSpeculativeEcho: true };
  const features = { fill: true };
  expect(resolveExecution(shape, supported, features)).toMatchObject({
    method: "speculative", mechanism: "continuous", fill: true });
  for (const disabled of [{ sharedSpeculativeEcho: false }, { groupedMethods: ["autoregressive"] }])
    expect(resolveExecution(shape, { ...supported, ...disabled }, features).fill).toBe(false);
  for (const extra of [{ hasVision: true }, { hasGrammar: true }, { wantsLogprobs: true }])
    expect(resolveExecution({ ...shape, ...extra }, supported, features).fill).toBe(false);
});

describe("interleaved denoising placement", () => {
  const denoising: ExecutionCapabilities = { method: "denoising",
    grammarBatch: false, checkpoints: true, sharedCheckpoints: true,
    groupedMethods: ["denoising"] };
  const features = { fill: false, compiledDecode: true, grammarJump: false };

  test("plain, seeded and adapted requests run continuously with AR-only reuse disabled", () => {
    for (const extra of [{}, { userSeed: true }, { hasAdapters: true }, { userSeed: true, hasAdapters: true }])
      expect(resolveExecution({ ...request, ...extra }, denoising, features)).toEqual({
        method: "denoising", mechanism: "continuous", promptCache: false, checkpoint: false,
        fill: false, compiledDecode: false, grammarJump: false, reasons: ["compiled-decode-unavailable-for-request"] });
  });

  test.each([
    [{ hasDraft: true }, {}, "draft-method-unsupported"],
    [{ hasGrammar: true }, {}, "grammar-batch-unsupported"],
    [{ wantsLogprobs: true }, {}, "logprobs-method-unsupported"],
    [{ hasRepetitionPenalty: true }, {}, "repetition-penalty-method-unsupported"],
    [{ hasLogitsExtras: true }, {}, "logits-extras-method-unsupported"],
    [{}, { fill: true }, "fill-method-unsupported"],
    [{ hasVision: true }, {}, "media-batch-unsupported"],
  ] as const)("a token-level or AR-state request is refused, never discarded: %#", (shape, feature, reason) => {
    const plan = resolveExecution({ ...request, ...shape }, denoising, { ...features, ...feature });
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
  [{ hasVision: true }, {}, "media-batch-unsupported"],
  [{ hasGrammar: true }, { grammarBatch: false }, "grammar-batch-unsupported"],
  [{}, { method: "denoising" }, "method-batch-unsupported"],
] as const)("an unsupported plan names its rejection reason: %#", (shape, capability, reason) => {
  const plan = resolveExecution({ ...request, ...shape }, { ...capabilities, ...capability }, { fill: false });
  expect(plan).toMatchObject({ mechanism: "unsupported", fill: false, grammarJump: false });
  expect(plan.reasons[0]).toBe(reason);
  expect(plan.reasons.filter(entry => entry.endsWith("-unsupported"))).toEqual([reason]);
});

test("committed grammar spans stay AR and do not replace verified grammar proposals", () => {
  const features = { fill: true, grammarJump: true };
  const grammar = { ...request, hasGrammar: true };
  const spans = { ...capabilities, sharedGrammarJump: true };
  expect(resolveExecution(grammar, spans, features)).toMatchObject({
    method: "autoregressive", mechanism: "continuous", grammarJump: true,
    checkpoint: false, promptCache: true, fill: false,
  });
  // An ignored draft is not an additional grammar restriction: the final
  // selected ordinary method is what main's direct grammar policy consumed.
  expect(resolveExecution({ ...grammar, hasAdapters: true, hasDraft: true }, spans, features))
    .toMatchObject({ method: "autoregressive", mechanism: "continuous", grammarJump: true });
  for (const [req, cap] of [[{ ...grammar, wantsLogprobs: true }, spans],
    [grammar, { ...spans, grammarBatch: false }], [request, spans],
    [grammar, { ...spans, sharedGrammarJump: false }]] as const)
    expect(resolveExecution(req, cap, features).grammarJump).toBe(false);
  const proposals = { ...spans, sharedGrammarProposals: true,
    groupedMethods: ["autoregressive", "speculative"] };
  expect(resolveExecution(grammar, proposals, features))
    .toMatchObject({ method: "speculative", mechanism: "continuous", grammarJump: true });
  expect(resolveExecution({ ...grammar, hasDraft: true }, proposals, features))
    .toMatchObject({ method: "speculative", mechanism: "continuous", grammarJump: false });
});
