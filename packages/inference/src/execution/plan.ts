import type { ExecutionCapabilities, ExecutionFeatures, ExecutionReason, ExecutionRequirements, ResolvedExecution } from "../contracts/portable/execution";

/** Select the request method, then place it using executor capabilities. The
 * composition's caches serve every method; nothing here reads how they store
 * keys and values. */
export function resolveExecution(
  request: ExecutionRequirements, capabilities: ExecutionCapabilities,
  features: ExecutionFeatures = { fill: false },
): ResolvedExecution {
  const reasons: ExecutionReason[] = [];
  const sharedRequestExclusions = [
    [request.hasVision && !capabilities.mediaBatch, "media-batch-unsupported"],
    [request.hasGrammar && !capabilities.grammarBatch, "grammar-batch-unsupported"],
  ] as const;
  const excluded = sharedRequestExclusions.some(([exclusion]) => exclusion);
  const sharedAdapterMethod = capabilities.sharedSpeculativeAdapters === true && !excluded &&
    capabilities.groupedMethods?.includes("speculative") === true;
  const grammarProposals = features.grammarJump === true && request.hasGrammar && !request.hasDraft && !request.hasVision &&
    (!request.wantsLogprobs || capabilities.speculativeLogprobs === true) && capabilities.method === "autoregressive" &&
    capabilities.sharedGrammarProposals === true && capabilities.groupedMethods?.includes("speculative") === true &&
    (!request.hasAdapters || sharedAdapterMethod) && !excluded;
  const speculative = grammarProposals || (capabilities.method === "autoregressive" && request.hasDraft &&
    !request.hasVision && (!request.hasAdapters || sharedAdapterMethod) &&
    (!request.wantsLogprobs || capabilities.speculativeLogprobs === true));
  const method = speculative ? "speculative" : capabilities.method;
  // Token-level request features act on sampled tokens. A method that does not
  // sample tokens refuses them instead of silently discarding them.
  const tokenMethod = capabilities.method === "autoregressive";
  const continuousExclusions = [
    [!(capabilities.groupedMethods ?? ["autoregressive"]).includes(method), "method-batch-unsupported"],
    [request.hasDraft && !tokenMethod, "draft-method-unsupported"],
    [request.wantsLogprobs && !tokenMethod, "logprobs-method-unsupported"],
    [request.hasRepetitionPenalty && !tokenMethod, "repetition-penalty-method-unsupported"],
    [request.hasLogitsExtras && !tokenMethod, "logits-extras-method-unsupported"],
    [features.fill && !tokenMethod, "fill-method-unsupported"],
    ...sharedRequestExclusions,
  ] as const;
  for (const [exclusion, reason] of continuousExclusions) if (exclusion) reasons.push(reason);
  const mechanism = reasons.length ? "unsupported" : "continuous";
  if (request.hasDraft && !speculative && tokenMethod) reasons.push("draft-incompatible-with-request");
  // Prefix reuse restores autoregressive token caches only.
  const promptCache = method === "autoregressive" && (!request.hasVision ||
    (mechanism === "continuous" && request.hasPreparedPrefixIdentity === true && capabilities.mediaPrefixCache === true));
  const sharedFill = mechanism === "continuous" && capabilities.sharedFill === true;
  const speculativeEcho = method === "speculative" && mechanism === "continuous" &&
    capabilities.sharedSpeculativeEcho === true;
  const fill = features.fill && !request.hasVision && !request.hasGrammar && !request.wantsLogprobs &&
    (speculativeEcho || (method === "autoregressive" && sharedFill && !request.hasDraft));
  if (features.fill && !fill && tokenMethod) reasons.push("fill-incompatible-with-request");
  const checkpoint = capabilities.checkpoints && method === "autoregressive" && !request.hasVision &&
    capabilities.sharedCheckpoints === true && promptCache && !request.hasGrammar &&
    !features.fill && !request.wantsLogprobs;
  const compiledDecode = features.compiledDecode === true && capabilities.compiledDecode === true &&
    method === "autoregressive" && !request.hasAdapters && !(sharedFill && fill);
  if (features.compiledDecode && !compiledDecode) reasons.push("compiled-decode-unavailable-for-request");
  const directGrammarJump = features.grammarJump === true && request.hasGrammar &&
    !request.wantsLogprobs && method === "autoregressive" && mechanism === "continuous" &&
    capabilities.sharedGrammarJump === true;
  const grammarJump = grammarProposals || directGrammarJump;
  if (features.grammarJump && request.hasGrammar && !grammarJump)
    reasons.push("grammar-jump-incompatible-with-request");
  return Object.freeze({ method, mechanism, promptCache, checkpoint, fill,
    compiledDecode, grammarJump,
    reasons: Object.freeze(reasons) });
}
