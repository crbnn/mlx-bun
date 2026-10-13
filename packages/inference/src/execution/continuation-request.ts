import { snapshotGenerationPolicy } from "./request-policy";
import type { GenerateOptions } from "../generation/index";
import type { ResolvedExecution } from "../contracts/portable/execution";
import { generationCheckpointKey } from "../generation/checkpoint-identity";
import { makeStepSampler } from "../sampling/index";
import { createRowSampling } from "./row-sampling";
import { type OrdinaryContinuation } from "./continuation-types";

/** Optional policy and sampler recovery for the existing ordinary driver. The
 * cache service resumes and checkpoints the generation under the key this
 * request names; KV layout and delayed conversion stay with the caches. */
export function createOrdinaryContinuationRequest(input: {
  options: GenerateOptions; prompt: number[]; execution: ResolvedExecution;
  /** Checkpoint every this many generated tokens (the cache service's `checkpointEveryTokens`). */
  interval: number;
  onToken: Parameters<typeof createRowSampling>[1];
}) {
  const options: GenerateOptions = snapshotGenerationPolicy(input.options);
  if (!Number.isSafeInteger(input.interval) || input.interval < 1) throw new Error("continuation interval must be positive");
  if (options.grammar || options.logprobs || options.topLogprobs || options.fill ||
      input.execution.method !== "autoregressive")
    throw new Error("ordinary continuation requires replayable ordinary sampling");
  const make = (seed = options.seed, history: readonly number[] = input.prompt) => createRowSampling(makeStepSampler(
    { ...options, seed }, { tokenRepresentation: "device", grammarWait: "external", historyUpdate: "after-sample",
      initialHistory: [...history] }), input.onToken);
  let sampling = make();
  const continuation: OrdinaryContinuation = {
    interval: input.interval,
    key: (namespace, identity) => generationCheckpointKey(input.prompt, options, namespace, input.execution, identity),
    seed: options.seed ?? 0,
    seedWasExplicit: options.seedWasExplicit === true,
    resumeSampling(state) {
      const next = make(state.seed, [...state.cacheTokens, state.pendingToken]);
      const previous = sampling; sampling = next; previous.dispose();
    },
  };
  return {
    continuation,
    sample: ((...args) => sampling.sample(...args)) as ReturnType<typeof createRowSampling>["sample"],
    // History-free greedy sampling needs no recovery work in the fast path.
    plainGreedy: sampling.plainGreedy,
    onToken: input.onToken,
    dispose: () => { sampling.dispose(); },
  };
}
