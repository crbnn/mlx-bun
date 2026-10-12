import type { ContinuationRequest, ContinuationState } from "../state/cache-service";

/** A replayable ordinary request: what the cache service checkpoints it under
 * (`ContinuationRequest`), how often the scheduler offers a checkpoint, and
 * the sampler recovery a resumed generation runs. No store knowledge enters
 * the driver; the cache service resumes and checkpoints it. */
export interface OrdinaryContinuation extends ContinuationRequest {
  readonly interval: number;
  resumeSampling(state: Omit<ContinuationState, "caches">): void;
}
