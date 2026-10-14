/** Sampling owns score processing, token selection, and request-local history.
 * Score/token representations and sync/device execution belong to the backend;
 * this contract has no scheduling, cache, or tensor-runtime dependency. */
export interface SampledToken<Token, Metadata> {
  token: Token;
  extras: Metadata | null;
}

export interface SamplingSession<Scores, DeviceToken, Result> {
  /** Optional stateless operation shared across compatible rows and positions.
   * It borrows scores and returns owned device tokens, without advancing history. */
  readonly independent?: { sample(scores: Scores): DeviceToken };
  /** Optional per-position operation for ONE request whose selection depends on
   * no grammar or probability capture: position p's draw depends only on its
   * scores, its step and its processor history. Borrows [N,V] scores and returns
   * owned device tokens [N] for the given steps, without advancing history.
   * A request with processor history (`needsHistory`) also passes the window's
   * inputs [N] (borrowed: the pending token, then the drafts): position p reads
   * its history extended by inputs[1..p], the history sample() would hold there
   * had every earlier position drawn its draft. Draws are those sample() would
   * make at each step: seeded sampling keys by (seed, step), never by a shared
   * stream, so unused positions cost nothing. `commit` takes the tokens a walk
   * over the window used, in order, and advances history as sample() would have
   * advanced it drawing each. */
  readonly positional?: {
    sample(scores: Scores, steps: readonly number[], inputs?: DeviceToken): DeviceToken;
    commit(tokens: readonly number[]): void;
  };
  /** Optional draw left on the device for a request whose selection depends on
   * its processor history but on no grammar or probability capture: borrows
   * [1,V] scores and returns the owned device token sample() would draw at
   * `step`, committing it to history unread when sample() would commit it. */
  readonly unread?: { sample(scores: Scores, step: number): DeviceToken };
  readonly isPlainGreedy: boolean;
  readonly capturesLogprobs: boolean;
  readonly needsHistory: boolean;
  /** Borrows scores; transfers ownership of the result to the caller. */
  sample(logits: Scores, step: number): Result;
  seedHistory(tokens: readonly number[]): void;
  /** Commit accepted tokens when configured for manual history updates. */
  commitDevice(token: DeviceToken): void;
  commitNumbers(tokens: readonly number[]): void;
  dispose(): void;
}
