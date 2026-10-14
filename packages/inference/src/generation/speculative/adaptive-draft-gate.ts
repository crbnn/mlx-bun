/** Cost-aware speculative draft scheduler.
 *
 * Adapted from mlx-optiq 0.5.8's AcceptanceGate (MIT). The scheduler models
 * the expected emitted tokens for a proposal depth and divides that by the
 * measured cost of its round (rows = depth + 1); a round without drafts is one
 * plain decode step, cost 1. It uses acceptance rather than
 * observed wall-clock throughput as the online signal, so unrelated machine
 * load does not make it oscillate.
 */

export type VerifyCostModel = ReadonlyMap<number, number>;

function interpolatedCost(costs: VerifyCostModel, positions: number): number {
  const keys = [...costs.keys()].sort((a, b) => a - b);
  if (keys.length < 2 || !costs.has(1))
    throw new Error("adaptive draft costs need position 1 and at least one larger position");
  const exact = costs.get(positions);
  if (exact !== undefined) return exact;
  if (positions > keys.at(-1)!) {
    const hi = keys.at(-1)!;
    const lo = keys.at(-2)!;
    return costs.get(hi)! +
      ((costs.get(hi)! - costs.get(lo)!) / (hi - lo)) * (positions - hi);
  }
  const lo = keys.filter((key) => key < positions).at(-1)!;
  const hi = keys.find((key) => key > positions)!;
  return costs.get(lo)! +
    ((costs.get(hi)! - costs.get(lo)!) / (hi - lo)) * (positions - lo);
}

export interface AdaptiveDraftGateOptions {
  maxDraftTokens: number;
  /** Whether rejection requires replaying the accepted prefix. */
  replayOnReject?: boolean;
  /** Every position's counts before any round is observed: accepted and rejected drafts (4.5 and 0.5, an acceptance
   *  of 0.9, so a new gate drafts deeply and backs off as rejections arrive). */
  priorAccepted?: number;
  priorRejected?: number;
  decay?: number;
  margin?: number;
  /** Rounds without drafts before `next` drafts one token to observe acceptance again (default 8). Each probe after
   *  which the gate still drafts nothing doubles the interval, up to `maxProbeInterval` (default 32). */
  probeInterval?: number;
  maxProbeInterval?: number;
}

/** Acceptance is modeled per draft position: the probability that the draft at position i is accepted given that
 *  every draft before it was. Each position's decayed counts are shrunk toward the position before it (weight 1;
 *  position 1 toward the prior), so a position seldom reached follows the ones that are, and a drafter whose later
 *  positions fall off (a recurrent MTP head trained for two steps) and one that stays flat (a block drafter) are
 *  both modeled. A round decays the positions it drafted. */
export class AdaptiveDraftGate {
  /** Decayed accepted and rejected drafts at each position (index 0 is position 1). */
  readonly #accepted: number[];
  readonly #rejected: number[];
  readonly #prior: number;
  readonly #priorWeight: number;
  readonly #decay: number;
  readonly #margin: number;
  readonly #maxDraftTokens: number;
  readonly #replayOnReject: boolean;
  readonly #probeInterval: number;
  readonly #maxProbeInterval: number;
  /** Rounds in a row `next` has run without drafts, and the rounds before its next probe. */
  #idle = 0;
  #interval: number;
  #probing = false;

  constructor(readonly costs: VerifyCostModel, options: AdaptiveDraftGateOptions) {
    this.#maxDraftTokens = options.maxDraftTokens;
    this.#replayOnReject = options.replayOnReject ?? false;
    const accepted = options.priorAccepted ?? 4.5, rejected = options.priorRejected ?? 0.5;
    this.#prior = accepted / (accepted + rejected);
    this.#priorWeight = accepted + rejected;
    this.#accepted = Array<number>(options.maxDraftTokens).fill(accepted);
    this.#rejected = Array<number>(options.maxDraftTokens).fill(rejected);
    this.#decay = options.decay ?? 0.85;
    this.#margin = options.margin ?? 0.03;
    this.#probeInterval = this.#interval = options.probeInterval ?? 8;
    this.#maxProbeInterval = options.maxProbeInterval ?? 32;
  }

  /** Forget what the gate observed: back to the prior, probing at the first interval. For new text (a request
   *  joining), whose acceptance the rounds before say little about. */
  reset(): void {
    this.#accepted.fill(this.#prior * this.#priorWeight);
    this.#rejected.fill((1 - this.#prior) * this.#priorWeight);
    this.#idle = 0;
    this.#interval = this.#probeInterval;
  }

  /** The probability that the draft at `position` (1-based) is accepted, given that the drafts before it were. */
  acceptance(position: number): number {
    let shrunk = this.#prior;
    for (let i = 0; i < position; i++)
      shrunk = (this.#accepted[i]! + shrunk) / (this.#accepted[i]! + this.#rejected[i]! + 1);
    return shrunk;
  }

  /** One round's outcome: `drafted` proposals, the first `accepted` of them accepted. Positions 1..accepted count an
   *  acceptance and the first rejected position a rejection, after the counts decay: every position's when every
   *  draft was accepted (the text may have turned predictable, so the positions past the round follow the ones it
   *  accepted), only the drafted positions' otherwise (a rejected probe says nothing about deeper drafts). A round
   *  without drafts observes nothing. */
  observe(drafted: number, accepted: number): void {
    const decayed = drafted > 0 && accepted === drafted ? this.#maxDraftTokens : Math.min(drafted, this.#maxDraftTokens);
    for (let i = 0; i < decayed; i++) {
      this.#accepted[i] = this.#accepted[i]! * this.#decay + Number(i < Math.min(accepted, drafted));
      this.#rejected[i] = this.#rejected[i]! * this.#decay + Number(i === accepted && accepted < drafted);
    }
  }

  /** Expected emitted tokens of a round drafting `k` per unit of cost (a plain decode step emits 1 per 1). */
  expectedRate(k: number): number {
    // survive: the probability that every draft up to a position is accepted; the round emits one token more
    // than it accepts, so tokens = 1 + the sum of survive over positions 1..k.
    let survive = 1, tokens = 1, replay = 0;
    for (let position = 1; position <= k; position++) {
      const next = survive * this.acceptance(position);
      // Exactly position - 1 accepted: the prefix is replayed.
      if (this.#replayOnReject) replay += (survive - next) * interpolatedCost(this.costs, position);
      tokens += next;
      survive = next;
    }
    return tokens / (interpolatedCost(this.costs, k + 1) + replay);
  }

  /** The next round's depth: `choose`'s, except that after a run of rounds without drafts it drafts one token,
   *  the cheapest round that observes acceptance, so the estimate keeps following the text (a round without drafts
   *  observes nothing). The run starts at `probeInterval` rounds and doubles after each probe that leaves the gate
   *  drafting nothing, up to `maxProbeInterval`: a probe costs more than the plain step it replaces, so text that
   *  stays hard to draft pays for fewer of them. */
  next(budget: number): number {
    this.#probing = false;
    const depth = this.choose(budget);
    if (depth > 0 || budget < 1) { this.#idle = 0; this.#interval = this.#probeInterval; return depth; }
    if (++this.#idle <= this.#interval) return 0;
    this.#idle = 0;
    this.#interval = Math.min(2 * this.#interval, this.#maxProbeInterval);
    this.#probing = true;
    return 1;
  }

  /** Whether the depth `next` last returned is a probe: drafts the estimate does not pay for, drafted only to observe
   *  acceptance. A drafter that can propose without committing may observe them without a verify round. */
  get probing(): boolean { return this.#probing; }

  choose(budget: number): number {
    const limit = Math.min(this.#maxDraftTokens, budget);
    let bestDepth = 0;
    let bestRate = 1 + this.#margin;
    for (let depth = 1; depth <= limit; depth++) {
      const rate = this.expectedRate(depth);
      if (rate > bestRate) {
        bestDepth = depth;
        bestRate = rate;
      }
    }
    return bestDepth;
  }
}
