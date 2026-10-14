import { describe, expect, test } from "bun:test";
import { AdaptiveDraftGate } from "../../src/generation/speculative/adaptive-draft-gate";

const costs = new Map([[1, 1], [2, 1.11], [3, 1.25], [5, 1.60], [9, 2.30]]);

describe("adaptive speculative draft gate", () => {
  test("drafts deeply when acceptance is high and gates when it collapses", () => {
    const gate = new AdaptiveDraftGate(costs, { maxDraftTokens: 8 });
    for (let i = 0; i < 8; i++) gate.observe(8, 8);
    expect(gate.choose(8)).toBeGreaterThan(1);
    for (let i = 0; i < 32; i++) gate.observe(8, 0);
    expect(gate.choose(8)).toBe(0);
  });

  test("charges recurrent replay and selects no deeper than the budget", () => {
    const plain = new AdaptiveDraftGate(costs, { maxDraftTokens: 8 });
    const replay = new AdaptiveDraftGate(costs, { maxDraftTokens: 8, replayOnReject: true });
    expect(replay.expectedRate(4)).toBeLessThan(plain.expectedRate(4));
    expect(plain.choose(2)).toBeLessThanOrEqual(2);
  });

  test("models acceptance per position: a drafter whose third position falls off drafts two", () => {
    const gate = new AdaptiveDraftGate(new Map([[1, 1], [2, 1.2], [3, 1.4], [4, 1.6], [5, 1.8]]), { maxDraftTokens: 4 });
    // Positions 1 and 2 always accepted, position 3 never: every round of four accepts two.
    for (let i = 0; i < 40; i++) gate.observe(4, 2);
    expect(gate.acceptance(1)).toBeGreaterThan(0.98);
    expect(gate.acceptance(3)).toBeLessThan(0.15);
    // Position 4 is never reached: its counts decayed away and it follows position 3.
    expect(Math.abs(gate.acceptance(4) - gate.acceptance(3))).toBeLessThan(0.01);
    expect(gate.choose(4)).toBe(2);
    // The expected tokens of a round are one plus the probability of reaching each drafted position.
    expect(gate.expectedRate(2)).toBeCloseTo((1 + gate.acceptance(1) + gate.acceptance(1) * gate.acceptance(2)) / 1.4, 9);
  });

  test("starts from the prior and backs off as rejections arrive", () => {
    const gate = new AdaptiveDraftGate(new Map([[1, 1], [2, 1.9], [8, 3.65]]), { maxDraftTokens: 7 });
    expect(gate.acceptance(1)).toBeCloseTo(0.9, 9);
    expect(gate.choose(7)).toBe(7);
    // The second position keeps failing: one draft, whose first position keeps paying for it.
    for (let i = 0; i < 8; i++) gate.observe(7, 1);
    expect(gate.choose(7)).toBe(1);
    for (let i = 0; i < 8; i++) gate.observe(1, 0);
    expect(gate.choose(7)).toBe(0);
  });

  test("without drafts, next drafts one token every probe interval to observe acceptance again", () => {
    const gate = new AdaptiveDraftGate(new Map([[1, 1], [2, 1.9], [3, 2.3]]), { maxDraftTokens: 2, probeInterval: 3,
      maxProbeInterval: 24 });
    for (let i = 0; i < 32; i++) gate.observe(2, 0);
    const probes: boolean[] = [];
    expect([0, 1, 2, 3, 4].map(() => { const depth = gate.next(2); probes.push(gate.probing); return depth; })).toEqual([0, 0, 0, 1, 0]);
    expect(probes).toEqual([false, false, false, true, false]);
    // The probe was rejected: the run before the next one doubles, then doubles again, up to its cap.
    gate.observe(1, 0);
    const runs: number[] = [];
    for (let probe = 0; probe < 4; probe++) {
      let run = 0;
      while (gate.next(2) === 0) run++;
      gate.observe(1, 0);
      runs.push(run);
    }
    expect(runs).toEqual([5, 12, 24, 24]);
    // Probes that keep accepting bring the drafts back.
    for (let i = 0; i < 16; i++) gate.observe(1, 1);
    for (let i = 0; i < 16; i++) gate.observe(2, 2);
    expect(gate.next(2)).toBeGreaterThan(0);
    // A budget of nothing drafts nothing.
    expect(gate.next(0)).toBe(0);
  });

  test("a rejected probe keeps the deeper positions' evidence; a fully accepted round lets it fade", () => {
    const gate = new AdaptiveDraftGate(new Map([[1, 1], [2, 2], [3, 2.4], [4, 2.6]]), { maxDraftTokens: 3 });
    for (let i = 0; i < 16; i++) gate.observe(3, 1);
    const second = gate.acceptance(2);
    expect(second).toBeLessThan(0.2);
    // Rejected probes at the first position do not wash out the second's rejections.
    for (let i = 0; i < 4; i++) gate.observe(1, 0);
    expect(gate.acceptance(2)).toBeLessThanOrEqual(second);
    for (let i = 0; i < 16; i++) gate.observe(1, 1);
    expect(gate.acceptance(2)).toBeGreaterThan(0.6);
  });

  test("reset returns to the prior and the first probe interval", () => {
    const gate = new AdaptiveDraftGate(new Map([[1, 1], [2, 1.9], [8, 3.65]]), { maxDraftTokens: 7, probeInterval: 2 });
    for (let i = 0; i < 16; i++) gate.observe(7, 0);
    expect(gate.choose(7)).toBe(0);
    gate.reset();
    expect(gate.acceptance(1)).toBeCloseTo(0.9, 9);
    expect(gate.choose(7)).toBe(7);
  });

  test("weighs a group's round by the rows its forward carries against a step without drafts over the same rows", () => {
    // A machine whose one-row step is cheap and whose wider forwards cost little more per row.
    const wide = new Map([[1, 1], [2, 2], [4, 2.5], [8, 3], [16, 3.4], [32, 6]]);
    const gate = new AdaptiveDraftGate(wide, { maxDraftTokens: 7 });
    for (let i = 0; i < 24; i++) gate.observe(3, 1);
    // One sequence: the same rate as before rows were counted.
    expect(gate.expectedRate(3, 1)).toBeCloseTo(gate.expectedRate(3), 12);
    // Four sequences drafting 3 verify 16 rows against a 4-row step: (1 + a1 + ...) * 2.5 / 3.4.
    let survive = 1, tokens = 1;
    for (let position = 1; position <= 3; position++) { survive *= gate.acceptance(position); tokens += survive; }
    expect(gate.expectedRate(3, 4)).toBeCloseTo(tokens * 2.5 / 3.4, 12);
    // Drafting pays at four sequences where it does not at one.
    expect(gate.choose(7, 1)).toBe(0);
    expect(gate.choose(7, 4)).toBeGreaterThan(0);
    // Past the widest measured forward a row costs what one of that forward does: 64 rows, 6 * 64 / 32.
    expect(gate.expectedRate(7, 8)).toBeCloseTo((1 + [1, 2, 3, 4, 5, 6, 7].reduce((sum, k) => {
      let p = 1; for (let position = 1; position <= k; position++) p *= gate.acceptance(position); return sum + p;
    }, 0)) * 3 / 12, 12);
  });
});
