import { expect, test } from "bun:test";
import type { MlxArray } from "@mlx-bun/mlx/array";
import { runMixedTokenIteration } from "../../src/execution/mixed-iteration";
import type { MlxForwardStep } from "../../src/contracts/mlx/forward-work";
import { type MlxLayerCapture } from "../../src/contracts/mlx/graph";
import * as ops from "@mlx-bun/mlx/ops";

/** One method forward as the mixed iteration's hook takes it: its phase, rows and own operation. */
const step = (phase: MlxForwardStep["phase"], ids: MlxArray, own: () => Promise<MlxArray>,
  capture: MlxLayerCapture | null = null): MlxForwardStep => {
  const rows = ids.shape[0]! * ids.shape[1]!;
  return { phase, rows, capture, own };
};
const unexpected = (what: string) => async (): Promise<MlxArray> => { throw new Error(what); };

test("mixed execution publishes and retires decode before preparation joins", async () => {
  using decode = ops.fromInt32([7], [1, 1]);
  using prompt = ops.fromInt32([11, 12, 13], [1, 3]);
  const order: string[] = [];
  const advanced = await runMixedTokenIteration({
    async prepare(forward) {
      using hidden = await forward(prompt, [], step("window", prompt, unexpected("separate forward was not expected")));
      expect(hidden.toIntTokens()).toEqual([11, 12, 13]);
      order.push("join");
    },
    async decode(forward) {
      using hidden = await forward(decode, [], step("decode", decode, unexpected("separate forward was not expected")));
      expect(hidden.toIntTokens()).toEqual([7]);
      order.push("publish", "retire");
    },
    mixed(groups, rows) {
      expect(groups.map(group => group.ids.shape)).toEqual([[1, 1], [1, 3]]);
      expect(groups.map(group => [group.phase, group.rows])).toEqual([["decode", 1], ["window", 3]]);
      expect(rows).toBe(4);
      order.push("mixed");
      return groups.map(group => ops.copyOf(group.ids));
    },
  });
  expect(advanced).toBe(true);
  expect(order).toEqual(["mixed", "publish", "retire", "join"]);
});

test("restored or cancelled preparation completes without waiting for a model call", async () => {
  const advanced = await runMixedTokenIteration({
    async prepare() {},
    async decode() { throw new Error("unexpected decode"); },
    mixed() { throw new Error("unexpected mixed forward"); },
  });
  expect(advanced).toBe(false);
});

test("a final pending decode token can retire while preparation still advances", async () => {
  using ids = ops.fromInt32([1, 2], [1, 2]);
  const events: string[] = [];
  expect(await runMixedTokenIteration({
    async prepare(forward) {
      using h = await forward(ids, [], step("window", ids, async () => { events.push("forward"); return ops.copyOf(ids); }));
      events.push("prepared");
    },
    async decode() { events.push("retired"); },
    mixed() { throw new Error("unexpected mixed forward"); },
  })).toBe(true);
  expect(events).toEqual(["retired", "forward", "prepared"]);
});

test("mixed work retains per-method taps and verification policy on every forward", async () => {
  using ids = ops.fromInt32([1], [1, 1]);
  const taps = [0, 0];
  const capture = taps.map((_, row) => () => { taps[row]!++; });
  const own = (tap: () => void) => async () => { tap(); return ops.copyOf(ids); };
  await runMixedTokenIteration({
    async prepare(forward) {
      using h = await forward(ids, [], step("window", ids, own(capture[1]!), capture[1]!));
      using tail = await forward(ids, [], step("window", ids, own(capture[1]!), capture[1]!));
    },
    async decode(forward) { using h = await forward(ids, [], step("verify", ids, own(capture[0]!), capture[0]!)); },
    mixed(groups) {
      expect(groups[0]!.phase).toBe("verify");
      expect(groups[1]!.phase).toBe("window");
      for (const group of groups) group.capture?.(0, group.ids);
      return groups.map(group => ops.copyOf(group.ids));
    },
  });
  expect(taps).toEqual([1, 2]);
});

test.each(["prepare", "mixed", "sample"])("%s failure unwinds both work producers without a stranded promise", async phase => {
  using ids = ops.fromInt32([1], [1, 1]);
  let preparationClosed = false;
  const own = async () => ops.copyOf(ids);
  await expect(runMixedTokenIteration({
    async prepare(forward) {
      try {
        if (phase === "prepare") throw new Error(phase);
        using h = await forward(ids, [], step("window", ids, own));
      } finally { preparationClosed = true; }
    },
    async decode(forward) { using h = await forward(ids, [], step("decode", ids, own)); throw new Error("sample"); },
    mixed(groups) {
      if (phase === "mixed") throw new Error(phase);
      return groups.map(group => ops.copyOf(group.ids));
    },
  })).rejects.toThrow(phase);
  expect(preparationClosed).toBe(true);
});
