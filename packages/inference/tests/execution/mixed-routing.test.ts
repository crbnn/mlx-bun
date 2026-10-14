import { expect, test } from "bun:test";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxMixedRange, MlxPhaseGroup, MlxSeparateMixed } from "../../src/contracts/mlx/graph";
import { dispatchMixedGroups } from "../../src/execution/mixed-iteration";

function output(label: string, disposed: string[] = []): MlxArray {
  return { label, dispose() { disposed.push(label); } } as unknown as MlxArray;
}
const label = (value: MlxArray) => (value as unknown as { label: string }).label;
const group = (phase: MlxPhaseGroup["phase"], rows: number): MlxPhaseGroup =>
  ({ phase, rows, ids: {} as MlxArray, state: [], capture: null });
function separate(): MlxSeparateMixed {
  const ranges = (phase: string) => [
    { from: 1, to: 4, operation: () => output(`${phase}:1-4`) },
    { from: 5, to: Infinity, operation: () => output(`${phase}:5+`) },
  ];
  return { decode: ranges("decode"), window: ranges("window"), verify: ranges("verify") };
}

test("mixed work chooses the packed operation using total rows", () => {
  const packed: MlxMixedRange[] = [
    { from: 1, to: 4, operation: () => [output("small"), output("small")] },
    { from: 5, to: Infinity, operation: () => [output("packed"), output("packed")] },
  ];
  expect(dispatchMixedGroups([group("decode", 1), group("window", 4)], 5, packed, separate()).map(label))
    .toEqual(["packed", "packed"]);
});

test("verify work retains each group's phase and row geometry", () => {
  const packed: MlxMixedRange[] = [{ from: 1, to: Infinity, operation() { throw new Error("packed verify"); } }];
  expect(dispatchMixedGroups([group("verify", 3), group("window", 8)], 11, packed, separate()).map(label))
    .toEqual(["verify:1-4", "window:5+"]);
});

test("a graph declaring only separate mixed operations still handles decode plus prompt", () => {
  expect(dispatchMixedGroups([group("decode", 2), group("window", 6)], 8, [], separate()).map(label))
    .toEqual(["decode:1-4", "window:5+"]);
});

test("a separate mixed failure releases earlier owned outputs", () => {
  const disposed: string[] = [], operations = separate();
  const failed: MlxSeparateMixed = { ...operations,
    decode: [{ from: 1, to: Infinity, operation: () => output("decode", disposed) }],
    window: [{ from: 1, to: Infinity, operation() { throw new Error("prompt failed"); } }],
  };
  expect(() => dispatchMixedGroups([group("decode", 1), group("window", 2)], 3, [], failed)).toThrow("prompt failed");
  expect(disposed).toEqual(["decode"]);
});
