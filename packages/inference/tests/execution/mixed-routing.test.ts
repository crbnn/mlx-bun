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

function verificationPacking(calls: string[] = [], disposed: string[] = []): readonly MlxMixedRange[] {
  const operation = (name: string) => (groups: readonly MlxPhaseGroup[], totalRows: number) => {
    calls.push(`${name}:${groups.reduce((sum, item) => sum + item.rows, 0)}:${totalRows}`);
    return groups.map(item => output(`${name}:${item.phase}:${item.rows}`, disposed));
  };
  return [
    { from: 1, to: 4, totalTo: 4, operation: operation("stock") },
    { from: 1, to: 4, totalFrom: 5, totalTo: 8, operation: operation("unfused") },
    { from: 5, to: 8, totalTo: 8, operation: operation("five") },
    { from: 1, to: 256, totalFrom: 9, operation: operation("wide") },
    { from: 257, to: Infinity, operation: operation("expanded") },
  ];
}

test("verification packing uses actual packed rows and the complete round's declared bounds", () => {
  const cases = [
    { groups: [group("window", 2), group("verify", 2), group("decode", 2)], name: "unfused", packed: 4, total: 6 },
    { groups: [group("verify", 40), group("window", 2), group("decode", 2)], name: "wide", packed: 4, total: 44 },
    { groups: [group("decode", 100), group("verify", 300), group("window", 100)], name: "wide", packed: 200, total: 500 },
    { groups: [group("decode", 128), group("verify", 1), group("window", 128)], name: "wide", packed: 256, total: 257 },
    { groups: [group("decode", 128), group("verify", 1), group("window", 129)], name: "expanded", packed: 257, total: 258 },
  ];
  for (const item of cases) {
    const calls: string[] = [];
    const packed = verificationPacking(calls);
    const result = dispatchMixedGroups(item.groups, item.total, [], { ...separate(), packed });
    expect(calls).toEqual([`${item.name}:${item.packed}:${item.total}`]);
    expect(result.map(label)).toEqual(item.groups.map(value => value.phase === "verify"
      ? `verify:${value.rows <= 4 ? "1-4" : "5+"}` : `${item.name}:${value.phase}:${value.rows}`));
  }
});

test("a missing verification packing declaration keeps every group separate", () => {
  const packed: MlxMixedRange[] = [{ from: 1, to: Infinity, operation() { throw new Error("ordinary packing called"); } }];
  expect(dispatchMixedGroups([group("window", 2), group("verify", 2), group("decode", 2)], 6, packed, separate()).map(label))
    .toEqual(["window:1-4", "verify:1-4", "decode:1-4"]);
});

test("one nonverify group uses its own phase geometry even with a verification packing declaration", () => {
  const calls: string[] = [];
  expect(dispatchMixedGroups([group("verify", 300), group("decode", 2)], 302, [],
    { ...separate(), packed: verificationPacking(calls) }).map(label)).toEqual(["verify:5+", "decode:1-4"]);
  expect(calls).toEqual([]);
});

test("failed separate verification releases packed outputs without disposing borrowed inputs", () => {
  const disposed: string[] = [], inputs: string[] = [];
  const groups = [group("window", 2), group("verify", 2), group("decode", 2)]
    .map((item, index) => ({ ...item, ids: output(`input:${index}`, inputs) }));
  const operations: MlxSeparateMixed = { ...separate(), packed: verificationPacking([], disposed),
    verify: [{ from: 1, to: Infinity, operation() { throw new Error("verify failed"); } }] };
  expect(() => dispatchMixedGroups(groups, 6, [], operations)).toThrow("verify failed");
  expect(disposed).toEqual(["unfused:window:2", "unfused:decode:2"]);
  expect(inputs).toEqual([]);
});

test("later verification failure releases earlier verification and packed outputs", () => {
  const disposed: string[] = [];
  const operations: MlxSeparateMixed = { ...separate(), packed: verificationPacking([], disposed),
    verify: [
      { from: 1, to: 1, operation: () => output("first verify", disposed) },
      { from: 2, to: Infinity, operation() { throw new Error("later verify failed"); } },
    ] };
  expect(() => dispatchMixedGroups([group("verify", 1), group("window", 2), group("verify", 2), group("decode", 2)], 7, [], operations))
    .toThrow("later verify failed");
  expect(disposed).toEqual(["first verify", "unfused:window:2", "unfused:decode:2"]);
});

test("verification-only work does not call a packed operation", () => {
  const calls: string[] = [];
  expect(dispatchMixedGroups([group("verify", 2), group("verify", 8)], 10, [],
    { ...separate(), packed: verificationPacking(calls) }).map(label)).toEqual(["verify:1-4", "verify:5+"]);
  expect(calls).toEqual([]);
});

test("packing failure does not run verification or dispose borrowed inputs", () => {
  const verificationCalls: string[] = [], inputs: string[] = [];
  const groups = [group("verify", 2), group("window", 2), group("decode", 2)]
    .map((item, index) => ({ ...item, ids: output(`input:${index}`, inputs) }));
  const operations: MlxSeparateMixed = { ...separate(),
    packed: [{ from: 1, to: Infinity, operation() { throw new Error("packing failed"); } }],
    verify: [{ from: 1, to: Infinity, operation() { verificationCalls.push("verify"); return output("verify"); } }],
  };
  expect(() => dispatchMixedGroups(groups, 6, [], operations)).toThrow("packing failed");
  expect(verificationCalls).toEqual([]);
  expect(inputs).toEqual([]);
});
