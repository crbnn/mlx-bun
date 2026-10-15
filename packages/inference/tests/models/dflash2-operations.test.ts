import { expect, test } from "bun:test";
import { mapOperations } from "../../src/models/speculative/dflash2-operations";
import type { MlxRowTable } from "../../src/contracts/mlx/row-table";
import { rowOperationFor } from "../../src/models/row-table";

test("mapping preserves only authored exact entries and an inspectable fixed default", () => {
  const listed = () => "listed", special = () => "special", fallback = () => "default";
  const source = Object.freeze({ 1: listed, 77: special, default: fallback });
  const bound: Function[] = [];
  const projections = mapOperations(source, operation => {
    bound.push(operation);
    return { forward: () => operation() };
  });
  const context = mapOperations(projections, projection => () => projection.forward());
  expect(Object.keys(projections)).toEqual(["1", "77", "default"]);
  expect(projections.default).toBeDefined();
  expect(bound).toEqual([listed, special, fallback]);
  for (const [rows, expected] of [[1, "listed"], [77, "special"], [65, "default"], [262144, "default"]] as const) {
    expect(rowOperationFor(projections, rows).forward()).toBe(expected);
    expect(rowOperationFor(context, rows)()).toBe(expected);
  }
  expect(rowOperationFor(context, 65)).toBe(context.default!);
  expect(context[65]).toBeUndefined(); // No Proxy manufactures an own/indexed entry.
  expect(bound).toHaveLength(3); // Lookups bind nothing.
  expect(Object.isFrozen(context)).toBe(true);
  expect(Reflect.set(context, "65", () => "replacement")).toBe(false);
});

test("missing entries, inherited defaults and invalid rows never manufacture an operation", () => {
  const source = Object.assign(Object.create({ 65: "inherited", default: "inherited" }), { 1: "one" }) as MlxRowTable<string>;
  const bound: string[] = [];
  const table = mapOperations(source, (operation: string) => { bound.push(operation); return () => operation; });
  expect(Object.keys(table)).toEqual(["1"]);
  expect(rowOperationFor(table, 1)()).toBe("one");
  for (const rows of [0, -1, 1.5, 65, Infinity, NaN])
    expect(() => rowOperationFor(table, rows)).toThrow("no declared operation");
  expect(() => rowOperationFor(source, 65)).toThrow("no declared operation");
  expect(bound).toEqual(["one"]);
  expect(table.default).toBeUndefined();
});

test("range mappings retain authored bounds including Infinity and bind each declared operation once", () => {
  const narrow = () => "narrow", expanded = () => "expanded", bound: Function[] = [];
  const ranges = mapOperations([{ from: 1, to: 4, operation: narrow }, { from: 5, to: Infinity, operation: expanded }], operation => {
    bound.push(operation); return () => operation();
  });
  expect(ranges.map(({ from, to }) => [from, to])).toEqual([[1, 4], [5, Infinity]]);
  expect(rowOperationFor(ranges, 1)()).toBe("narrow");
  expect(rowOperationFor(ranges, 262144)()).toBe("expanded");
  expect(bound).toEqual([narrow, expanded]);
  expect(Object.isFrozen(ranges)).toBe(true);
  expect(() => rowOperationFor(ranges, 0)).toThrow("no declared operation");
  expect(() => rowOperationFor(ranges, Infinity)).toThrow("no declared operation");
});
