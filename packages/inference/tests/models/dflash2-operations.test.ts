import { expect, test } from "bun:test";
import { mapOperations } from "../../src/models/speculative/dflash2-operations";
import { rowOperationFor, rowTableWithDefault } from "../../src/models/row-table";

test("draft projection and context mappings retain a custom table's unbounded default", () => {
  const listed = () => "listed", special = () => "special", fallback = () => "default";
  const source = rowTableWithDefault({ 1: listed, 77: special }, fallback);
  const projectionBindings: number[] = [], contextBindings: number[] = [];
  const projections = mapOperations(source, (operation, rows) => {
    projectionBindings.push(rows);
    return { forward: () => `${rows}:${operation()}` };
  });
  const context = mapOperations(source, (_operation, rows) => {
    contextBindings.push(rows);
    const projection = rowOperationFor(projections, rows);
    return () => projection.forward();
  });
  expect(projectionBindings).toEqual([1, 77]);
  expect(contextBindings).toEqual([1, 77]);
  for (const rows of [1, 65, 77, 262144]) {
    const operation = rowOperationFor(context, rows);
    expect(operation()).toBe(`${rows}:${rows === 1 ? "listed" : rows === 77 ? "special" : "default"}`);
    expect(rowOperationFor(context, rows)).toBe(operation);
  }
  expect(projectionBindings).toEqual([1, 77, 65, 262144]);
  expect(contextBindings).toEqual([1, 77, 65, 262144]);
  expect(Object.keys(context)).toEqual(["1", "77"]);
  expect(Object.isFrozen(context)).toBe(true);
  expect(Reflect.set(context, "65", () => "replacement")).toBe(false);
});

test("absent or invalid rows never bind a mapped operation", () => {
  const bindings: number[] = [];
  const mapped = mapOperations({ 1: "listed" }, (operation, rows) => {
    bindings.push(rows);
    return () => operation;
  });
  for (const key of ["65", "0", "-1", "1.5", "01", "1e2", "Infinity", "then", "constructor", Symbol.iterator])
    expect(Reflect.get(mapped, key)).toBeUndefined();
  expect(() => rowOperationFor(mapped, 65)).toThrow("no declared operation for 65 rows");
  expect(bindings).toEqual([1]);
});

test("range mapping keeps the declared boundaries and binds once per range", () => {
  const bindings: number[] = [];
  const mapped = mapOperations([
    { from: 1, to: 3, operation: "narrow" },
    { from: 4, to: Infinity, operation: "wide" },
  ], (operation, rows) => {
    bindings.push(rows);
    return () => operation;
  });
  expect(rowOperationFor(mapped, 3)()).toBe("narrow");
  expect(rowOperationFor(mapped, 4)()).toBe("wide");
  expect(rowOperationFor(mapped, 262144)).toBe(rowOperationFor(mapped, 4));
  expect(bindings).toEqual([1, 4]);
  expect((mapped as readonly { from: number; to: number }[]).map(({ from, to }) => [from, to]))
    .toEqual([[1, 3], [4, Infinity]]);
});
