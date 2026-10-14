import type { DrafterOperations } from "../../contracts/mlx/drafter";
import type { MlxRowTable } from "../../contracts/mlx/row-table";

/** Bind the declared operations while retaining a numeric table's positive-row lookups. */
export function mapOperations<Input, Output>(operations: DrafterOperations<Input>,
  bind: (operation: Input, rows: number) => Output): DrafterOperations<Output> {
  if (Array.isArray(operations))
    return operations.map(range => ({ from: range.from, to: range.to, operation: bind(range.operation, range.from) }));
  const source = operations as MlxRowTable<Input>;
  const entries: Record<number, Output> = {};
  for (const [key, operation] of Object.entries(source)) {
    const rows = positiveRow(key);
    if (rows !== undefined && operation !== undefined) entries[rows] = bind(operation, rows);
  }
  const table = Object.freeze(entries), bound = new Map<number, Output>();
  return new Proxy(table, {
    get(target, key) {
      const rows = positiveRow(key);
      if (rows === undefined) return undefined;
      if (Object.hasOwn(target, key)) return target[rows];
      if (bound.has(rows)) return bound.get(rows);
      const operation = source[rows];
      if (operation === undefined) return undefined;
      const result = bind(operation, rows);
      bound.set(rows, result);
      return result;
    },
  });
}

function positiveRow(key: string | symbol): number | undefined {
  if (typeof key !== "string") return undefined;
  const rows = Number(key);
  return Number.isSafeInteger(rows) && rows > 0 && String(rows) === key ? rows : undefined;
}
