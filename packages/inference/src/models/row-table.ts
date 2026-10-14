import type { MlxRowTable } from "../contracts/mlx/row-table";

/** Index the file's declared rows or contiguous ranges; it never chooses a class. */
export function rowOperationFor<Operation>(operations: readonly {readonly from:number;readonly to:number;readonly operation:Operation}[],rows:number):Operation;
export function rowOperationFor<Operation>(operations: Readonly<Record<number,Operation>>,rows:number):Operation;
export function rowOperationFor<Operation>(operations: Readonly<Record<number,Operation>> | readonly {readonly from:number;readonly to:number;readonly operation:Operation}[],rows:number):Operation;
export function rowOperationFor<Operation>(operations: Readonly<Record<number, Operation>> | readonly {readonly from:number;readonly to:number;readonly operation:Operation}[], rows: number): Operation {
  const operation = Array.isArray(operations)
    ? operations.find(range => rows >= range.from && rows <= range.to)?.operation
    : (operations as Readonly<Record<number,Operation>>)[rows];
  if (!operation) throw new Error(`no declared operation for ${rows} rows`);
  return operation;
}

/** Literal exact-row entries plus one fixed operation for every other positive row count. */
export function rowTableWithDefault<Operation>(entries:MlxRowTable<Operation>,defaultOperation:Operation):MlxRowTable<Operation> {
  const table=Object.freeze({...entries});
  return new Proxy(table,{
    get(target,key) {
      if (typeof key!=="string") return undefined;
      const rows=Number(key);
      if (!Number.isSafeInteger(rows) || rows<1 || String(rows)!==key) return undefined;
      return target[rows] ?? defaultOperation;
    },
  });
}
