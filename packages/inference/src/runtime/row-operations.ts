type RowTable<Operation> = Readonly<Record<number, Operation>>;
type RowOperations<Operation> = RowTable<Operation> |
  readonly { readonly from: number; readonly to: number; readonly operation: Operation }[];

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
export function rowTableWithDefault<Operation>(entries:RowTable<Operation>,defaultOperation:Operation):RowTable<Operation> {
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

/** Bind the declared operations while retaining a numeric table's positive-row lookups. */
export function mapOperations<Input, Output>(operations: RowOperations<Input>,
  bind: (operation: Input, rows: number) => Output): RowOperations<Output> {
  if (Array.isArray(operations))
    return operations.map(range => ({ from: range.from, to: range.to, operation: bind(range.operation, range.from) }));
  const source = operations as RowTable<Input>;
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
