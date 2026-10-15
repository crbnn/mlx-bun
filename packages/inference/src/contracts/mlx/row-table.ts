/** Authored exact positive row entries and, only when explicitly written, a default operation. */
export type MlxRowTable<Operation> = Readonly<Partial<Record<number, Operation>> & { readonly default?: Operation }>;
