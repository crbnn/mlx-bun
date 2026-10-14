/** Operations keyed by the exact positive token-row count supplied by a caller. */
export type MlxRowTable<Operation> = Readonly<Record<number, Operation>>;
