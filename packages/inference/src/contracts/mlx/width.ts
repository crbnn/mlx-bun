/** One operation per width class, where a forward's rows are its sequences
 * times its tokens per sequence. The classes are the row-count boundaries the
 * shipped kernels were measured at: one row, two or three rows sharing each
 * decoded weight, four rows (where the matrix-unit down projection takes over
 * from the shared-row one), five to eight rows on the matrix unit, and wider.
 * The caller picks the entry by the rows it is loading (`widthClass`) and hands
 * an entry only rows of its class; an entry never inspects its input to learn
 * its class. A graph with one forward supplies the same operation in every
 * entry; a specialized graph supplies the operation it built for each class. */
export interface MlxWidthTable<Operation> {
  /** Exactly one row. */
  readonly one: Operation;
  /** Two or three rows. */
  readonly twoToThree: Operation;
  /** Exactly four rows. */
  readonly four: Operation;
  /** Five to eight rows. */
  readonly fiveToEight: Operation;
  /** Nine or more rows. */
  readonly wider: Operation;
}

/** The name of one width class. */
export type MlxWidthClass = keyof MlxWidthTable<unknown>;

/** The width class of a forward of `rows` rows (sequences times tokens per
 * sequence). The caller's choice, made from the rows it is loading; a graph
 * never calls this on its input. */
export function widthClass(rows: number): MlxWidthClass {
  if (!Number.isSafeInteger(rows) || rows < 1) throw new Error(`invalid forward row count ${rows}`);
  return rows === 1 ? "one" : rows <= 3 ? "twoToThree" : rows === 4 ? "four" : rows <= 8 ? "fiveToEight" : "wider";
}
