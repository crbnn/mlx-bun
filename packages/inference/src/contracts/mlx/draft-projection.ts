import type { MlxArray } from "@mlx-bun/mlx/array";
import type { Dtype } from "@mlx-bun/mlx/ffi";
import type { MlxWidthTable } from "./width";

/** Numerical ports are backend-specific; returned arrays are caller-owned. */
export interface DraftProjection {
  readonly embed: {
    encode(ids: MlxArray): MlxArray;
    readonly scales: { readonly dtype: Dtype };
  };
  /** The target's output head per width class: the drafter indexes it by the
   *  rows it projects. Borrows the hidden; the logits belong to the caller. */
  readonly logits: MlxWidthTable<(hidden: MlxArray) => MlxArray>;
  /** Declared by a graph whose composition measured a drafter selecting over the
   *  target's frequency-ranked draft vocabulary: its ids [N] (ascending, uint32,
   *  borrowed) and the head's projection onto those rows per width class
   *  (logits [..., N], caller-owned). A drafter that selects by its scores alone
   *  selects over these rows; the target still verifies over its full head.
   *  Absent: the full head. */
  readonly vocabulary?: {
    readonly ids: MlxArray;
    readonly logits: MlxWidthTable<(hidden: MlxArray) => MlxArray>;
  };
}
