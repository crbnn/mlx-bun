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
}
