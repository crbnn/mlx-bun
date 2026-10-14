import type { MlxArray } from "@mlx-bun/mlx/array";

/** The output head over a post-final-norm hidden [.., hidden-width] (borrowed)
 * to logits [.., vocabulary] the caller owns, possibly unevaluated. */
export type MlxHeadForward = (hidden: MlxArray) => MlxArray;
export interface MlxHeadRange { readonly from: number; readonly to: number; readonly operation: MlxHeadForward }
