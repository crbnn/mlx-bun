import type { MlxArray } from "@mlx-bun/mlx/array";
import type { Cache } from "./cache";
import type { MlxLayerCapture, MlxPhaseGroup } from "./graph";

/** One forward a method hands to the mixed iteration: the phase its rows run
 * in, their count and width class, the drafter's capture, and the forward
 * those rows run on their own, which the iteration calls when it does not pack
 * them with another group. */
export interface MlxForwardStep {
  readonly phase: MlxPhaseGroup["phase"];
  /** Rows: sequences times positions. */
  readonly rows: number;
  readonly capture: MlxLayerCapture | null;
  own(): Promise<MlxArray>;
}

export type MlxForwardWork = (ids: MlxArray, cache: Cache[], step: MlxForwardStep) => Promise<MlxArray>;

export interface MlxPreparationWork {
  /** Real prompt tokens across all rows, not a padded sequence width. */
  readonly maxTokens?: number;
  readonly forward?: MlxForwardWork;
}
