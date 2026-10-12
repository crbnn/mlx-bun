import type { SpeculativeRowLayout } from "../contracts/mlx/cache";

/** The verification rows a storage's row-layout declaration names
 * (`state/layout`'s table). */
export type TargetRowLayout = SpeculativeRowLayout;

export { targetRowLayoutFactory } from "./layout";
