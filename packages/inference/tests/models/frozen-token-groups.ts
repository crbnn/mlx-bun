// NOT A PATTERN TO FOLLOW: the mixed-forward groups the frozen reference graphs
// take, as contracts/mlx/token-work.ts and input/token-groups.ts defined them
// before E2 replaced them with the named phase groups (MlxPhaseGroup). Kept
// only so the frozen references stay byte-for-byte their old selves; delete
// with them.
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { Cache } from "../../src/contracts/mlx/cache";
import { mapPackedTokens } from "../../src/input/token-groups";

export interface TokenGroup {
  readonly ids: MlxArray;
  readonly cache: Cache[];
  /** Verification keeps its established GEMV/GEMM geometry. */
  readonly preserveTokenGeometry?: boolean;
  /** Borrows a residual stream; the layer-count index is the post-final-norm output. */
  readonly captureLayer?: (layer: number, hidden: MlxArray) => void;
}

/** The old `mapTokenGroups`: groups that may share a tokenwise block packed into
 *  one call, a geometry-preserving group (and every group when `pack` is off) on its own. */
export function mapTokenGroups(groups: readonly TokenGroup[], inputs: readonly MlxArray[],
  operation: (hidden: MlxArray) => MlxArray, pack = true): MlxArray[] {
  const results: MlxArray[] = [];
  const packedRows = groups.flatMap((group, row) => pack && !group.preserveTokenGeometry ? [row] : []);
  try {
    const outputs = packedRows.length ? mapPackedTokens(packedRows.map(row => inputs[row]!), operation) : [];
    for (const [index, row] of packedRows.entries()) results[row] = outputs[index]!;
    for (let row = 0; row < groups.length; row++) results[row] ??= operation(inputs[row]!);
    return results;
  } catch (error) { for (const result of results) result?.dispose(); throw error; }
}
