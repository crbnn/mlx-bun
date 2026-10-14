import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { Cache } from "../../contracts/mlx/cache";
import type { MlxPromptRange } from "../../contracts/mlx/graph";
import type { Gemma4Phase } from "./blocks";

/** Compatibility boundary for a caller's whole-sequence scoring operation.
 * The table is the graph's declared prompt ranges, indexed by caller rows. */
export function writtenSequence(ranges: readonly MlxPromptRange[], ids: MlxArray, caches: Cache[], rows: number): MlxArray {
  const range = ranges.find(range => rows >= range.from && rows <= range.to)!;
  return range.operation(ids, caches, null) as MlxArray;
}

/** Prepared request data is normalized before entering the named media phase.
 * The row range selects only an operation the file declared. */
export function writtenMedia(graph: {
  readonly embedScale: number;
  readonly perLayerWidth: number;
  readonly embeddingPrompt: readonly { from: number; to: number; operation(hidden: MlxArray, caches: Cache[], phase: Gemma4Phase, ids: MlxArray): MlxArray }[];
}, embeds: MlxArray, caches: Cache[], bidir: MlxArray | null, ids: MlxArray, multimodal: MlxArray | null): MlxArray {
  const hidden = ops.mulScalar(embeds, graph.embedScale);
  const zeroing = multimodal ?? bidir;
  let inputIds = ids;
  if (graph.perLayerWidth && zeroing) {
    using condition = ops.reshape(zeroing, ids.shape);
    using zero = ops.zeros(ids.shape, ids.dtype);
    inputIds = ops.where(condition, zero, ids);
  }
  const rows = ids.shape[0]! * ids.shape[1]!;
  const range = graph.embeddingPrompt.find(range => rows >= range.from && rows <= range.to)!;
  try { return range.operation(hidden, caches, bidir ? { bidirectional: bidir } : "window", inputIds); }
  finally { if (inputIds !== ids) inputIds.dispose(); }
}
