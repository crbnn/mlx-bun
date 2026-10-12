import type { MlxArray } from "@mlx-bun/mlx/array";
import type { Cache } from "../contracts/mlx/cache";
import type { MlxPromptInput } from "../contracts/mlx/media";
import type { MlxWidthClass } from "../contracts/mlx/width";

/** Preserve the ordinary generator's projection geometry for a prepared
 * embedding sequence: process the whole prompt, then project only its tip. */
export function bindEmbeddingsInput(
  prefill: (ids: MlxArray, caches: Cache[], start: number, phase: "tail" | "decode", width: MlxWidthClass) => MlxArray,
): MlxPromptInput {
  return { prefill(ids, caches, start, phase, width) {
    using hidden = prefill(ids, caches, start, phase, width);
    const [batch, length, hiddenWidth] = hidden.shape;
    return hidden.slice([0, length! - 1, 0], [batch!, length!, hiddenWidth!]);
  } };
}
