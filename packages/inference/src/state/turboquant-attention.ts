import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { AttentionRead } from "../contracts/mlx/cache";
import * as tq from "../kernels/turboquant/ops";
import { sdpaRead, withLease, type MaskLease } from "./attention-read";

/** One append's window decoded with its values still in the codec's rotated
 * domain, and the transform that returns an attention output over those
 * values to the values' own domain. The caller owns `keys` and `values`. */
export interface RotatedWindow {
  readonly keys: MlxArray;
  readonly values: MlxArray;
  /** Borrows an attention output [B, Hq, L, Dv] and returns an owned one. */
  readonly restore: (output: MlxArray) => MlxArray;
}

/** The two ways TurboQuant storage can append one call's keys and values and
 * decode its window. A read calls exactly one of them, once. */
export interface TurboQuantFetches {
  /** Append, then decode the window with keys and values in their own domain
   * (`updateAndFetch`). The caller owns both. */
  decoded(): [MlxArray, MlxArray];
  /** Append, then decode the window with values left rotated
   * (`updateAndFetchDeferredV`). */
  rotated(): RotatedWindow;
}

/** How a TurboQuant cache attends, fixed when the cache is built, like the
 * affine kernel sets. The cache leases the mask for the phase and offers its
 * fetches; the set decides which fetch runs and what attends over it. Both sets
 * below are a graph's served numbers today, so the graph that composes the
 * cache names its set: `denseTurboQuantKernels` for Qwen 3.5's,
 * `deferredTurboQuantKernels` for Gemma 4's and MiniCPM5's. */
export interface TurboQuantKernels {
  /** The fetch `read` calls: the window decoded to the values' own domain, or
   * with values left rotated. A compiled decode step decodes the stored prefix
   * in this domain outside its recorded graph (`TurboQuantKVCache`). */
  readonly fetch: "decoded" | "rotated";
  /** Appends through `fetches` and returns the read under `mask`. Takes
   * ownership of `mask` (released if the fetch throws) and of what the fetch
   * returns. */
  read(fetches: TurboQuantFetches, mask: MaskLease): AttentionRead;
}

/** The read over fetched keys and values still in the codec's rotated domain:
 * the stock fused SDPA under the leased mask, then `restore` on its output
 * (the inverse value rotation, f32 through the transform, unless the storage
 * says otherwise). Attention is linear in V, so the caller receives the output
 * in the values' own domain. Takes ownership of `keys`, `values` and `mask`. */
function rotatedValueRead(keys: MlxArray, values: MlxArray, mask: MaskLease,
  restore: (output: MlxArray) => MlxArray = tq.unrotateValues): AttentionRead {
  return {
    attend(q, scale) {
      const rotated = ops.sdpa(q, keys, values, scale, mask.mask.mode, mask.mask.arr);
      try { return restore(rotated); } finally { rotated.dispose(); }
    },
    dispose() { keys.dispose(); values.dispose(); mask.release(); },
  };
}

/** Decode keys and values back to their own domain, then the stock fused SDPA
 * under the mask. Qwen 3.5's graph composes this set: its attention block read
 * TurboQuant storage this way (`updateAndFetch`, then `ops.sdpa`) before it read
 * through the cache, and these are its served numbers. */
export const denseTurboQuantKernels: TurboQuantKernels = Object.freeze({
  fetch: "decoded" as const,
  read: (fetches: TurboQuantFetches, mask: MaskLease) => withLease(mask, held => {
    const [keys, values] = fetches.decoded();
    return sdpaRead(keys, values, held);
  }),
});

/** Leave values rotated, run the stock fused SDPA under the mask on them, then
 * undo the rotation on the output (B1c's read). Gemma 4's and MiniCPM5's graphs
 * compose this set: it is the sequence they run after
 * `updateAndFetchDeferredV` today, so it keeps their served numbers when they
 * read through the cache. */
export const deferredTurboQuantKernels: TurboQuantKernels = Object.freeze({
  fetch: "rotated" as const,
  read: (fetches: TurboQuantFetches, mask: MaskLease) => withLease(mask, held => {
    const { keys, values, restore } = fetches.rotated();
    return rotatedValueRead(keys, values, held, restore);
  }),
});

/** The set of plain storage whose graph composes no TurboQuant read: every
 * graph but Qwen 3.5's, Gemma 4's and MiniCPM5's reads converted TurboQuant
 * storage through the deprecated fetches, which take no set. A TurboQuant cache
 * converted from such storage serves those fetches; its named reads throw. */
export const uncomposedTurboQuantKernels: TurboQuantKernels = Object.freeze({
  fetch: "decoded" as const,
  read(_fetches: TurboQuantFetches, mask: MaskLease): AttentionRead {
    mask.release();
    throw new Error("this TurboQuant cache was converted from storage whose graph composed no TurboQuant read: " +
      "the graph's makeCache passes denseTurboQuantKernels or deferredTurboQuantKernels (state/turboquant-attention.ts) to its KVCache");
  },
});
