// `quantizedSdpa` as the graphs called it before E3, frozen for the references
// and gates that compare against those graphs: a window (L > 1) through the
// N-tiled port where `fusedSdpaSupported` held, which read
// MLX_BUN_NO_FUSED_SDPA on every call, and every other read through the
// unfused port. E3 composes that choice into the KV scheme
// (`KvStorageScheme.kernels`, `composedAffineKernels`); no source file reads
// the flag.
import type { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import type { Cache, KvAttentionView, Mask } from "../../src/contracts/mlx/cache";
import { quantizedAppendAttention } from "../../src/layers/quantized-append-attention";
import { fusedSdpaConfigSupported, quantizedSdpaTiled, quantizedSdpaUnfused } from "../../src/layers/quantized-attention";
import { disposeTriple } from "../../src/state/quantized-tensor";
import { isCompiledTrace } from "../../src/runtime/compiled-trace";
import { runtimeValue } from "../../src/runtime/config";
import { kvStorage } from "../../src/state/kv-storage";

/** The runtime half of `fusedSdpaSupported` before E3 (the flag, the activation
 * dtype and the mask), which the frozen pre-D2 generated graphs
 * (gemma4-generated-reference) call with the (bits, group size) half baked in. */
export function fusedSdpaRuntimeOk(q: MlxArray, mask: Mask): boolean {
  if (runtimeValue("MLX_BUN_NO_FUSED_SDPA") === "1") return false;
  if (q.dtype !== Dtype.bfloat16 && q.dtype !== Dtype.float16) return false;
  if (mask.mode === "causal" || mask.mode === "") return true;
  if (mask.mode === "array")
    return mask.causalEquivalent === true && mask.arr !== null &&
      mask.arr.shape.length === 2 && mask.arr.dtype === Dtype.bool;
  return false;
}

function fusedSdpaSupported(q: MlxArray, mask: Mask, groupSize: number, bits: number): boolean {
  return fusedSdpaConfigSupported(groupSize, bits, q.dtype) && fusedSdpaRuntimeOk(q, mask);
}

/** `captureKvAttention` as the frozen graphs called it before E3: a row
 * layout's own view, affine storage through `quantizedSdpa` above (or the
 * committed append for independent positions), plain storage through SDPA. */
export function referenceKvAttention(cache: Cache, k: MlxArray, v: MlxArray): KvAttentionView {
  if (kvStorage(cache).attentionState) return kvStorage(cache).attentionState!.appendAndFetch(k, v);
  const quantized = kvStorage(cache).quantizedAttention!;
  if (quantized) {
    const { groupSize, bits } = quantized;
    const [keys, values] = quantized.updateAndFetchQuantized(k, v);
    return {
      attend: (q, scale, mask, independentPositions) => independentPositions && q.shape[2]! > 1
        ? quantizedAppendAttention(q, keys, values, scale, groupSize, bits)
        : quantizedSdpa(q, keys, values, scale, mask, groupSize, bits),
      dispose() { disposeTriple(keys); disposeTriple(values); },
    };
  }
  const [keys, values] = kvStorage(cache).updateAndFetch(k, v);
  return {
    attend: (q, scale, mask) => ops.sdpa(q, keys, values, scale, mask.mode, mask.arr),
    dispose() { keys.dispose(); values.dispose(); },
  };
}

export function quantizedSdpa(
  q: MlxArray, kq: ops.QuantizedTensor, vq: ops.QuantizedTensor,
  scale: number, mask: Mask, groupSize: number, bits: number,
): MlxArray {
  if (q.shape[2]! > 1 && fusedSdpaSupported(q, mask, groupSize, bits)) {
    if (isCompiledTrace())
      throw new Error("quantizedSdpaTiled inside a compiled trace: the tile loop bakes trace-time N");
    return quantizedSdpaTiled(q, kq, vq, scale, mask, groupSize, bits);
  }
  return quantizedSdpaUnfused(q, kq, vq, scale, mask, groupSize, bits);
}
