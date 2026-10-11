import { kv4DecodeAttention } from "../layers/kv4-decode-attention";
import { foldedQuantizedSdpa } from "../layers/quantized-attention";
import { unfusedKernel, type AffineKernels } from "./affine-attention";
import { AttentionMasks } from "./attention-read";
import { QuantizedKVCache } from "./quantized-kv";

/** The kernels of the Qwen3.8-27B Trellis M4 Pro composition's full-attention
 * layers for one sequence: the fused one-row KV4 decode kernel
 * (`kv4DecodeAttention`, each packed key and value row read once for every
 * query head of its KV head) and the folded-GQA window kernel
 * (`foldedQuantizedSdpa`, one quantized matmul per KV head for its query rows,
 * causal over the last L positions). Neither reads the mask, and both take one
 * row (B = 1), head dim 256 and an even GQA factor. `multiRow` is the
 * composition's affine set, which a row layout reads with whenever it holds
 * more than one row or one behind left padding: the old graph served several
 * sequences on the generic path. A masked window on serial storage takes the
 * unfused port, which this storage never needs (its mask is plain causal). */
export function kv4Head256Kernels(multiRow: AffineKernels): AffineKernels {
  return {
    decode: (q, keys, values, scale) => kv4DecodeAttention(q, keys, values, scale, 64, 4),
    window: (q, keys, values, scale) => foldedQuantizedSdpa(q, keys, values, scale, 64, 4),
    maskedWindow: unfusedKernel(64, 4),
    multiRow,
  };
}

/** The 4-bit group-64 head-dim-256 full-attention storage the M4 Pro Qwen3.8
 * composition reads, quantized from its first append: `QuantizedKVCache`
 * storage with `kv4Head256Kernels`. The served lego is the bf16-first
 * `Bf16FirstKv4Head256Cache` (`state/bf16-first-kv.ts`), whose converted
 * storage reads with these kernels; this cache is the quantized-from-token-zero
 * composition. The committed read is `QuantizedKVCache`'s.
 *
 * Each read is bit-identical to the kernel the old graph injected for one
 * sequence; neither kernel is bit-identical to the unfused port (f32 online
 * softmax, regrouped rows: bf16-level). A one-row window takes the folded
 * kernel, where the old graph's one-row plan took the KV4 decode kernel. Row
 * layouts made from this cache keep its kernels, `multiRow` included. */
export class Kv4Head256Cache extends QuantizedKVCache {
  /** `multiRow`: the composition's affine kernels for 4-bit group 64. */
  constructor(multiRow: AffineKernels, masks = new AttentionMasks()) {
    super(64, 4, kv4Head256Kernels(multiRow), masks);
  }
}
