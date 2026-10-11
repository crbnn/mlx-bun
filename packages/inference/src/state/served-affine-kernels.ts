import type { Dtype } from "@mlx-bun/mlx/ffi";
import { fusedSdpaConfigSupported } from "../layers/quantized-attention";
import { runtimeValue } from "../runtime/config";
import { tiledCausalAffineKernels, unfusedAffineKernels, type AffineKernels } from "./affine-attention";

/** The kernel set an affine cache that the loader did not compose attends
 * with: the numbers the graphs' deprecated `quantizedSdpa` dispatch gives today
 * under the flag the app writes for its KV scheme. `MLX_BUN_NO_FUSED_SDPA=1`
 * (uniform `--kv-quant N`, mlx-lm parity): unfused. Otherwise (`--kv-quant
 * config`, OptiQ parity): tiled while the cache's own mask is causal, which
 * needs a configuration the tiled port takes; `quantizedSdpa` runs any other
 * configuration unfused, and so does the cache. Read where `kv-maintenance`
 * converts a cache and where `persistence` restores one.
 * E3 transition: E3 composes the set from the KV scheme at load and deletes
 * the flag; this read of it goes then. */
export function servedAffineKernels(): (bits: number, groupSize: number, dtype: Dtype) => AffineKernels {
  if (runtimeValue("MLX_BUN_NO_FUSED_SDPA") === "1") return unfusedAffineKernels;
  return (bits, groupSize, dtype) => fusedSdpaConfigSupported(groupSize, bits, dtype)
    ? tiledCausalAffineKernels(bits, groupSize, dtype) : unfusedAffineKernels(bits, groupSize, dtype);
}
