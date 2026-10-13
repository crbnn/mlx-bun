import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { MetalKernel } from "@mlx-bun/mlx/metal-kernel";
import { HEADER } from "./codebook";
import type { TrellisGeometry } from "./geometry";

// Down projection on the simdgroup matrix unit for any number of activation
// rows: down-mma's sub-tiles, splits and accumulation (each output row bit for
// bit the same), with every decoded B fragment feeding an MMA for MT = 2
// blocks of 8 rows; grid z walks chunks of 16 rows. The row count is read at
// run time (one pipeline per geometry). SPLITS f32 partials [M, SPLITS, C]
// fold in one MLX sum. Two kernels, one per code layout: row-major
// [rows, cols·K/32] and the 3-bit block interleave [cols/512, rows, 48].
// Measured on the M1 Max at the Qwen down shape (17408 -> 5120, 3-bit
// interleaved): 9 rows 0.84 ms, 16 0.85, 32 1.35, 64 2.56, 128 4.99, against
// 1.74, 1.67, 2.16, 2.89 and 4.10 for weight expansion and GEMM.
const SG_TG = 4, MT = 2, S = 16, SPLITS = 8;

const SOURCE = String.raw`
  const uint M = (uint)x_shape[0];
  const uint lane = thread_index_in_simdgroup;
  const uint id = threadgroup_position_in_grid.y * (uint)SG_TG + simdgroup_index_in_threadgroup;
  const uint groupsPerBlock = 32u / (uint)S;
  const uint nGroups = ((uint)C / 256u) * groupsPerBlock;
  if (id >= nGroups * (uint)SPLITS) return;
  const uint cg = id % nGroups, split = id / nGroups;
  const uint block = cg / groupsPerBlock, pg = (cg % groupsPerBlock) * 8u * (uint)S;
  const uint fm = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
  const uint fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
  const uint wpb = 8u * (uint)K;
  const ulong rowWords = (ulong)(INTERLEAVE ? 2u : (uint)C / 256u) * wpb;
  const device uint32_t* words = codes + (INTERLEAVE
    ? (ulong)(block / 2u) * (ulong)R * 2u * wpb + (block % 2u) * wpb : (ulong)block * wpb);
  // This lane's 2S positions start at p0; realign its bit window to bit 0.
  const uint p0 = pg + (uint)S * fn;
  const uint bit0 = p0 * (uint)K, w0 = bit0 >> 5, off0 = bit0 & 31u;
  const uint mBase = threadgroup_position_in_grid.z * 8u * (uint)MT;
  const uint rowsPer = (((uint)R + (uint)SPLITS - 1u) / (uint)SPLITS + 7u) & ~7u;
  const uint r0 = split * rowsPer;
  const uint r1 = metal::min((uint)R, r0 + rowsPer);
  metal::simdgroup_float8x8 acc[S][MT];
  #pragma clang loop unroll(full)
  for (uint s = 0; s < (uint)S; ++s)
    #pragma clang loop unroll(full)
    for (uint bk = 0; bk < (uint)MT; ++bk) acc[s][bk] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint r = r0; r < r1; r += 8u) {
    const uint rw = r + fm;
    const device uint32_t* rowCodes = words + (ulong)metal::min(rw, (uint)R - 1u) * rowWords;
    uint raw[NW + 1];
    #pragma clang loop unroll(full)
    for (uint i = 0; i <= (uint)NW; ++i) {
      const uint wi = w0 + i;
      raw[i] = rowCodes[wi % wpb];
    }
    uint w[NW];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < (uint)NW; ++i) w[i] = off0 ? (raw[i] >> off0) | (raw[i + 1] << (32u - off0)) : raw[i];
    metal::simdgroup_float8x8 a[MT];
    const uint ra = r + fn;
    const float sa = ra < r1 ? float(scales[ra]) * TRELLIS_RCP : 0.0f;
    const float sb = ra + 1u < r1 ? float(scales[ra + 1u]) * TRELLIS_RCP : 0.0f;
    #pragma clang loop unroll(full)
    for (uint bk = 0; bk < (uint)MT; ++bk) {
      const uint m = mBase + 8u * bk + fm;
      const bool ok = m < M;
      const device T* xm = x + (ulong)(ok ? m : 0u) * (ulong)R;
      a[bk].thread_elements()[0] = ok && ra < r1 ? float(xm[ra]) * sa : 0.0f;
      a[bk].thread_elements()[1] = ok && ra + 1u < r1 ? float(xm[ra + 1u]) * sb : 0.0f;
    }
    const bool rowOk = rw < r1;
    #pragma clang loop unroll(full)
    for (uint s = 0; s < (uint)S; ++s) {
      metal::simdgroup_float8x8 b;
      b.thread_elements()[0] = rowOk ? trellis_y_f32((block_state<K, L>(w, s))) : 0.0f;
      b.thread_elements()[1] = rowOk ? trellis_y_f32((block_state<K, L>(w, s + (uint)S))) : 0.0f;
      #pragma clang loop unroll(full)
      for (uint bk = 0; bk < (uint)MT; ++bk)
        metal::simdgroup_multiply_accumulate(acc[s][bk], a[bk], b, acc[s][bk]);
    }
  }
  #pragma clang loop unroll(full)
  for (uint bk = 0; bk < (uint)MT; ++bk) {
    const uint m = mBase + 8u * bk + fm;
    if (m >= M) continue;
    device float* out = partial + ((ulong)m * (ulong)SPLITS + split) * (ulong)C + (ulong)block * 256u + 255u - p0;
    #pragma clang loop unroll(full)
    for (uint s = 0; s < (uint)S; ++s) {
      out[-(int)s] = acc[s][bk].thread_elements()[0];
      out[-(int)((uint)S + s)] = acc[s][bk].thread_elements()[1];
    }
  }
`;

// State at position t of a realigned window (bit t·K of w, LSB first).
const STATE_HEADER = String.raw`
template <int K, int L_>
inline uint block_state(thread const uint* w, uint t) {
  const uint b = t * (uint)K;
  const uint wi = b >> 5;
  const uint off = b & 31u;
  uint win = w[wi] >> off;
  if (off + (uint)L_ > 32u) win |= w[wi + 1] << (32u - off);
  return win & ((1u << (uint)L_) - 1u);
}
`;

function launch(kernel: MetalKernel, x2: MlxArray, codes: MlxArray, scales: MlxArray, g: TrellisGeometry, interleave: 0 | 1): MlxArray {
  const M = x2.shape[0]!;
  if (x2.ndim !== 2 || M < 1) throw new Error("downMmaWide: expects [rows, inputs]");
  if (g.axis !== 0 || g.T !== 256 || g.cols % 256 !== 0 || g.L > 12) throw new Error("downMmaWide: unsupported geometry");
  const nGroups = (g.cols / 256) * (32 / S);
  // Words covering 2S positions plus the L-bit window, after realignment.
  const NW = Math.ceil((2 * S * g.k + g.L - 1) / 32) + 1;
  const [partial] = kernel.apply([x2, codes, scales], {
    outputs: [{ shape: [M, SPLITS, g.cols], dtype: Dtype.float32 }],
    grid: [32 * SG_TG, Math.ceil((nGroups * SPLITS) / SG_TG), Math.ceil(M / (8 * MT))], threadGroup: [32 * SG_TG, 1, 1],
    templateDtypes: { T: x2.dtype },
    templateInts: { R: g.rows, C: g.cols, K: g.k, L: g.L, SPLITS, SG_TG, INTERLEAVE: interleave, MT, S, NW },
  });
  using sum = ops.sumAxis(partial!, 1, false);
  partial!.dispose();
  return sum.astype(x2.dtype);
}

let rowMajor: MetalKernel | undefined, interleaved: MetalKernel | undefined;

/** x2 [M, inFeatures] · W → [M, outFeatures] in x2's dtype for row-major axis-0 codes (T 256), any number of rows.
 *  Borrows inputs. */
export function downMmaWide(x2: MlxArray, codes: MlxArray, scales: MlxArray, g: TrellisGeometry): MlxArray {
  if (g.blockInterleave || codes.ndim !== 2) throw new Error("downMmaWide: needs row-major codes");
  rowMajor ??= new MetalKernel({ name: "mlx_bun_trellis_down_mma_wide", inputNames: ["x", "codes", "scales"],
    outputNames: ["partial"], source: SOURCE, header: HEADER + STATE_HEADER, ensureRowContiguous: true });
  return launch(rowMajor, x2, codes, scales, g, 0);
}

/** downMmaWide for 3-bit block-interleaved codes [cols/512, rows, 48]. */
export function downK3InterleavedMmaWide(x2: MlxArray, codes: MlxArray, scales: MlxArray, g: TrellisGeometry): MlxArray {
  if (g.blockInterleave !== 2 || g.k !== 3) throw new Error("downK3InterleavedMmaWide: needs 3-bit block-interleaved codes");
  interleaved ??= new MetalKernel({ name: "mlx_bun_trellis_down_k3i_mma_wide", inputNames: ["x", "codes", "scales"],
    outputNames: ["partial"], source: SOURCE, header: HEADER + STATE_HEADER, ensureRowContiguous: true });
  return launch(interleaved, x2, codes, scales, g, 1);
}
