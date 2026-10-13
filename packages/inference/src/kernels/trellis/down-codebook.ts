import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { MetalKernel } from "@mlx-bun/mlx/metal-kernel";
import type { TrellisGeometry } from "./geometry";
import { HEADER } from "./codebook";

// Axis-0 (down) projection of one activation row over T 256 Trellis codes,
// decoding through a threadgroup codebook (every code value y·(1/147.8) of the
// 2^L states, generated once per threadgroup). Each lane owns P consecutive
// positions of one block whose bits start on a word boundary (P·K a multiple of
// 32: NW words plus the next lane's first word for the last windows), so every
// window shift is a compile-time constant; it loads U input rows' words ahead
// (no lane exchanges data, so lanes past the last block return at once).
// Split-K over input rows into SPLITS f32 partials folded by one MLX sum. Every
// position accumulates the same f32 code·scale weights over the same rows in
// the same order as trellisScatter's single-row kernels at variant 13 (the
// balanced one for the 3-bit block interleave), so the output is theirs bit
// for bit.
const SPLITS = 128, THREADS = 256, U = 2;

const SOURCE = String.raw`
  threadgroup float codebook[1 << L];
  for (uint i = thread_position_in_threadgroup.x; i < (1u << (uint)L); i += (uint)THREADS)
    codebook[i] = trellis_unrefined_bits_y(trellis_y(i));
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  constexpr uint NW = (uint)(P * K / 32);            // words a lane owns
  constexpr uint LB = 256u / (uint)P;                // lanes per block
  constexpr uint WPB = 8u * (uint)K;                 // words per block
  constexpr uint NB = (uint)(C / 256);               // blocks
  constexpr uint SGS = (NB + 32u / LB - 1u) / (32u / LB); // SIMD groups per split
  constexpr ulong ROWW = INTERLEAVE ? 2ul * WPB : (ulong)(C / 256) * WPB;
  const uint lane = thread_index_in_simdgroup;
  const uint group = thread_position_in_grid.y * (uint)SG_TG + simdgroup_index_in_threadgroup;
  const uint split = group / SGS;
  const uint li = lane % LB;
  const uint blk = (group - split * SGS) * (32u / LB) + lane / LB;
  if (blk >= NB) return;
  const device uint32_t* blockWords = codes + (INTERLEAVE
    ? (ulong)(blk / 2u) * (ulong)R * 2ul * WPB + (blk % 2u) * WPB : (ulong)blk * WPB);
  const device uint32_t* own = blockWords + li * NW;
  const device uint32_t* spill = blockWords + ((li + 1u) * NW) % WPB;
  const uint rowsPer = ((uint)R + (uint)SPLITS - 1u) / (uint)SPLITS;
  const uint r0 = split * rowsPer;
  const uint r1 = metal::min((uint)R, r0 + rowsPer);
  float acc[P];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < (uint)P; ++i) acc[i] = 0.0f;
  uint r = r0;
  for (; r + (uint)U <= r1; r += (uint)U) {
    uint w[U][NW + 1];
    float xr[U], sr[U];
    #pragma clang loop unroll(full)
    for (uint u = 0; u < (uint)U; ++u) {
      #pragma clang loop unroll(full)
      for (uint i = 0; i < NW; ++i) w[u][i] = own[(ulong)(r + u) * ROWW + i];
      w[u][NW] = spill[(ulong)(r + u) * ROWW];
      xr[u] = float(x[r + u]);
      sr[u] = float(scales[r + u]);
    }
    #pragma clang loop unroll(full)
    for (uint u = 0; u < (uint)U; ++u) {
      #pragma clang loop unroll(full)
      for (uint i = 0; i < (uint)P; ++i) {
        const uint bit = i * (uint)K, wi = bit >> 5, off = bit & 31u;
        uint win = w[u][wi] >> off;
        if (off + (uint)L > 32u) win |= w[u][wi + 1] << (32u - off);
        win &= (1u << (uint)L) - 1u;
        acc[i] = metal::fma(codebook[win] * sr[u], xr[u], acc[i]);
      }
    }
  }
  for (; r < r1; ++r) {
    uint w[NW + 1];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < NW; ++i) w[i] = own[(ulong)r * ROWW + i];
    w[NW] = spill[(ulong)r * ROWW];
    const float xr = float(x[r]);
    const float sr = float(scales[r]);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < (uint)P; ++i) {
      const uint bit = i * (uint)K, wi = bit >> 5, off = bit & 31u;
      uint win = w[wi] >> off;
      if (off + (uint)L > 32u) win |= w[wi + 1] << (32u - off);
      win &= (1u << (uint)L) - 1u;
      acc[i] = metal::fma(codebook[win] * sr, xr, acc[i]);
    }
  }
  const ulong out = (ulong)split * (ulong)C + (ulong)blk * 256ul + 255ul - li * (uint)P;
  #pragma clang loop unroll(full)
  for (uint i = 0; i < (uint)P; ++i) partial[out - i] = acc[i];
`;

let kernel: MetalKernel | undefined;

function launch(x2: MlxArray, codes: MlxArray, scales: MlxArray, g: TrellisGeometry, positions: number, interleave: boolean): MlxArray {
  const groupsPerSplit = Math.ceil((g.cols / 256) / (positions / 8));
  if (x2.shape[0] !== 1) throw new Error("downCodebookRow: expects one activation row");
  if (g.axis !== 0 || g.T !== 256 || g.L > 12 || g.cols % 256) throw new Error("downCodebookRow: needs axis-0 codes at T 256, L <= 12");
  kernel ??= new MetalKernel({ name: "mlx_bun_trellis_down_codebook_row", inputNames: ["x", "codes", "scales"],
    outputNames: ["partial"], source: SOURCE, header: HEADER, ensureRowContiguous: true });
  const [partial] = kernel.apply([x2, codes, scales], {
    outputs: [{ shape: [1, SPLITS, g.cols], dtype: Dtype.float32 }],
    grid: [THREADS, SPLITS * groupsPerSplit / (THREADS / 32), 1],
    threadGroup: [THREADS, 1, 1],
    templateDtypes: { T: x2.dtype },
    templateInts: { R: g.rows, C: g.cols, K: g.k, L: g.L, P: positions, U, SPLITS, SG_TG: THREADS / 32, THREADS,
      INTERLEAVE: Number(interleave) },
  });
  const sum = ops.sumAxis(partial!, 1, false);
  partial!.dispose();
  const out = sum.astype(x2.dtype);
  sum.dispose();
  return out;
}

/** x2 [1, inFeatures] · W → [1, outFeatures] in x2's dtype over row-major 2- or
 *  4-bit codes [rows, cols·K/32]. Borrows inputs. */
export function downCodebookRow(x2: MlxArray, codes: MlxArray, scales: MlxArray, g: TrellisGeometry): MlxArray {
  if (g.blockInterleave || codes.ndim !== 2 || (g.k !== 2 && g.k !== 4)) throw new Error("downCodebookRow: needs row-major 2- or 4-bit codes");
  return launch(x2, codes, scales, g, 16, false);
}

/** The same over 3-bit block-interleaved codes [cols/512, rows, 48]. */
export function downK3InterleavedCodebookRow(x2: MlxArray, codes: MlxArray, scales: MlxArray, g: TrellisGeometry): MlxArray {
  if (g.blockInterleave !== 2 || g.k !== 3) throw new Error("downK3InterleavedCodebookRow: needs 3-bit block-interleaved codes");
  return launch(x2, codes, scales, g, 32, true);
}
