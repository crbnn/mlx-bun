import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { MetalKernel } from "@mlx-bun/mlx/metal-kernel";
import type { TrellisGeometry } from "./geometry";
import { CODEBOOK_WINDOW, HEADER } from "./codebook";
import { TRELLIS_CODEBOOK_DOWN_THREADS } from "./launch";

// Axis-0 (down) projection of one activation row over T 256 Trellis codes,
// decoding through a threadgroup codebook (every code value y·(1/147.8) of the
// 2^L states, generated once per threadgroup). Each lane owns P = 16
// consecutive positions of one block: it loads the words they span (plus the
// next lane's first word for the last windows) for U input rows ahead, aligns
// them to its first bit once per row where P·K is not a multiple of 32 (the
// 3-bit codes, whose odd lanes start at bit 16), then reads every window with
// one bit-field extract at a compile-time shift (CODEBOOK_WINDOW, codebook.ts).
// No lane exchanges data, so lanes past the last block return at once.
// Split-K over input rows into SPLITS f32 partials folded by one MLX sum.
// Every position accumulates the same f32 code·scale weights over the same
// rows in the same order as trellisScatter's single-row kernels at variant 13
// (the balanced one for the 3-bit block interleave), so the output is theirs
// bit for bit.
const SPLITS = 128, P = 16, U = 2;

const SOURCE = String.raw`
  threadgroup float codebook[1 << L];
  for (uint i = thread_position_in_threadgroup.x; i < (1u << (uint)L); i += (uint)THREADS)
    codebook[i] = trellis_unrefined_bits_y(trellis_y(i));
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  constexpr uint BITS = (uint)(P * K);               // a lane's bits
  constexpr uint SHIFT = BITS % 32u;                 // 0, or 16: odd lanes start at bit 16 of a word
  constexpr bool ALIGNED = SHIFT == 0u;
  constexpr uint NA = (BITS + (uint)L - (uint)K + 31u) / 32u;  // words its windows span from its first bit
  constexpr uint NL = ALIGNED ? NA : NA + 1u;        // words it loads
  constexpr uint LB = 256u / (uint)P;                // lanes per block
  constexpr uint WPB = 8u * (uint)K;                 // words per block
  constexpr uint NB = (uint)(C / 256);               // blocks
  constexpr uint SGS = (NB + 32u / LB - 1u) / (32u / LB); // SIMD groups per split
  constexpr ulong ROWW = INTERLEAVE ? 2ul * WPB : (ulong)NB * WPB;
  constexpr uint ROWS_PER = ((uint)R + (uint)SPLITS - 1u) / (uint)SPLITS;
  const uint lane = thread_index_in_simdgroup;
  const uint group = thread_position_in_grid.y * (uint)SG_TG + simdgroup_index_in_threadgroup;
  const uint split = group / SGS;
  const uint li = lane % LB;
  const uint blk = (group - split * SGS) * (32u / LB) + lane / LB;
  if (blk >= NB) return;
  const device uint32_t* blockWords = codes + (INTERLEAVE
    ? (ulong)(blk / 2u) * (ulong)R * 2ul * WPB + (blk % 2u) * WPB : (ulong)blk * WPB);
  const uint first = (li * BITS) >> 5;
  const bool shifted = !ALIGNED && (li & 1u);
  uint idx[NL];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < NL; ++i) idx[i] = (first + i) % WPB;
  const uint r0 = split * ROWS_PER;
  const uint rows = (uint)R % (uint)SPLITS == 0u ? ROWS_PER : r0 >= (uint)R ? 0u : metal::min(ROWS_PER, (uint)R - r0);
  const threadgroup uchar* table = (const threadgroup uchar*)codebook;
  float acc[P];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < (uint)P; ++i) acc[i] = 0.0f;
  uint r = 0;
  for (; r + (uint)U <= rows; r += (uint)U) {
    uint ld[U][NL];
    float xr[U], sr[U];
    #pragma clang loop unroll(full)
    for (uint u = 0; u < (uint)U; ++u) {
      const device uint32_t* rowWords = blockWords + (ulong)(r0 + r + u) * ROWW;
      #pragma clang loop unroll(full)
      for (uint i = 0; i < NL; ++i) ld[u][i] = rowWords[idx[i]];
      xr[u] = float(x[r0 + r + u]);
      sr[u] = float(scales[r0 + r + u]);
    }
    #pragma clang loop unroll(full)
    for (uint u = 0; u < (uint)U; ++u) {
      uint w[NA], v[NA];
      #pragma clang loop unroll(full)
      for (uint i = 0; i < NA; ++i) w[i] = shifted ? (ld[u][i] >> SHIFT) | (ld[u][i + 1] << (32u - SHIFT)) : ld[u][i];
      v[0] = w[0] << 16;
      #pragma clang loop unroll(full)
      for (uint i = 1; i < NA; ++i) v[i] = (w[i - 1] >> 16) | (w[i] << 16);
      #pragma clang loop unroll(full)
      for (uint i = 0; i < (uint)P; ++i) {
        const float code = trellis_codebook_window(table, w, v, i * (uint)K, (uint)L);
        acc[i] = metal::fma(code * sr[u], xr[u], acc[i]);
      }
    }
  }
  for (; r < rows; ++r) {
    const device uint32_t* rowWords = blockWords + (ulong)(r0 + r) * ROWW;
    uint ld[NL];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < NL; ++i) ld[i] = rowWords[idx[i]];
    const float xr = float(x[r0 + r]);
    const float sr = float(scales[r0 + r]);
    uint w[NA], v[NA];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < NA; ++i) w[i] = shifted ? (ld[i] >> SHIFT) | (ld[i + 1] << (32u - SHIFT)) : ld[i];
    v[0] = w[0] << 16;
    #pragma clang loop unroll(full)
    for (uint i = 1; i < NA; ++i) v[i] = (w[i - 1] >> 16) | (w[i] << 16);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < (uint)P; ++i) {
      const float code = trellis_codebook_window(table, w, v, i * (uint)K, (uint)L);
      acc[i] = metal::fma(code * sr, xr, acc[i]);
    }
  }
  const ulong out = (ulong)split * (ulong)C + (ulong)blk * 256ul + 255ul - li * (uint)P;
  #pragma clang loop unroll(full)
  for (uint i = 0; i < (uint)P; ++i) partial[out - i] = acc[i];
`;

let kernel: MetalKernel | undefined;

function launch(x2: MlxArray, codes: MlxArray, scales: MlxArray, g: TrellisGeometry, interleave: boolean): MlxArray {
  const groupsPerSplit = Math.ceil((g.cols / 256) / (P / 8));
  if (x2.shape[0] !== 1) throw new Error("downCodebookRow: expects one activation row");
  if (g.axis !== 0 || g.T !== 256 || g.L > 12 || g.cols % 256) throw new Error("downCodebookRow: needs axis-0 codes at T 256, L <= 12");
  kernel ??= new MetalKernel({ name: "mlx_bun_trellis_down_codebook_row", inputNames: ["x", "codes", "scales"],
    outputNames: ["partial"], source: SOURCE, header: HEADER + CODEBOOK_WINDOW, ensureRowContiguous: true });
  const [partial] = kernel.apply([x2, codes, scales], {
    outputs: [{ shape: [1, SPLITS, g.cols], dtype: Dtype.float32 }],
    grid: [TRELLIS_CODEBOOK_DOWN_THREADS, SPLITS * groupsPerSplit / (TRELLIS_CODEBOOK_DOWN_THREADS / 32), 1],
    threadGroup: [TRELLIS_CODEBOOK_DOWN_THREADS, 1, 1],
    templateDtypes: { T: x2.dtype },
    templateInts: { R: g.rows, C: g.cols, K: g.k, L: g.L, P, U, SPLITS, SG_TG: TRELLIS_CODEBOOK_DOWN_THREADS / 32,
      THREADS: TRELLIS_CODEBOOK_DOWN_THREADS, INTERLEAVE: Number(interleave) },
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
  return launch(x2, codes, scales, g, false);
}

/** The same over 3-bit block-interleaved codes [cols/512, rows, 48]. */
export function downK3InterleavedCodebookRow(x2: MlxArray, codes: MlxArray, scales: MlxArray, g: TrellisGeometry): MlxArray {
  if (g.blockInterleave !== 2 || g.k !== 3) throw new Error("downK3InterleavedCodebookRow: needs 3-bit block-interleaved codes");
  return launch(x2, codes, scales, g, true);
}
