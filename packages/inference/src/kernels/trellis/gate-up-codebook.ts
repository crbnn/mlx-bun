import { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import { MetalKernel } from "@mlx-bun/mlx/metal-kernel";
import type { TrellisWeights } from "./geometry";
import { HEADER } from "./codebook";
import { TRELLIS_CODEBOOK_THREADS, TRELLIS_CODEBOOK_SG_PER_TG } from "./launch";
import type { MixedGateUpTail } from "./mixed-gate-up";

// Fused gate/up/SwiGLU for one activation row over axis-1 Trellis codes whose
// gate and up widths may differ (KG, KU), decoding through a threadgroup
// codebook: every code value y·(1/147.8) of the 2^L states, generated once per
// threadgroup and shared by the rows its sixteen SIMD groups take in turn (at
// most THREADGROUPS threadgroups, so each table serves several rows). Same
// lanes, runs, accumulation order and f32 code·scale weights as
// fusedGateUpSwiglu (same widths) and fusedGateUpSwigluMixed (different
// widths), bit for bit; TAIL selects the mixed kernel's SwiGLU arithmetic (1:
// MLX's compiled swiglu over the bf16 projections; 0: the float32 sigmoid of
// the same-width kernel).
const SOURCE = String.raw`
  threadgroup float codebook[1 << L];
  for (uint i = thread_position_in_threadgroup.x; i < (1u << (uint)L); i += (uint)THREADS)
    codebook[i] = trellis_unrefined_bits_y(trellis_y(i));
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const uint lane = thread_index_in_simdgroup;
  const uint gWpb = (uint)(BT * KG / 32);
  const uint uWpb = (uint)(BT * KU / 32);
  const uint runsPerBlock = (uint)BT / 32u;
  const uint nRuns = (uint)C / 32u;
  const uint rowStep = threadgroups_per_grid.y * (uint)ROWS_TG;
  for (uint row = thread_position_in_grid.y * (uint)ROWS_TG + simdgroup_index_in_threadgroup; row < (uint)R; row += rowStep) {
    const device uint32_t* gRow = gcodes + (ulong)row * (ulong)((C / BT) * gWpb);
    const device uint32_t* uRow = ucodes + (ulong)row * (ulong)((C / BT) * uWpb);
    const float gScale = float(gscales[row]);
    const float uScale = float(uscales[row]);
    float gAcc = 0.0f, gAcc2 = 0.0f, uAcc = 0.0f, uAcc2 = 0.0f;
    for (uint run = lane; run < nRuns; run += 32u) {
      const uint blk = run / runsPerBlock;
      const uint t0 = (run - blk * runsPerBlock) * 32u;
      const device uint32_t* gw = gRow + blk * gWpb;
      const device uint32_t* uw = uRow + blk * uWpb;
      const uint gW0 = ((uint)BT - 32u - t0) * (uint)KG / 32u;
      const uint uW0 = ((uint)BT - 32u - t0) * (uint)KU / 32u;
      uint g[KG + 1];
      uint u[KU + 1];
      #pragma clang loop unroll(full)
      for (uint i = 0; i <= (uint)KG; ++i) { const uint wi = gW0 + i; g[i] = gw[wi >= gWpb ? wi - gWpb : wi]; }
      #pragma clang loop unroll(full)
      for (uint i = 0; i <= (uint)KU; ++i) { const uint wi = uW0 + i; u[i] = uw[wi >= uWpb ? wi - uWpb : wi]; }
      const device metal::vec<T, 4>* x4 = (const device metal::vec<T, 4>*)(x + blk * (uint)BT + t0);
      #pragma clang loop unroll(full)
      for (uint j4 = 0; j4 < 8u; ++j4) {
        const float4 xv4 = float4(x4[j4]);
        #pragma clang loop unroll(full)
        for (uint q = 0; q < 4u; ++q) {
          const uint j = j4 * 4u + q;
          const uint gb = (31u - j) * (uint)KG, ub = (31u - j) * (uint)KU;
          const uint gwi = gb >> 5, goff = gb & 31u;
          const uint uwi = ub >> 5, uoff = ub & 31u;
          uint gwin = g[gwi] >> goff;
          uint uwin = u[uwi] >> uoff;
          if (goff + (uint)L > 32u) gwin |= g[gwi + 1] << (32u - goff);
          if (uoff + (uint)L > 32u) uwin |= u[uwi + 1] << (32u - uoff);
          gwin &= (1u << (uint)L) - 1u;
          uwin &= (1u << (uint)L) - 1u;
          const float xv = xv4[q];
          const float gv = codebook[gwin] * gScale;
          const float uv = codebook[uwin] * uScale;
          if (q & 1u) { gAcc2 = metal::fma(gv, xv, gAcc2); uAcc2 = metal::fma(uv, xv, uAcc2); }
          else { gAcc = metal::fma(gv, xv, gAcc); uAcc = metal::fma(uv, xv, uAcc); }
        }
      }
    }
    const float gate = metal::simd_sum(gAcc + gAcc2);
    const float up = metal::simd_sum(uAcc + uAcc2);
    if (lane == 0u) {
      const T gateT = T(gate);
      const T upT = T(up);
      if ((TAIL) == 1) {
        const T sigY = 1 / (1 + metal::exp(metal::abs(gateT)));
        const T sigmoidT = (gateT < 0) ? sigY : 1 - sigY;
        const T siluT = gateT * sigmoidT;
        mid[row] = siluT * upT;
      } else {
        const T sigmoidT = T(1.0f / (1.0f + metal::precise::exp(-float(gateT))));
        const T siluT = T(float(gateT) * float(sigmoidT));
        mid[row] = T(float(siluT) * float(upT));
      }
    }
  }
`;

/** Threadgroups at most: ten per M1 Max GPU core, measured against one per sixteen rows (the codebook generated
 *  1088 times per projection at the artifact's 17408 rows). */
const THREADGROUPS = 320;

let kernel: MetalKernel | undefined;

/** silu(gate(x))·up(x) for exactly one activation row over matching axis-1
 *  geometry (L at most 12; the bit widths may differ). `tail` applies where the
 *  widths differ; same widths take the float32 sigmoid. Borrows x and the
 *  weights; returns owned lazy output with x's leading dimensions. */
export function gateUpCodebookRow(x: MlxArray, gate: TrellisWeights, up: TrellisWeights, tail: MixedGateUpTail): MlxArray {
  const g = gate.geometry, u = up.geometry;
  if (g.axis !== 1 || u.axis !== 1 || g.rows !== u.rows || g.cols !== u.cols || g.T !== u.T || g.L !== u.L || g.L > 12)
    throw new Error("gateUpCodebookRow: gate and up need matching axis-1 geometry with L <= 12");
  if (x.size !== x.shape.at(-1)) throw new Error("gateUpCodebookRow: expects one activation row");
  kernel ??= new MetalKernel({ name: "mlx_bun_trellis_gateup_codebook_row",
    inputNames: ["x", "gcodes", "gscales", "ucodes", "uscales"], outputNames: ["mid"],
    source: SOURCE, header: HEADER, ensureRowContiguous: true });
  const lead = x.shape.slice(0, -1);
  const x2 = ops.reshape(x, [1, g.inFeatures]);
  const [mid] = kernel.apply([x2, gate.codes, gate.scales, up.codes, up.scales], {
    outputs: [{ shape: [1, g.rows], dtype: x.dtype }],
    grid: [TRELLIS_CODEBOOK_THREADS, Math.min(THREADGROUPS, Math.ceil(g.rows / TRELLIS_CODEBOOK_SG_PER_TG)), 1],
    threadGroup: [TRELLIS_CODEBOOK_THREADS, 1, 1],
    templateDtypes: { T: x.dtype },
    templateInts: { R: g.rows, C: g.cols, BT: g.T, KG: g.k, KU: u.k, L: g.L, ROWS_TG: TRELLIS_CODEBOOK_SG_PER_TG,
      THREADS: TRELLIS_CODEBOOK_THREADS, TAIL: g.k !== u.k && tail === "split" ? 1 : 0 },
  });
  x2.dispose();
  const out = ops.reshape(mid!, [...lead, g.rows]);
  mid!.dispose();
  return out;
}
