// The serial convert-after reference the delayed row layouts are gated
// against: a plain entry holding at least `start` positions converts in place
// with the conversion the legos run after the append that reaches their start
// (`state/bf16-first-kv.ts`); every other entry stays. Before E3 the served
// path's KV maintenance did this to a solo request's caches between forwards.
import type { Cache } from "../../src/contracts/mlx/cache";
import { unfusedAffineKernels, type AffineKernelSet } from "../../src/state/affine-attention";
import { convertToAffine, convertToTurboQuant, transitionDue } from "../../src/state/bf16-first-kv";
import { KVCache } from "../../src/state/kv";
import { RotatingKVCache } from "../../src/state/rotating-kv";
import { turboQuantFusedDecode } from "../../src/state/turboquant-codec";

/** Affine: full and rotating plain entries, `groupSize` 64 unless named; the
 * converted storage reads with `kernels` (unfused unless named). */
export function settleAffine(rows: Cache[], bits: number, start: number, groupSize = 64,
  kernels: AffineKernelSet = unfusedAffineKernels): void {
  rows.forEach((row, index) => {
    if (!transitionDue(row.offset, start)) return;
    if (row instanceof KVCache) rows[index] = convertToAffine(row, groupSize, bits, start, kernels);
    else if (row instanceof RotatingKVCache) rows[index] = convertToAffine(row, groupSize, bits, start, kernels);
  });
}

/** TurboQuant: full-attention plain entries only; rotating ones stay bf16. */
export function settleTurbo(rows: Cache[], kBits: number, vBits: number, start: number): void {
  rows.forEach((row, index) => {
    if (row instanceof KVCache && transitionDue(row.offset, start))
      rows[index] = convertToTurboQuant(row, kBits, vBits, start, turboQuantFusedDecode());
  });
}
