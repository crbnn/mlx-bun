// The folded-GQA window read over a bf16 cache (layers/folded-attention, the M1 Max Qwen3.8 composition's bf16
// window read through state/attention-read's foldedWindowRead) at the Qwen3.8 attention shape (24 query heads, 4 KV
// heads, head dim 256) against MLX's fused SDPA on the windows that SDPA hands to its unfused fallback: 6 to 8 causal
// queries per row, one and two rows, over cache views sliced from a longer buffer as the caches hand them out. On the
// M1 Max, where it was measured and is selected, it is that fallback bit for bit; elsewhere it agrees to bf16
// precision. Shorter windows, which MLX's vector kernel takes, and longer ones are not eligible.
import { expect, test } from "bun:test";
import { Dtype, MlxArray, ops } from "@mlx-bun/mlx";
import { deviceArchitecture } from "@mlx-bun/mlx/ffi";
import { foldedSdpa, foldedSdpaEligible } from "../../src/layers/folded-attention";
import { causalLease, foldedWindowRead, sdpaRead } from "../../src/state/attention-read";
import { KVCache } from "../../src/state/kv";
import { BatchedKVCache } from "../../src/state/batched-kv";
import { targetCacheLayout } from "../../src/state/layout";

const H = 24, KVH = 4, D = 256, SCALE = 1 / 16;
const noise = (shape: number[], seed: number) => {
  const n = shape.reduce((a, b) => a * b, 1);
  using wide = MlxArray.fromFloat32(Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.7311 + seed) + Math.cos(i * 0.0137 * seed)), shape);
  return wide.astype(Dtype.bfloat16);
};
/** 0 when the bytes agree, else the largest difference. */
function diff(actual: MlxArray, expected: MlxArray): number {
  expect(actual.shape).toEqual(expected.shape);
  using a32 = actual.astype(Dtype.float32), b32 = expected.astype(Dtype.float32);
  const a = a32.toFloat32(), b = b32.toFloat32();
  let worst = 0;
  for (let i = 0; i < b.length; i++) worst = Math.max(worst, Math.abs(a[i]! - b[i]!));
  return worst;
}
const measured = deviceArchitecture() === "applegpu_g13s";

test("6 to 8 causal queries per row over a bf16 cache: MLX's fused SDPA (its unfused fallback) on the M1 Max bit for bit", () => {
  for (const B of [1, 2]) for (const N of [300, 1100]) {
    using kFull = noise([B, KVH, N + 256, D], 1 + N), vFull = noise([B, KVH, N + 256, D], 2 + N);
    using keys = kFull.slice([0, 0, 0, 0], [B, KVH, N, D]), values = vFull.slice([0, 0, 0, 0], [B, KVH, N, D]);
    for (const L of [6, 7, 8]) {
      using q = noise([B, H, L, D], 3 + L);
      expect(foldedSdpaEligible(q, keys)).toBe(true);
      using folded = foldedSdpa(q, keys, values, SCALE);
      using stock = ops.sdpa(q, keys, values, SCALE, "causal", null);
      const worst = diff(folded, stock);
      if (measured) expect(worst, `B${B} N${N} L${L}`).toBe(0);
      else expect(worst, `B${B} N${N} L${L}`).toBeLessThan(2e-2);
    }
  }
}, 60_000);

test("windows MLX's vector kernel takes (up to 32 query vectors per KV head) and wider ones are not eligible", () => {
  using keys = noise([1, KVH, 64, D], 4);
  for (const [L, eligible] of [[1, false], [5, false], [6, true], [8, true], [9, false]] as const) {
    using q = noise([1, H, L, D], 5);
    expect(foldedSdpaEligible(q, keys), `L${L}`).toBe(eligible);
  }
  // No more queries than keys.
  using few = noise([1, KVH, 7, D], 6), q = noise([1, H, 8, D], 7);
  expect(foldedSdpaEligible(q, few)).toBe(false);
});

test("foldedWindowRead reads every window as sdpaRead does; a KVCache's row layouts and clones read with its window read", () => {
  const caches = [new KVCache(undefined, undefined, foldedWindowRead), new KVCache()];
  try {
    for (const [L, seed] of [[40, 10], [8, 11], [3, 12], [6, 13]] as const) {
      using k = noise([1, KVH, L, D], seed), v = noise([1, KVH, L, D], seed + 1), q = noise([1, H, L, D], seed + 2);
      const [folded, stock] = caches.map(cache => cache.appendWindow(k, v));
      try {
        using a = folded!.attend(q, SCALE), b = stock!.attend(q, SCALE);
        const worst = diff(a, b);
        if (measured || L < 6 || L > 8) expect(worst, `L${L}`).toBe(0);
        else expect(worst, `L${L}`).toBeLessThan(2e-2);
      } finally { folded!.dispose(); stock!.dispose(); }
    }
    const layout = targetCacheLayout(caches[0]!) as BatchedKVCache;
    expect(layout).toBeInstanceOf(BatchedKVCache);
    expect(layout.windowRead).toBe(foldedWindowRead);
    expect(layout.makeEmptyBatch().windowRead).toBe(foldedWindowRead);
    expect((targetCacheLayout(caches[1]!) as BatchedKVCache).windowRead).toBe(sdpaRead);
    // A one-query window is not eligible: the stock read's.
    using kk = noise([1, KVH, 16, D], 20), vv = noise([1, KVH, 16, D], 21), q = noise([1, H, 1, D], 22);
    const read = foldedWindowRead(ops.contiguous(kk), ops.contiguous(vv), causalLease);
    try { using a = read.attend(q, SCALE), b = ops.sdpa(q, kk, vv, SCALE, "causal", null); expect(diff(a, b)).toBe(0); }
    finally { read.dispose(); }
  } finally { for (const cache of caches) cache.dispose(); }
});
