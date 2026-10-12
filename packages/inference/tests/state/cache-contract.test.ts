// The cache contract, run against every cache lego the composition can name
// (PLAN B3: "every cache lego implements the whole contract, so the cache is
// swappable and nothing else knows or cares"). Model-free: synthetic keys,
// values and queries at small shapes. A lego either passes every check here or
// it is not a lego; there is no skip list and no capability table.
//
// Each attention lego is the cache the composition builds for one layer
// (`composedFullAttentionCache` / `composedSlidingAttentionCache` for the KV
// scheme, the M4 Pro composition's KV4 cache, the paged cache the request
// policy pages a full-attention layer with), driven the way the scheduler and
// the methods drive it, through `state/` only:
// - phases: prefill chunk, prefill tail, decode and verify windows on the
//   serial cache, compared with the bf16 lego of the same layer (bit for bit
//   where the lego stores bf16, within its quantization otherwise);
// - known tokens: the committed read on the serial cache and on the
//   verification rows, against the same positions read one at a time;
// - donor reads: the drafters' captured attention over the stored rows;
// - snapshot: a clone continues bit for bit like its source;
// - SSD: a saved and reloaded cache continues bit for bit like its source;
// - decode rows: the layout continuous decoding keeps the lego's rows in
//   (`ownedCacheLayout` after the scheme's `prepareBatch`), joined from serial
//   rows of different lengths; each row's reads follow its serial twin's and an
//   extracted row continues as its twin;
// - prefill rows: the layout a prefill cohort keeps the rows in
//   (`prefillCacheLayout` after `preparePrefill`);
// - verification rows and the speculative transaction (`targetCacheLayout`,
//   `bindRowCacheRollback`): rollback to each row's accepted length leaves the
//   rows bit for bit as twins that appended only the kept positions, commit as
//   twins that appended all of them.
// The recurrent state (`SSMCache`, the gated-DeltaNet contract) runs its own
// operations: recurrence per phase, rows, transaction, snapshot and SSD; so
// does the latent attention state (`MLACache`, GLM-5.2's compressed MLA state,
// with and without DSA index keys): the latent read per phase, snapshot, SSD,
// rows and the transaction.
import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import type {
  AttentionCache, AttentionRead, BatchableCache, BidirectionalAttentionCache, BlockAttentionCache, Cache, CommittedAttentionCache,
  EncoderAttentionCache, GatedDeltaCache, GatedDeltaParameters,
} from "../../src/contracts/mlx/cache";
import { BF16_KV, type KvStorageScheme } from "../../src/contracts/portable/composition";
import { composedAffineKernels } from "../../src/state/affine-attention";
import { AttentionMasks } from "../../src/state/attention-read";
import { Bf16FirstKv4Head256Cache, composedFullAttentionCache, composedSlidingAttentionCache } from "../../src/state/bf16-first-kv";
import { createKvMaintenance } from "../../src/state/kv-maintenance";
import { captureKvDonorAttention } from "../../src/state/kv-attention-view";
import { ownedCacheLayout, prefillCacheLayout, targetCacheLayout } from "../../src/state/layout";
import { PagedKVCache, type PagedQuantization } from "../../src/state/paged/cache";
import { maybePageKv } from "../../src/state/request-policy";
import { cloneKvCaches, loadKvCache, saveKvCache } from "../../src/state/persistence";
import { bindRowCacheRollback, type RowSpeculativeState } from "../../src/state/rollback";
import { SSMCache } from "../../src/state/ssm";
import { deferredTurboQuantKernels, denseTurboQuantKernels } from "../../src/state/turboquant-attention";
import { leaseCacheStates } from "../../src/state/leases";
import { composedLatentCache, latentLegoCodecs } from "../../src/state/latent-legos";
import type { LatentAttentionCache, LatentAttentionState } from "../../src/contracts/mlx/cache";

// ---------------------------------------------------------------------------
// Inputs and comparisons

interface Geometry { readonly heads: number; readonly kvHeads: number; readonly dim: number; readonly window: number | null }
const FULL: Geometry = { heads: 4, kvHeads: 2, dim: 64, window: null };
const SLIDING: Geometry = { ...FULL, window: 8 };
const HEAD256: Geometry = { heads: 4, kvHeads: 2, dim: 256, window: null };

const noise = (count: number, seed: number) => Float32Array.from({ length: count },
  (_, i) => Math.sin(i * 0.731 + seed * 1.37) * 0.9 + Math.cos(i * 0.137 + seed * 0.53) * 0.4);
function bf16(values: Float32Array, shape: number[]): MlxArray {
  using wide = MlxArray.fromFloat32(values, shape);
  return wide.astype(Dtype.bfloat16);
}
/** Keys and values of `L` positions for one row; `seed` names the row's stream. */
function rowKv(g: Geometry, L: number, seed: number): [MlxArray, MlxArray] {
  const n = g.kvHeads * L * g.dim;
  return [bf16(noise(n, seed), [1, g.kvHeads, L, g.dim]), bf16(noise(n, seed + 0.5), [1, g.kvHeads, L, g.dim])];
}
function rowQ(g: Geometry, L: number, seed: number): MlxArray {
  return bf16(noise(g.heads * L * g.dim, seed + 0.25), [1, g.heads, L, g.dim]);
}
function rowsOf(parts: MlxArray[]): MlxArray {
  try { return parts.length === 1 ? ops.contiguous(parts[0]!) : ops.concatAxis(parts, 0); }
  finally { for (const part of parts) part.dispose(); }
}
const kvRows = (g: Geometry, L: number, seeds: readonly number[]): [MlxArray, MlxArray] => {
  const pairs = seeds.map(seed => rowKv(g, L, seed));
  return [rowsOf(pairs.map(pair => pair[0])), rowsOf(pairs.map(pair => pair[1]))];
};
const qRows = (g: Geometry, L: number, seeds: readonly number[]) => rowsOf(seeds.map(seed => rowQ(g, L, seed)));
/** The first `count` positions of a [B, H, L, D] tensor. */
const head = (a: MlxArray, from: number, to: number) => a.slice([0, 0, from, 0], [a.shape[0]!, a.shape[1]!, to, a.shape[3]!]);
const rowOf = (a: MlxArray, row: number) => a.slice([row, 0, 0, 0], [row + 1, ...a.shape.slice(1)]);

function floats(a: MlxArray): Float32Array {
  using wide = a.astype(Dtype.float32);
  using flat = ops.contiguous(wide);
  return flat.toFloat32();
}
function maxDiff(a: MlxArray, b: MlxArray): number {
  expect(a.shape).toEqual(b.shape);
  const x = floats(a), y = floats(b);
  let worst = 0;
  for (let i = 0; i < x.length; i++) {
    const d = Math.abs(x[i]! - y[i]!);
    if (!(d <= worst)) worst = Number.isNaN(d) ? Infinity : d;
  }
  return worst;
}
function finite(a: MlxArray): boolean { return floats(a).every(Number.isFinite); }

const SCALE = (g: Geometry) => 1 / Math.sqrt(g.dim);

/** One append and attend: owned output [B, Hq, L, D]. */
function readOnce(read: AttentionRead, q: MlxArray, g: Geometry): MlxArray {
  try { return read.attend(q, SCALE(g)); } finally { read.dispose(); }
}
type Phase = "decode" | "window" | "committed";
function step(cache: Cache, g: Geometry, phase: Phase, L: number, seeds: readonly number[]): MlxArray {
  const [k, v] = kvRows(g, L, seeds);
  using q = qRows(g, L, seeds);
  try {
    const attention = cache as CommittedAttentionCache;
    const read = phase === "decode" ? attention.appendDecode(k, v) : phase === "window" ? attention.appendWindow(k, v)
      : attention.appendCommitted(k, v);
    return readOnce(read, q, g);
  } finally { k.dispose(); v.dispose(); }
}
/** Append `L` positions of `seed` without reading (a prefill chunk whose
 * output nobody needs). */
function fill(cache: Cache, g: Geometry, L: number, seed: number): void {
  step(cache, g, "window", L, [seed]).dispose();
}
function dispose(...caches: (Cache | undefined | null)[]): void { for (const cache of caches) cache?.dispose(); }

// ---------------------------------------------------------------------------
// The legos

interface Lego {
  readonly name: string;
  readonly g: Geometry;
  /** The composition's KV scheme: what kv maintenance runs over the rows. */
  readonly scheme: KvStorageScheme;
  /** The layer's cache as the composition (or the paging request policy)
   * builds it. */
  make(masks?: AttentionMasks): Cache;
  /** Every read equals the bf16 lego's bit for bit (bf16 storage). */
  readonly exact: boolean;
  /** A first window is stored bf16 (every bf16-first lego), so the first-window
   * reads equal the bf16 lego's bit for bit. */
  readonly firstWindowBf16: boolean;
  /** Bound on a read's difference from the bf16 lego's (quantized storage). */
  readonly tolerance: number;
}

const affine = (bits: number, start = 0, kernels: "unfused" | "tiled-causal" = "unfused"): KvStorageScheme =>
  ({ kind: "affine", bits, groupSize: 64, kernels, ...(start ? { quantizedKvStart: start } : {}) });
const perLayer = (bits: number): KvStorageScheme =>
  ({ kind: "affine-layers", layers: [{ layer: 0, bits, groupSize: 64 }], kernels: "tiled-causal" });
const turbo = (start = 0): KvStorageScheme => ({ kind: "turbo", kBits: 8, vBits: 3, ...(start ? { quantizedKvStart: start } : {}) });

const full = (name: string, scheme: KvStorageScheme, kernels = denseTurboQuantKernels, tolerance = 0.2): Lego => ({
  name, g: FULL, scheme, exact: scheme.kind === "bf16", firstWindowBf16: true, tolerance,
  make: masks => composedFullAttentionCache(scheme, 0, masks ?? new AttentionMasks(), kernels),
});
const sliding = (name: string, scheme: KvStorageScheme, tolerance = 0.2): Lego => ({
  name, g: SLIDING, scheme, exact: scheme.kind === "bf16" || scheme.kind === "turbo", firstWindowBf16: true, tolerance,
  make: masks => composedSlidingAttentionCache(scheme, 0, SLIDING.window!, masks ?? new AttentionMasks()),
});
const paged = (name: string, quantization?: PagedQuantization, direct = false): Lego => ({
  name, g: FULL, scheme: quantization ? affine(quantization.bits) : BF16_KV, exact: !quantization && !direct,
  firstWindowBf16: !quantization,
  tolerance: quantization ? 0.2 : 0.07,
  make: () => new PagedKVCache(256, 16, direct, quantization),
});

/** The paged lego the request policy pages layer `layer` of a composition with
 * (`maybePageKv` over the graph's cache for that layer), its pages storing what
 * `scheme` names there. */
const pagedAs = (name: string, scheme: KvStorageScheme, layer: number, tolerance: number, exact = false,
  kernels = denseTurboQuantKernels): Lego => ({
  name, g: FULL, scheme, exact, firstWindowBf16: exact, tolerance,
  make: masks => {
    const held = masks ?? new AttentionMasks();
    const caches = Array.from({ length: layer + 1 }, (_, index) => composedFullAttentionCache(scheme, index, held, kernels));
    maybePageKv(caches, scheme, { blockSize: 16 }, 256);
    for (const cache of caches.slice(0, layer)) cache.dispose();
    expect(caches[layer]).toBeInstanceOf(PagedKVCache);
    return caches[layer]!;
  },
});

const LEGOS: readonly Lego[] = [
  full("bf16 full (KVCache)", BF16_KV),
  sliding("bf16 sliding (RotatingKVCache)", BF16_KV),
  full("affine 4-bit full, bf16 first", affine(4)),
  full("affine 8-bit full, bf16 for 6 positions", affine(8, 6)),
  sliding("affine 4-bit sliding, bf16 first", affine(4)),
  sliding("affine 8-bit sliding, bf16 for 6 positions", affine(8, 6)),
  full("affine per-layer full (tiled-causal)", perLayer(8)),
  sliding("affine per-layer sliding (tiled-causal)", perLayer(4)),
  full("TurboQuant dense full, bf16 first", turbo(), denseTurboQuantKernels, 0.5),
  full("TurboQuant deferred full, bf16 first", turbo(), deferredTurboQuantKernels, 0.5),
  full("TurboQuant dense full, bf16 for 6 positions", turbo(6), denseTurboQuantKernels, 0.5),
  sliding("TurboQuant sliding (bf16)", turbo()),
  { name: "KV4 head-256 (M4 Pro composition), bf16 first", g: HEAD256, scheme: affine(4), exact: false, firstWindowBf16: true, tolerance: 0.2,
    make: masks => new Bf16FirstKv4Head256Cache(0, composedAffineKernels("unfused"), masks ?? new AttentionMasks()) },
  paged("paged bf16"),
  paged("paged bf16, direct read (Lab)", undefined, true),
  paged("paged 4-bit", { bits: 4, groupSize: 64 }),
  paged("paged 8-bit", { bits: 8, groupSize: 64 }),
  // Pages as the request policy builds them for the composition's scheme (`maybePageKv`), per layer.
  pagedAs("paged per-layer, a listed 8-bit layer", perLayer(8), 0, 0.2),
  pagedAs("paged per-layer, an unlisted layer (bf16 pages)", perLayer(8), 1, 0, true),
  pagedAs("paged 3-bit (a width the direct read cannot take)", affine(3), 0, 0.5),
  pagedAs("paged TurboQuant 8/3, dense read", turbo(), 0, 0.5, false, denseTurboQuantKernels),
  pagedAs("paged TurboQuant 8/3, deferred read", turbo(), 0, 0.5, false, deferredTurboQuantKernels),
  pagedAs("paged TurboQuant 4/2, deferred read", { kind: "turbo", kBits: 4, vBits: 2 }, 0, 1, false, deferredTurboQuantKernels),
];

/** The bf16 lego of the same layer geometry: the numerical reference. */
function reference(lego: Lego): Cache {
  const masks = new AttentionMasks();
  return lego.g.window === null ? composedFullAttentionCache(BF16_KV, 0, masks, denseTurboQuantKernels)
    : composedSlidingAttentionCache(BF16_KV, 0, lego.g.window, masks);
}

/** A serial row: the lego's cache holding `length` positions of stream `seed`,
 * filled by one prefill chunk then one tail (the scheduler's two prefill
 * phases). */
function makeRow(lego: Lego, length: number, seed: number, masks?: AttentionMasks): Cache {
  const cache = lego.make(masks);
  try {
    const chunk = Math.max(1, length - 2);
    fill(cache, lego.g, chunk, seed);
    if (length > chunk) step(cache, lego.g, "window", length - chunk, [seed + 100]).dispose();
    return cache;
  } catch (error) { cache.dispose(); throw error; }
}

/** The rows a running batch joins (`batch-group`): the scheme's `prepareBatch`
 * over each row's caches, the first row adopted as is, each joiner merged into
 * the decode layout its own storage names. */
function joinDecode(lego: Lego, rows: Cache[]): BatchableCache {
  const maintain = createKvMaintenance(lego.scheme);
  for (let row = 0; row < rows.length; row++) { const one = [rows[row]!]; maintain.prepareBatch?.(one); rows[row] = one[0]!; }
  let running: Cache = rows[0]!;
  for (const joiner of rows.slice(1)) {
    const layout = ownedCacheLayout(joiner);
    if (!layout) throw new Error(`${joiner.signature()} names no decode layout`);
    layout.mergeRows([running, joiner]);
    if (running !== rows[0]) running.dispose();
    running = layout;
  }
  return running as BatchableCache;
}

/** Verification rows (`speculative-group`): `prepareBatch`, then each row
 * appended into the target layout its storage names, one membership change per
 * admission (`MlxStateRows.prepareAppend`). */
function joinVerify(lego: Lego, rows: Cache[]): BatchableCache & RowSpeculativeState {
  const maintain = createKvMaintenance(lego.scheme);
  for (let row = 0; row < rows.length; row++) { const one = [rows[row]!]; maintain.prepareBatch?.(one); rows[row] = one[0]!; }
  let layout: BatchableCache = targetCacheLayout(rows[0]!) as unknown as BatchableCache;
  let count = 0;
  for (const row of rows) {
    const merged = layout.makeEmptyBatch();
    merged.mergeRows([...(count ? [layout] : []), row]);
    layout.dispose(); layout = merged; count++;
  }
  return layout as BatchableCache & RowSpeculativeState;
}

/** A prefill cohort (`prefill-rows`): `preparePrefill` over each row, the
 * layout the first row's storage names, one merge of every row. */
function joinPrefill(lego: Lego, rows: Cache[]): BatchableCache {
  const maintain = createKvMaintenance(lego.scheme);
  for (let row = 0; row < rows.length; row++) { const one = [rows[row]!]; maintain.preparePrefill?.(one); rows[row] = one[0]!; }
  const layout = prefillCacheLayout(rows[0]!).makeEmptyBatch();
  layout.mergeRows(rows);
  return layout;
}

function evaluate(caches: readonly Cache[]): void {
  const state = leaseCacheStates(caches);
  try { if (state.borrow().length) ops.evalAll([...state.borrow()]); } finally { state.close(); }
}

/** Compare each row of a batched output with its serial twin's output. */
function expectRows(label: string, batched: MlxArray, twins: MlxArray[], tolerance: number, exact = false): void {
  for (const [row, twin] of twins.entries()) {
    using mine = rowOf(batched, row);
    const diff = maxDiff(mine, twin);
    if (exact) expect(diff, `${label} row ${row}`).toBe(0);
    else expect(diff, `${label} row ${row}: ${diff}`).toBeLessThanOrEqual(tolerance);
  }
}

const tmp = mkdtempSync(join(tmpdir(), "mlx-bun-cache-contract-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

// Within one lego's own arithmetic, batching moves only the last bits (the
// bit-stability decision): joined rows reduce over a padded length.
const BATCHED = 0.03;

for (const lego of LEGOS) describe(lego.name, () => {
  const g = lego.g;

  test("phases: prefill chunk, tail, decode, verify", () => {
    const cache = lego.make(), bf16Cache = reference(lego);
    try {
      let position = 0;
      for (const [phase, L, seed] of [["window", 5, 1], ["window", 3, 2], ["decode", 1, 3], ["decode", 1, 4],
        ["window", 3, 5], ["decode", 1, 6], ["window", 4, 7]] as const) {
        using out = step(cache, g, phase, L, [seed]);
        using expected = step(bf16Cache, g, phase, L, [seed]);
        position += L;
        expect(out.shape).toEqual([1, g.heads, L, g.dim]);
        expect(cache.offset).toBe(position);
        expect(finite(out)).toBe(true);
        const diff = maxDiff(out, expected);
        if (lego.exact) expect(diff, `${phase} ${L} at ${position}`).toBe(0);
        else expect(diff, `${phase} ${L} at ${position}: ${diff}`).toBeLessThanOrEqual(lego.tolerance);
      }
    } finally { dispose(cache, bf16Cache); }
  });

  /** A first window's flags: positions 2..4 attend each other both ways. */
  const flags = (L: number) => {
    using ints = ops.fromInt32(Array.from({ length: L }, (_, i) => Number(i >= 2 && i <= 4)), [L]);
    return ints.astype(Dtype.bool);
  };
  const expectRead = (label: string, out: MlxArray, want: MlxArray, exact: boolean) => {
    const diff = maxDiff(out, want);
    if (exact) expect(diff, label).toBe(0);
    else expect(diff, `${label}: ${diff}`).toBeLessThanOrEqual(lego.tolerance);
  };

  test("vision prefill: the bidirectional first window", () => {
    const cache = lego.make(), bf16Cache = reference(lego);
    const L = 7;
    try {
      using bidirectional = flags(L);
      const [k, v] = rowKv(g, L, 81);
      using q = rowQ(g, L, 81);
      try {
        using out = readOnce((cache as BidirectionalAttentionCache).appendBidirectional(k, v, bidirectional), q, g);
        using want = readOnce((bf16Cache as BidirectionalAttentionCache).appendBidirectional(k, v, bidirectional), q, g);
        expectRead("bidirectional", out, want, lego.firstWindowBf16);
      } finally { k.dispose(); v.dispose(); }
      expect(cache.offset).toBe(L);
      using a = step(cache, g, "decode", 1, [82]);
      using b = step(bf16Cache, g, "decode", 1, [82]);
      expectRead("decode after", a, b, lego.exact);
    } finally { dispose(cache, bf16Cache); }
  });

  test("encoder: the first window, which the sliding window does not limit", () => {
    const cache = lego.make(), bf16Cache = reference(lego);
    const L = 12;
    try {
      using bidirectional = flags(L);
      const [k, v] = rowKv(g, L, 83);
      using q = rowQ(g, L, 83);
      try {
        using out = readOnce((cache as EncoderAttentionCache).appendEncoderWindow(k, v, bidirectional), q, g);
        using want = readOnce((bf16Cache as EncoderAttentionCache).appendEncoderWindow(k, v, bidirectional), q, g);
        expectRead("encoder window", out, want, lego.firstWindowBf16);
      } finally { k.dispose(); v.dispose(); }
      expect(cache.offset).toBe(L);
    } finally { dispose(cache, bf16Cache); }
  });

  test("block read: the stored positions and the block, nothing appended", () => {
    const cache = makeRow(lego, 11, 84), bf16Cache = reference(lego);
    try {
      fill(bf16Cache, g, 9, 84); step(bf16Cache, g, "window", 2, [184]).dispose();
      const [k, v] = rowKv(g, 4, 85);
      using q = rowQ(g, 4, 85);
      try {
        using out = readOnce((cache as BlockAttentionCache).readBlock(k, v), q, g);
        using again = readOnce((cache as BlockAttentionCache).readBlock(k, v), q, g);
        using want = readOnce((bf16Cache as BlockAttentionCache).readBlock(k, v), q, g);
        expect(cache.offset).toBe(11);
        expect(maxDiff(out, again)).toBe(0);
        expectRead("block", out, want, lego.exact);
      } finally { k.dispose(); v.dispose(); }
    } finally { dispose(cache, bf16Cache); }
  });

  test("known tokens: the committed read keeps one-position reads", () => {
    const cache = makeRow(lego, 6, 11), twin = makeRow(lego, 6, 11);
    try {
      using out = step(cache, g, "committed", 3, [12]);
      expect(out.shape).toEqual([1, g.heads, 3, g.dim]);
      expect(cache.offset).toBe(9);
      const [k, v] = rowKv(g, 3, 12);
      using q = rowQ(g, 3, 12);
      try {
        for (let i = 0; i < 3; i++) {
          using ki = head(k, i, i + 1), vi = head(v, i, i + 1), qi = head(q, i, i + 1);
          using one = readOnce((twin as AttentionCache).appendDecode(ki, vi), qi, g);
          using mine = head(out, i, i + 1);
          const diff = maxDiff(mine, one);
          expect(diff, `position ${i}: ${diff}`).toBeLessThanOrEqual(BATCHED);
        }
      } finally { k.dispose(); v.dispose(); }
      using next = step(cache, g, "decode", 1, [13]);
      using twinNext = step(twin, g, "decode", 1, [13]);
      expect(maxDiff(next, twinNext)).toBeLessThanOrEqual(BATCHED);
    } finally { dispose(cache, twin); }
  });

  test("donor reads: the drafters' captured attention", () => {
    const cache = makeRow(lego, 7, 21), bf16Cache = reference(lego);
    try {
      fill(bf16Cache, g, 5, 21); step(bf16Cache, g, "window", 2, [121]).dispose();
      const donor = captureKvDonorAttention(cache), expected = captureKvDonorAttention(bf16Cache);
      try {
        expect(donor.offsets).toEqual(expected.offsets);
        using q = rowQ(g, 1, 22);
        using out = donor.attend(q, SCALE(g), { mode: "", arr: null });
        using want = expected.attend(q, SCALE(g), { mode: "", arr: null });
        const diff = maxDiff(out, want);
        if (lego.exact) expect(diff).toBe(0);
        else expect(diff, `${diff}`).toBeLessThanOrEqual(lego.tolerance);
      } finally { donor.dispose(); expected.dispose(); }
    } finally { dispose(cache, bf16Cache); }
  });

  test("snapshot: a clone continues as its source", () => {
    const cache = makeRow(lego, 7, 31);
    const [clone] = cloneKvCaches([cache]);
    try {
      expect(clone!.offset).toBe(cache.offset);
      for (const [phase, L, seed] of [["decode", 1, 32], ["window", 3, 33], ["decode", 1, 34]] as const) {
        using a = step(cache, g, phase, L, [seed]);
        using b = step(clone!, g, phase, L, [seed]);
        expect(maxDiff(a, b), `${phase} ${L}`).toBe(0);
      }
    } finally { dispose(cache, clone); }
  });

  test("SSD: a saved and reloaded cache continues as its source", () => {
    const cache = makeRow(lego, 9, 41), path = join(tmp, `${lego.name.replace(/\W+/g, "-")}.kv`);
    try {
      evaluate([cache]);
      saveKvCache(path, Array.from({ length: cache.offset }, (_, i) => i), [cache]);
      const loaded = loadKvCache(path, { makeCache: () => [lego.make()] });
      const restored = loaded.caches[0]!;
      try {
        expect(loaded.tokens.length).toBe(9);
        expect(restored.offset).toBe(cache.offset);
        for (const [phase, L, seed] of [["decode", 1, 42], ["window", 3, 43], ["decode", 1, 44]] as const) {
          using a = step(cache, g, phase, L, [seed]);
          using b = step(restored, g, phase, L, [seed]);
          expect(maxDiff(a, b), `${phase} ${L}`).toBe(0);
        }
      } finally { dispose(restored); }
    } finally { dispose(cache); }
  });

  test("decode rows: join, read, extract, filter", () => {
    const lengths = [5, 11], seeds = [51, 52];
    const rows = lengths.map((length, row) => makeRow(lego, length, seeds[row]!));
    const twins = lengths.map((length, row) => makeRow(lego, length, seeds[row]!));
    let layout: BatchableCache | undefined, extracted: Cache | undefined;
    try {
      layout = joinDecode(lego, rows);
      expect(layout.batchSize).toBe(2);
      expect([...layout.rowOffsets]).toEqual(lengths);
      for (const [phase, L, seed] of [["decode", 1, 53], ["decode", 1, 54], ["window", 2, 55], ["decode", 1, 56]] as const) {
        using out = step(layout, g, phase, L, seeds.map(s => s + seed));
        const expected = twins.map((twin, row) => step(twin, g, phase, L, [seeds[row]! + seed]));
        try { expectRows(`${phase} ${L}`, out, expected, BATCHED); } finally { for (const e of expected) e.dispose(); }
        (layout as { releaseRopeArr?: () => void }).releaseRopeArr?.();
      }
      expect([...layout.rowOffsets]).toEqual(lengths.map(length => length + 5));
      extracted = layout.extractRow(1);
      expect(extracted.offset).toBe(16);
      using a = step(extracted, g, "decode", 1, [57]);
      using b = step(twins[1]!, g, "decode", 1, [57]);
      expect(maxDiff(a, b)).toBe(0);
      layout.filterRows([0]);
      expect(layout.batchSize).toBe(1);
      using c = step(layout, g, "decode", 1, [58]);
      using d = step(twins[0]!, g, "decode", 1, [58]);
      expect(maxDiff(c, d)).toBeLessThanOrEqual(BATCHED);
    } finally { dispose(layout, extracted, ...rows, ...twins); }
  });

  test("prefill rows: a cohort chunk, then each row back to serial use", () => {
    const seeds = [61, 62];
    const rows = seeds.map(() => lego.make()), twins = seeds.map(() => lego.make());
    let layout: BatchableCache | undefined;
    const extracted: Cache[] = [];
    try {
      layout = joinPrefill(lego, rows);
      expect(layout.batchSize).toBe(2);
      const maintenance = layout.prefillMaintenance;
      maintenance?.beginPrefill();
      for (const [L, seed] of [[4, 63], [3, 64]] as const) {
        using out = step(layout, g, "window", L, seeds.map(s => s + seed));
        const expected = twins.map((twin, row) => step(twin, g, "window", L, [seeds[row]! + seed]));
        try { expectRows(`chunk ${L}`, out, expected, BATCHED); } finally { for (const e of expected) e.dispose(); }
        evaluate([layout]);
        maintenance?.commitPrefill([0, 1]);
      }
      maintenance?.endPrefill();
      for (let row = 0; row < 2; row++) extracted.push(layout.extractRow(row));
      for (const [row, cache] of extracted.entries()) {
        expect(cache.offset).toBe(7);
        using a = step(cache, g, "decode", 1, [65 + row]);
        using b = step(twins[row]!, g, "decode", 1, [65 + row]);
        expect(maxDiff(a, b), `row ${row}`).toBe(0);
      }
    } finally { dispose(layout, ...extracted, ...rows, ...twins); }
  });

  test("verification rows: rollback to each row's accepted length, commit", () => {
    const lengths = [5, 11], seeds = [71, 72];
    const rows = lengths.map((length, row) => makeRow(lego, length, seeds[row]!));
    const twins = lengths.map((length, row) => makeRow(lego, length, seeds[row]!));
    let layout: (BatchableCache & RowSpeculativeState) | undefined;
    const extracted: Cache[] = [];
    try {
      layout = joinVerify(lego, rows);
      expect(layout.batchSize).toBe(2);
      const transaction = bindRowCacheRollback([layout], 2);
      // Round 1: three drafts; row 0 accepts none (keeps the pending token),
      // row 1 accepts two (keeps three).
      transaction.begin(3);
      using verify = step(layout, g, "window", 4, seeds.map(s => s + 73));
      expect(verify.shape).toEqual([2, g.heads, 4, g.dim]);
      transaction.resolve([0, 2]);
      expect([...layout.rowOffsets]).toEqual([lengths[0]! + 1, lengths[1]! + 3]);
      for (const [row, keep] of [1, 3].entries()) {
        const [k, v] = rowKv(g, 4, seeds[row]! + 73);
        using q = rowQ(g, keep, seeds[row]! + 73);
        try {
          using kk = head(k, 0, keep), vv = head(v, 0, keep);
          readOnce((twins[row] as AttentionCache).appendWindow(kk, vv), q, g).dispose();
        } finally { k.dispose(); v.dispose(); }
      }
      // Round 2: every draft accepted, the round commits.
      transaction.begin(1);
      step(layout, g, "window", 2, seeds.map(s => s + 74)).dispose();
      transaction.resolve([1, 1]);
      for (const [row, twin] of twins.entries()) step(twin, g, "window", 2, [seeds[row]! + 74]).dispose();
      expect([...layout.rowOffsets]).toEqual([lengths[0]! + 3, lengths[1]! + 5]);
      // Known tokens and donor reads on the verification rows.
      using committed = step(layout, g, "committed", 2, seeds.map(s => s + 75));
      expect(committed.shape).toEqual([2, g.heads, 2, g.dim]);
      for (const [row, twin] of twins.entries()) step(twin, g, "committed", 2, [seeds[row]! + 75]).dispose();
      const donor = captureKvDonorAttention(layout);
      try {
        using q = qRows(g, 1, [76, 77]);
        using out = donor.attend(q, SCALE(g), { mode: "", arr: null });
        expect(out.shape).toEqual([2, g.heads, 1, g.dim]);
      } finally { donor.dispose(); }
      // Each row, extracted, is its twin: the rolled-back round left nothing.
      for (let row = 0; row < 2; row++) extracted.push(layout.extractRow(row));
      for (const [row, cache] of extracted.entries()) {
        expect(cache.offset).toBe(twins[row]!.offset);
        using a = step(cache, g, "decode", 1, [78 + row]);
        using b = step(twins[row]!, g, "decode", 1, [78 + row]);
        expect(maxDiff(a, b), `row ${row}`).toBe(0);
      }
    } finally { dispose(layout, ...extracted, ...rows, ...twins); }
  });
});

// ---------------------------------------------------------------------------
// The recurrent state: the gated-DeltaNet contract (`GatedDeltaCache`)

describe("recurrent state (SSMCache)", () => {
  const HK = 2, HV = 4, DK = 32, DV = 16, K = 4, CONV = 2 * HK * DK + HV * DV;
  const owned: MlxArray[] = [];
  afterAll(() => { for (const array of owned) array.dispose(); });
  const keep = (a: MlxArray) => { owned.push(a); return a; };
  const convWeight = keep(bf16(noise(CONV * K, 1).map(x => x * 0.3), [CONV, K, 1]));
  const aLog = keep(MlxArray.fromFloat32(noise(HV, 2).map(x => x * 0.5), [HV]));
  const dtBias = keep(bf16(noise(HV, 3).map(x => x * 0.5), [HV]));
  const layer: GatedDeltaParameters = {
    convWeight, aLog, dtBias,
    heads(convolved) {
      const [B, S] = convolved.shape as [number, number, number];
      using act = ops.silu(convolved);
      const part = (from: number, to: number, heads: number, dim: number, scale: number) => {
        using view = act.slice([0, 0, from], [B, S, to]);
        using shaped = ops.reshape(view, [B, S, heads, dim]);
        return ops.mulScalar(shaped, scale);
      };
      return { q: part(0, HK * DK, HK, DK, 1 / DK), k: part(HK * DK, 2 * HK * DK, HK, DK, 1 / Math.sqrt(DK)),
        v: part(2 * HK * DK, CONV, HV, DV, 1) };
    },
  };
  const inputs = (S: number, seeds: readonly number[]) => {
    const per = (seed: number) => [bf16(noise(S * CONV, seed), [1, S, CONV]), bf16(noise(S * HV, seed + 0.3), [1, S, HV]),
      bf16(noise(S * HV, seed + 0.6), [1, S, HV])] as const;
    const parts = seeds.map(per);
    return [0, 1, 2].map(index => rowsOf(parts.map(part => part[index]!))) as [MlxArray, MlxArray, MlxArray];
  };
  const recur = (cache: Cache, phase: "decode" | "window", S: number, seeds: readonly number[]) => {
    const [qkv, a, b] = inputs(S, seeds);
    const state = cache as GatedDeltaCache;
    return phase === "decode" ? state.recurDecode(qkv, a, b, layer) : state.recurWindow(qkv, a, b, layer);
  };
  const row = (length: number, seed: number) => { const cache = new SSMCache(); recur(cache, "window", length, [seed]).dispose(); return cache; };
  const statesEqual = (x: Cache, y: Cache) => {
    const a = x.state(), b = y.state();
    expect(a.length).toBe(b.length);
    for (let i = 0; i < a.length; i++) expect(maxDiff(a[i]!, b[i]!)).toBe(0);
  };

  test("phases: window, decode, window", () => {
    const cache = new SSMCache();
    try {
      for (const [phase, S, seed] of [["window", 5, 1], ["decode", 1, 2], ["window", 3, 3]] as const) {
        using out = recur(cache, phase, S, [seed]);
        expect(out.shape).toEqual([1, S, HV, DV]);
        expect(finite(out)).toBe(true);
      }
      expect(cache.offset).toBe(9);
    } finally { cache.dispose(); }
  });

  test("snapshot and SSD continue as the source", () => {
    const cache = row(6, 11), path = join(tmp, "ssm.kv");
    const [clone] = cloneKvCaches([cache]);
    try {
      evaluate([cache]);
      saveKvCache(path, Array.from({ length: 6 }, (_, i) => i), [cache]);
      const restored = loadKvCache(path, { makeCache: () => [new SSMCache()] }).caches[0]!;
      try {
        for (const other of [clone!, restored]) statesEqual(cache, other);
        using a = recur(clone!, "decode", 1, [12]);
        using b = recur(restored, "decode", 1, [12]);
        expect(maxDiff(a, b)).toBe(0);
      } finally { restored.dispose(); }
    } finally { dispose(cache, clone); }
  });

  test("rows and the speculative transaction", () => {
    const rows = [row(5, 21), row(9, 22)], twins = [row(5, 21), row(9, 22)];
    let layout: (BatchableCache & RowSpeculativeState) | undefined;
    const extracted: Cache[] = [];
    try {
      const decode = ownedCacheLayout(rows[0]!), prefill = prefillCacheLayout(rows[0]!);
      expect(decode).toBeDefined();
      dispose(decode, prefill);
      layout = targetCacheLayout(rows[0]!) as unknown as BatchableCache & RowSpeculativeState;
      layout.mergeRows(rows);
      const transaction = bindRowCacheRollback([layout], 2);
      transaction.begin(2);
      recur(layout, "window", 3, [23, 24]).dispose();
      transaction.resolve([0, 2]);
      for (const [index, keepCount] of [1, 3].entries()) {
        const [qkv, a, b] = inputs(3, [23 + index]);
        using qkvKeep = qkv.slice([0, 0, 0], [1, keepCount, CONV]);
        using aKeep = a.slice([0, 0, 0], [1, keepCount, HV]), bKeep = b.slice([0, 0, 0], [1, keepCount, HV]);
        qkv.dispose(); a.dispose(); b.dispose();
        (twins[index] as GatedDeltaCache).recurWindow(ops.contiguous(qkvKeep), ops.contiguous(aKeep), ops.contiguous(bKeep), layer).dispose();
      }
      for (let index = 0; index < 2; index++) extracted.push(layout.extractRow(index));
      for (const [index, cache] of extracted.entries()) statesEqual(cache, twins[index]!);
    } finally { dispose(layout, ...extracted, ...rows, ...twins); }
  });
});

// ---------------------------------------------------------------------------
// The latent attention state: GLM-5.2's compressed MLA state
// (`LatentAttentionCache`), with and without DSA index keys, under every lego a
// KV scheme and paging compose for it (`composedLatentCache`): f32 (`MLACache`),
// encoded f32-first (affine, TurboQuant, a later start) and paged (f32,
// affine, TurboQuant).

interface LatentLego {
  readonly name: string;
  readonly scheme: KvStorageScheme;
  readonly paged: boolean;
  /** Every read equals the stored f32 inputs bit for bit (f32 storage). */
  readonly exact: boolean;
  /** Positions the lego reads in f32 before it converts (encoded f32-first legos). */
  readonly f32Until: number;
  /** Bound on a decoded read's difference from its f32 inputs. */
  readonly tolerance: number;
}
const LATENT_LEGOS: readonly LatentLego[] = [
  { name: "f32 (MLACache)", scheme: BF16_KV, paged: false, exact: true, f32Until: Infinity, tolerance: 0 },
  { name: "affine 4, f32 first", scheme: affine(4), paged: false, exact: false, f32Until: 0, tolerance: 0.3 },
  { name: "affine 8, f32 for 6 positions", scheme: affine(8, 6), paged: false, exact: false, f32Until: 6, tolerance: 0.03 },
  { name: "TurboQuant 8/3, f32 first", scheme: turbo(), paged: false, exact: false, f32Until: 0, tolerance: 1 },
  { name: "paged f32", scheme: BF16_KV, paged: true, exact: true, f32Until: Infinity, tolerance: 0 },
  { name: "paged affine 4", scheme: affine(4), paged: true, exact: false, f32Until: -1, tolerance: 0.3 },
  { name: "paged TurboQuant 8/3", scheme: turbo(), paged: true, exact: false, f32Until: -1, tolerance: 1 },
];

for (const lego of LATENT_LEGOS) for (const dsa of [false, true])
describe(`latent attention state: ${lego.name}${dsa ? ", DSA index keys" : ""}`, () => {
  const RANK = 64, ROPE_DIM = 64, INDEX = 64;
  const make = (): Cache => composedLatentCache(lego.scheme, lego.paged ? { blockSize: 4 } : null, 0,
    { kvLoraRank: RANK, ropeHeadDim: ROPE_DIM, maxTokens: 64, ...(dsa ? { indexHeadDim: INDEX } : {}) });
  /** `L` float32 positions of each row's stream (`seeds` name the rows). */
  const inputs = (L: number, seeds: readonly number[]) => {
    const part = (width: number, shift: number) =>
      rowsOf(seeds.map(seed => MlxArray.fromFloat32(noise(L * width, seed + shift), [1, L, width])));
    return { latent: part(RANK, 0), rope: part(ROPE_DIM, 0.3), indexKeys: dsa ? part(INDEX, 0.6) : null };
  };
  const release = (state: LatentAttentionState) => { state.latent.dispose(); state.rope.dispose(); state.indexKeys?.dispose(); };
  const appendRows = (cache: Cache, L: number, seeds: readonly number[]): LatentAttentionState => {
    const { latent, rope, indexKeys } = inputs(L, seeds);
    try { return (cache as LatentAttentionCache).appendLatent(latent, rope, indexKeys); }
    finally { latent.dispose(); rope.dispose(); indexKeys?.dispose(); }
  };
  const expectStates = (label: string, a: LatentAttentionState, b: LatentAttentionState) => {
    expect(maxDiff(a.latent, b.latent), `${label} latent`).toBe(0);
    expect(maxDiff(a.rope, b.rope), `${label} rope`).toBe(0);
    if (dsa) expect(maxDiff(a.indexKeys!, b.indexKeys!), `${label} index keys`).toBe(0);
    else expect([a.indexKeys, b.indexKeys]).toEqual([null, null]);
  };
  const row = (length: number, seed: number) => { const cache = make(); release(appendRows(cache, length, [seed])); return cache; };
  const tail = (a: MlxArray, from: number, to: number) => a.slice([0, from, 0], [a.shape[0]!, to, a.shape[2]!]);
  /** A read of stored f32 positions `want`: bit for bit while the lego reads f32, within its tolerance once it decodes. */
  const expectRead = (label: string, read: MlxArray, want: MlxArray, f32: boolean) => {
    const diff = maxDiff(read, want);
    if (lego.exact || f32) expect(diff, label).toBe(0);
    else expect(diff, `${label}: ${diff}`).toBeLessThanOrEqual(lego.tolerance);
  };

  test("phases: window, decode, window; each read returns every stored position, f32 first where the lego converts", () => {
    const cache = make() as Cache & LatentAttentionCache;
    const stored: { latent: MlxArray; rope: MlxArray; indexKeys: MlxArray | null }[] = [];
    try {
      let position = 0;
      for (const [L, seed] of [[5, 1], [1, 2], [3, 3], [1, 4]] as const) {
        const before = cache.storedIndexKeys();
        const appended = inputs(L, [seed]);
        stored.push(appended);
        const state = cache.appendLatent(appended.latent, appended.rope, appended.indexKeys);
        try {
          expect(cache.offset).toBe(position + L);
          expect(state.latent.shape).toEqual([1, position + L, RANK]);
          // The lego reads f32 through the append that leaves it at its start (bf16-first's rule).
          const f32 = lego.f32Until >= 0 && position < Math.max(1, lego.f32Until);
          using allLatent = ops.concatAxis(stored.map(part => part.latent), 1);
          using allRope = ops.concatAxis(stored.map(part => part.rope), 1);
          expectRead(`latent at ${position}`, state.latent, allLatent, f32);
          expectRead(`rope at ${position}`, state.rope, allRope, f32);
          if (dsa) {
            using allIndex = ops.concatAxis(stored.map(part => part.indexKeys!), 1);
            expectRead(`index keys at ${position}`, state.indexKeys!, allIndex, f32);
            if (position === 0) expect(before).toBeNull();
            else { using prior = tail(state.indexKeys!, 0, position); expectRead("stored index keys", before!, prior, f32); }
          } else expect([state.indexKeys, before]).toEqual([null, null]);
          position += L;
        } finally { release(state); before?.dispose(); }
      }
    } finally { cache.dispose(); for (const part of stored) { part.latent.dispose(); part.rope.dispose(); part.indexKeys?.dispose(); } }
  });

  test("snapshot and SSD continue as the source", () => {
    const cache = row(6, 11), path = join(tmp, `latent-${lego.name.replace(/\W+/g, "-")}${dsa ? "-dsa" : ""}.kv`);
    const [clone] = cloneKvCaches([cache], latentLegoCodecs);
    try {
      evaluate([cache]);
      saveKvCache(path, Array.from({ length: 6 }, (_, i) => i), [cache], {}, latentLegoCodecs);
      const restored = loadKvCache(path, { makeCache: () => [make()] }, {}, latentLegoCodecs).caches[0]!;
      try {
        expect(restored.offset).toBe(6);
        expect(restored.signature()).toBe(cache.signature());
        const source = appendRows(cache, 1, [12]);
        try {
          for (const [label, other] of [["clone", clone!], ["restored", restored]] as const) {
            const next = appendRows(other, 1, [12]);
            try { expectStates(label, next, source); } finally { release(next); }
          }
        } finally { release(source); }
      } finally { restored.dispose(); }
    } finally { dispose(cache, clone); }
  });

  test("rows and the speculative transaction", () => {
    const rows = [row(5, 21), row(9, 22)], twins = [row(5, 21), row(9, 22)];
    let layout: (BatchableCache & RowSpeculativeState) | undefined;
    const extracted: Cache[] = [];
    try {
      layout = targetCacheLayout(rows[0]!) as unknown as BatchableCache & RowSpeculativeState;
      layout.mergeRows(rows);
      expect([...layout.rowOffsets]).toEqual([5, 9]);
      // A merged layout reads each row as its serial twin does.
      {
        const a = appendRows(layout, 1, [26, 27]);
        const twinReads = twins.map((twin, index) => appendRows(twin, 1, [26 + index]));
        try {
          for (const [index, read] of twinReads.entries()) {
            const pad = layout.leftPad[index]!, width = layout.offset;
            using mine = a.latent.slice([index, pad, 0], [index + 1, width, RANK]);
            expect(maxDiff(mine, read.latent), `merged row ${index}`).toBe(0);
          }
        } finally { release(a); for (const read of twinReads) release(read); }
      }
      // Two drafts: row 0 accepts none (keeps one position), row 1 accepts both (keeps three).
      const transaction = bindRowCacheRollback([layout], 2);
      transaction.begin(2);
      release(appendRows(layout, 3, [23, 24]));
      transaction.resolve([0, 2]);
      expect([...layout.rowOffsets]).toEqual([7, 13]);
      for (const [index, keep] of [1, 3].entries()) {
        const { latent, rope, indexKeys } = inputs(3, [23 + index]);
        try {
          using keptLatent = tail(latent, 0, keep), keptRope = tail(rope, 0, keep);
          using keptIndex = indexKeys ? tail(indexKeys, 0, keep) : null;
          release((twins[index] as LatentAttentionCache).appendLatent(keptLatent, keptRope, keptIndex));
        } finally { latent.dispose(); rope.dispose(); indexKeys?.dispose(); }
      }
      for (let index = 0; index < 2; index++) extracted.push(layout.extractRow(index));
      for (const [index, cache] of extracted.entries()) {
        expect(cache.offset).toBe(twins[index]!.offset);
        const a = appendRows(cache, 1, [25 + index]), b = appendRows(twins[index]!, 1, [25 + index]);
        try { expectStates(`row ${index}`, a, b); } finally { release(a); release(b); }
      }
      layout.filterRows([1]);
      expect([...layout.rowOffsets]).toEqual([13]);
    } finally { dispose(layout, ...extracted, ...rows, ...twins); }
  });
});
