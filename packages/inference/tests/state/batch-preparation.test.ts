import { expect, test } from "bun:test";
import { createKvMaintenance, unchangedKv } from "../../src/state/layout";
import { KVCache } from "../../src/state/kv";
import { unfusedAffineKernels } from "../../src/state/affine-attention";
import { AttentionMasks } from "../../src/state/attention-read";
import { composedFullAttentionCache, composedSlidingAttentionCache } from "../../src/state/bf16-first-kv";
import { KvScheme } from "../../src/state/kv-scheme";
import { RotatingKVCache } from "../../src/state/rotating-kv";
import { QuantizedKVCache } from "../../src/state/quantized-kv";
import { RotatingQuantizedKVCache } from "../../src/state/rotating-quantized-kv";
import { TurboQuantKVCache } from "../../src/state/turboquant-kv";
import { denseTurboQuantKernels } from "../../src/state/turboquant-attention";
import { type Cache, type BatchableCache } from "../../src/contracts/mlx/cache";
import type { KvStorageScheme } from "../../src/contracts/portable/composition";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { kvStorage } from "../../src/state/kv-storage";

function populate(cache: Cache, length: number): void {
  const source = MlxArray.fromFloat32(Float32Array.from({ length: length * 64 }, (_, i) => Math.sin(i * 0.13)), [1, 1, length, 64]);
  const value = source.astype(Dtype.bfloat16);
  source.dispose();
  const fetched = kvStorage(cache).updateAndFetch(value, value);
  ops.evalAll(fetched);
  for (const array of fetched) array.dispose();
  value.dispose();
}

/** What a graph's `makeCache` composes for `kv`: full-attention layers, and a
 * sliding layer of 8 positions where `sliding` names it. */
function compose(kv: KvStorageScheme, layers: readonly ("full" | "sliding")[]): Cache[] {
  const masks = new AttentionMasks();
  return layers.map((kind, layer) => kind === "sliding"
    ? composedSlidingAttentionCache(kv, layer, 8, masks)
    : composedFullAttentionCache(kv, layer, masks, denseTurboQuantKernels));
}

const storageOf = (cache: Cache): Cache => cache.composed?.storage() ?? cache;

test("composed legos keep the populated boundary, layer policy and converted bytes", () => {
  const config = [{ layer: 0, bits: 4, groupSize: 32 }, { layer: 1, bits: 8, groupSize: 64 }];
  const kv: KvStorageScheme = { kind: "affine-layers", layers: config, quantizedKvStart: 3, kernels: "unfused" };
  const caches = compose(kv, ["full", "sliding", "full"]);
  const references = [new KVCache(), new RotatingKVCache(8)];
  try {
    expect(caches.map(cache => cache.maxAppendTokens?.() ?? Infinity)).toEqual([3, 3, Infinity]);
    expect(caches[2]).toBeInstanceOf(KVCache);
    for (const cache of [...caches, ...references]) populate(cache, 2);
    expect(caches.map(cache => cache.maxAppendTokens?.() ?? Infinity)).toEqual([1, 1, Infinity]);
    expect(storageOf(caches[0]!)).toBeInstanceOf(KVCache);
    expect(storageOf(caches[1]!)).toBeInstanceOf(RotatingKVCache);
    for (const cache of [...caches, ...references]) populate(cache, 1);
    expect(storageOf(caches[0]!)).toBeInstanceOf(QuantizedKVCache);
    expect(storageOf(caches[1]!)).toBeInstanceOf(RotatingQuantizedKVCache);
    expect(caches[2]).toBeInstanceOf(KVCache);
    expect(caches.map(cache => cache.maxAppendTokens?.() ?? Infinity)).toEqual([Infinity, Infinity, Infinity]);
    for (let layer = 0; layer < 2; layer++) {
      const entry = config[layer]!;
      const expected = references[layer]!.toQuantized(entry.groupSize, entry.bits, unfusedAffineKernels);
      try {
        expect(caches[layer]!.minimumReusableOffset).toBe(3);
        const actual = caches[layer]!.state();
        const wanted = expected.state();
        expect(actual.length).toBe(wanted.length);
        actual.forEach((array, i) => expect(Buffer.from(array.rawBytesView())).toEqual(Buffer.from(wanted[i]!.rawBytesView())));
      } finally { expected.dispose(); }
    }
  } finally {
    for (const cache of caches) cache.dispose();
    for (const cache of references) cache.dispose();
  }
});

test("uniform schemes default to a 5000-token start; a zero start converts after the first append, never while empty", () => {
  expect(new KvScheme("affine-uniform", { kvBits: 4 }).storage)
    .toEqual({ kind: "affine", bits: 4, groupSize: 64, quantizedKvStart: 5000, kernels: "unfused" });
  const waiting = compose(new KvScheme("affine-uniform", { kvBits: 4 }).storage, ["full", "sliding"]);
  const eager = compose({ kind: "affine", bits: 4, groupSize: 64, kernels: "unfused" }, ["full", "sliding"]);
  try {
    expect(storageOf(eager[0]!)).toBeInstanceOf(KVCache);
    expect(storageOf(eager[1]!)).toBeInstanceOf(RotatingKVCache);
    for (const cache of [...waiting, ...eager]) populate(cache, 3);
    expect(storageOf(waiting[0]!)).toBeInstanceOf(KVCache);
    expect(storageOf(waiting[1]!)).toBeInstanceOf(RotatingKVCache);
    expect(storageOf(eager[0]!)).toBeInstanceOf(QuantizedKVCache);
    expect(storageOf(eager[1]!)).toBeInstanceOf(RotatingQuantizedKVCache);
  } finally { for (const cache of [...waiting, ...eager]) cache.dispose(); }
});

test("the bf16 scheme composes plain caches and needs no row maintenance", () => {
  const caches = compose({ kind: "bf16" }, ["full", "sliding"]);
  try {
    expect(caches[0]).toBeInstanceOf(KVCache);
    expect(caches[1]).toBeInstanceOf(RotatingKVCache);
    expect(createKvMaintenance({ kind: "bf16" })).toBe(unchangedKv);
  } finally { for (const cache of caches) cache.dispose(); }
});

test("TurboQuant append work stops at its conversion boundary", () => {
  const caches = compose({ kind: "turbo", kBits: 8, vBits: 3, quantizedKvStart: 3 }, ["full"]);
  try {
    populate(caches[0]!, 2);
    expect(caches[0]!.maxAppendTokens!()).toBe(1);
    populate(caches[0]!, 1);
    expect(storageOf(caches[0]!)).toBeInstanceOf(TurboQuantKVCache);
    expect(caches[0]!.minimumReusableOffset).toBe(3);
    expect(caches[0]!.maxAppendTokens!()).toBe(Number.POSITIVE_INFINITY);
  } finally { for (const cache of caches) cache.dispose(); }
});

for (const format of ["affine", "rotating", "turboquant"] as const) {
  test(`${format} shared append limits follow logical row offsets through retirement`, () => {
    const kv: KvStorageScheme = format === "turboquant" ? { kind: "turbo", kBits: 8, vBits: 3, quantizedKvStart: 5 }
      : { kind: "affine", bits: 4, groupSize: 64, quantizedKvStart: 5, kernels: "unfused" };
    const maintain = createKvMaintenance(kv);
    const rows: Cache[] = [2, 4, 7].map(length => {
      const row = compose(kv, [format === "rotating" ? "sliding" : "full"])[0]!;
      populate(row, length); return row;
    });
    let group: BatchableCache | undefined;
    try {
      maintain.prepareBatch!(rows);
      group = (rows[0] as BatchableCache).makeEmptyBatch();
      group.mergeRows(rows);
      expect(group.rowOffsets).toEqual([2, 4, 7]);
      // The physical batch offset is seven, but two rows still await conversion.
      expect(maintain.maxAppendTokens!([group])).toBe(1);
      group.filterRows([0, 2]);
      expect(maintain.maxAppendTokens!([group])).toBe(3);
      group.filterRows([1]);
      expect(maintain.maxAppendTokens!([group])).toBe(Infinity);
    } finally { group?.dispose(); for (const row of rows) row.dispose(); }
  });
}

test("delayed batch preparation binds layer precision before interpreting row membership", async () => {
  const { DelayedQuantizedKVCache } = await import("../../src/state/delayed-quantized-kv");
  const { cloneKvCaches } = await import("../../src/state/persistence");
  const kv: KvStorageScheme = { kind: "affine-layers", layers: [
    { layer: 0, bits: 4, groupSize: 32 },
    { layer: 2, bits: 8, groupSize: 64 },
  ], quantizedKvStart: 3, kernels: "unfused" };
  const maintain = createKvMaintenance(kv);
  const rows: Cache[][] = [2, 4].map(length => compose(kv, ["full", "full", "full"]).map(cache => {
    populate(cache, length); return cache;
  }));
  // The serial reference: the same legos, cloned before the rows join a layout.
  const reference = rows.map(row => cloneKvCaches(row));
  const groups: Cache[] = [];
  try {
    for (const row of rows) maintain.prepareBatch!(row);
    expect(rows[0]![1]).toBeInstanceOf(KVCache);
    for (const [layer, bits, groupSize] of [[0, 4, 32], [2, 8, 64]]) {
      const first = rows[0]![layer!] as InstanceType<typeof DelayedQuantizedKVCache>;
      expect(first).toBeInstanceOf(DelayedQuantizedKVCache);
      const group = first.makeEmptyBatch(); groups.push(group);
      group.mergeRows(rows.map(row => row[layer!]!));
      group.specRoundBegin(); group.specRoundCommit();
      expect(group.rowOffsets).toEqual([2, 4]);
      for (let row = 0; row < rows.length; row++) {
        const actual = group.extractRow(row), expected = reference[row]![layer!]!;
        try {
          // A row handed back to serial use is the lego again.
          expect(actual.constructor).toBe(expected.constructor);
          expect(actual.signature()).toBe(expected.signature());
          expect(actual.minimumReusableOffset ?? 0).toBe(expected.minimumReusableOffset ?? 0);
          const storage = storageOf(actual);
          if (storage instanceof QuantizedKVCache) {
            expect(storage.bits).toBe(bits!); expect(storage.groupSize).toBe(groupSize!);
          }
          expect(actual.state().map(a => Buffer.from(a.rawBytesView())))
            .toEqual(expected.state().map(a => Buffer.from(a.rawBytesView())));
        } finally { actual.dispose(); }
      }
    }
  } finally {
    for (const cache of [...groups, ...rows.flat(), ...reference.flat()]) cache.dispose();
  }
});
