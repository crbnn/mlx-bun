// Every KV storage answers the dense fetch (`updateAndFetch`: the window an
// append leaves, as keys and values arrays), which a read that attends them that
// way (Gemma 2's softcapped read) takes: plain storage as stored, TurboQuant
// decoded, affine storage dequantized, before and after a row's transition.
//
// The oracle for affine storage: the plain storage the lego starts as (KVCache,
// RotatingKVCache, or the plain row layouts), with a converted row's fetched
// window affine-quantized and dequantized again by mlx itself. Groups run along
// the head dimension, so a position's codes do not depend on when it was
// quantized: quantizing the window at the conversion and each later append on
// its own equals quantizing the fetched window at every read. A row converts
// after the append that leaves it holding the start (the bf16-first rule).
import { expect, test } from "bun:test";
import * as ops from "@mlx-bun/mlx/ops";
import { Dtype } from "@mlx-bun/mlx/ffi";
import type { MlxArray } from "@mlx-bun/mlx/array";
import { KVCache } from "../../src/state/kv";
import { RotatingKVCache } from "../../src/state/rotating-kv";
import { BatchedRotatingCache } from "../../src/state/batched-rotating";
import { DelayedQuantizedKVCache } from "../../src/state/delayed-quantized-kv";
import { DelayedRotatingQuantizedKVCache } from "../../src/state/delayed-rotating-quantized-kv";
import { DelayedTurboQuantKVCache } from "../../src/state/delayed-turboquant-kv";
import { unfusedAffineKernels } from "../../src/state/affine-attention";
import { deferredTurboQuantKernels } from "../../src/state/turboquant-attention";
import { turboQuantFusedDecode } from "../../src/state/turboquant-codec";
import { AttentionMasks } from "../../src/state/attention-read";
import { composedFullAttentionCache, composedSlidingAttentionCache, transitionDue } from "../../src/state/bf16-first-kv";
import type { Cache } from "../../src/contracts/mlx/cache";
import type { KvStorageScheme } from "../../src/contracts/portable/composition";
import { kvStorage } from "../../src/state/kv-storage";
import { disposeResources } from "../../src/runtime/resources";

const W = 8, GROUP = 64;
const tensor = (b: number, n: number, seed: number) => {
  using key = ops.randomKey(BigInt(seed)); return ops.randomNormal([b, 2, n, GROUP], Dtype.bfloat16, 0, 1, key);
};
/** `x` affine-quantized at `bits` and dequantized again (owned). */
const requantized = (x: MlxArray, bits: number): MlxArray => {
  const q = ops.quantize(x, GROUP, bits);
  try { return ops.dequantize(q.packed, q.scales, q.biases, { groupSize: GROUP, bits, mode: "affine" }); }
  finally { q.packed.dispose(); q.scales.dispose(); q.biases.dispose(); }
};
/** The same shape, dtype and bytes. */
const expectSame = (label: string, actual: MlxArray, expected: MlxArray) => {
  expect([actual.shape, actual.dtypeName], label).toEqual([expected.shape, expected.dtypeName]);
  // Boolean masks compare as integers (no raw byte view of bool).
  const bytes = (x: MlxArray) => x.dtypeName === "bool" ? x.astype(Dtype.int32) : ops.contiguous(x);
  using a = bytes(actual), b = bytes(expected);
  expect(Buffer.from(a.rawBytes()).equals(Buffer.from(b.rawBytes())), `${label}: bytes`).toBe(true);
};
const fetch = (cache: Cache, k: MlxArray, v: MlxArray) => kvStorage(cache).updateAndFetch(k, v);
/** Appends of these lengths cross the ring's window and the starts below. */
const APPENDS = [4, 1, 1, 3, 1, 1, 1, 5, 1, 2];

for (const rotating of [false, true]) for (const bits of [4, 8]) for (const start of [0, 3, 6, 11]) {
  test(`${rotating ? "sliding" : "full"} affine ${bits} from ${start}: the lego's dense fetch is its bf16 window, then its window dequantized`, () => {
    const kv: KvStorageScheme = { kind: "affine", bits, groupSize: GROUP, quantizedKvStart: start, kernels: "unfused" };
    const masks = new AttentionMasks();
    const lego = rotating ? composedSlidingAttentionCache(kv, 0, W, masks) : composedFullAttentionCache(kv, 0, masks, deferredTurboQuantKernels);
    const plain = rotating ? new RotatingKVCache(W) : new KVCache();
    let converted = false, seed = 100;
    try {
      for (const [step, length] of APPENDS.entries()) {
        const label = `append ${step} (${length} at ${plain.offset})`;
        const mask = kvStorage(lego).makeMask(length, rotating ? W : null), expectedMask = plain.makeMask(length, rotating ? W : null);
        try {
          expect(mask.mode, label).toBe(expectedMask.mode);
          if (expectedMask.arr) expectSame(`${label} mask`, mask.arr!, expectedMask.arr);
        } finally { mask.arr?.dispose(); expectedMask.arr?.dispose(); }
        using k = tensor(1, length, seed++), v = tensor(1, length, seed++);
        const actual = fetch(lego, k, v), stored = plain.updateAndFetch(k, v);
        const expected = converted ? stored.map(x => requantized(x, bits)) : stored;
        try { for (const index of [0, 1]) expectSame(`${label} ${index ? "values" : "keys"}`, actual[index]!, expected[index]!); }
        finally { disposeResources([...actual, ...stored, ...(converted ? expected : [])]); }
        converted ||= transitionDue(plain.offset, start);
      }
      expect((lego.composed!.storage()).constructor.name).toBe(rotating ? "RotatingQuantizedKVCache" : "QuantizedKVCache");
    } finally { lego.dispose(); plain.dispose(); }
  });
}

test("per-layer affine: a listed layer converts and fetches dequantized; an unlisted layer stays bf16", () => {
  const kv: KvStorageScheme = { kind: "affine-layers", layers: [{ layer: 0, bits: 4, groupSize: GROUP }], quantizedKvStart: 3, kernels: "unfused" };
  const masks = new AttentionMasks();
  const listed = composedFullAttentionCache(kv, 0, masks, deferredTurboQuantKernels), unlisted = composedFullAttentionCache(kv, 1, masks, deferredTurboQuantKernels);
  const plain = [new KVCache(), new KVCache()];
  let seed = 200;
  try {
    for (const [step, length] of [2, 1, 2, 1].entries()) {
      for (const [index, cache] of [listed, unlisted].entries()) {
        const before = plain[index]!.offset;
        using k = tensor(1, length, seed++), v = tensor(1, length, seed++);
        const actual = fetch(cache, k, v), stored = plain[index]!.updateAndFetch(k, v);
        const converted = index === 0 && transitionDue(before, 3);
        const expected = converted ? stored.map(x => requantized(x, 4)) : stored;
        try { for (const plane of [0, 1]) expectSame(`step ${step} layer ${index}`, actual[plane]!, expected[plane]!); }
        finally { disposeResources([...actual, ...stored, ...(converted ? expected : [])]); }
      }
    }
    expect([listed.composed!.storage().constructor.name, unlisted.constructor.name]).toEqual(["QuantizedKVCache", "KVCache"]);
  } finally { disposeResources([listed, unlisted, ...plain]); }
});

/** `lengths` rows, each prefilled with its own tokens, in the delayed layout
 * `make` builds and, as plain rows, in `plainLayout` (the plain row layout the
 * same rows join, the oracle's storage). */
function rowsIn<L extends Cache>(make: () => L, plainLayout: () => Cache & { mergeRows(rows: readonly Cache[]): void },
  rotating: boolean, lengths: readonly number[], seed: number) {
  const rows = lengths.map((length, index) => {
    const row = rotating ? new RotatingKVCache(W) : new KVCache();
    using k = tensor(1, length, seed + 2 * index), v = tensor(1, length, seed + 2 * index + 1);
    disposeResources(row.updateAndFetch(k, v));
    return row;
  });
  const layout = make() as L & { mergeRows(rows: readonly Cache[]): void };
  const plain = plainLayout();
  try { layout.mergeRows(rows); plain.mergeRows(rows); } finally { disposeResources(rows); }
  return { layout, plain };
}

for (const rotating of [false, true]) for (const [start, lengths] of [[4, [2, 6]], [5, [1, 4, 7]], [0, [3, 3]], [40, [2, 5]]] as const) {
  test(`delayed ${rotating ? "sliding" : "full"} affine rows ${lengths} from ${start}: each row fetches bf16 until its transition, then dequantized, mixed and packed alike`, () => {
    const make = (): Cache => rotating ? new DelayedRotatingQuantizedKVCache(W, GROUP, 4, start, unfusedAffineKernels)
      : new DelayedQuantizedKVCache(GROUP, 4, start, unfusedAffineKernels);
    // The plain row layout: the one a full layout pads its rows in (zeros on
    // the left), or the plain ring layout.
    const plainLayout = rotating ? () => new BatchedRotatingCache(W, []) : () => new DelayedQuantizedKVCache(GROUP, 4, Number.MAX_SAFE_INTEGER, unfusedAffineKernels);
    const { layout, plain } = rowsIn(make, plainLayout, rotating, lengths, 300);
    const B = lengths.length, offsets = [...lengths] as number[];
    const converted = offsets.map(offset => transitionDue(offset, start));
    let seed = 400;
    try {
      for (const [step, length] of [1, 1, 2, 1, 1, 3, 1].entries()) {
        using k = tensor(B, length, seed++), v = tensor(B, length, seed++);
        const actual = fetch(layout, k, v), stored = fetch(plain, k, v);
        const expected = stored.map(plane => {
          const parts = Array.from({ length: B }, (_, row) => {
            const slice = plane.slice([row, 0, 0, 0], [row + 1, plane.shape[1]!, plane.shape[2]!, plane.shape[3]!]);
            if (!converted[row]) return slice;
            // A full row's left padding stays zeros; its stored positions requantize.
            const pad = rotating ? 0 : plane.shape[2]! - (offsets[row]! + length);
            if (!pad) { try { return requantized(slice, 4); } finally { slice.dispose(); } }
            using padding = slice.slice([0, 0, 0, 0], [1, plane.shape[1]!, pad, plane.shape[3]!]);
            using held = slice.slice([0, 0, pad, 0], [1, plane.shape[1]!, plane.shape[2]!, plane.shape[3]!]);
            using dense = requantized(held, 4);
            slice.dispose();
            return ops.concatAxis([padding, dense], 2);
          });
          try { return ops.concatAxis(parts, 0); } finally { disposeResources(parts); }
        });
        try { for (const index of [0, 1]) expectSame(`step ${step} ${index ? "values" : "keys"}`, actual[index]!, expected[index]!); }
        finally { disposeResources([...actual, ...stored, ...expected]); }
        for (let row = 0; row < B; row++) { offsets[row]! += length; converted[row] ||= transitionDue(offsets[row]!, start); }
      }
    } finally { layout.dispose(); plain.dispose(); }
  });
}

test("delayed TurboQuant rows fetch decoded before, across and after their transition", () => {
  const group = new DelayedTurboQuantKVCache(8, 3, 3, turboQuantFusedDecode(), deferredTurboQuantKernels);
  const rows = [2, 5].map((length, index) => {
    const row = new KVCache();
    using k = tensor(1, length, 500 + 2 * index), v = tensor(1, length, 501 + 2 * index);
    disposeResources(row.updateAndFetch(k, v));
    return row;
  });
  try {
    group.mergeRows(rows);
    for (const [step, length] of [1, 1, 2].entries()) {
      using k = tensor(2, length, 600 + 2 * step), v = tensor(2, length, 601 + 2 * step);
      const [keys, values] = fetch(group, k, v);
      try {
        expect([keys.shape[0], keys.shape[2], values.dtypeName]).toEqual([2, Math.max(...group.rowOffsets), keys.dtypeName]);
        expect(keys.toFloat32().every(Number.isFinite) && values.toFloat32().every(Number.isFinite)).toBe(true);
      } finally { keys.dispose(); values.dispose(); }
    }
  } finally { disposeResources([group, ...rows]); }
});
