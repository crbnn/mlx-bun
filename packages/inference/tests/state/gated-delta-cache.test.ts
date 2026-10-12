// The recurrent caches own the gated-DeltaNet read (GatedDeltaCache). A block
// on small generated weights lends the cache its recurrence (`recurrence`: the
// weights and the glue between the convolution and the kernel), and the test
// drives `recurDecode`/`recurWindow` with the block's projections: padded rows
// equal their unpadded one-row runs, and a speculative round's replay equals a
// fresh run over the accepted positions. No model, no download.
import { afterAll, describe, expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import type { ModelConfig } from "../../src/artifacts/config";
import type { Weights } from "../../src/artifacts/weights";
import type { GatedDeltaCache } from "../../src/contracts/mlx/cache";
import { disposing } from "../../src/layers/helpers";
import { GatedDeltaNet, compiledPreciseSwiglu, type AttentionLinear } from "../../src/models/qwen/blocks";
import { BatchedSSMCache } from "../../src/state/batched-ssm";
import { SSMCache } from "../../src/state/ssm";

const HIDDEN = 64, HK = 2, HV = 4, DK = 64, DV = 32, KERNEL = 4;
const KEY_DIM = HK * DK, VALUE_DIM = HV * DV, CONV_DIM = 2 * KEY_DIM + VALUE_DIM;

const owned: MlxArray[] = [];
afterAll(() => { for (const array of owned) array.dispose(); });

const noise = (count: number, phase: number, scale: number, shift = 0) => Float32Array.from({ length: count },
  (_, i) => shift + Math.sin(i * 0.37 + phase) * scale + Math.cos(i * 0.11 + 2 * phase) * scale);
const tensor = (values: Float32Array, shape: number[], dtype = Dtype.bfloat16): MlxArray => {
  using wide = MlxArray.fromFloat32(values, shape);
  return wide.astype(dtype);
};
const input = (B: number, S: number, phase: number) => tensor(noise(B * S * HIDDEN, phase, 0.8), [B, S, HIDDEN]);

class DenseLinear implements AttentionLinear {
  constructor(readonly weight: MlxArray) {}
  forward(x: MlxArray): MlxArray { return ops.matmul(x, this.weight); }
}

function makeBlock(): GatedDeltaNet<DenseLinear> {
  const keep = (array: MlxArray) => { owned.push(array); return array; };
  const tensors = new Map<string, MlxArray>([
    ["layer.conv1d.weight", keep(tensor(noise(CONV_DIM * KERNEL, 1, 0.3), [CONV_DIM, KERNEL, 1]))],
    ["layer.A_log", keep(tensor(noise(HV, 2, 0.5), [HV], Dtype.float32))],
    ["layer.dt_bias", keep(tensor(noise(HV, 3, 0.5), [HV]))],
    ["layer.norm.weight", keep(tensor(noise(DV, 4, 0.2, 1), [DV]))],
  ]);
  const weights = { tensor: (name: string) => tensors.get(name)! } as unknown as Weights;
  const config = { text: {
    linearNumKeyHeads: HK, linearNumValueHeads: HV, linearKeyHeadDim: DK, linearValueHeadDim: DV,
    linearConvKernelDim: KERNEL, rmsNormEps: 1e-6,
  } } as unknown as ModelConfig;
  const sizes: Record<string, [number, number]> = {
    in_proj_qkv: [HIDDEN, CONV_DIM], in_proj_z: [HIDDEN, VALUE_DIM], in_proj_b: [HIDDEN, HV],
    in_proj_a: [HIDDEN, HV], out_proj: [VALUE_DIM, HIDDEN],
  };
  let seed = 10;
  return new GatedDeltaNet<DenseLinear>(weights, config, "layer", (_weights, path) => {
    const [rows, cols] = sizes[path.split(".").pop()!]!;
    return new DenseLinear(keep(tensor(noise(rows * cols, ++seed, 0.15), [rows, cols])));
  });
}

interface Projected { qkv: MlxArray; z: MlxArray; a: MlxArray; b: MlxArray }

/** The block's projections, owned by the caller. */
function project(block: GatedDeltaNet<DenseLinear>, x: MlxArray): Projected {
  const [B, S] = x.shape as [number, number, number];
  const qkv = block.inProjQkv.forward(x);
  let z = block.inProjZ.forward(x);
  z = disposing(z, ops.reshape(z, [B, S, block.numVHeads, block.headVDim]));
  return { qkv, z, b: block.inProjB.forward(x), a: block.inProjA.forward(x) };
}

/** Copies of positions [from, to) of rows [row, row + rows), so a reference
 *  run reads exactly the projected values the windowed run read. */
function cut(array: MlxArray, from: number, to: number, row = 0, rows = array.shape[0]!): MlxArray {
  const start = array.shape.map((_, axis) => axis === 0 ? row : axis === 1 ? from : 0);
  const stop = array.shape.map((size, axis) => axis === 0 ? row + rows : axis === 1 ? to : size);
  using view = array.slice(start, stop);
  return ops.contiguous(view);
}
function cutProjected(p: Projected, from: number, to: number, row?: number, rows?: number): Projected {
  return { qkv: cut(p.qkv, from, to, row, rows), z: cut(p.z, from, to, row, rows),
    a: cut(p.a, from, to, row, rows), b: cut(p.b, from, to, row, rows) };
}

/** The block's output from its projections around one recurrence call. The
 *  call consumes qkv, a and b; z is consumed here. */
function finish(block: GatedDeltaNet<DenseLinear>, p: Projected, recur: (qkv: MlxArray, a: MlxArray, b: MlxArray) => MlxArray): MlxArray {
  const [B, S] = p.qkv.shape as [number, number, number];
  const out = recur(p.qkv, p.a, p.b);
  const xn = ops.rmsNorm(out, block.normWeight, block.eps);
  const gated = compiledPreciseSwiglu(out, p.z, xn);
  xn.dispose();
  out.dispose();
  p.z.dispose();
  const merged = ops.reshape(gated, [B, S, block.valueDim]);
  gated.dispose();
  const result = block.outProj.forward(merged);
  merged.dispose();
  return result;
}

type Phase = "decode" | "window";
function viaCache(block: GatedDeltaNet<DenseLinear>, x: MlxArray, cache: GatedDeltaCache, phase: Phase): MlxArray {
  return finish(block, project(block, x), (qkv, a, b) =>
    phase === "decode" ? cache.recurDecode(qkv, a, b, block.recurrence) : cache.recurWindow(qkv, a, b, block.recurrence));
}

/** Largest absolute difference; the raw bytes must also agree. */
function maxDiff(actual: MlxArray | null, expected: MlxArray | null): number {
  expect(actual === null).toBe(expected === null);
  if (!actual || !expected) return 0;
  expect(actual.shape).toEqual(expected.shape);
  expect(actual.dtype).toBe(expected.dtype);
  using left = ops.contiguous(actual);
  using right = ops.contiguous(expected);
  expect(Buffer.from(left.rawBytes()).equals(Buffer.from(right.rawBytes()))).toBe(true);
  using leftWide = left.astype(Dtype.float32);
  using rightWide = right.astype(Dtype.float32);
  const a = leftWide.toFloat32(), b = rightWide.toFloat32();
  expect(b.some(value => value !== 0)).toBe(true);
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i]! - b[i]!));
  return worst;
}

function stateDiff(actual: SSMCache, expected: SSMCache): { conv: number; recurrent: number } {
  expect(actual.offset).toBe(expected.offset);
  expect(actual.offsets).toEqual(expected.offsets);
  return { conv: maxDiff(actual.conv, expected.conv), recurrent: maxDiff(actual.recurrent, expected.recurrent) };
}

function rowState(cache: SSMCache, row: number): { conv: MlxArray; recurrent: MlxArray } {
  return { conv: cut(cache.conv!, 0, cache.conv!.shape[1]!, row, 1), recurrent: cut(cache.recurrent!, 0, HV, row, 1) };
}

const report = (name: string, values: Record<string, number>) =>
  console.log(`gated-delta-cache ${name}: ${Object.entries(values).map(([key, value]) => `${key} maxDiff ${value}`).join(", ")}`);

const worst = (results: Record<string, number>[]) => results.reduce((acc, result) => {
  for (const [key, value] of Object.entries(result)) acc[key] = Math.max(acc[key] ?? 0, value);
  return acc;
}, {} as Record<string, number>);

const zeros = (result: Record<string, number>) => { for (const value of Object.values(result)) expect(value).toBe(0); };

describe("BatchedSSMCache honours prepared padding", () => {
  const block = makeBlock();
  const lengths = [3, 7, 5];

  /** Each row's state equals a one-row cache run over that row's real positions. */
  function expectRowsMatchUnpadded(cache: SSMCache, p: Projected, starts: readonly number[], ends: readonly number[]) {
    const results: Record<string, number>[] = [];
    for (let row = 0; row < lengths.length; row++) {
      const single = new SSMCache();
      try {
        const real = cutProjected(p, starts[row]!, ends[row]!, row, 1);
        real.z.dispose();
        single.recurWindow(real.qkv, real.a, real.b, block.recurrence).dispose();
        const found = rowState(cache, row);
        try { results.push({ conv: maxDiff(found.conv, single.conv), recurrent: maxDiff(found.recurrent, single.recurrent) }); }
        finally { found.conv.dispose(); found.recurrent.dispose(); }
        expect(single.offset).toBe(lengths[row]!);
      } finally { single.dispose(); }
    }
    return worst(results);
  }

  test("3 rows of real lengths 3, 7, 5, right padded, in chunks of 4 and 3, then decode: each row advances by its real length", () => {
    const cache = new BatchedSSMCache();
    try {
      cache.preparePrefill({ lengths, rightPadding: lengths.map(n => 7 - n) });
      using x = input(3, 7, 60);
      using head = x.slice([0, 0, 0], [3, 4, HIDDEN]);
      using tail = x.slice([0, 4, 0], [3, 7, HIDDEN]);
      using chunk0 = ops.contiguous(head);
      using chunk1 = ops.contiguous(tail);
      viaCache(block, chunk0, cache, "window").dispose();
      viaCache(block, chunk1, cache, "window").dispose();
      expect(cache.rowOffsets).toEqual(lengths);
      cache.finalizePrefill();
      using token = input(3, 1, 61);
      viaCache(block, token, cache, "decode").dispose();
      expect(cache.rowOffsets).toEqual(lengths.map(n => n + 1));
    } finally { cache.dispose(); }
  });

  test("padded positions change neither state: each row equals its unpadded one-row run", () => {
    for (const side of ["right", "left"] as const) {
      const cache = new BatchedSSMCache();
      try {
        cache.preparePrefill(side === "right"
          ? { lengths, rightPadding: lengths.map(n => 7 - n) }
          : { lengths: [7, 7, 7], leftPadding: lengths.map(n => 7 - n) });
        using x = input(3, 7, 80);
        const p = project(block, x);
        const kept = cutProjected(p, 0, 7);
        kept.z.dispose();
        cache.recurWindow(kept.qkv, kept.a, kept.b, block.recurrence).dispose();
        if (side === "left") expect(cache.rowOffsets).toEqual(lengths);
        const starts = side === "right" ? [0, 0, 0] : lengths.map(n => 7 - n);
        const ends = side === "right" ? lengths : [7, 7, 7];
        const result = expectRowsMatchUnpadded(cache, p, starts, ends);
        for (const array of [p.qkv, p.z, p.a, p.b]) array.dispose();
        report(`batched ${side} padding vs unpadded rows`, result);
        zeros(result);
      } finally { cache.dispose(); }
    }
  });
});

describe("speculative rounds replay inside the cache", () => {
  const block = makeBlock();

  test("rollback to 2 accepted of 5 drafted, replayed, equals a fresh run over those 2", () => {
    const cache = new SSMCache(), reference = new SSMCache();
    try {
      using prior = input(1, 5, 90);
      viaCache(block, prior, cache, "window").dispose();
      viaCache(block, prior, reference, "window").dispose();

      using draft = input(1, 5, 91);
      cache.specRoundBegin();
      viaCache(block, draft, cache, "window").dispose();
      cache.specRoundRollback(2);
      expect(cache.specRound).toBeNull();
      expect(cache.offset).toBe(7);

      const p = project(block, draft);
      const accepted = cutProjected(p, 0, 2);
      for (const array of [p.qkv, p.z, p.a, p.b, accepted.z]) array.dispose();
      reference.recurWindow(accepted.qkv, accepted.a, accepted.b, block.recurrence).dispose();
      const replayVsFresh = stateDiff(cache, reference);

      using next = input(1, 1, 92);
      using fromReplay = viaCache(block, next, cache, "decode");
      using fromFresh = viaCache(block, next, reference, "decode");
      const continuedVsFresh = { output: maxDiff(fromReplay, fromFresh), ...stateDiff(cache, reference) };

      report("spec rollback(2) replay vs fresh run over 2", replayVsFresh);
      report("spec decode after replay vs fresh", continuedVsFresh);
      for (const result of [replayVsFresh, continuedVsFresh]) zeros(result);
    } finally { for (const set of [cache, reference]) set.dispose(); }
  });

  test("unequal row prefixes (2 and 4 of 5) replay through the state-only kernel", () => {
    const keep = [2, 4];
    const cache = new BatchedSSMCache();
    const references = keep.map(() => new SSMCache());
    try {
      cache.preparePrefill({ lengths: [5, 5] });
      using prior = input(2, 5, 100);
      viaCache(block, prior, cache, "window").dispose();
      cache.finalizePrefill();

      using draft = input(2, 5, 101);
      cache.specRoundBegin();
      viaCache(block, draft, cache, "window").dispose();
      cache.specRoundRollback(keep);
      expect(cache.rowOffsets).toEqual([7, 9]);

      const priorProjected = project(block, prior), draftProjected = project(block, draft);
      const results: Record<string, number>[] = [];
      for (const [row, reference] of references.entries()) {
        const before = cutProjected(priorProjected, 0, 5, row, 1);
        const accepted = cutProjected(draftProjected, 0, keep[row]!, row, 1);
        before.z.dispose();
        accepted.z.dispose();
        reference.recurWindow(before.qkv, before.a, before.b, block.recurrence).dispose();
        reference.recurWindow(accepted.qkv, accepted.a, accepted.b, block.recurrence).dispose();
        const found = rowState(cache, row);
        try { results.push({ conv: maxDiff(found.conv, reference.conv), recurrent: maxDiff(found.recurrent, reference.recurrent) }); }
        finally { found.conv.dispose(); found.recurrent.dispose(); }
      }
      for (const array of [...Object.values(priorProjected), ...Object.values(draftProjected)]) array.dispose();
      const replayVsFresh = worst(results);

      report("spec rows rollback([2,4]) replay vs fresh one-row runs", replayVsFresh);
      zeros(replayVsFresh);
    } finally { for (const set of [cache, ...references]) set.dispose(); }
  });
});
