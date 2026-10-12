// A cache drives the compiled decode step through its own protocol: prepare the
// step, declare a slot, feed the closure inputs, place a stand-in in the traced
// graph, then apply the closure's outputs. Run eagerly (no compile), the stand-in
// must fetch exactly what the real cache's append fetches and leave the cache in
// exactly the state the real append leaves, in both fetch phases (growing concat,
// then the rotating ring at steady state). Every plane is compared byte for byte
// against a twin cache that took the ordinary append.
import { unfusedAffineKernels } from "../../src/state/affine-attention";
import { expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import type { Cache, CompiledDecodeCache, DecodeSlot } from "../../src/contracts/mlx/cache";
import type { KvStorageScheme } from "../../src/contracts/portable/composition";
import { CompiledFunction } from "@mlx-bun/mlx/compile";
import { setCompiledTrace } from "../../src/runtime/compiled-trace";
import { composedAffineKernels } from "../../src/state/affine-attention";
import { Bf16FirstQuantizedKVCache, Bf16FirstRotatingQuantizedKVCache, Bf16FirstTurboQuantKVCache } from "../../src/state/bf16-first-kv";
import { PagedKVCache } from "../../src/state/paged/cache";
import { createKvMaintenance } from "../../src/state/kv-maintenance";
import { decodeStepInputs } from "../../src/state/decode-step-inputs";
import { BatchedKVCache } from "../../src/state/batched-kv";
import { KVCache } from "../../src/state/kv";
import { deferredTurboQuantKernels, denseTurboQuantKernels } from "../../src/state/turboquant-attention";
import { QuantizedKVCache } from "../../src/state/quantized-kv";
import { RotatingKVCache } from "../../src/state/rotating-kv";
import { RotatingQuantizedKVCache } from "../../src/state/rotating-quantized-kv";
import { TurboQuantKVCache } from "../../src/state/turboquant-kv";
import { kvStorage } from "../../src/state/kv-storage";

const W = 8, H = 2, D = 64, GROUP = 32, BITS = 4;
type Kind = { name: string; make: () => CompiledDecodeCache & Cache; quantized: boolean; ring: boolean };
const KINDS: Kind[] = [
  { name: "KVCache", make: () => new KVCache(), quantized: false, ring: false },
  { name: "QuantizedKVCache", make: () => new QuantizedKVCache(GROUP, BITS, unfusedAffineKernels(BITS, GROUP, Dtype.bfloat16)), quantized: true, ring: false },
  { name: "RotatingKVCache", make: () => new RotatingKVCache(W), quantized: false, ring: true },
  { name: "RotatingQuantizedKVCache", make: () => new RotatingQuantizedKVCache(W, GROUP, BITS, unfusedAffineKernels(BITS, GROUP, Dtype.bfloat16)), quantized: true, ring: true },
];

/** [1, H, n, D] rows whose values depend on the position tag, head and lane. */
function rows(from: number, n: number, sign: 1 | -1): MlxArray {
  const data = new Float32Array(H * n * D);
  for (let h = 0; h < H; h++) for (let i = 0; i < n; i++) for (let d = 0; d < D; d++)
    data[(h * n + i) * D + d] = sign * ((from + i) * 0.37 + h * 0.25 + d / 128);
  const wide = MlxArray.fromFloat32(data, [1, H, n, D]);
  try { return wide.astype(Dtype.bfloat16); } finally { wide.dispose(); }
}

const bytes = (a: MlxArray): Uint8Array => {
  const c = ops.contiguous(a);
  try { return Uint8Array.from(c.rawBytes()); } finally { c.dispose(); }
};
const same = (a: MlxArray[], b: MlxArray[]) => {
  expect(a.length).toBe(b.length);
  a.forEach((x, i) => { expect(x.shape).toEqual(b[i]!.shape); expect(Buffer.compare(bytes(x), bytes(b[i]!))).toBe(0); });
};
const planes = (t: { packed: MlxArray; scales: MlxArray; biases: MlxArray }) => [t.packed, t.scales, t.biases];

/** The ordinary append; returns the attended tensors. */
function append(cache: Cache, quantized: boolean, k: MlxArray, v: MlxArray): MlxArray[] {
  if (quantized) {
    const [kq, vq] = (cache as QuantizedKVCache).updateAndFetchQuantized(k, v);
    return [...planes(kq), ...planes(vq)];
  }
  return kvStorage(cache).updateAndFetch(k, v).slice();
}

for (const kind of KINDS) {
  for (const prefill of [3, 12]) {
    test(`${kind.name} after a ${prefill}-token prefill: the stand-in fetches and commits what the ordinary append does`, () => {
      const real = kind.make(), driven = kind.make();
      const phases = new Set<string>();
      try {
        const kp = rows(0, prefill, 1), vp = rows(0, prefill, -1);
        for (const cache of [real, driven])
          for (const a of append(cache, kind.quantized, kp, vp)) a.dispose();
        kp.dispose(); vp.dispose();

        for (let step = 0; step < 14; step++) {
          const at = prefill + step;
          const k = rows(at, 1, 1), v = rows(at, 1, -1);
          const expected = append(real, kind.quantized, k, v);

          const phase = driven.decodePhase();
          const plan = driven.prepareDecodeStep();
          expect(plan.fetch).toBe(phase);
          phases.add(plan.fetch);
          const slot = driven.decodeSlot(plan);
          expect(slot.fetch).toBe(plan.fetch);
          const temps: MlxArray[] = [];
          const inputs = driven.decodeInputs(plan, decodeStepInputs(temps));
          expect(inputs.length).toBe(slot.inputs);
          const ropeOffset = ops.fromInt32([at], []);
          const trace = slot.trace([...inputs], ropeOffset);
          const fetched = append(trace, kind.quantized, k, v);
          expect(trace.outs.length).toBe(slot.outputs);
          same(fetched, expected);
          const roots = driven.commitDecodeStep(slot, trace.outs);
          for (const t of [...temps, ropeOffset, ...fetched]) t.dispose();

          expect(driven.offset).toBe(real.offset);
          same(driven.state(), real.state());
          expect(roots.length).toBe(slot.fetch === "concat" ? driven.state().length : 0);
          if (kind.ring) expect((driven as RotatingKVCache).ringIdx).toBe((real as RotatingKVCache).ringIdx);
          for (const a of [k, v, ...expected]) a.dispose();
        }
        // a ring past its window at the prefill only ever runs the ring fetch
        expect([...phases].sort()).toEqual(kind.ring && prefill >= W ? ["ring"] : kind.ring ? ["concat", "ring"] : ["concat"]);
      } finally { real.dispose(); driven.dispose(); }
    });
  }
}

/** The protocol's members, as the compiled step calls them. */
const drivesCompiledDecode = (cache: Cache) => (["prepareDecodeStep", "decodePhase", "decodeSlot", "decodeInputs",
  "commitDecodeStep"] as const).every(name => typeof (cache as Partial<CompiledDecodeCache>)[name] === "function");

// ---------------------------------------------------------------------------
// Every lego the composition can name, replayed through recorded graphs.
//
// A lego is the cache a composition's graph holds for one layer while a lone
// row decodes: the graph's composed lego (bf16, affine and TurboQuant, bf16
// first, converting at the first append or at a later start), its storage after
// the scheduler's batch preparation (`prepareBatch`: the storage itself, or
// the one-row delayed layout of a later start), and the paged lego the request
// state pages a full-attention layer with. Each is driven twice from the same
// prompt: through its ordinary decode read, and through the compiled protocol
// replayed by one shapeless recorded graph per closure key (a decode read and
// its attention over a query), as the compiled step replays it. Every step's
// output is compared byte for byte; after the steps, both continue with
// ordinary decodes, which must also agree (the compiled steps left the same
// state).

const LW = 8, LH = 2, LD = 64, LHQ = 4, SCALE = 0.125;
function lrows(from: number, n: number, sign: number, heads = LH): MlxArray {
  const data = new Float32Array(heads * n * LD);
  for (let h = 0; h < heads; h++) for (let i = 0; i < n; i++) for (let d = 0; d < LD; d++)
    data[(h * n + i) * LD + d] = sign * Math.sin((from + i) * 0.37 + h * 0.25 + d / 17);
  const wide = MlxArray.fromFloat32(data, [1, heads, n, LD]);
  try { return wide.astype(Dtype.bfloat16); } finally { wide.dispose(); }
}
type Read = { attend(q: MlxArray, scale: number): MlxArray; dispose(): void };
type Decoding = Cache & { appendDecode(k: MlxArray, v: MlxArray): Read; appendWindow(k: MlxArray, v: MlxArray): Read };
interface Lego { readonly name: string; make(): Cache; readonly scheme?: KvStorageScheme }
const unfused = composedAffineKernels("unfused");
const affine = (bits: number, start: number): KvStorageScheme => ({ kind: "affine", bits, groupSize: 64, kernels: "unfused", quantizedKvStart: start });
const turbo = (start: number): KvStorageScheme => ({ kind: "turbo", kBits: 8, vBits: 3, quantizedKvStart: start });
const LEGOS: Lego[] = [
  { name: "bf16 KVCache", make: () => new KVCache() },
  { name: "bf16 RotatingKVCache", make: () => new RotatingKVCache(LW) },
  { name: "affine QuantizedKVCache", make: () => new QuantizedKVCache(64, 4, unfused(4, 64, Dtype.bfloat16)) },
  { name: "affine RotatingQuantizedKVCache", make: () => new RotatingQuantizedKVCache(LW, 64, 8, unfused(8, 64, Dtype.bfloat16)) },
  { name: "TurboQuant 8/3, deferred read", make: () => new TurboQuantKVCache(8, 3, deferredTurboQuantKernels, true) },
  { name: "TurboQuant 8/3, dense read", make: () => new TurboQuantKVCache(8, 3, denseTurboQuantKernels, true) },
  { name: "TurboQuant 4/2, deferred read, eager codec", make: () => new TurboQuantKVCache(4, 2, deferredTurboQuantKernels, false) },
  { name: "bf16-first affine 4, converting at the first append", make: () => new Bf16FirstQuantizedKVCache(64, 4, 0, unfused) },
  { name: "bf16-first affine 4 from 12", make: () => new Bf16FirstQuantizedKVCache(64, 4, 12, unfused) },
  { name: "bf16-first rotating affine 8 from 12", make: () => new Bf16FirstRotatingQuantizedKVCache(LW, 64, 8, 12, unfused) },
  { name: "bf16-first TurboQuant 8/3 from 12", make: () => new Bf16FirstTurboQuantKVCache(8, 3, 12, true, deferredTurboQuantKernels) },
  { name: "bf16-first TurboQuant 8/3, converting at the first append", make: () => new Bf16FirstTurboQuantKVCache(8, 3, 0, true, deferredTurboQuantKernels) },
  { name: "delayed affine 4 from 12 (one row)", scheme: affine(4, 12), make: () => new Bf16FirstQuantizedKVCache(64, 4, 12, unfused) },
  { name: "delayed rotating affine 8 from 12 (one row)", scheme: affine(8, 12), make: () => new Bf16FirstRotatingQuantizedKVCache(LW, 64, 8, 12, unfused) },
  { name: "delayed TurboQuant 8/3 from 12 (one row)", scheme: turbo(12), make: () => new Bf16FirstTurboQuantKVCache(8, 3, 12, true, deferredTurboQuantKernels) },
  { name: "paged bf16", make: () => new PagedKVCache(64, 4, false) },
  { name: "paged affine 4", make: () => new PagedKVCache(64, 4, false, { bits: 4, groupSize: 64 }) },
  { name: "paged affine 8", make: () => new PagedKVCache(64, 4, false, { bits: 8, groupSize: 64 }) },
];

/** One recorded graph per closure key: [ropeOffset, k, v, ...slot inputs, q] -> [output, ...slot outputs]. */
function recordedRead(slot: DecodeSlot): CompiledFunction {
  return new CompiledFunction((inputs: MlxArray[]) => {
    setCompiledTrace(true);
    try {
      const trace = slot.trace(inputs.slice(3, 3 + slot.inputs), inputs[0]!) as unknown as Decoding & { outs: MlxArray[] };
      const read = trace.appendDecode(inputs[1]!, inputs[2]!);
      try { return [read.attend(inputs[3 + slot.inputs]!, SCALE), ...trace.outs]; } finally { read.dispose(); }
    } finally { setCompiledTrace(false); }
  }, true);
}

/** `cache` after the scheduler's batch preparation for a lone row of `scheme` (none: the cache itself). */
function prepared(cache: Cache, scheme: KvStorageScheme | undefined): Cache {
  if (!scheme) return cache;
  const one = [cache];
  createKvMaintenance(scheme).prepareBatch?.(one);
  return one[0]!;
}

for (const lego of LEGOS) {
  test(`${lego.name}: replayed compiled decode steps read and leave what ordinary decodes do`, () => {
    let real = lego.make(), driven = lego.make();
    const closures = new Map<string, CompiledFunction>();
    try {
      using kp = lrows(0, 6, 1), vp = lrows(0, 6, -1);
      for (const cache of [real, driven]) (cache as Decoding).appendWindow(kp, vp).dispose();
      real = prepared(real, lego.scheme); driven = prepared(driven, lego.scheme);
      expect(drivesCompiledDecode(driven)).toBe(true);
      for (let step = 0; step < 18; step++) {
        const at = 6 + step;
        using k = lrows(at, 1, 1), v = lrows(at, 1, -1), q = lrows(at, 1, 0.5, LHQ);
        const read = (real as Decoding).appendDecode(k, v);
        using expected = read.attend(q, SCALE);
        read.dispose();
        const phase = (driven as CompiledDecodeCache).decodePhase();
        const plan = (driven as CompiledDecodeCache).prepareDecodeStep();
        expect(plan.fetch).toBe(phase);
        const slot = (driven as CompiledDecodeCache).decodeSlot(plan);
        const temps: MlxArray[] = [];
        const inputs = (driven as CompiledDecodeCache).decodeInputs(plan, decodeStepInputs(temps));
        let closure = closures.get(slot.key);
        if (!closure) { closure = recordedRead(slot); closures.set(slot.key, closure); }
        using rope = ops.fromInt32([at], []);
        const outs = closure.apply([rope, k, v, ...inputs, q]);
        using actual = outs[0]!;
        const roots = (driven as CompiledDecodeCache).commitDecodeStep(slot, outs.slice(1));
        ops.evalAll([actual, expected, ...roots]);
        for (const temp of temps) temp.dispose();
        expect(Buffer.compare(bytes(actual), bytes(expected)), `${lego.name} step ${step} (${slot.key})`).toBe(0);
        expect(driven.offset).toBe(real.offset);
      }
      for (const [key, closure] of closures) expect(closure.traceCount, key).toBe(1);
      // Ordinary decodes after the compiled steps (a lone row's layout takes its row back).
      for (let step = 0; step < 4; step++) {
        const at = 24 + step;
        using k = lrows(at, 1, 1), v = lrows(at, 1, -1), q = lrows(at, 1, 0.5, LHQ);
        const a = (real as Decoding).appendDecode(k, v), b = (driven as Decoding).appendDecode(k, v);
        using x = a.attend(q, SCALE), y = b.attend(q, SCALE);
        a.dispose(); b.dispose();
        expect(Buffer.compare(bytes(x), bytes(y)), `${lego.name} ordinary decode ${step}`).toBe(0);
      }
    } finally {
      real.dispose(); driven.dispose();
      for (const closure of closures.values()) closure.dispose();
    }
  });
}

test("the compiled step drives every lego; batched layouts of several rows are not lone rows", () => {
  const caches = LEGOS.map(lego => prepared(lego.make(), lego.scheme));
  try {
    for (const cache of caches) expect(drivesCompiledDecode(cache), cache.signature()).toBe(true);
    expect(drivesCompiledDecode(new BatchedKVCache())).toBe(false);
  } finally { for (const cache of caches) cache.dispose(); }
});

test("a lone row's layout answers the coming fetch from its own positions while ordinary decodes drive it", () => {
  // The segmented compiled step asks every layer's fetch before choosing which
  // layers it records; a layer whose fetch is the concat runs its ordinary read.
  for (const lego of LEGOS.filter(lego => lego.scheme)) {
    const twin = lego.make(), layout0 = lego.make();
    try {
      using kp = lrows(0, 3, 1), vp = lrows(0, 3, -1);
      for (const cache of [twin, layout0]) (cache as Decoding).appendWindow(kp, vp).dispose();
      const layout = prepared(layout0, lego.scheme);
      try {
        for (let step = 0; step < 14; step++) {
          expect((layout as CompiledDecodeCache).decodePhase(), `${lego.name} at ${layout.offset}`)
            .toBe((twin as CompiledDecodeCache).decodePhase());
          using k = lrows(3 + step, 1, 1), v = lrows(3 + step, 1, -1);
          (twin as Decoding).appendDecode(k, v).dispose();
          (layout as Decoding).appendDecode(k, v).dispose();
        }
      } finally { layout.dispose(); }
    } finally { twin.dispose(); }
  }
});
