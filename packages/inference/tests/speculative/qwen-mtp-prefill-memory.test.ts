import { expect, test } from "bun:test";
import { MlxArray, gpuStream } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import { activeMemory, clearCache, Dtype, synchronize } from "@mlx-bun/mlx/ffi";
import type { Cache } from "../../src/contracts/mlx/cache";
import { QwenMtpRows } from "../../src/generation/speculative/bindings/qwen-mtp-rows";
import { atEveryWidth } from "../../src/models/graph";
import { mtpModule } from "../models/phase-forward";

type Module = ConstructorParameters<typeof QwenMtpRows>[1];

test("MTP prefill releases the prompt backing buffer and preserves its final hidden row", async () => {
  const length = 8193, width = 256;
  let pendingHidden: number[] = [];
  const target = {
    hiddenSize: width, layerCount: 1,
    embed(ids: MlxArray) { return ops.zeros([1, ids.shape[1]!, width], Dtype.float32); },
    logits: atEveryWidth(() => ops.zeros([1, 1, 8], Dtype.float32)),
  };
  // A small lazy KV projection lets this ownership test exercise the real
  // rows and cache without model weights or a full attention operation.
  const module = mtpModule((_embeds: MlxArray, hidden: MlxArray, cache: Cache) => {
    const count = hidden.shape[1]!;
    if (count === 1) pendingHidden = [...hidden.toFloat32Host()];
    using column = hidden.slice([0, 0, 0], [1, count, 1]);
    using kv = ops.reshape(column, [1, 1, count, 1]);
    for (const view of cache.updateAndFetch(kv, kv)) view.dispose();
    return ops.zeros([1, count, width], Dtype.float32);
  }) as unknown as Module;
  const liveBytes = () => { synchronize(gpuStream); clearCache(); return activeMemory(); };
  Bun.gc(true);
  await Bun.sleep(0);
  const baseline = liveBytes();
  const rows = new QwenMtpRows(target, module,
    { selection: { kind: "sampled" }, sample: () => MlxArray.fromInt32(new Int32Array([1]), [1]) }, [null]);
  const values = Float32Array.from({ length: length * width }, (_, i) => i % 1009);
  try {
    {
      using ids = ops.fromInt32(Array(length).fill(7), [1, length]);
      using context = MlxArray.fromFloat32(values, [1, length, width]);
      rows.prefill(ids, context);
    }
    // KV is at most 64 KiB and the pending row is 1 KiB. Neither the lazy
    // prefill graph nor that row may keep the 8 MiB prompt buffer alive.
    expect(liveBytes() - baseline).toBeLessThan(256 * 1024);
    rows.draft([7], 1, [0]);
    expect(pendingHidden).toEqual([...values.slice(-width)]);
  } finally { rows.dispose(); }
  expect(liveBytes() - baseline).toBeLessThan(4096);
});

test("a failed MTP sample releases its logits and in-flight module output", async () => {
  const width = 65536;
  const array = (length: number) =>
    MlxArray.fromFloat32(new Float32Array(length), [1, 1, length]);
  const target = {
    hiddenSize: width, layerCount: 1,
    embed() { return array(width); },
    logits: atEveryWidth(() => array(width)),
  };
  const module = mtpModule((_embeds: MlxArray, _hidden: MlxArray, cache: Cache) => {
    using kv = MlxArray.fromFloat32(new Float32Array([1]), [1, 1, 1, 1]);
    for (const view of cache.updateAndFetch(kv, kv)) view.dispose();
    return array(width);
  }) as unknown as Module;
  const liveBytes = () => { synchronize(gpuStream); clearCache(); return activeMemory(); };
  Bun.gc(true);
  await Bun.sleep(0);
  const baseline = liveBytes();
  const rows = new QwenMtpRows(target, module,
    { selection: { kind: "sampled" }, sample: () => { throw new Error("sample failed"); } }, [null]);
  try {
    {
      using ids = ops.fromInt32([7], [1, 1]);
      using context = array(width);
      rows.prefill(ids, context);
    }
    expect(() => rows.draft([7], 1, [0])).toThrow("sample failed");
  } finally { rows.dispose(); }
  // No GC after the failure: the rows must release these arrays themselves.
  expect(liveBytes() - baseline).toBeLessThan(4096);
});
