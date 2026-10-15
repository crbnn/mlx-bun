import { expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { QwenMtpRows } from "../../src/generation/speculative/bindings/qwen-mtp-rows";
import type { MtpModule } from "../../src/models/qwen/mtp";
import type { Cache } from "../../src/contracts/mlx/cache";
import { rowTable } from "../../src/models/graph";
import { mtpModule } from "../models/phase-forward";
import { toLogprobs } from "../../src/sampling/index";
import { createRuntimeConfig, withRuntimeConfig } from "../../src/runtime/config";
import { kvStorage } from "../../src/state/kv-storage";
import { draftSamplerOptions } from "../../src/generation/speculative/draft-policy";
import { COMPOSITION_DEFAULTS } from "../../src/contracts/portable/composition";

const V = 6;
/** Row r at chained step s favors token (r * 2 + s) % V by a clear margin. */
function fixture(sampling: ConstructorParameters<typeof QwenMtpRows>[2]) {
  let step = 0;
  const target = { hiddenSize: 1, layerCount: 1,
    embed(ids: MlxArray) {
      using floats = ids.astype(Dtype.float32);
      return ops.reshape(floats, [...ids.shape, 1]);
    },
    logits: rowTable((hidden: MlxArray): MlxArray => {
      const B = hidden.shape[0]!;
      const values = Float32Array.from({ length: B * V }, (_, i) => {
        const row = Math.floor(i / V), token = i % V;
        return (token === (row * 2 + step) % V ? 4 : 0) + ((token * 3 + row) % 4) * 0.1;
      });
      step++;
      return MlxArray.fromFloat32(values, [B, 1, V]);
    }),
  };
  const module = mtpModule((embeds: MlxArray, hidden: MlxArray, cache: Cache) => {
    const [B, N] = embeds.shape;
    using k = ops.reshape(embeds, [B!, 1, N!, 1]);
    using v = ops.reshape(hidden, [B!, 1, N!, 1]);
    for (const view of kvStorage(cache).updateAndFetch(k, v)) view.dispose();
    return ops.copyOf(hidden);
  }) as unknown as MtpModule;
  const rows = new QwenMtpRows(target, module, sampling, []);
  rows.append([null, null]);
  using ids = ops.fromInt32([1, 2, 3, 11, 12, 13], [2, 3]);
  using context = MlxArray.fromFloat32(Float32Array.from([10, 20, 30, 110, 120, 130]), [2, 3, 1]);
  rows.prefill(ids, context);
  return rows;
}

test("greedy draft rows select the fused argmax without calling the request sampler", () => {
  let sampled = 0;
  const greedy = fixture({ selection: { kind: "greedy" }, sample() { sampled++; throw new Error("greedy rows must not sample"); } });
  const sampledRows = fixture({ selection: { kind: "sampled" }, sample(logprobs) { sampled++; return ops.argmaxAxis(logprobs, -1); } });
  try {
    const expected = [[0, 1, 2], [2, 3, 4]];
    expect(greedy.draft([4, 14], 3, [0, 10])).toEqual(expected);
    expect(sampled).toBe(0);
    expect(sampledRows.draft([4, 14], 3, [0, 10])).toEqual(expected);
    expect(sampled).toBe(3);
  } finally { greedy.dispose(); sampledRows.dispose(); }
});

/** H = 64 hidden lanes, V = 8 tokens: token v reads lanes j with j % 8 = v. The
 *  module passes its hidden through, so every position drafts from the prompt's
 *  last hidden row: row 0 favors token 3, row 1 token 2 (and token 5 second). */
function listedFixture(sampling: ConstructorParameters<typeof QwenMtpRows>[2], vocabularyHead = true) {
  const H = 64, VL = 8;
  const dense = Float32Array.from({ length: VL * H }, (_, i) => i % H % VL === Math.floor(i / H) ? 1 : 0);
  using weights = MlxArray.fromFloat32(dense, [VL, H]);
  const q = ops.quantize(weights, 64, 4);
  using transposed = ops.transposeAxes(weights, [1, 0]);
  const projection = ops.copyOf(transposed);
  const target = { hiddenSize: H, layerCount: 1,
    embed(ids: MlxArray) { return ops.zeros([...ids.shape, H], Dtype.float32); },
    logits: rowTable((hidden: MlxArray): MlxArray => ops.matmul(hidden, projection)),
    ...(vocabularyHead ? { vocabularyHead: { w: q.packed, scales: q.scales, biases: q.biases,
      spec: { bits: 4, groupSize: 64, mode: "affine" }, vocabSize: VL } } : {}),
  };
  const module = mtpModule((embeds: MlxArray, hidden: MlxArray, cache: Cache) => {
    const [B, N] = embeds.shape;
    using k = ops.reshape(embeds, [B!, 1, N!, H]);
    using v = ops.reshape(hidden, [B!, 1, N!, H]);
    for (const view of kvStorage(cache).updateAndFetch(k, v)) view.dispose();
    return ops.copyOf(hidden);
  }) as unknown as MtpModule;
  const release = () => { projection.dispose(); q.packed.dispose(); q.scales.dispose(); q.biases.dispose(); };
  let rows: QwenMtpRows;
  try { rows = new QwenMtpRows(target, module, sampling, []); }
  catch (error) { release(); throw error; }
  rows.append([null, null]);
  const lane = (row: number, j: number) => row === 0 ? Number(j % VL === 3) : j % VL === 2 ? 1 : j % VL === 5 ? 0.5 : 0;
  using ids = ops.fromInt32([1, 2, 3, 11, 12, 13], [2, 3]);
  using context = MlxArray.fromFloat32(Float32Array.from({ length: 2 * 3 * H }, (_, i) =>
    Math.floor(i / H) % 3 === 2 ? lane(Math.floor(i / (3 * H)), i % H) : 0), [2, 3, H]);
  rows.prefill(ids, context);
  return { rows, release };
}

test("a listed draft vocabulary binds its projection once and draws over the listed rows only", () => {
  const ids = [1, 3, 5];
  let drawn = 0, sampled = 0;
  const listed = listedFixture({ sample() { sampled++; throw new Error("listed rows draw over the subset"); }, selection: { kind: "listed", ids,
    sample(logits, head) { drawn++; using index = ops.argmaxAxis(logits, -1); return ops.takeAxis(head.ids, index, 0); } } });
  const full = listedFixture({ selection: { kind: "sampled" }, sample(logprobs) { return ops.argmaxAxis(logprobs, -1); } });
  try {
    // Token 2 is not listed: the listed draw takes token 5, the full head token 2.
    expect(listed.rows.draft([4, 14], 2, [0, 10])).toEqual([[3, 3], [5, 5]]);
    expect([drawn, sampled]).toEqual([2, 0]);
    expect(full.rows.draft([4, 14], 2, [0, 10])).toEqual([[3, 3], [2, 2]]);
  } finally { listed.rows.dispose(); listed.release(); full.rows.dispose(); full.release(); }
  // The projection is the target's: a target without a quantized head refuses
  // the listed selection when the rows open, before any draft.
  expect(() => listedFixture({ sample() { throw new Error("unused"); }, selection: { kind: "listed", ids,
    sample() { throw new Error("unused"); } } }, false)).toThrow("needs the target's quantized vocabulary head");
});

test("a preview proposes the head's first draft under a round of its cache that rolls back, leaving the rows' state", () => {
  // The module reads its whole cache: an appended row the preview left would move every later draft.
  const H = 64, VL = 8;
  const make = () => {
    const dense = Float32Array.from({ length: VL * H }, (_, i) => Math.sin(i * 0.7) * 0.5);
    using weights = MlxArray.fromFloat32(dense, [VL, H]);
    using transposed = ops.transposeAxes(weights, [1, 0]);
    const projection = ops.copyOf(transposed);
    const target = { hiddenSize: H, layerCount: 1,
      embed(ids: MlxArray) {
        using floats = ids.astype(Dtype.float32);
        using column = ops.reshape(floats, [...ids.shape, 1]);
        return ops.broadcastTo(column, [...ids.shape, H]);
      },
      logits: rowTable((hidden: MlxArray): MlxArray => ops.matmul(hidden, projection)),
    };
    const module = mtpModule((embeds: MlxArray, hidden: MlxArray, cache: Cache) => {
      const [B, N] = embeds.shape;
      using k = ops.reshape(embeds, [B!, 1, N!, H]);
      using v = ops.reshape(hidden, [B!, 1, N!, H]);
      const [keys, values] = kvStorage(cache).updateAndFetch(k, v);
      using total = ops.sumAxis(keys!, 2, true); keys!.dispose(); values!.dispose();
      using scaled = ops.mulScalar(total, 0.01);
      using mixed = ops.reshape(scaled, [B!, N! === 1 ? 1 : 1, H]);
      return ops.add(hidden, mixed);
    }) as unknown as MtpModule;
    const rows = new QwenMtpRows(target, module, { selection: { kind: "greedy" }, sample() { throw new Error("greedy"); } }, []);
    rows.append([null, null]);
    using ids = ops.fromInt32([1, 2, 3, 4, 5, 6], [2, 3]);
    using context = MlxArray.fromFloat32(Float32Array.from({ length: 2 * 3 * H }, (_, i) => Math.cos(i * 0.31)), [2, 3, H]);
    rows.prefill(ids, context);
    return { rows, release: () => projection.dispose() };
  };
  const previewed = make(), control = make();
  try {
    const before = previewed.rows.extractRow(1);
    using tokens = previewed.rows.preview([4, 7], 3, [0, 10])!;
    const after = previewed.rows.extractRow(1);
    try {
      expect(after.cache.offset).toBe(before.cache.offset);
      expect(tokens.shape).toEqual([2, 1]);
      const drafted = previewed.rows.draft([4, 7], 3, [0, 10]);
      expect(tokens.toIntTokens()).toEqual(drafted.map(row => row[0]!));
      // The same draft as rows that never previewed, and the next round's too.
      expect(drafted).toEqual(control.rows.draft([4, 7], 3, [0, 10]));
      using context = MlxArray.fromFloat32(Float32Array.from({ length: 2 * 4 * H }, (_, i) => Math.sin(i * 0.13)), [2, 4, H]);
      previewed.rows.commit([1, 2], context); control.rows.commit([1, 2], context);
      expect(previewed.rows.draft([5, 6], 2, [4, 14])).toEqual(control.rows.draft([5, 6], 2, [4, 14]));
    } finally { for (const state of [before, after]) { state.cache.dispose(); state.hidden.dispose(); } }
    expect(previewed.rows.preview([1, 1], 0, [0, 0])).toBeNull();
  } finally { for (const fixture of [previewed, control]) { fixture.rows.dispose(); fixture.release(); } }
});

test("greedy drafting is the composition's opt-in field: it argmaxes the draft sampler, and the runtime configuration does not reach it", () => {
  expect(COMPOSITION_DEFAULTS.greedyDraft).toBe(false);
  const options = { temperature: 0.7, topP: 0.9 };
  expect(draftSamplerOptions(options, true)).toEqual({ temperature: 0, topP: 0.9 });
  expect(withRuntimeConfig(createRuntimeConfig({ MLX_BUN_SPEC_GREEDY_DRAFT: "1" }), () => draftSamplerOptions(options, false))).toBe(options);
});

test("normalized argmax and sampled argmax agree on the draft logits", () => {
  using logits = MlxArray.fromFloat32(Float32Array.from([0.1, 4.0, 0.2, 0.3, 0.1, 0.0, 3.9, 0.2, 4.1, 0.3, 0.1, 0.0]), [2, V]);
  using logprobs = toLogprobs(logits);
  using viaLogprobs = ops.argmaxAxis(logprobs, -1);
  expect(viaLogprobs.toIntTokens()).toEqual([1, 2]);
});
