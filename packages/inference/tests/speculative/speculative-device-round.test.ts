import { expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { makeStepSampler } from "../../src/sampling/index";
import { advanceSpeculativeRows } from "../../src/generation/speculative/round";
import { createRuntimeConfig, withRuntimeConfig } from "../../src/runtime/config";

// Two rows, depth 2, a 16-token vocabulary. Logits favor `choices` with a clear
// margin but sampling stays stochastic; both orders must make the same draws.
const proposals = [[2, 3], [5, 6]];
const choices = [[2, 9, 4], [5, 6, 7]];
const logits = () => MlxArray.fromFloat32(Float32Array.from({ length: 2 * 3 * 16 }, (_, i) => {
  const row = Math.floor(i / 48), position = Math.floor(i / 16) % 3, token = i % 16;
  return (choices[row]![position] === token ? 3 : 0) + ((token * 5 + position) % 4) * 0.25;
}), [2, 3, 16]);

async function round(deviceFirst: boolean, options: Parameters<typeof makeStepSampler>[0], { walk = false } = {}) {
  const samplers = [0, 1].map(() => makeStepSampler(options, {
    tokenRepresentation: "number", grammarWait: "before-sample", historyUpdate: "after-sample", initialHistory: [7, 2],
  }));
  // The position-by-position walk: sample() at each position, committing each draw.
  if (walk) for (const sampling of samplers) Object.assign(sampling, { independent: undefined, positional: undefined });
  const calls = { draft: 0, draftDevice: 0, resolved: [] as number[][], forwardIds: [] as number[] };
  const run = () => advanceSpeculativeRows(samplers.map((sampling, row) => ({
    pending: row + 1, step: row * 10 + 2, remaining: 20, eosTokenIds: [], sampling,
  })), 2, {
    draft() { calls.draft++; return proposals; },
    draftDevice() {
      calls.draftDevice++;
      return { tokens: ops.fromInt32(proposals.flat(), [2, 2]).astype(Dtype.uint32),
        resolve(read) { calls.resolved.push([...read]); return proposals; } };
    },
    commit() {},
  }, {
    transaction: { canBegin: () => true, begin() {}, resolve() {} },
    async forward(ids) { calls.forwardIds = ids.toIntTokens(); return { logits: logits(), context: ops.zeros([2, 3, 2], Dtype.float32) }; },
  });
  try {
    const result = await withRuntimeConfig(createRuntimeConfig({ MLX_BUN_SPEC_DEVICE_ROUND: deviceFirst ? "1" : "0" }), run);
    // The draw each row makes next over flat scores, which its processor history decides.
    const next: number[] = [];
    for (const sampling of samplers) {
      using flat = MlxArray.fromFloat32(Float32Array.from({ length: 16 }, (_, token) => (token % 3) * 0.01), [1, 16]);
      next.push((await sampling.sample(flat, 40)).token);
    }
    return { calls, next, emitted: result.map(row => row.acceptance.emitted), accepted: result.map(row => row.acceptance.accepted),
      correction: result.map(row => row.acceptance.correction) };
  } finally { for (const sampler of samplers) sampler.dispose(); }
}

for (const [name, options] of [
  ["sampled", { temperature: 0.8, topP: 0.95, topK: 6, seed: 23 }],
  ["greedy", { temperature: 0 }],
] as const) {
  test(`device-first round makes the same decisions as the host-first round (${name})`, async () => {
    const host = await round(false, options), device = await round(true, options);
    expect(host.calls).toMatchObject({ draft: 1, draftDevice: 0 });
    expect(device.calls).toMatchObject({ draft: 0, draftDevice: 1 });
    expect(device.calls.resolved).toEqual([proposals.flat()]);
    // The verify window is [pending, ...drafts] per row in both orders.
    expect(device.calls.forwardIds).toEqual([1, 2, 3, 2, 5, 6]);
    expect(host.calls.forwardIds).toEqual(device.calls.forwardIds);
    expect(device.emitted).toEqual(host.emitted);
    expect(device.accepted).toEqual(host.accepted);
    expect(device.correction).toEqual(host.correction);
  });
}

for (const [name, options] of [
  ["Qwen's instruct settings (top-k within the 16-token vocabulary)", { temperature: 0.7, topP: 0.8, topK: 6, presencePenalty: 1.5, seed: 7 }],
  ["repetition penalty", { temperature: 0.8, seed: 3, repetitionPenalty: 1.2 }],
] as const) test(`rows whose sampling reads processor history verify device-first with the draws of the position walk (${name})`, async () => {
  const device = await round(true, options), host = await round(false, options);
  expect(device.calls).toMatchObject({ draft: 0, draftDevice: 1 });
  expect(device.emitted).toEqual(host.emitted);
  expect(device.accepted).toEqual(host.accepted);
  expect(device.correction).toEqual(host.correction);
  const walked = await round(false, options, { walk: true });
  expect(walked.emitted).toEqual(host.emitted);
  expect(walked.correction).toEqual(host.correction);
  // Each walk commits the tokens it used: the next draw reads the same history.
  expect(device.next).toEqual(walked.next);
  expect(host.next).toEqual(walked.next);
});
