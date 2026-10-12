import * as ops from "@mlx-bun/mlx/ops";
import { expect, test } from "bun:test";
import type { TargetView } from "../../src/generation/speculative/source";
import { projectedDraftGroups } from "../../src/generation/speculative/bindings/projected-draft-rows";
import { deepspecGroups } from "../../src/generation/speculative/bindings/deepspec-rows";
import { MlxArray } from "@mlx-bun/mlx/array";
import { bindSpeculativeTargetModel } from "../../src/generation/speculative/bindings/binding";
import { declareGraph } from "../../src/models/capabilities";
import { atEveryWidth } from "../../src/models/graph";
import { stubPhases } from "../models/phase-forward";
import { assistantGroups } from "../../src/generation/speculative/bindings/assistant-rows";
import { DRAFT_CHECKPOINT_SCHEMA } from "../../src/contracts/mlx/draft-checkpoint";
import type { AssistantDraftStep } from "../../src/contracts/mlx/drafter";

/** Assistant draft rows holding one row whose true hidden is `hidden`, over `target`. */
function assistantRows(forward: AssistantDraftStep, target: TargetView, hidden: MlxArray) {
  const checkpoint = { processedTokens: 4, attachment: { schema: DRAFT_CHECKPOINT_SCHEMA.assistant, metadata: { processedTokens: 4 }, tensors: [hidden] } };
  return assistantGroups({ forwardRows: atEveryWidth(forward), dispose() {} }, "assistant").open({ target, checkpoints: [checkpoint],
    sampling: { selection: { kind: "sampled" }, sample() { throw new Error("the assistant drafts with its own head"); } } });
}

test("assistant rows use an independent target's ports and release each borrowed view", async () => {
  const released: number[] = [];
  const tensor = (id: number) => ({ dispose() { released.push(id); }, [Symbol.dispose]() { released.push(id); } }) as unknown as MlxArray;
  const positions: number[] = [];
  let step = 0;
  using hidden = MlxArray.fromFloat32(new Float32Array([0.5]), [1, 1, 1]);
  const rows = assistantRows((_embedding, _hidden, _donors, position) => {
    positions.push(position as number);
    const token = 4 + step;
    return { tokens: ops.fromInt32([token], [1]), nextHidden: tensor(20 + step++) };
  }, {
    identity: {},
    assistantRows: {
      hiddenSize: 1, embed: (ids) => tensor(ops.itemUint32(ids)),
      readDonors: () => ({ positions: [9], sliding: {} as never, full: {} as never, dispose() { released.push(10, 11, 12, 13); } }),
    },
  }, hidden);
  try {
    expect(await rows.draft([3], 2, [0])).toEqual([[4, 5]]);
    expect(positions).toEqual([9, 10]);
    expect(released.sort((a, b) => a - b)).toEqual([3, 4, 10, 11, 12, 13, 20, 21]);
  } finally { rows.dispose(); }
});

test("an assistant draft failure releases retained donors and its current embedding", async () => {
  const released: number[] = [];
  const tensor = (id: number) => ({ dispose() { released.push(id); }, [Symbol.dispose]() { released.push(id); } }) as unknown as MlxArray;
  using hidden = MlxArray.fromFloat32(new Float32Array([0.5]), [1, 1, 1]);
  const rows = assistantRows(() => { throw new Error("draft failed"); }, {
    identity: {}, assistantRows: {
      hiddenSize: 1, embed: () => tensor(1),
      readDonors: () => ({ positions: [0], sliding: {} as never, full: {} as never, dispose() { released.push(2, 3, 4, 5); } }),
    },
  }, hidden);
  try {
    expect(() => rows.draft([1], 1, [0])).toThrow("draft failed");
    expect(released).toEqual([1, 2, 3, 4, 5]);
  } finally { rows.dispose(); }
});

test("grouped rows refuse a target that lacks their port before touching the drafter", () => {
  const target: TargetView = { identity: {} };
  for (const open of [(groups: ReturnType<typeof assistantGroups>) => groups.openPrefill({ target, checkpoints: [] }),
    (groups: ReturnType<typeof assistantGroups>) => groups.open({ target, checkpoints: [], sampling: {} as never })]) {
    expect(() => open(assistantGroups(undefined as never, "assistant"))).toThrow("target graph does not provide assistantRows");
    expect(() => open(deepspecGroups({ tapLayers: [1], cfg: { num_target_layers: 2 } } as never, "deepspec")))
      .toThrow("target graph does not provide hiddenLayerTaps");
  }
});

test("projected draft providers declare the immutable target tap list their rows consume", () => {
  const layers = [3, 8, 12];
  const grouped = projectedDraftGroups("fixture", layers, () => ({ namespace: "fixture", schema: "fixture-v1", layers: 1,
    project: () => { throw new Error("unused"); }, draft: () => [] }));
  layers.push(99); // declared at construction, not read through later
  const target = { identity: {} } as TargetView, declared = grouped.targetTapLayers!(target);
  expect(declared).toEqual([3, 8, 12]);
  expect(Object.isFrozen(declared)).toBe(true);
  for (const open of [grouped.open, grouped.openPrefill]) {
    const rows = open({ target, checkpoints: [], sampling: {} as never });
    try { expect(rows.tapLayers).toBe(declared); } finally { rows.dispose(); }
  }
  // DSpark-style context providers declare their drafter's own list.
  const deepspec = deepspecGroups({ tapLayers: [1, 4], cfg: { num_hidden_layers: 1 } } as never, "deepspec");
  expect(deepspec.targetTapLayers!(target)).toEqual([1, 4]);
});

test("the target binding taps its graph's layers and supplies an uncaptured post-norm sentinel from the forward output", async () => {
  const H = 2, L = 3, created: MlxArray[] = [];
  const filled = (value: number) => { const array = MlxArray.fromFloat32(new Float32Array(L * H).fill(value), [1, L, H]); created.push(array); return array; };
  // A graph whose prompt phase hands its capture layers 0..N-1 and returns the
  // post-final-norm output without handing it over.
  const model = { config: { modelType: "fixture", eosTokenIds: [], text: { enableMoeBlock: false, numHiddenLayers: 2 } },
    graphCapabilities: declareGraph({ hiddenLayerTaps: true }), makeCache: () => [], logitsFromHidden: (hidden: MlxArray) => hidden,
    ...stubPhases((_ids, _caches, capture) => {
      for (let layer = 0; layer < 2; layer++) capture?.(layer, filled(layer + 1));
      return ops.contiguous(filled(9));
    }, hidden => hidden) };
  const binding = bindSpeculativeTargetModel(model as never);
  expect([binding.supportsTapLayers([]), binding.supportsTapLayers([1, 2]), binding.supportsTapLayers([3]),
    binding.supportsTapLayers([-1])]).toEqual([true, true, false, false]);
  using ids = ops.fromInt32([1, 2, 3], [1, 3]);
  const { hidden, ctxML } = await binding.prefill(ids, [], { phase: "tail", sequences: 1, tokens: L }, [1, 2]);
  try {
    expect(ctxML!.shape).toEqual([1, L, 2 * H]);
    expect([...ctxML!.toFloat32()]).toEqual(Array.from({ length: L }, () => [2, 2, 9, 9]).flat());
    expect([...hidden.toFloat32()]).toEqual(new Array(L * H).fill(9)); // the caller keeps the forward output
  } finally { hidden.dispose(); ctxML?.dispose(); for (const array of created) array.dispose(); }
  // A graph that declares no hidden-layer taps supports only an empty declaration.
  const plain = bindSpeculativeTargetModel({ ...model, graphCapabilities: declareGraph() } as never);
  expect([plain.supportsTapLayers([]), plain.supportsTapLayers([0])]).toEqual([true, false]);
});
