import { expect, test } from "bun:test";
import * as ops from "@mlx-bun/mlx/ops";
import { Dtype } from "@mlx-bun/mlx/ffi";
import { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxCompiledDecodeStep } from "../../src/contracts/mlx/graph";
import { MlxBatchExecutionGroup } from "../../src/execution/batch-group";
import { bindMlxAutoregressiveGraph } from "../../src/generation/bindings/autoregressive";
import { compiledDecodeStepOf, declareGraph } from "../../src/models/capabilities";
import type { MlxTokenGraph } from "../../src/models/graph";
import { createRuntimeConfig } from "../../src/runtime/config";
import { KVCache } from "../../src/state/kv";
import type { Cache } from "../../src/contracts/mlx/cache";
import { stubPhases } from "../models/phase-forward";
import { testComposition } from "../models/test-composition";

// A graph that is none of the model classes and declares its own compiled step:
// the scheduler and the autoregressive binding reach it only through that
// declaration. Ordinary forwards produce token 0; the declared step, token 3.
const VOCAB = 8, COMPILED_TOKEN = 3;

function graph(declared: Parameters<typeof declareGraph>[0], extra: Partial<MlxTokenGraph> = {}) {
  const forwards: number[] = [];
  const append = (caches: Cache[], width: number) => {
    using kv = ops.zeros([1, 1, width, 4], Dtype.float32);
    for (const cache of caches) { const [k, v] = cache.updateAndFetch(kv, kv); k.dispose(); v.dispose(); }
  };
  const model: MlxTokenGraph = {
    weightsBytes: 0,
    config: { modelType: "fixture", eosTokenIds: [], text: { numHiddenLayers: 1, layerTypes: ["full_attention"],
      numGlobalKeyValueHeads: 1, globalHeadDim: 4, slidingWindow: 0 } } as never,
    makeCache: () => [new KVCache()], requiredDenseKvLayers: [], graphCapabilities: declareGraph(declared),
    ...stubPhases((ids, caches) => { forwards.push(ids.shape[1]!); append(caches, ids.shape[1]!); return ops.zeros([1, ids.shape[1]!, VOCAB], Dtype.float32); },
      hidden => ops.copyOf(hidden)),
    ...extra,
  };
  return { model, forwards, append };
}

function countingStep(model: { append: (caches: Cache[], width: number) => void }, options: { failAt?: number } = {}) {
  const calls = { steps: 0 };
  const step: MlxCompiledDecodeStep = {
    step(_token: MlxArray, caches: Cache[]) {
      if (calls.steps++ === options.failAt) throw new Error("compiled step failed");
      model.append(caches, 1);
      const row = new Float32Array(VOCAB); row[COMPILED_TOKEN] = 1;
      return { logits: MlxArray.fromFloat32(row, [1, 1, VOCAB]), evalWith: [] };
    },
  };
  return { step, calls };
}

async function decode(model: MlxTokenGraph, options: { compiledDecode?: boolean } = {},
  tokens: number[] = []) {
  const group = new MlxBatchExecutionGroup(model, { maxBatch: 2, runtime: createRuntimeConfig({}) });
  try {
    await group.submit({ promptIds: [1, 2, 3], maxTokens: 6, eosTokenIds: [], sample: logits => ops.argmaxAxis(logits, -1),
      ...(options.compiledDecode === undefined ? {} : { compiledDecode: options.compiledDecode }),
      onToken(token) { tokens.push(token); return undefined; } });
  } finally { await group.close(); }
  return tokens;
}

/** A cache the compiled step cannot drive: it declares no decode slots. */
const unreplayable = () => [Object.assign(new KVCache(), { prepareDecodeStep: undefined }) as unknown as Cache];

test("a lone row decodes through the step its graph declares, and other graphs never get one", async () => {
  const fixture = graph({ compiledDecode: true });
  const declared = countingStep(fixture);
  const model = { ...fixture.model, compiledDecodeStep: () => declared.step };
  const tokens = await decode(model);
  expect(tokens[0]).toBe(0); // token zero comes from the prefill projection
  expect(tokens.slice(1)).toEqual(Array(5).fill(COMPILED_TOKEN));
  expect(declared.calls.steps).toBeGreaterThan(0);
  expect(fixture.forwards).toEqual([2, 1]); // the prompt (tail split off) only; no decode forward ran

  // A graph that declares no compiled decode is never asked for a step, even if it has the method.
  const undeclared = graph({});
  const unused = countingStep(undeclared);
  expect(await decode({ ...undeclared.model, compiledDecodeStep: () => unused.step })).toEqual(Array(6).fill(0));
  expect(unused.calls).toEqual({ steps: 0 });
  expect(undeclared.forwards.length).toBeGreaterThan(1);
});

test("the request, the composition and caches the step cannot drive keep decoding on the decode operation", async () => {
  for (const [name, options, extra] of [
    ["request opt-out", { compiledDecode: false }, {}],
    ["composition off", {}, { composition: testComposition(undefined, { compiledDecode: false }) }],
    ["caches without decode slots", {}, { makeCache: unreplayable }],
  ] as const) {
    const fixture = graph({ compiledDecode: true }, extra);
    const declared = countingStep(fixture);
    expect(await decode({ ...fixture.model, compiledDecodeStep: () => declared.step }, options), name).toEqual(Array(6).fill(0));
    expect(declared.calls.steps, name).toBe(0);
  }
});

test("a step that fails fails its request: the compiled step has no fallback", async () => {
  const fixture = graph({ compiledDecode: true });
  const declared = countingStep(fixture, { failAt: 2 });
  const tokens: number[] = [];
  await expect(decode({ ...fixture.model, compiledDecodeStep: () => declared.step }, {}, tokens)).rejects.toThrow("compiled step failed");
  // Decode is pipelined: the second step's token is read only after the third step dispatches, which fails.
  expect(tokens).toEqual([0, COMPILED_TOKEN]);
  expect(declared.calls.steps).toBe(3);
  expect(fixture.forwards).toEqual([2, 1]); // no decode forward replayed the token
});

test("a graph that promises a compiled step and provides none is refused where it binds", () => {
  const fixture = graph({ compiledDecode: true });
  expect(() => compiledDecodeStepOf(fixture.model)).toThrow("provides no compiledDecodeStep");
  expect(() => new MlxBatchExecutionGroup(fixture.model, { maxBatch: 1 })).toThrow("provides no compiledDecodeStep");
  expect(compiledDecodeStepOf(graph({}).model)).toBeNull();
});

test("the autoregressive binding offers the declared step for a run whose caches it drives, unless adapters or paging rule replay out", () => {
  const fixture = graph({ compiledDecode: true });
  const declared = countingStep(fixture);
  const binding = bindMlxAutoregressiveGraph({ ...fixture.model, compiledDecodeStep: () => declared.step });
  const caches = fixture.model.makeCache();
  try {
    expect(binding.createDecode!({ hasAdapters: true, pagedKv: false }, caches)).toBeNull();
    expect(binding.createDecode!({ hasAdapters: false, pagedKv: true }, caches)).toBeNull();
    expect(binding.createDecode!({ hasAdapters: false, pagedKv: false }, unreplayable())).toBeNull();
    const decoder = binding.createDecode!({ hasAdapters: false, pagedKv: false }, caches)!;
    try {
      using token = MlxArray.fromFloat32(new Float32Array([0]), [1]);
      const result = decoder.step(token, caches);
      expect(result.logits.shape).toEqual([1, 1, VOCAB]);
      result.logits.dispose();
      expect(caches[0]!.offset).toBe(1);
      expect(declared.calls.steps).toBe(1);
    } finally { void decoder.close(); }
  } finally { for (const cache of caches) cache.dispose(); }
  // A graph declaring nothing has no decoder, whatever methods it carries.
  expect(bindMlxAutoregressiveGraph({ ...graph({}).model, compiledDecodeStep: () => declared.step })
    .createDecode!({ hasAdapters: false, pagedKv: false }, caches)).toBeNull();
});
