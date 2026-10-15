import { expect, test } from "bun:test";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { ModelConfig } from "../../../src/artifacts/config";
import type { Glm52Config } from "../../../src/artifacts/glm52-config";
import type { Glm52WeightSource } from "../../../src/contracts/mlx/glm52-weights";
import { rowOperationFor } from "../../../src/models/row-table";
import { GLM52_PHASE_READS, type Glm52PhaseReads } from "../../../src/models/glm52/mla";
import { Glm52Model } from "../../../src/models/glm52/model";

function fixture() {
  const context = 1 << 30;
  const reads: Glm52PhaseReads[] = [];
  const fixed: boolean[] = [];
  let releases = 0;
  const output = {} as MlxArray;
  const embedding = { dispose: () => { releases++; } } as MlxArray;
  // Zero decoder layers and opaque handles keep this constructor/dispatch
  // fixture host-only: no native array allocation or tensor operation.
  const weights = {
    weightsBytes: 0,
    tensor: () => output,
    embedding: () => embedding,
  } as unknown as Glm52WeightSource;
  const config = { modelType: "glm_moe_dsa", text: { numHiddenLayers: 0 } } as ModelConfig;
  const glmConfig = {
    modelType: "glm_moe_dsa", numHiddenLayers: 0,
    hiddenSize: 1, vocabSize: 1, maxPositionEmbeddings: context,
  } as Glm52Config;
  const runtime = {
    executor: { setFixedKernelFamily: (value: boolean) => { fixed.push(value); } },
  } as unknown as ConstructorParameters<typeof Glm52Model>[5];
  const model = new Glm52Model(weights, config, glmConfig, { dsa: false, mtpMetadata: false }, null, runtime);
  model.forwardEmbeddingsAsync = async (_embeddings, _cache, phase) => {
    reads.push(phase);
    return output;
  };
  model.logitsFromHidden = () => output;
  const ids = (batch: number, tokens: number) => ({ shape: [batch, tokens] }) as MlxArray;
  return { model, context, reads, fixed, output, ids, releases: () => releases };
}

test("GLM phase declarations stay compact for a billion-position context and preserve aggregate-row lookup", async () => {
  const f = fixture();
  expect(Array.isArray(f.model.appendKnown)).toBe(true);
  expect(f.model.appendKnown).toHaveLength(2);
  for (const table of [f.model.decode, f.model.verify, f.model.logits]) {
    expect(Object.keys(table)).toEqual(["default"]);
    expect(rowOperationFor<Function>(table,2 * f.context)).toBe(table.default!);
    expect(table[0]).toBeUndefined();
  }
  expect(await rowOperationFor(f.model.decode,2 * f.context)(f.ids(2 * f.context, 1), [], null)).toBe(f.output);
  expect(await rowOperationFor(f.model.appendKnown, 1)(f.ids(1, 1), [])).toBe(f.output);
  expect(await rowOperationFor(f.model.appendKnown, f.context)(f.ids(1, f.context), [])).toBe(f.output);
  expect(await rowOperationFor(f.model.verify,2 * f.context)(f.ids(2, f.context), [], null)).toBe(f.output);
  expect(rowOperationFor(f.model.logits,2 * f.context)(f.output)).toBe(f.output);
  expect(f.reads).toEqual([
    GLM52_PHASE_READS.oneToken, GLM52_PHASE_READS.oneToken,
    GLM52_PHASE_READS.window, GLM52_PHASE_READS.window,
  ]);
  expect(f.fixed).toEqual([true, false]);
  expect(f.releases()).toBe(4);
});

test("GLM's unbounded verify operation restores the expert kernel family when forwarding fails", async () => {
  const f = fixture();
  f.model.forwardEmbeddingsAsync = async () => { throw new Error("forward failed"); };
  await expect(rowOperationFor(f.model.verify,2 * f.context)(f.ids(2, 8), [], null)).rejects.toThrow("forward failed");
  expect(f.fixed).toEqual([true, false]);
  expect(f.releases()).toBe(1);
});
