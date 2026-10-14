import { afterAll, beforeAll, expect, spyOn, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Weights } from "../../src/artifacts/weights";
import { openModel } from "../../src/models/factory";

let dir: string;
beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "mlx-explicit-graph-"));
  writeFileSync(join(dir, "config.json"), JSON.stringify({ model_type: "custom_decisions", hidden_size: 8,
    num_hidden_layers: 2, num_attention_heads: 2, num_key_value_heads: 2, intermediate_size: 16,
    vocab_size: 32, max_position_embeddings: 64, tie_word_embeddings: true }));
  const header = JSON.stringify({ "model.norm.weight": { dtype: "BF16", shape: [8], data_offsets: [0, 16] } });
  const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(header.length));
  writeFileSync(join(dir, "model.safetensors"), Buffer.concat([length, Buffer.from(header), Buffer.alloc(16)]));
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

test("an explicit load function needs no family registration or decode operation", async () => {
  const graph = await openModel(dir, { graph: (weights, config) => ({
    weights, config, prefill: () => [1, 2], score: () => [0.25, 0.75],
  }) });
  try {
    expect(graph.config.modelType).toBe("custom_decisions");
    expect(graph.prefill()).toEqual([1, 2]);
    expect(graph.score()).toEqual([0.25, 0.75]);
    expect(graph).not.toHaveProperty("decode");
  } finally { graph.weights.dispose(); }
});

test("a failed explicit load releases the weights it borrowed", async () => {
  const dispose = spyOn(Weights.prototype, "dispose");
  try {
    await expect(openModel(dir, { graph: () => { throw new Error("graph attachment failed"); } }))
      .rejects.toThrow("graph attachment failed");
    expect(dispose).toHaveBeenCalledTimes(1);
  } finally { dispose.mockRestore(); }
});
