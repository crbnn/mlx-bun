// The loader's selection on real artifacts, through openModel: a written Gemma 4
// artifact gets the generic graph for the default composition (bf16 KV) and its
// written graph for the kv_config composition (`--kv-quant config`); the
// published Qwen3.8 Trellis artifact gets the M4 Pro graph on applegpu_g16s and
// the generated M1 Max graph on applegpu_g13s for the default composition. The graphs are
// constructed (weights open, nothing runs a forward). Opt in with
//   MLX_BUN_TEST_GRAPH_TABLE_MODEL=/snapshot
// naming one such artifact. Unset skips; a blank value, a missing config.json or
// an artifact with no expectation here fails. One artifact per process: the
// Trellis graph creates Neural Engine programs when the bridge is present.
import { expect, test } from "bun:test";
import { strict as assert } from "node:assert";
import { existsSync } from "node:fs";
import { join } from "node:path";

const MODEL = "MLX_BUN_TEST_GRAPH_TABLE_MODEL";

export function optIn(env: Record<string, string | undefined>) {
  const model = env[MODEL];
  if (model === undefined) return null;
  if (!model.trim()) throw new Error(`${MODEL} is blank`);
  assert(existsSync(join(model, "config.json")), `${MODEL}: no config.json in ${model}`);
  return { model };
}

const inputs = optIn(Bun.env);

test.skipIf(!inputs)("openModel builds the graph the table selects for the composition", async () => {
  const { deviceArchitecture } = await import("@mlx-bun/mlx/ffi");
  const { loadModelConfig, configFingerprint } = await import("../../src/artifacts");
  const { openModel, resolveComposition } = await import("../../src/models");
  const { Gemma4Model } = await import("../../src/models/gemma4/model");
  const { Qwen38TrellisM4Pro } = await import("../../src/models/qwen/qwen38-27b-trellis-m4pro");
  const { Qwen38TrellisM1Max } = await import("../../src/models/qwen/qwen38-27b-trellis-m1max");
  const { qwen35GraphFingerprint } = await import("../../src/models/qwen/checkpoint");
  const [twelve, e4b, moe] = await Promise.all([import("../../src/models/gemma4/gemma4-12b"),
    import("../../src/models/gemma4/gemma4-e4b"), import("../../src/models/gemma4/gemma4-26b")]);
  const dir = inputs!.model;
  const config = await loadModelConfig(dir);
  const device = deviceArchitecture();
  /** Open with `composition`, read the built graph's class and release what the graph owns. */
  const built = async (composition?: ReturnType<typeof resolveComposition>) => {
    const model = await openModel(dir, composition ? { composition } : {});
    try { return model.constructor; } finally { (model as { dispose?: () => void }).dispose?.(); }
  };

  const written = [{ fingerprint: twelve.FINGERPRINT, Graph: twelve.Gemma4TwelveB },
    { fingerprint: e4b.FINGERPRINT, Graph: e4b.Gemma4E4b },
    { fingerprint: moe.FINGERPRINT, Graph: moe.Gemma4Moe }];
  const gemma = written.find(entry => entry.fingerprint === configFingerprint(config));
  if (gemma) {
    assert(config.kvQuant?.length, `${MODEL}: a written Gemma 4 artifact ships kv_config.json`);
    expect(await built()).toBe(Gemma4Model);
    expect(await built(resolveComposition(config, { kv: { override: "config" } }))).toBe(gemma.Graph);
    console.log(`[graph-table] ${dir}: default composition -> Gemma4Model; kv_config composition -> written ${gemma.fingerprint}`);
    return;
  }
  assert(qwen35GraphFingerprint(config) === "cfa205c8f5af046a", `${MODEL}: no selection expectation for ${dir}`);
  assert(device === "applegpu_g16s" || device === "applegpu_g13s", `${MODEL}: the Trellis artifact has a written graph for g16s and g13s, not ${device}`);
  const expected = device === "applegpu_g16s" ? Qwen38TrellisM4Pro : Qwen38TrellisM1Max;
  expect(await built()).toBe(expected);
  console.log(`[graph-table] ${dir} on ${device}: default composition -> ${expected.name}`);
}, 900_000);

test("opt-in: unset skips; blank or incomplete artifacts fail (CPU only)", () => {
  expect(optIn({})).toBeNull();
  expect(() => optIn({ [MODEL]: " " })).toThrow("blank");
  expect(() => optIn({ [MODEL]: "/nonexistent-graph-table-model" })).toThrow("no config.json");
});
