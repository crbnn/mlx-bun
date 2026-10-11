// Generated Gemma4 graphs against the dedicated graph on real weights. A
// generated graph (12B, e4b or 26B, selected by config fingerprint) is an entire
// graph built for the artifact's kv_config composition; it reads every cache
// through that cache's own read and must equal the monolith bit for bit. This
// ports `02d723a:tests/parity/generated-parity.test.ts` (12B and e4b cells) to
// every registered fingerprint:
// - createModel selects the generated graph registered for the fingerprint;
// - three uncompiled steps over caches converted to kv_config before the first
//   forward, from a prompt past the sliding window: byte-identical full
//   vocabulary vectors;
// - 24-token greedy trajectories under kv_config (quantized from token 0; stop
//   tokens off, since e4b's first token for this raw prompt is EOS) are
//   identical uncompiled and with compiled decode, whose step records the
//   generated graph's own layers (e4b whole-graph, 12B segmented);
// - under plain bf16 caches, which the composition does not name, the
//   generated graph reads them through their own reads: identical vectors;
// - two different rows in one forward over kv_config caches equal the monolith's.
// Each case logs a digest of the generated graph's outputs, for comparing runs
// across trees. This is within-tree specialization identity, not
// external-oracle parity; the monolith's oracle consumer is gemma4-parity.test.ts.
// Opt in with MLX_BUN_TEST_GENERATED_MODEL=/gemma4/snapshot (an artifact whose
// fingerprint has a generated graph and which ships kv_config.json). Unset
// skips; a blank value or an artifact without kv_config.json fails. No downloads.
import { expect, test } from "bun:test";
import { strict as assert } from "node:assert";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { configureRuntime } from "@mlx-bun/inference/runtime/config";
import { releaseAll, sha256 } from "./real-weight-inputs";

type A = any;
const MODEL = "MLX_BUN_TEST_GENERATED_MODEL";
const STEPS = 3, TOKENS = 24, ROW_TOKENS = 16;

export function optIn(env: Record<string, string | undefined>) {
  const model = env[MODEL];
  if (model === undefined) return null;
  if (!model.trim()) throw new Error(`${MODEL} is blank`);
  assert(existsSync(join(model, "config.json")), `${MODEL}: no config.json in ${model}`);
  assert(existsSync(join(model, "kv_config.json")), `${MODEL}: generated graphs serve the kv_config layout; no kv_config.json in ${model}`);
  const raw = JSON.parse(readFileSync(join(model, "config.json"), "utf8"));
  assert(String(raw.model_type).startsWith("gemma4"), `${MODEL}: not a Gemma4 artifact (${raw.model_type})`);
  return { model };
}

/** Main's prompt lengths: 1100 tokens for the 12B (window 1024), 700 for e4b (512). */
export const promptLength = (window: number) => Math.max(700, window + 76);

const inputs = optIn(Bun.env);

test.skipIf(!inputs)("generated Gemma4 graph equals the monolith under kv_config and under plain caches", async () => {
  const ffi = await import("@mlx-bun/mlx/ffi");
  const ops = await import("@mlx-bun/mlx/ops");
  const { loadModelConfig, Weights, configFingerprint } = await import("@mlx-bun/inference/artifacts");
  const { createModel } = await import("@mlx-bun/inference/models");
  const { Gemma4Model } = await import("@mlx-bun/inference/models/gemma4");
  const { CompiledDecode } = await import("@mlx-bun/inference/models/gemma4/compiled-decode");
  const { generate } = await import("@mlx-bun/inference/generation");
  const { loadTokenizer } = await import("@mlx-bun/inference/input");
  // The kernel set kv-maintenance composes for the flag the run has (both graphs read through it).
  const { servedAffineKernels } = await import("../../src/state/served-affine-kernels");
  const { GEMMA4_GENERATED_GRAPHS } = await import("../../src/models/gemma4/generated");
  const path = inputs!.model;
  const config = await loadModelConfig(path);
  const fingerprint = configFingerprint(config);
  const generated = GEMMA4_GENERATED_GRAPHS.find(entry => entry.composition.fingerprint === fingerprint);
  assert(generated, `no generated graph is registered for fingerprint ${fingerprint}`);
  const kvConfig = config.kvQuant;
  assert(kvConfig?.length, "the artifact's kv_config.json did not load");
  const tokenizer = await loadTokenizer(path);
  let message = "Explain the memory hierarchy of a modern computer in detail.";
  const filler = "Context: caches, DRAM, bandwidth, and latency all interact. ";
  const length = promptLength(config.text.slidingWindow);
  while (tokenizer.encode(message).length < length) message = filler + message;
  const prompt = tokenizer.encode(message).slice(0, length);
  const weights = await Weights.open(path);
  let mono: A, gen: A;
  const digest = (label: string, parts: (Buffer | number[])[]) =>
    console.log(`[generated ${fingerprint}] ${label} digest ${sha256(Buffer.concat(parts.map(part =>
      Array.isArray(part) ? Buffer.from(JSON.stringify(part)) : part)))}`);
  /** The last position's full vector (bytes) and its argmax. */
  const lastVector = (logits: A): { bytes: Buffer; next: number } => {
    const [, L, V] = logits.shape as [number, number, number];
    using last = logits.slice([0, L - 1, 0], [1, L, V]);
    using exact = ops.contiguous(last);
    using arg = ops.argmaxAxis(last, -1);
    return { bytes: Buffer.from(exact.rawBytes()), next: arg.toIntTokens()[0]! };
  };
  /** Fresh caches with each layer the kv_config names converted before any forward (the serve scenario). */
  const converted = (model: A): A[] => {
    const caches: A[] = model.makeCache();
    for (let layer = 0; layer < caches.length; layer++) {
      const entry = kvConfig.find(item => item.layerIdx === layer);
      if (entry) caches[layer] = caches[layer].toQuantized(entry.groupSize, entry.bits, servedAffineKernels());
    }
    return caches;
  };
  try {
    mono = new Gemma4Model(weights, config);
    gen = createModel(weights, config);
    expect(gen).toBeInstanceOf(generated.Graph);
    expect(mono).not.toBeInstanceOf(generated.Graph);

    // 1. Per-step vectors over caches converted before the first forward.
    const restore = configureRuntime({ MLX_BUN_COMPILED_DECODE: "0" });
    try {
      const steps = (model: A) => {
        const caches = converted(model);
        try {
          const vectors: Buffer[] = [];
          let tokens = prompt;
          for (let step = 0; step < STEPS; step++) {
            const logits = model.forward(tokens, caches);
            try { const { bytes, next } = lastVector(logits); vectors.push(bytes); tokens = [next]; }
            finally { logits.dispose(); }
          }
          return vectors;
        } finally { releaseAll(caches.map(cache => () => cache.dispose())); ffi.clearCache(); }
      };
      const specialized = steps(gen);
      const dedicated = steps(mono);
      specialized.forEach((vector, step) => expect({ step, equal: vector.equals(dedicated[step]!) }).toEqual({ step, equal: true }));
      digest("kv_config step vectors", specialized);
    } finally { restore(); }

    // 2. Greedy trajectories under kv_config, uncompiled and compiled.
    for (const compiled of [false, true]) {
      const reset = configureRuntime({ MLX_BUN_COMPILED_DECODE: compiled ? "1" : "0" });
      try {
        const run = async (model: A) => {
          const out: number[] = [];
          // No stop tokens: every trajectory is TOKENS long whatever the artifact's first token (e4b's is EOS here).
          const generation = generate(model, prompt, { maxTokens: TOKENS, temperature: 0, kvConfig, quantizedKvStart: 0, eosTokenIds: [] });
          for await (const token of generation) out.push(token.token);
          ffi.clearCache();
          return out;
        };
        const before = CompiledDecode.stepsExecuted;
        const specialized = await run(gen);
        const steps = CompiledDecode.stepsExecuted - before;
        const dedicated = await run(mono);
        expect(specialized.length).toBe(TOKENS);
        expect({ compiled, specialized }).toEqual({ compiled, specialized: dedicated });
        // The compiled run took compiled steps where the composition declares them (not the MoE 26B).
        if (!compiled || !gen.graphCapabilities.compiledDecode) expect(steps).toBe(0);
        else expect(steps).toBeGreaterThan(0);
        digest(`compiled ${compiled} trajectory`, [specialized]);
        console.log(`[generated ${fingerprint}] compiled ${compiled}: ${specialized.length} tokens identical, ${steps} compiled steps`);
      } finally { reset(); }
    }

    // 3. Plain caches: read through their own reads.
    const plain = (model: A) => {
      const caches: A[] = model.makeCache();
      try {
        const logits = model.forward(prompt.slice(0, 64), caches);
        try { return lastVector(logits).bytes; } finally { logits.dispose(); }
      } finally { releaseAll(caches.map(cache => () => cache.dispose())); }
    };
    const plainVector = plain(gen);
    expect(plainVector.equals(plain(mono))).toBe(true);
    digest("bf16 vector", [plainVector]);

    // 4. Two different rows through one forward over kv_config caches: the batch axis is not
    // assumed to be one (e4b slices its per-layer inputs per layer; row 1 must get its own).
    const rows = (model: A) => {
      const caches = converted(model);
      try {
        const ids = ops.fromInt32([...prompt.slice(0, ROW_TOKENS), ...prompt.slice(-ROW_TOKENS)], [2, ROW_TOKENS]);
        try {
          const hidden = model.forwardHidden(ids, caches);
          try { using exact = ops.contiguous(hidden); return Buffer.from(exact.rawBytes()); } finally { hidden.dispose(); }
        } finally { ids.dispose(); }
      } finally { releaseAll(caches.map(cache => () => cache.dispose())); ffi.clearCache(); }
    };
    const batched = rows(gen);
    expect(batched.equals(rows(mono))).toBe(true);
    digest("two rows hidden", [batched]);
  } finally {
    // Compiled closures borrow weight arrays: release their tapes before the weights.
    try { releaseAll([mono, gen].filter(Boolean).map(model => () => CompiledDecode.release(model))); }
    finally {
      try { weights.dispose(); }
      finally { releaseAll([...weights.shards.files.values()].map(file => () => file.mmap.unmap()).concat([() => ffi.clearCache()])); }
    }
  }
}, 900_000);

// ---- CPU-only validation (no native libraries) ------------------------------------------
test("opt-in: unset skips; blank or incomplete artifacts fail (CPU only)", () => {
  expect(optIn({})).toBeNull();
  expect(() => optIn({ [MODEL]: " " })).toThrow("blank");
  expect(() => optIn({ [MODEL]: "/nonexistent-generated-model" })).toThrow("no config.json");
});

test("prompt lengths reproduce main's and pass the window (CPU only)", () => {
  expect(promptLength(1024)).toBe(1100);
  expect(promptLength(512)).toBe(700);
  for (const window of [512, 1024, 4096]) expect(promptLength(window)).toBeGreaterThan(window);
});
