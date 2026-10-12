// Generated Gemma4 graphs against the dedicated graph on real weights. A
// generated graph (12B, e4b or 26B, selected by config fingerprint) is an entire
// graph built for the artifact's kv_config composition; it composes that
// scheme's caches and reads every cache through that cache's own read, and must
// equal the generic Gemma 4 graph bit for bit. This ports
// `02d723a:tests/parity/generated-parity.test.ts` (12B and e4b cells) to every
// registered fingerprint:
// - createModel selects the generated graph registered for the fingerprint, for
//   the composition it serves (`--kv-quant config`: the artifact's kv_config),
//   and the generic graph for the default composition (bf16);
// - three uncompiled steps over caches converted to kv_config before the first
//   forward, from a prompt past the sliding window: byte-identical full
//   vocabulary vectors;
// - 24-token greedy trajectories under the kv_config composition (each graph's
//   own caches: the bf16-first affine lego on every cache the kv_config lists;
//   stop tokens off, since e4b's first token for this raw prompt is EOS) are
//   identical uncompiled and with compiled decode, whose step records the
//   generated graph's own layers (e4b whole-graph, 12B segmented). Uncompiled,
//   the generated graph's forward is the one that runs on every forward (it
//   extends nothing), each forward reads every cache once, and from the second
//   forward on every listed cache answers from its quantized storage;
// - under plain bf16 caches, which the composition does not name, the
//   generated graph reads them through their own reads: identical vectors;
// - two different rows in one forward over kv_config caches equal the generic graph's.
// Each case logs a digest of the generated graph's outputs, for comparing runs
// across trees. This is within-tree specialization identity, not
// external-oracle parity; the generic graph's oracle consumer is gemma4-parity.test.ts.
// Opt in with MLX_BUN_TEST_GENERATED_MODEL=/gemma4/snapshot (an artifact whose
// fingerprint has a generated graph and which ships kv_config.json). Unset
// skips; a blank value or an artifact without kv_config.json fails. No downloads.
import { expect, test } from "bun:test";
import { strict as assert } from "node:assert";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
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

/** Affine storage, full or rotating (`kv:quant:…`, `kv:rotating-quant:…`). */
export const isQuantizedStorage = (signature: string) => /^kv:(rotating-)?quant:/.test(signature);

/** `caches` with every append recorded in `log[cache]`: the signature of the storage
 *  the cache held when the append arrived, so the storage that answered the read. */
export function recordReads(caches: A[], storageOf: (cache: A) => { signature(): string }, log: string[][]): A[] {
  return caches.map((cache, index) => new Proxy(cache, {
    get(target, prop) {
      if (prop === "appendDecode" || prop === "appendWindow" || prop === "appendBidirectional")
        return (...args: unknown[]) => { log[index]!.push(storageOf(target).signature()); return target[prop](...args); };
      const value = Reflect.get(target, prop, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
    set: (target, prop, value) => Reflect.set(target, prop, value, target),
  }));
}

const inputs = optIn(Bun.env);

test.skipIf(!inputs)("generated Gemma4 graph equals the generic graph under kv_config and under plain caches", async () => {
  const ffi = await import("@mlx-bun/mlx/ffi");
  const ops = await import("@mlx-bun/mlx/ops");
  const { loadModelConfig, Weights, configFingerprint } = await import("@mlx-bun/inference/artifacts");
  const { createModel, resolveComposition } = await import("@mlx-bun/inference/models");
  const { Gemma4Model } = await import("@mlx-bun/inference/models/gemma4");
  const { CompiledDecode } = await import("@mlx-bun/inference/models/gemma4/compiled-decode");
  const { generate } = await import("@mlx-bun/inference/generation");
  const { loadTokenizer } = await import("@mlx-bun/inference/input");
  const { phaseHidden, phaseLogits } = await import("../models/phase-forward");
  // The kernel set `--kv-quant config` composes (tiled-causal), which the caches converted
  // before the first forward read with in both graphs.
  const { composedAffineKernels } = await import("@mlx-bun/inference/state");
  const { composedStorageOf } = await import("../../src/state/capabilities");
  const configKernels = composedAffineKernels("tiled-causal");
  const { GENERATED_GRAPHS } = await import("../../src/models/gemma4/generated");
  const path = inputs!.model;
  const config = await loadModelConfig(path);
  const fingerprint = configFingerprint(config);
  const registered = GENERATED_GRAPHS.find(item => item.serves.fingerprint === fingerprint);
  assert(registered, `no generated graph is registered for fingerprint ${fingerprint}`);
  const generated = await import(`../../src/models/gemma4/generated/${registered.id}`);
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
  /** Plain caches (the bf16 composition's) with each layer the kv_config names converted before any forward. */
  const converted = (): A[] => {
    const caches: A[] = mono.makeCache();
    for (let layer = 0; layer < caches.length; layer++) {
      const entry = kvConfig.find(item => item.layerIdx === layer);
      if (entry) caches[layer] = caches[layer].toQuantized(entry.groupSize, entry.bits, configKernels);
    }
    return caches;
  };
  try {
    mono = new Gemma4Model(weights, config, resolveComposition(config));
    gen = createModel(weights, config, resolveComposition(config, { kv: { override: "config" } }));
    expect(gen).toBeInstanceOf(generated.GeneratedGemma4);
    // An entire graph: no class to fall back to.
    expect(Object.getPrototypeOf(generated.GeneratedGemma4.prototype)).toBe(Object.prototype);
    expect(mono).not.toBeInstanceOf(generated.GeneratedGemma4);
    {
      // The default composition (bf16) builds the generic graph.
      const plainGraph = createModel(weights, config);
      try { expect(plainGraph.constructor).toBe(Gemma4Model); } finally { (plainGraph as A).dispose?.(); }
    }

    // 1. Per-step vectors over caches converted before the first forward.
    {
      const steps = (model: A) => {
        const caches = converted();
        try {
          const vectors: Buffer[] = [];
          let tokens = prompt;
          for (let step = 0; step < STEPS; step++) {
            const logits = phaseLogits(model, tokens, caches);
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
    }

    // 2. Greedy trajectories under the kv_config composition, uncompiled and compiled: both
    // graphs composed with the artifact's per-layer scheme, each reading its own caches.
    const configComposition = resolveComposition(config, { kv: { override: "config", quantizedKvStart: 0 } });
    const configured = { mono: new Gemma4Model(weights, config, configComposition), gen: createModel(weights, config, configComposition) as A };
    expect(configured.gen).toBeInstanceOf(generated.GeneratedGemma4);
    try {
      for (const compiled of [false, true]) {
        {
          const run = async (model: A, log: string[][] | null) => {
            const out: number[] = [];
            const caches: A[] = model.makeCache();
            const read = log ? recordReads(caches, composedStorageOf, log) : caches;
            try {
              // No stop tokens: every trajectory is TOKENS long whatever the artifact's first token (e4b's is EOS here).
              const generation = generate(model, prompt, { maxTokens: TOKENS, temperature: 0, eosTokenIds: [], cache: read,
                ...(compiled ? {} : { decodePolicy: { compiledDecode: false, grammarJump: false } }) });
              for await (const token of generation) out.push(token.token);
            } finally { releaseAll(caches.map(cache => () => cache.dispose())); ffi.clearCache(); }
            return out;
          };
          // Uncompiled, every append of the generated graph's caches is recorded.
          const log: string[][] | null = compiled ? null : Array.from({ length: configured.gen.numDonors }, () => []);
          const before = CompiledDecode.stepsExecuted;
          const specialized = await run(configured.gen, log);
          const steps = CompiledDecode.stepsExecuted - before;
          const dedicated = await run(configured.mono, null);
          expect(specialized.length).toBe(TOKENS);
          expect({ compiled, specialized }).toEqual({ compiled, specialized: dedicated });
          if (log) {
            // Every forward read every cache once, through the generated graph (the only forward it has); the
            // first append of a listed cache is bf16, every later one is answered by its quantized storage.
            const forwards = log[0]!.length;
            expect(forwards).toBeGreaterThanOrEqual(TOKENS);
            log.forEach((reads, cache) => {
              expect({ cache, reads: reads.length }).toEqual({ cache, reads: forwards });
              const listed = kvConfig.some(item => item.layerIdx === cache);
              expect({ cache, first: isQuantizedStorage(reads[0]!) }).toEqual({ cache, first: false });
              const later = reads.slice(1).filter(signature => isQuantizedStorage(signature) !== listed);
              expect({ cache, listed, otherwise: later }).toEqual({ cache, listed, otherwise: [] });
            });
            console.log(`[generated ${fingerprint}] uncompiled: ${forwards} generated forwards, every listed cache quantized after the first`);
          }
          // The compiled run took compiled steps where the composition declares them (not the MoE 26B).
          if (!compiled || !configured.gen.graphCapabilities.compiledDecode) expect(steps).toBe(0);
          else expect(steps).toBeGreaterThan(0);
          digest(`compiled ${compiled} trajectory`, [specialized]);
          console.log(`[generated ${fingerprint}] compiled ${compiled}: ${specialized.length} tokens identical, ${steps} compiled steps`);
        }
      }
    } finally {
      try { releaseAll([configured.mono, configured.gen].map(model => () => CompiledDecode.release(model))); }
      finally { releaseAll([configured.mono, configured.gen].map(model => () => model.dispose?.())); }
    }

    // 3. Plain caches (the bf16 composition's): read through their own reads.
    const plain = (model: A) => {
      const caches: A[] = mono.makeCache();
      try {
        const logits = phaseLogits(model, prompt.slice(0, 64), caches);
        try { return lastVector(logits).bytes; } finally { logits.dispose(); }
      } finally { releaseAll(caches.map(cache => () => cache.dispose())); }
    };
    const plainVector = plain(gen);
    expect(plainVector.equals(plain(mono))).toBe(true);
    digest("bf16 vector", [plainVector]);

    // 4. Two different rows through one forward over kv_config caches: the batch axis is not
    // assumed to be one (e4b slices its per-layer inputs per layer; row 1 must get its own).
    const rows = (model: A) => {
      const caches = converted();
      try {
        const ids = ops.fromInt32([...prompt.slice(0, ROW_TOKENS), ...prompt.slice(-ROW_TOKENS)], [2, ROW_TOKENS]);
        try {
          const hidden = phaseHidden(model, ids, caches);
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

test("the read recorder logs the storage that answered each append and leaves the cache's behavior alone (CPU only)", () => {
  const appended: string[] = [];
  let signature = "kv:plain";
  const cache = {
    offset: 0,
    appendDecode(k: string) { appended.push(k); signature = "kv:quant:4:64"; return k; },
    appendWindow(k: string) { appended.push(k); return k; },
    get storage() { return { signature: () => signature }; },
  };
  const log: string[][] = [[]];
  const [wrapped] = recordReads([cache], c => c.storage, log);
  expect(wrapped.appendDecode("a")).toBe("a");
  expect(wrapped.appendWindow("b")).toBe("b");
  wrapped.offset = 3;
  expect(cache.offset).toBe(3);
  expect(appended).toEqual(["a", "b"]);
  expect(log).toEqual([["kv:plain", "kv:quant:4:64"]]);
  expect(["kv:quant:4:64", "kv:rotating-quant:8:64", "kv:plain", "kv:rotating-plain", "kv:bf16-first:kv:quant:4:64"]
    .map(isQuantizedStorage)).toEqual([true, true, false, false, false]);
});
