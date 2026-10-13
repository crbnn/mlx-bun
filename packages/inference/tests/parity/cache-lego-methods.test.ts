// Opt-in, real weights: methods on cache legos that did not serve them before
// PLAN B3 (speculation over paged KV, TurboQuant and delayed affine KV), against
// the same method on the bf16 lego. Each lego is a composition the loader
// resolves (`resolveComposition`: its KV scheme and paging); the gateway binding
// plans every request and gives each row its request state, as the app does.
// - Teacher-forced verification rounds: one row, then two rows of different
//   lengths, are prefilled on each lego's request state (a prompt tail), joined
//   into its verification rows (`targetCacheLayout`) and run through rounds of
//   fixed windows on the graph's verify phase (a fixed greedy continuation)
//   under the speculative transaction with a fixed
//   acceptance schedule. Every kept position's logits are compared with the
//   bf16 lego's: max |Δlogit|, argmax agreement, mean and max KL. The KL of the
//   served affine lego of the same width, and of the tiled affine kernels
//   against the unfused ones (the accepted tiled null), are printed beside the
//   newly served legos' for calibration.
// - The speculative group with the assistant drafter on every lego, placed by
//   the binding's plan and submitted with its request state: one row and a
//   four-row cohort, tokens and acceptance against the same group over the
//   bf16 lego.
// Opt in with MLX_BUN_TEST_LEGO_TARGET=/gemma4/snapshot and
// MLX_BUN_TEST_LEGO_DRAFT=/assistant/snapshot (a Gemma 4 target and its
// assistant drafter). No downloads.
import { expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { join } from "node:path";

type A = any;
const target = Bun.env.MLX_BUN_TEST_LEGO_TARGET, draft = Bun.env.MLX_BUN_TEST_LEGO_DRAFT;
for (const [name, dir] of [["TARGET", target], ["DRAFT", draft]] as const)
  if (dir !== undefined && !existsSync(join(dir, "config.json"))) throw new Error(`MLX_BUN_TEST_LEGO_${name}: no config.json in ${dir}`);
if ((target === undefined) !== (draft === undefined)) throw new Error("MLX_BUN_TEST_LEGO_TARGET and MLX_BUN_TEST_LEGO_DRAFT come together");

const BLOCK = 256, CAPACITY = 1024, DEPTH = 3, ROUNDS = 6;

/** A lego: the composition's KV request, and whether it pages. */
interface Lego { readonly name: string; readonly kv?: A; readonly paged: boolean }
const LEGOS: readonly Lego[] = [
  { name: "bf16", paged: false },
  { name: "paged bf16", paged: true },
  { name: "affine 4 (unfused)", kv: { override: 4, fusedSdpa: false }, paged: false },
  { name: "affine 4 (tiled-causal)", kv: { override: 4, fusedSdpa: true }, paged: false },
  // Converts inside the rounds: the prompts hold 15 to 30 tokens, the rounds add 24.
  { name: "affine 4 from 32 (unfused)", kv: { override: 4, fusedSdpa: false, quantizedKvStart: 32 }, paged: false },
  { name: "paged 4", kv: { override: 4 }, paged: true },
  { name: "affine 8 (unfused)", kv: { override: 8, fusedSdpa: false }, paged: false },
  { name: "paged 8", kv: { override: 8 }, paged: true },
  { name: "TurboQuant 8/3", kv: { turboQuant: { kBits: 8, vBits: 3 } }, paged: false },
  { name: "affine config (tiled-causal)", kv: { override: "config" }, paged: false },
  // Pages under a per-layer scheme and under TurboQuant (B4): newly served, KL-gated.
  { name: "paged config", kv: { override: "config" }, paged: true },
  { name: "paged TurboQuant 8/3", kv: { turboQuant: { kBits: 8, vBits: 3 } }, paged: true },
];
const PAIRS: readonly [string, string][] = [
  ["bf16", "paged bf16"],
  ["affine 4 (unfused)", "affine 4 (tiled-causal)"],
  ["bf16", "affine 4 (unfused)"], ["bf16", "affine 4 from 32 (unfused)"], ["bf16", "paged 4"],
  ["bf16", "affine 8 (unfused)"], ["bf16", "paged 8"],
  ["bf16", "TurboQuant 8/3"],
  ["bf16", "affine config (tiled-causal)"], ["bf16", "paged config"], ["affine config (tiled-causal)", "paged config"],
  ["bf16", "paged TurboQuant 8/3"], ["TurboQuant 8/3", "paged TurboQuant 8/3"],
];
const SHAPE = { hasVision: false, hasAdapters: false, hasRepetitionPenalty: false, userSeed: false,
  hasLogitsExtras: false, hasGrammar: false, wantsLogprobs: false, hasDraft: false };

/** Log-softmax of one logit row, f64 on the host. */
function logSoftmax(row: Float32Array): Float64Array {
  let max = -Infinity;
  for (const x of row) if (x > max) max = x;
  let sum = 0;
  for (const x of row) sum += Math.exp(x - max);
  const lse = max + Math.log(sum), out = new Float64Array(row.length);
  for (let i = 0; i < row.length; i++) out[i] = row[i]! - lse;
  return out;
}
function compare(a: readonly Float32Array[], b: readonly Float32Array[]) {
  let maxAbs = 0, agree = 0, klSum = 0, klMax = 0;
  for (let p = 0; p < a.length; p++) {
    const x = a[p]!, y = b[p]!;
    let ax = 0, ay = 0;
    for (let i = 0; i < x.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(x[i]! - y[i]!));
      if (x[i]! > x[ax]!) ax = i;
      if (y[i]! > y[ay]!) ay = i;
    }
    agree += Number(ax === ay);
    const lx = logSoftmax(x), ly = logSoftmax(y);
    let kl = 0;
    for (let i = 0; i < x.length; i++) kl += Math.exp(lx[i]!) * (lx[i]! - ly[i]!);
    klSum += kl; klMax = Math.max(klMax, kl);
  }
  return { positions: a.length, maxAbs, argmaxAgree: agree / a.length, klMean: klSum / a.length, klMax };
}

async function loadTarget() {
  const { loadModelConfig, Weights, createModel, resolveComposition, loadTokenizer, generate } = await import("@mlx-bun/inference");
  const config = await loadModelConfig(target!);
  const weights = await Weights.open(target!);
  const tokenizer = await loadTokenizer(target!);
  /** The graph the loader builds for `lego`'s composition (bf16 when none is named). */
  const graph = (lego?: Lego) => createModel(weights, config, resolveComposition(config, {
    ...(lego?.kv ? { kv: lego.kv } : {}), ...(lego?.paged ? { pagedKv: { blockSize: BLOCK } } : {}) })) as A;
  const release = async () => {
    const { clearCache } = await import("@mlx-bun/mlx/ffi");
    try { weights.dispose(); } finally { for (const file of weights.shards.files.values()) file.mmap.unmap(); clearCache(); }
  };
  return { config, tokenizer, graph, generate, release };
}

/** One request's state as the scheduler gives it to a row: fresh state from the
 * composition's cache service (`KvCacheService` over the bound graph: its paging
 * over the graph's caches), for a request of `CAPACITY` tokens. */
function requestState(cache: A, binding: A, execution: A): A[] {
  const row = { promptIds: [], maxTokens: CAPACITY, reusesPrefix: binding.reusesPrefix(execution) };
  try { return cache.acquire(row, "").caches; } finally { cache.release(row, false); }
}

test.skipIf(!target)("teacher-forced verification rounds on every lego's verification rows", async () => {
  const ops = await import("@mlx-bun/mlx/ops");
  const { clearCache, Dtype } = await import("@mlx-bun/mlx/ffi");
  const { createKvMaintenance } = await import("../../src/state/layout");
  const { bindMlxGateway } = await import("@mlx-bun/inference/execution");
  const { KvCacheService } = await import("@mlx-bun/inference/state");
  const { targetCacheLayout } = await import("../../src/state/layout");
  const { MlxStateRows } = await import("../../src/state/rows");
  const { bindRowCacheRollback } = await import("../../src/state/rollback");
  const { phaseHidden } = await import("../models/phase-forward");
  const { widthClass } = await import("../../src/contracts/mlx/graph");
  const loaded = await loadTarget();
  try {
    const prompts = ["Write a short paragraph about lighthouses and the people who kept them.",
      "List three facts about the moon, then explain which one surprised scientists the most and why it mattered for the Apollo program."]
      .map(text => loaded.tokenizer.encode(text, true));
    // The fixed continuation each row is forced through: the bf16 graph's greedy tokens.
    const reference = loaded.graph();
    const continuations: number[][] = [];
    for (const prompt of prompts) {
      const tokens: number[] = [];
      for await (const token of loaded.generate(reference, prompt, { maxTokens: ROUNDS * (DEPTH + 1) + 2, temperature: 0 })) tokens.push(token.token);
      continuations.push(tokens);
    }
    clearCache();
    const accepted = (round: number, row: number) => (round * 7 + row * 3 + 1) % (DEPTH + 1);
    const report: Record<string, Record<string, unknown>> = {};
    // One row reads aligned windows (the tiled kernels' own case); two rows of
    // different lengths read padded ones.
    for (const cohort of [1, 2]) {
      const used = prompts.slice(0, cohort);
      const logits = new Map<string, Float32Array[]>();
      for (const lego of LEGOS) {
        const model = loaded.graph(lego);
        const binding = bindMlxGateway(model);
        // The planner places the row whatever the lego; its request state is the composition's.
        const execution = binding.plan(SHAPE, {}, { checkpoints: false });
        expect(execution.mechanism, lego.name).toBe("continuous");
        const maintain = createKvMaintenance(model.composition?.kv ?? { kind: "bf16" });
        const cache = new KvCacheService(binding.graph, {}, { runtime: binding.runtime });
        const rows: A[][] = [];
        for (const prompt of used) {
          const caches = requestState(cache, binding, execution);
          if (lego.paged) expect(caches.some((cache: A) => cache.signature() === "kv:paged"), lego.name).toBe(true);
          using ids = ops.fromInt32(prompt, [1, prompt.length]);
          phaseHidden(model, ids, caches).dispose();
          maintain.prepareBatch?.(caches);
          rows.push(caches);
        }
        const state = new MlxStateRows(rows[0]!.map((cache: A) => targetCacheLayout(cache)));
        const kept: Float32Array[] = [];
        try {
          for (const row of rows) state.append(row);
          const rollback = bindRowCacheRollback(state.caches, rows.length);
          const position = rows.map(() => 0);
          for (let round = 0; round < ROUNDS; round++) {
            rollback.begin(DEPTH);
            using ids = ops.fromInt32(position.flatMap((at, row) => continuations[row]!.slice(at, at + DEPTH + 1)), [rows.length, DEPTH + 1]);
            // The verify phase at the class of the window's rows (rows times depth + 1), as the speculative group calls it.
            const width = widthClass(rows.length * (DEPTH + 1));
            using hidden = model.verify[width](ids, state.caches, null);
            using out = model.logits[width](hidden);
            const counts = rows.map((_, row) => accepted(round, row));
            for (const [row, count] of counts.entries()) for (let p = 0; p <= count; p++) {
              using one = out.slice([row, p, 0], [row + 1, p + 1, out.shape[2]!]);
              using wide = one.astype(Dtype.float32);
              kept.push(wide.toFloat32());
            }
            rollback.resolve(counts);
            for (const [row, count] of counts.entries()) position[row]! += count + 1;
            expect([...state.caches[0]!.rowOffsets]).toEqual(used.map((prompt, row) => prompt.length + position[row]!));
          }
        } finally { state.dispose(); for (const row of rows) for (const cache of row) cache.dispose(); clearCache(); }
        logits.set(lego.name, kept);
      }
      const pairs: Record<string, unknown> = report[`B${cohort}`] = {};
      for (const [a, b] of PAIRS) pairs[`${a} | ${b}`] = compare(logits.get(a)!, logits.get(b)!);
      // Bf16 pages store what the bf16 lego stores; their verification rows join
      // the rows the way the decode rows do, which reduces over a padded length.
      expect((pairs["bf16 | paged bf16"] as ReturnType<typeof compare>).argmaxAgree).toBe(1);
    }
    console.log(JSON.stringify({ target, depth: DEPTH, rounds: ROUNDS, report }, null, 1));
  } finally { await loaded.release(); }
}, 1_800_000);

test.skipIf(!target)("the speculative group with the assistant drafter on every lego, placed by the planner", async () => {
  const { bindMlxGateway } = await import("@mlx-bun/inference/execution");
  const speculative = await import("@mlx-bun/inference/generation/speculative");
  const { clearCache } = await import("@mlx-bun/mlx/ffi");
  const loaded = await loadTarget();
  try {
    const provider = (await speculative.defaultDraftProviders().load("assistant", { dir: draft,
      target: { vocabSize: loaded.config.text.vocabSize, tokenizer: loaded.tokenizer } })).provider as A;
    const prompts = ["Describe a quiet harbor at dawn.", "Name four rivers in Europe and one fact about each of them.",
      "What is the boiling point of water at high altitude, and why?", "Tell a two-sentence story about a clockmaker."]
      .map(text => loaded.tokenizer.encode(text, true));
    const options: A = { temperature: 0, maxTokens: 32, eosTokenIds: [] };
    /** `count` rows on `lego` as the app runs them: the binding plans each
     * request and submits it with its method and request state. */
    const run = async (lego: Lego, count: number) => {
      const model = loaded.graph(lego);
      const binding = bindMlxGateway(model, { provider, numDraftTokens: 2 });
      const plan = binding.plan({ ...SHAPE, hasDraft: true }, options, { checkpoints: false });
      expect({ lego: lego.name, method: plan.method, mechanism: plan.mechanism })
        .toEqual({ lego: lego.name, method: "speculative", mechanism: "continuous" });
      let held = true;
      const group = binding.createBatchGroup({ maxBatch: 4, admissionHeld: () => held });
      const tokens = Array.from({ length: count }, () => [] as number[]);
      try {
        const pending = tokens.map((output, row) => group.submit({ method: binding.methodRequest!(plan, options)!,
          reusesPrefix: binding.reusesPrefix(plan),
          compiledDecode: plan.compiledDecode,
          promptIds: prompts[row]!, maxTokens: options.maxTokens, eosTokenIds: [], onToken(token: number) { output.push(token); } }));
        held = false; group.kick();
        const stats = await Promise.all(pending);
        return { tokens, spec: stats.map((stat: A) => ({ drafted: stat.spec?.drafted, accepted: stat.spec?.accepted, rounds: stat.spec?.rounds })) };
      } finally { await group.close(); clearCache(); }
    };
    const report: Record<string, Record<string, unknown>> = {};
    const legos = LEGOS.filter(lego => lego.name !== "affine 4 (tiled-causal)");
    for (const count of [1, 4]) {
      const cohort: Record<string, unknown> = report[`B${count}`] = {};
      const reference = await run(legos[0]!, count);
      cohort.bf16 = { spec: reference.spec };
      for (const lego of legos.slice(1)) {
        const served = await run(lego, count);
        cohort[lego.name] = { tokensEqual: JSON.stringify(reference.tokens) === JSON.stringify(served.tokens),
          spec: served.spec, firstDifference: reference.tokens.map((row, index) => row.findIndex((token, at) => served.tokens[index]![at] !== token)) };
        for (const stats of served.spec) expect(stats.rounds, lego.name).toBeGreaterThan(0);
        // Bf16 pages store what the bf16 lego stores: one row reads them alike.
        if (lego.name === "paged bf16" && count === 1) expect(served.tokens).toEqual(reference.tokens);
      }
    }
    console.log(JSON.stringify({ target, draft, report }, null, 1));
  } finally { await loaded.release(); }
}, 1_800_000);

