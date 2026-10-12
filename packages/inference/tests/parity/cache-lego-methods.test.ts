// Opt-in, real weights: methods bound directly onto cache legos that did not
// serve them before PLAN B3 (paged KV under speculation, and every lego's
// verification rows), against the same method on the bf16 lego. The planner
// still refuses these pairs until E2 deletes its cache-format terms, so the
// methods are bound here, bypassing it.
// - Teacher-forced verification rounds: two rows of different lengths are
//   prefilled on each lego, joined into its verification rows
//   (`targetCacheLayout`) and run through rounds of fixed verify windows (a
//   fixed greedy continuation) under the speculative transaction with a fixed
//   acceptance schedule. Every kept position's logits are compared with the
//   bf16 lego's: max |Δlogit|, argmax agreement, mean and max KL. The KL of the
//   served affine lego of the same width, and of the tiled affine kernels
//   against the unfused ones (the accepted tiled null), are printed beside the
//   paged quantized legos' for calibration.
// - The speculative group with the assistant drafter over paged bf16 KV (the
//   request policy's pages in place of the graph's full-attention caches): one
//   row and a four-row cohort, tokens and acceptance against the same group
//   over the bf16 lego.
// Opt in with MLX_BUN_TEST_LEGO_TARGET=/gemma4/snapshot and
// MLX_BUN_TEST_LEGO_DRAFT=/assistant/snapshot (a Gemma 4 target and its
// assistant drafter). No downloads.
import { expect, spyOn, test } from "bun:test";
import { existsSync } from "node:fs";
import { join } from "node:path";

type A = any;
const target = Bun.env.MLX_BUN_TEST_LEGO_TARGET, draft = Bun.env.MLX_BUN_TEST_LEGO_DRAFT;
for (const [name, dir] of [["TARGET", target], ["DRAFT", draft]] as const)
  if (dir !== undefined && !existsSync(join(dir, "config.json"))) throw new Error(`MLX_BUN_TEST_LEGO_${name}: no config.json in ${dir}`);
if ((target === undefined) !== (draft === undefined)) throw new Error("MLX_BUN_TEST_LEGO_TARGET and MLX_BUN_TEST_LEGO_DRAFT come together");

const BLOCK = 256, CAPACITY = 1024, DEPTH = 3, ROUNDS = 6;

/** A lego: the composition's KV request, and whether the request policy pages
 * its full-attention layers. */
interface Lego { readonly name: string; readonly kv?: A; readonly paged: boolean }
const LEGOS: readonly Lego[] = [
  { name: "bf16", paged: false },
  { name: "paged bf16", paged: true },
  { name: "affine 4 (unfused)", kv: { override: 4, fusedSdpa: false }, paged: false },
  { name: "affine 4 (tiled-causal)", kv: { override: 4, fusedSdpa: true }, paged: false },
  { name: "paged 4", kv: { override: 4 }, paged: true },
  { name: "affine 8 (unfused)", kv: { override: 8, fusedSdpa: false }, paged: false },
  { name: "paged 8", kv: { override: 8 }, paged: true },
  { name: "TurboQuant 8/3", kv: { turboQuant: { kBits: 8, vBits: 3 } }, paged: false },
];
const PAIRS: readonly [string, string][] = [
  ["bf16", "paged bf16"],
  ["affine 4 (unfused)", "affine 4 (tiled-causal)"],
  ["bf16", "affine 4 (unfused)"], ["bf16", "paged 4"],
  ["bf16", "affine 8 (unfused)"], ["bf16", "paged 8"],
  ["bf16", "TurboQuant 8/3"],
];

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
  const graph = (kv?: A) => createModel(weights, config, resolveComposition(config, kv ? { kv } : {})) as A;
  const release = async () => {
    const { clearCache } = await import("@mlx-bun/mlx/ffi");
    try { weights.dispose(); } finally { for (const file of weights.shards.files.values()) file.mmap.unmap(); clearCache(); }
  };
  return { config, tokenizer, graph, generate, release };
}

/** `model`'s caches, with the request policy's pages in place of the
 * full-attention storage when `paged` (`maybePageKv`, as `--paged-kv` pages a
 * request). */
async function cachesOf(model: A, paged: boolean) {
  const { maybePageKv } = await import("../../src/state/request-policy");
  const caches = model.makeCache();
  if (paged) maybePageKv(caches, model.composition?.kv ?? { kind: "bf16" }, { pagedKv: { blockSize: BLOCK } }, CAPACITY);
  return caches;
}

test.skipIf(!target)("teacher-forced verification rounds on every lego's verification rows", async () => {
  const ops = await import("@mlx-bun/mlx/ops");
  const { clearCache, Dtype } = await import("@mlx-bun/mlx/ffi");
  const { createKvMaintenance } = await import("../../src/state/kv-maintenance");
  const { targetCacheLayout } = await import("../../src/state/layout");
  const { MlxStateRows } = await import("../../src/state/rows");
  const { bindRowCacheRollback } = await import("../../src/state/rollback");
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
    const logits = new Map<string, Float32Array[]>();
    for (const lego of LEGOS) {
      const model = loaded.graph(lego.kv);
      const maintain = createKvMaintenance(model.composition?.kv ?? { kind: "bf16" });
      const rows: A[][] = [];
      for (const prompt of prompts) {
        const caches = await cachesOf(model, lego.paged);
        using ids = ops.fromInt32(prompt, [1, prompt.length]);
        model.forwardHidden(ids, caches).dispose();
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
          using hidden = model.forwardHidden(ids, state.caches);
          using out = model.logitsFromHidden(hidden);
          const counts = rows.map((_, row) => accepted(round, row));
          for (const [row, count] of counts.entries()) for (let p = 0; p <= count; p++) {
            using one = out.slice([row, p, 0], [row + 1, p + 1, out.shape[2]!]);
            using wide = one.astype(Dtype.float32);
            kept.push(wide.toFloat32());
          }
          rollback.resolve(counts);
          for (const [row, count] of counts.entries()) position[row]! += count + 1;
          expect([...state.caches[0]!.rowOffsets]).toEqual(prompts.map((prompt, row) => prompt.length + position[row]!));
        }
      } finally { state.dispose(); for (const row of rows) for (const cache of row) cache.dispose(); clearCache(); }
      logits.set(lego.name, kept);
    }
    const report: Record<string, unknown> = {};
    for (const [a, b] of PAIRS) report[`${a} | ${b}`] = compare(logits.get(a)!, logits.get(b)!);
    console.log(JSON.stringify({ target, depth: DEPTH, rounds: ROUNDS, report }, null, 1));
    // Bf16 pages store what the bf16 lego stores; their verification rows join
    // the rows the way the decode rows do, which reduces over a padded length.
    const paged = report["bf16 | paged bf16"] as ReturnType<typeof compare>;
    expect(paged.argmaxAgree).toBe(1);
  } finally { await loaded.release(); }
}, 1_800_000);

test.skipIf(!target)("the speculative group with the assistant drafter over paged bf16 KV", async () => {
  const { bindMlxGateway } = await import("@mlx-bun/inference/execution");
  const speculative = await import("@mlx-bun/inference/generation/speculative");
  const { clearCache } = await import("@mlx-bun/mlx/ffi");
  const { maybePageKv } = await import("../../src/state/request-policy");
  const loaded = await loadTarget();
  try {
    const provider = (await speculative.defaultDraftProviders().load("assistant", { dir: draft,
      target: { vocabSize: loaded.config.text.vocabSize, tokenizer: loaded.tokenizer } })).provider as A;
    const prompts = ["Describe a quiet harbor at dawn.", "Name four rivers in Europe and one fact about each of them.",
      "What is the boiling point of water at high altitude, and why?", "Tell a two-sentence story about a clockmaker."]
      .map(text => loaded.tokenizer.encode(text, true));
    const options: A = { temperature: 0, maxTokens: 32, eosTokenIds: [] };
    const run = async (paged: boolean, count: number) => {
      const model = loaded.graph();
      // The paged lego in place of the graph's caches, before anything binds.
      if (paged) {
        const make = model.makeCache.bind(model);
        spyOn(model, "makeCache").mockImplementation(() => {
          const caches = make();
          maybePageKv(caches, model.composition?.kv ?? { kind: "bf16" }, { pagedKv: { blockSize: BLOCK } }, CAPACITY);
          return caches;
        });
      }
      const binding = bindMlxGateway(model, { provider, numDraftTokens: 2 });
      const plan = binding.plan({ hasVision: false, hasAdapters: false, hasRepetitionPenalty: false, userSeed: false,
        hasLogitsExtras: false, hasGrammar: false, wantsLogprobs: false, hasDraft: true }, options,
        { continuous: true, quantizedBatch: false, checkpoints: false });
      expect(plan.method).toBe("speculative");
      let held = true;
      const group = binding.createBatchGroup({ maxBatch: 4, admissionHeld: () => held });
      const tokens = Array.from({ length: count }, () => [] as number[]);
      try {
        const pending = tokens.map((output, row) => group.submit({ method: binding.methodRequest!(plan, options)!,
          promptIds: prompts[row]!, maxTokens: options.maxTokens, eosTokenIds: [], onToken(token: number) { output.push(token); } }));
        held = false; group.kick();
        const stats = await Promise.all(pending);
        return { tokens, spec: stats.map((stat: A) => ({ drafted: stat.spec?.drafted, accepted: stat.spec?.accepted, rounds: stat.spec?.rounds })) };
      } finally { await group.close(); clearCache(); }
    };
    const report: Record<string, unknown> = {};
    for (const count of [1, 4]) {
      const plain = await run(false, count), paged = await run(true, count);
      report[`B${count}`] = { tokensEqual: JSON.stringify(plain.tokens) === JSON.stringify(paged.tokens),
        bf16: plain.spec, paged: paged.spec,
        firstDifference: plain.tokens.map((row, index) => row.findIndex((token, at) => paged.tokens[index]![at] !== token)) };
      for (const stats of paged.spec) expect(stats.rounds).toBeGreaterThan(0);
    }
    console.log(JSON.stringify({ target, draft, report }, null, 1));
  } finally { await loaded.release(); }
}, 1_800_000);

