// Qwen3.8-27B packed-Trellis graph and its native MTP draft head against main's
// own outputs. The packed-Trellis artifact has no external oracle (mlx-lm cannot
// load it), so the reference is what main at 02d723a computes for the same
// artifacts, produced outside this repository by
// packages/inference/scripts/main-qwen-trellis-reference.ts. This test never
// starts main or Python. Opt in with all four of
//   MLX_BUN_TEST_QWEN_TRELLIS_TARGET=/packed/trellis/snapshot
//   MLX_BUN_TEST_QWEN_TRELLIS_DRAFT=/folded/mtp/snapshot
//   MLX_BUN_TEST_QWEN_TRELLIS_REFERENCE=/path/main-qwen-trellis.json
//   MLX_BUN_TEST_QWEN_TRELLIS_REFERENCE_SHA256=<its SHA-256>
// None set skips; any other combination fails. The reference and both artifacts
// (metadata files and shards, by content) are verified before native libraries
// load, and the runtime must match the reference's MLX version and GPU family.
// Checked, batch one, plain KV, per prompt:
//  - the tokenizer and template render the reference's prompt IDs;
//  - greedy trajectory: the prompt once and each selected token on the same live
//    cache through `model.forward`; every step's full last-position float32
//    logits hash to main's (bit for bit) and are finite, and every selected
//    token is main's;
//  - verify windows against one-row decode on this tree, main's MTP run
//    teacher-forced: after the prompt, each recorded round's window [pending,
//    ...proposals] runs through `verify` at its width class, the caches'
//    speculative rollback drops it again, and the round's accepted tokens are
//    decoded one row at a time on the same state. The one-row decode selects
//    every token main emitted (main's speculative output was its plain greedy
//    output, and plain decode here is main's to the bit). Each window row whose
//    context was accepted (the pending token's and each accepted proposal's) is
//    compared with the one-row logits of its position: KL(one row || window)
//    within KL_ROW per row and KL_MEAN on average, and the same argmax unless the
//    one-row top is a near-tie (NEAR_TIE), where the window may pick a tied
//    token. Rows after a rejected proposal read a context no output depends on
//    and are not compared; the fully accepted rounds compare every window row;
//  - MTP speculation, depth 2, greedy, stop tokens off, through the public gateway
//    binding. Where every compared window row is bit-identical to its one-row
//    decode, as main's M1 graph's were, the run equals main's shared batch-one
//    gateway: the emitted tokens, every target forward's input IDs (so every
//    verify row, accepted or rejected) and the round and draft counters. Where
//    they are not, a run's last bits depend on its window widths, which Josh's
//    2026-10-11 width-class decision accepts, KL-gated, as the M4 Pro graph
//    serves: since 2026-10-13 this tree's M1 Max graph verifies two to three rows
//    on the factored and affine row kernels, which reassociate relative to its
//    one-row codebook kernels (main's ran the reduce and scatter row kernels
//    there). Then the emitted tokens equal plain greedy (main's) up to their
//    first difference, which must fall on a near-tie of the one-row logits, and
//    the counters are those the run's own target forwards and tokens give. The
//    check keys on that measured identity, not on the graph's name.
// Continuation state and append geometry are the runtime-oracle plan's job
// (README, "Repeatable runtime comparison"). This is not a claim about other
// artifacts, batching above one, quantized KV, sampling, or speed.
import { expect, test } from "bun:test";
import { strict as assert } from "node:assert";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { isSha256, optInAll, releaseAll, sha256, verifyArtifact } from "./real-weight-inputs";

const OPT_IN = ["MLX_BUN_TEST_QWEN_TRELLIS_TARGET", "MLX_BUN_TEST_QWEN_TRELLIS_DRAFT", "MLX_BUN_TEST_QWEN_TRELLIS_REFERENCE",
  "MLX_BUN_TEST_QWEN_TRELLIS_REFERENCE_SHA256"] as const;
const SPEC_KEYS = ["drafted", "accepted", "rejected", "targetCalls", "rounds", "acceptanceLengths", "draftedByPos", "acceptedByPos"] as const;
const TARGET_METADATA = ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"];
// A window row against its one-row decode, where the two are not bit-identical. Measured 2026-10-13 on the M1 Max
// graph over main's three runs (192 rows): mean KL 2.4e-4, the largest 1.1e-2 (after the end of turn, stop tokens
// off), the top two logits moved by at most 0.75, one argmax difference, at an exact bf16 tie. Faults put into the
// two-to-four-row factored down kernel of the 24 row-major layers: one input row of each split's sum skipped, mean
// 2.6e-3 and largest 1.4e-1; every row reading the first row's activations, mean 3.3 and argmax differences at
// margins up to 11.
const KL_ROW = 5e-2, KL_MEAN = 1e-3;
/** A logit this close to the top one is tied with it: four bf16 steps of a logit between 16 and 32, two above 32
 *  (these runs' top logits lie between 20 and 34). */
const NEAR_TIE = 0.5;
type A = any;

interface Inputs { target: string; draft: string; reference: string; referenceSha256: string }
interface Pins { files: Record<string, string>; weights: Record<string, string> }
interface Reference {
  producer: string; schema: 1; mainRevision: string; graph: string;
  runtime: { mlx: string; architecture: string };
  artifacts: { target: Pins; draft: Pins };
  prompts: { text: string; ids: number[] }[];
  trajectories: { prompt: string; promptIds: number[]; greedy: number[]; logitsSha256: string[]; finite: boolean; vocab: number }[];
  mtp: { execution: "shared-b1"; depth: number; maxTokens: number; runs: { prompt: string; tokens: number[]; forwards: number[][]; spec: Record<string, unknown> }[] };
}

function optIn(env: Record<string, string | undefined>): Inputs | null {
  const values = optInAll(env, OPT_IN, "Qwen Trellis parity");
  if (!values) return null;
  const referenceSha256 = values[OPT_IN[3]];
  if (!isSha256(referenceSha256)) throw new Error(`${OPT_IN[3]} must be a lowercase SHA-256`);
  return { target: values[OPT_IN[0]], draft: values[OPT_IN[1]], reference: values[OPT_IN[2]], referenceSha256 };
}

/** A greedy speculative run's rounds and counters as its target forwards (the prompt, then [pending, ...proposals]
 *  per round) and emitted tokens give them: a proposal is accepted exactly when it is the next emitted token. */
function speculationOf(forwards: readonly (readonly number[])[], tokens: readonly number[], depth: number, what: string) {
  const rounds: { start: number; drafts: number; accepted: number }[] = [];
  let start = 0;
  for (const window of forwards.slice(1)) {
    assert.equal(window[0], tokens[start], `${what} round ${rounds.length}: the pending token is not the next emitted token`);
    let accepted = 0;
    while (accepted + 1 < window.length && window[accepted + 1] === tokens[start + accepted + 1]) accepted++;
    rounds.push({ start, drafts: window.length - 1, accepted });
    start += accepted + 1;
  }
  // The last round's own sample is emitted unless the token budget ended the run first.
  assert(start === tokens.length || start === tokens.length - 1, `${what}: the rounds do not account for the emitted tokens`);
  const sum = (count: (round: typeof rounds[number]) => number) => rounds.reduce((total, round) => total + count(round), 0);
  const byPosition = (count: (round: typeof rounds[number]) => number) =>
    Array.from({ length: depth }, (_, position) => rounds.filter(round => count(round) > position).length);
  const drafted = sum(round => round.drafts), accepted = sum(round => round.accepted);
  return { rounds, spec: { drafted, accepted, rejected: drafted - accepted, targetCalls: forwards.length, rounds: rounds.length,
    acceptanceLengths: rounds.map(round => round.accepted), draftedByPos: byPosition(round => round.drafts),
    acceptedByPos: byPosition(round => round.accepted) } };
}

/** The tokens of a logits row within NEAR_TIE of its top, the top (the first maximum, as argmax picks) first. */
function nearTop(row: Float32Array): number[] {
  let top = 0;
  for (let i = 1; i < row.length; i++) if (row[i]! > row[top]!) top = i;
  const tied = [top];
  for (let i = 0; i < row.length; i++) if (i !== top && row[i]! >= row[top]! - NEAR_TIE) tied.push(i);
  return tied;
}

/** Verify the reference and both artifacts it pins, wherever they now live. */
async function verifyInputs(inputs: Inputs): Promise<Reference> {
  const bytes = readFileSync(inputs.reference);
  assert.equal(sha256(bytes), inputs.referenceSha256, "reference SHA-256 differs from the supplied pin");
  const reference = JSON.parse(bytes.toString("utf8")) as Reference;
  assert.equal(reference.producer, "main-qwen-trellis-reference", "not a main-qwen-trellis-reference file");
  assert.equal(reference.schema, 1, "unsupported reference schema");
  assert(typeof reference.runtime?.mlx === "string" && reference.runtime.mlx && typeof reference.runtime?.architecture === "string" && reference.runtime.architecture,
    "reference runtime identity missing");
  const ids = (value: unknown, what: string) => assert(Array.isArray(value) && value.length > 0 && value.every(n => Number.isSafeInteger(n) && n >= 0), `${what}: expected token IDs`);
  assert(reference.prompts?.length > 0, "reference has no prompts");
  reference.prompts.forEach((p, i) => ids(p.ids, `prompt ${i}`));
  assert.equal(reference.trajectories?.length, reference.prompts.length, "a trajectory per prompt is required");
  reference.trajectories.forEach((t, i) => {
    assert.deepEqual(t.promptIds, reference.prompts[i]!.ids, `trajectory ${i}: prompt IDs differ from the prompt`);
    ids(t.greedy, `trajectory ${i} greedy`);
    assert.equal(t.logitsSha256?.length, t.greedy.length, `trajectory ${i}: a logits hash per step is required`);
    t.logitsSha256.forEach(h => assert(isSha256(h), `trajectory ${i}: invalid logits SHA-256`));
    assert.equal(t.finite, true, `trajectory ${i}: main's logits were not all finite`);
    assert(Number.isSafeInteger(t.vocab) && t.vocab > 0, `trajectory ${i}: vocabulary`);
  });
  assert.equal(reference.mtp?.execution, "shared-b1", "MTP reference must use the same shared batch-one execution shape");
  assert.equal(reference.mtp?.runs?.length, reference.prompts.length, "an MTP run per prompt is required");
  assert(Number.isSafeInteger(reference.mtp.depth) && reference.mtp.depth >= 1, "MTP depth");
  reference.mtp.runs.forEach((run, i) => {
    ids(run.tokens, `mtp run ${i} tokens`);
    assert.equal(run.tokens.length, reference.mtp.maxTokens, `mtp run ${i}: expected ${reference.mtp.maxTokens} tokens`);
    assert(Array.isArray(run.forwards) && run.forwards.length > 1 && run.forwards.every(row => Array.isArray(row) && row.length > 0), `mtp run ${i}: target forwards`);
    for (const key of SPEC_KEYS) assert(key in (run.spec ?? {}), `mtp run ${i}: spec.${key} missing`);
    assert.deepEqual(run.forwards[0], reference.prompts[i]!.ids, `mtp run ${i}: the first target forward is not the prompt`);
    // Its windows are teacher-forced below with the acceptances its tokens give.
    assert.deepEqual(speculationOf(run.forwards, run.tokens, reference.mtp.depth, `mtp run ${i}`).spec,
      Object.fromEntries(SPEC_KEYS.map(key => [key, run.spec[key]])), `mtp run ${i}: counters disagree with its forwards and tokens`);
  });
  await verifyArtifact(inputs.target, reference.artifacts?.target, TARGET_METADATA);
  await verifyArtifact(inputs.draft, reference.artifacts?.draft, ["config.json"]);
  return reference;
}

const inputs = optIn(Bun.env);

test.skipIf(!inputs)("Qwen3.8 packed-Trellis against main: logits, greedy tokens, verify windows and MTP speculation", async () => {
  const reference = await verifyInputs(inputs!);
  const ffi = await import("@mlx-bun/mlx/ffi");
  assert(ffi.MLX_VERSION === reference.runtime.mlx && ffi.deviceArchitecture() === reference.runtime.architecture,
    `reference/runtime mismatch: the reference was produced with MLX ${reference.runtime.mlx} on ${reference.runtime.architecture}, ` +
    `this runtime is MLX ${ffi.MLX_VERSION} on ${ffi.deviceArchitecture()}; supply a reference generated on a matching runtime`);
  const { loadModelConfig, loadTokenizer, ChatTemplate, Weights, createModel } = await import("@mlx-bun/inference");
  const { QwenMtpProvider } = await import("@mlx-bun/inference/generation/speculative");
  const { bindMlxGateway } = await import("@mlx-bun/inference/execution");
  const { argmaxLastPosition, klPerToken, lastPositionLogits } = await import("@mlx-bun/inference/scoring");
  const { rollbackVerifyBlock } = await import("@mlx-bun/inference/state");
  const { widthClass } = await import("@mlx-bun/inference/contracts");
  const ops = await import("@mlx-bun/mlx/ops");
  const { Qwen38TrellisM1Max } = await import("@mlx-bun/inference/models/qwen38-27b-trellis-m1max");
  const { clearCache } = await import("@mlx-bun/mlx/ffi");
  const { observedPhases, phaseLogits } = await import("../models/phase-forward");

  const config = await loadModelConfig(inputs!.target);
  const tokenizer = await loadTokenizer(inputs!.target), template = await ChatTemplate.load(inputs!.target);
  for (const prompt of reference.prompts) {
    const ids: number[] = tokenizer.encode(template.render([{ role: "user", content: prompt.text }], { enableThinking: false }));
    expect(ids[0] === ids[1] && ids[0] === tokenizer.bosTokenId ? ids.slice(1) : ids, prompt.text).toEqual(prompt.ids);
  }

  const weights = await Weights.open(inputs!.target);
  const releases: Array<() => void> = [];
  const report: Record<string, unknown> = {};
  try {
    const model = createModel(weights, config) as A;
    // Main selected its hand-written graph for this artifact on the M1 Max; this tree selects the graph generated
    // for that composition (the reference runtime's GPU family is checked above).
    expect(model).toBeInstanceOf(Qwen38TrellisM1Max);
    report.graph = model.constructor.name;

    // Greedy trajectories, full logits by hash.
    let steps = 0;
    for (const [i, trajectory] of reference.trajectories.entries()) {
      const cache = model.makeCache();
      try {
        let tokens = trajectory.promptIds;
        for (let step = 0; step < trajectory.greedy.length; step++) {
          const out = phaseLogits(model, tokens, cache);
          let next: number;
          try {
            const last: Float32Array = lastPositionLogits(out);
            expect(last.length, `trajectory ${i} step ${step}: width`).toBe(trajectory.vocab);
            expect(last.every(Number.isFinite), `trajectory ${i} step ${step}: finite`).toBe(true);
            expect(sha256(new Uint8Array(last.buffer, last.byteOffset, last.byteLength)), `trajectory ${i} step ${step}: logits differ from main's bytes`)
              .toBe(trajectory.logitsSha256[step]!);
            next = argmaxLastPosition(out);
          } finally { out.dispose(); }
          expect(next, `trajectory ${i} step ${step}: greedy token`).toBe(trajectory.greedy[step]!);
          tokens = [next];
          steps++;
        }
      } finally { releaseAll(cache.map((c: A) => () => c.dispose())); clearCache(); }
    }
    report.trajectorySteps = steps;

    // Main's MTP windows teacher-forced: each verified on the state its accepted predecessors built one row at a time,
    // dropped, then its accepted tokens decoded one row at a time and their logits compared.
    const windows = { rows: 0, identical: 0, klSum: 0, klMax: 0, tieFlips: 0 };
    const plainTies: number[][][] = []; // per run, per emitted token: the one-row near-top tokens at its position
    for (const [i, run] of reference.mtp.runs.entries()) {
      const cache = model.makeCache();
      const ties: number[][] = [];
      plainTies.push(ties);
      try {
        // The one-row logits [1, 1, V] after `tokens`, at the position of emitted token `ties.length`.
        const plain = (tokens: readonly number[]) => {
          using out = phaseLogits(model, tokens, cache);
          const [, L, V] = out.shape as number[];
          const last = out.slice([0, L! - 1, 0], [1, L!, V!]);
          ties.push(nearTop(last.toFloat32()));
          return last;
        };
        plain(reference.prompts[i]!.ids).dispose();
        const { rounds } = speculationOf(run.forwards, run.tokens, reference.mtp.depth, `mtp run ${i}`);
        for (const [r, round] of rounds.entries()) {
          const window = run.forwards[r + 1]!, width = widthClass(window.length), kept = round.accepted + 1;
          for (const c of cache) c.specRoundBegin?.();
          using ids = ops.fromInt32([...window], [1, window.length]);
          using hidden = await model.verify[width](ids, cache, null);
          using verified = model.logits[width](hidden);
          ops.evalAll([verified]);
          rollbackVerifyBlock(cache, 0, window.length);
          const V = verified.shape[2]!;
          using rows = verified.slice([0, 0, 0], [1, kept, V]);
          const decoded = Array.from({ length: kept }, (_, j) => plain([run.tokens[round.start + j]!]));
          using one = ops.concatAxis(decoded, 1);
          releaseAll(decoded.map(row => () => row.dispose()));
          const kl = klPerToken(one, rows), oneRows = one.toFloat32(), windowRows = rows.toFloat32();
          for (let j = 0; j < kept; j++) {
            const where = `mtp run ${i} round ${r}: window row ${j} (emitted token ${round.start + j + 1})`;
            const a = oneRows.subarray(j * V, (j + 1) * V), b = windowRows.subarray(j * V, (j + 1) * V);
            windows.rows++;
            if (Buffer.from(a.buffer, a.byteOffset, a.byteLength).equals(Buffer.from(b.buffer, b.byteOffset, b.byteLength))) {
              windows.identical++;
              continue;
            }
            expect(kl[j]!, `${where}: KL(one row || window)`).toBeLessThanOrEqual(KL_ROW);
            windows.klSum += kl[j]!; windows.klMax = Math.max(windows.klMax, kl[j]!);
            const near = nearTop(a), picked = nearTop(b)[0]!;
            expect(near, `${where}: the window's argmax ${picked} is not the one-row top or tied with it`).toContain(picked);
            if (picked !== near[0]) windows.tieFlips++;
          }
        }
        // The last accepted window's last row may predict one past the run.
        expect(ties.slice(0, run.tokens.length).map(near => near[0]), `mtp run ${i}: one-row decode selects main's emitted tokens`)
          .toEqual(run.tokens);
      } finally { releaseAll(cache.map((c: A) => () => c.dispose())); clearCache(); }
    }
    const exact = windows.identical === windows.rows;
    expect(windows.klSum / windows.rows, "mean KL(one row || window) over the compared rows").toBeLessThanOrEqual(KL_MEAN);
    report.windows = { rows: windows.rows, identical: windows.identical, meanKl: windows.klSum / windows.rows,
      maxKl: windows.klMax, tieFlips: windows.tieFlips };

    // Native MTP through the gateway binding, one request at a time.
    const provider = await QwenMtpProvider.load(inputs!.draft);
    releases.push(() => provider.dispose());
    let forwards: number[][] = [];
    // The scheduler binds to a view whose phases record each target forward's tokens.
    const view = observedPhases(model, { forward(input: A) { forwards.push(input.toIntTokens() as number[]); } });
    const binding = bindMlxGateway(view, { provider, numDraftTokens: reference.mtp.depth });
    const options = { temperature: 0, maxTokens: reference.mtp.maxTokens };
    const rounds: unknown[] = [];
    for (const [i, run] of reference.mtp.runs.entries()) {
      forwards = [];
      const plan = binding.plan({ hasVision: false, hasAdapters: false, hasRepetitionPenalty: false, userSeed: false,
        hasLogitsExtras: false, hasGrammar: false, wantsLogprobs: false, hasDraft: true }, options,
      { checkpoints: false });
      assert.equal(plan.method, "speculative");
      const method = binding.methodRequest!(plan, options);
      assert(method, "the MTP method must bind");
      const group = binding.createBatchGroup({ maxBatch: 1 });
      const tokens: number[] = [];
      let result: A;
      try {
        result = await group.submit({ method, promptIds: reference.prompts[i]!.ids, maxTokens: reference.mtp.maxTokens, eosTokenIds: [],
          onToken(token: number) { tokens.push(token); } });
      } finally { await group.close(); }
      const spec = Object.fromEntries(SPEC_KEYS.map(key => [key, result.spec?.[key] ?? null]));
      const mains = Object.fromEntries(SPEC_KEYS.map(key => [key, run.spec[key]]));
      const summary: Record<string, unknown> = { accepted: spec.accepted, drafted: spec.drafted, rounds: spec.rounds };
      if (exact) {
        expect(tokens, `mtp run ${i}: emitted tokens`).toEqual(run.tokens);
        expect(forwards, `mtp run ${i}: target forwards (prefill, then [pending, ...proposals] per round)`).toEqual(run.forwards);
        expect(spec, `mtp run ${i}: speculation counters`).toEqual(mains);
      } else {
        expect(tokens.length, `mtp run ${i}: emitted tokens`).toBe(reference.mtp.maxTokens);
        expect(forwards[0], `mtp run ${i}: prefill forward`).toEqual(reference.prompts[i]!.ids);
        expect(spec, `mtp run ${i}: speculation counters`).toEqual(speculationOf(forwards, tokens, reference.mtp.depth, `mtp run ${i}`).spec);
        // Plain greedy is main's emitted tokens (the one-row decode above selected each of them).
        const differs = tokens.findIndex((token, m) => token !== run.tokens[m]);
        if (differs >= 0) expect(plainTies[i]![differs], `mtp run ${i}: emitted token ${differs} differs from plain greedy's ` +
          `${run.tokens[differs]} beyond a near-tie`).toContain(tokens[differs]!);
        Object.assign(summary, { differsAt: differs < 0 ? null : differs, forwardsEqualMain: Bun.deepEquals(forwards, run.forwards),
          countersEqualMain: Bun.deepEquals(spec, mains) });
      }
      rounds.push(summary);
      clearCache();
    }
    report.mtp = rounds;
  } finally {
    try { releaseAll(releases.reverse()); } finally {
      try { weights.dispose(); } finally {
        releaseAll([...weights.shards.files.values()].map(file => () => file.mmap.unmap()).concat([() => clearCache()]));
      }
    }
  }
  console.log(JSON.stringify({ status: "pass", ...report, referenceSha256: inputs!.referenceSha256, mainRevision: reference.mainRevision,
    mlx: ffi.MLX_VERSION, architecture: ffi.deviceArchitecture(), bun: Bun.version }));
}, 3_600_000);

// ---- CPU-only validation (synthetic inputs; no native libraries) ------------------------
function synthetic() {
  const root = mkdtempSync(join(tmpdir(), "qwen-trellis-parity-"));
  const target = join(root, "target"), draft = join(root, "draft");
  mkdirSync(target); mkdirSync(draft);
  const files = Object.fromEntries(TARGET_METADATA.map(name => [name, `${name} contents`]));
  files["model.safetensors.index.json"] = JSON.stringify({ weight_map: { "a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors" } });
  for (const [name, text] of Object.entries(files)) writeFileSync(join(target, name), text);
  writeFileSync(join(target, "model-00001-of-00002.safetensors"), "shard-one");
  writeFileSync(join(target, "model-00002-of-00002.safetensors"), "shard-two");
  writeFileSync(join(draft, "config.json"), "draft config");
  writeFileSync(join(draft, "model.safetensors"), "draft weights");
  const h = "a".repeat(64);
  const reference: Reference = {
    producer: "main-qwen-trellis-reference", schema: 1, mainRevision: "rev", graph: "Qwen38TrellisTQ", runtime: { mlx: "0.32.2", architecture: "applegpu_g13s" },
    artifacts: {
      target: { files: Object.fromEntries(Object.entries(files).map(([name, text]) => [name, sha256(text)])),
        weights: { "model-00001-of-00002.safetensors": sha256("shard-one"), "model-00002-of-00002.safetensors": sha256("shard-two") } },
      draft: { files: { "config.json": sha256("draft config") }, weights: { "model.safetensors": sha256("draft weights") } },
    },
    prompts: [{ text: "p", ids: [1, 2, 3] }],
    trajectories: [{ prompt: "p", promptIds: [1, 2, 3], greedy: [4, 5], logitsSha256: [h, h], finite: true, vocab: 8 }],
    mtp: { execution: "shared-b1", depth: 2, maxTokens: 3, runs: [{ prompt: "p", tokens: [4, 5, 6], forwards: [[1, 2, 3], [4, 5, 6]],
      spec: { drafted: 2, accepted: 2, rejected: 0, targetCalls: 2, rounds: 1, acceptanceLengths: [2], draftedByPos: [1, 1], acceptedByPos: [1, 1] } }] },
  };
  const pin = () => { const text = JSON.stringify(reference); writeFileSync(join(root, "reference.json"), text); return sha256(text); };
  return { root, target, draft, reference, pin, inputs: (): Inputs => ({ target, draft, reference: join(root, "reference.json"), referenceSha256: pin() }) };
}
type Synthetic = ReturnType<typeof synthetic>;
async function rejects(change: (s: Synthetic) => Inputs | void, message: string) {
  const s = synthetic();
  try {
    const changed = change(s) ?? s.inputs();
    await expect(verifyInputs(changed)).rejects.toThrow(message);
  } finally { rmSync(s.root, { recursive: true, force: true }); }
}

test("opt-in is all or nothing (CPU only)", () => {
  expect(optIn({})).toBeNull();
  const full = { [OPT_IN[0]]: "/t", [OPT_IN[1]]: "/d", [OPT_IN[2]]: "/r.json", [OPT_IN[3]]: "a".repeat(64) };
  expect(optIn(full)).toEqual({ target: "/t", draft: "/d", reference: "/r.json", referenceSha256: "a".repeat(64) });
  expect(() => optIn({ [OPT_IN[0]]: "/t" })).toThrow("missing or blank");
  expect(() => optIn({ ...full, [OPT_IN[1]]: " " })).toThrow(`missing or blank: ${OPT_IN[1]}`);
  expect(() => optIn({ ...full, [OPT_IN[3]]: "A".repeat(64) })).toThrow("lowercase SHA-256");
});

test("a relocated identical reference and artifacts verify (CPU only)", async () => {
  const s = synthetic();
  try {
    const moved = join(s.root, "moved");
    cpSync(s.target, join(moved, "t"), { recursive: true });
    cpSync(s.draft, join(moved, "d"), { recursive: true });
    const verified = await verifyInputs({ ...s.inputs(), target: join(moved, "t"), draft: join(moved, "d") });
    expect(verified.trajectories[0]!.greedy).toEqual([4, 5]);
  } finally { rmSync(s.root, { recursive: true, force: true }); }
});

test("altered, incomplete or mismatched inputs fail before any native load (CPU only)", async () => {
  await rejects(s => ({ ...s.inputs(), referenceSha256: "0".repeat(64) }), "reference SHA-256 differs");
  await rejects(s => { s.reference.schema = 2 as 1; }, "unsupported reference schema");
  await rejects(s => { s.reference.producer = "other"; }, "not a main-qwen-trellis-reference file");
  await rejects(s => { s.reference.trajectories[0]!.logitsSha256.pop(); }, "a logits hash per step");
  await rejects(s => { s.reference.trajectories[0]!.logitsSha256[1] = "xyz"; }, "invalid logits SHA-256");
  await rejects(s => { s.reference.trajectories[0]!.finite = false; }, "were not all finite");
  await rejects(s => { s.reference.trajectories[0]!.promptIds = [9]; }, "prompt IDs differ from the prompt");
  await rejects(s => { s.reference.trajectories = []; }, "a trajectory per prompt");
  await rejects(s => { s.reference.mtp.runs[0]!.tokens.pop(); }, "expected 3 tokens");
  await rejects(s => { delete s.reference.mtp.runs[0]!.spec.acceptanceLengths; }, "spec.acceptanceLengths missing");
  await rejects(s => { s.reference.mtp.execution = "serial" as "shared-b1"; }, "same shared batch-one execution shape");
  await rejects(s => { s.reference.mtp.runs = []; }, "an MTP run per prompt");
  await rejects(s => { s.reference.mtp.runs[0]!.forwards[0] = [1, 2]; }, "the first target forward is not the prompt");
  await rejects(s => { s.reference.mtp.runs[0]!.forwards[1] = [5, 5, 6]; }, "the pending token is not the next emitted token");
  await rejects(s => { s.reference.mtp.runs[0]!.forwards[1] = [4]; }, "the rounds do not account for the emitted tokens");
  await rejects(s => { s.reference.mtp.runs[0]!.spec.acceptanceLengths = [1]; }, "counters disagree with its forwards and tokens");
  await rejects(s => { s.reference.runtime.architecture = ""; }, "runtime identity missing");
  await rejects(s => { writeFileSync(join(s.target, "tokenizer.json"), "changed"); }, "model file tokenizer.json differs");
  await rejects(s => { writeFileSync(join(s.target, "model-00002-of-00002.safetensors"), "other"); }, "weights model-00002-of-00002.safetensors differ");
  await rejects(s => { writeFileSync(join(s.draft, "model.safetensors"), "other"); }, "weights model.safetensors differ");
  await rejects(s => { rmSync(join(s.draft, "model.safetensors")); }, "model.safetensors");
  await rejects(s => { delete s.reference.artifacts.target.files["chat_template.jinja"]; }, "artifact pins do not include chat_template.jinja");
});
