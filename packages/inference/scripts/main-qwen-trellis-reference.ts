/** Reference producer for the Qwen3.8-27B packed-Trellis parity test: runs the
 * pre-refactor main checkout's own purpose-built graph and native shared MTP path
 * and records what they compute in a JSON file outside this repository.
 *
 *   cd <isolated main checkout at the pinned current main revision>
 *   MLX_BUN_LIBMLXC=<libmlxc.dylib> bun --no-env-file <this file> \
 *     --target <qwen38 packed-Trellis snapshot> --draft <folded MTP snapshot> --out main-qwen-trellis.json
 *
 * Main's sources are imported by path; nothing is written to or changed in the
 * main checkout. It records, for the prompts below rendered with the artifact's
 * chat template (thinking off):
 *  - greedy trajectories: the prompt once, then each selected token on the same
 *    live cache through that checkout's `MlxPhaseDispatch`, with the SHA-256 of every step's full
 *    last-position float32 logits;
 *  - native MTP speculation (shared batch-one gateway, depth 2, greedy, stop tokens off):
 *    the emitted tokens, every target forward's input ids (the verify rows
 *    [pending, ...proposals], accepted and rejected proposals alike) and the
 *    speculation counters.
 * The artifacts are pinned by the content of their metadata files and shards.
 * The test compares a checkout of this tree against the produced file, which is
 * pinned by its SHA-256 and never committed. */
import { createHash } from "node:crypto";
import { createReadStream, existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { observeTargetPhases } from "./reference-phases";
import { logitsEvidence, cachePositions, acceptedRounds, validateTeacherforce, validateNativeHeads, type LogitsEvidence, type TeacherforceEvidence } from "./reference-logits";
import { parseArgs } from "node:util";

const sha = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
type Loose = any; // main's modules are imported by path at run time, so they have no static types here
const load = (main: string, path: string): Promise<Loose> => import(join(main, "packages", "inference", "src", path));
async function fileSha(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

const PROMPTS = [
  "List the planets of the solar system in order from the Sun.",
  "Write a bash one-liner that counts lines in all .ts files, then explain it briefly.",
  "Explain in two sentences why the sky is blue.",
];
const TRAJECTORY_STEPS = 32;
const MTP_TOKENS = 64;
const MTP_DEPTH = 2;
const METADATA = ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"];

/** Pin an artifact by content: its metadata files and exactly the shards its
 * index names (or its one model.safetensors). */
async function pin(dir: string) {
  const files: Record<string, string> = {};
  for (const name of readdirSync(dir).sort())
    if (METADATA.includes(name) || name === "draft_vocab.json") files[name] = await fileSha(join(dir, name));
  let names = ["model.safetensors"];
  if (existsSync(join(dir,"model.safetensors.index.json"))) {
    const index = JSON.parse(readFileSync(join(dir,"model.safetensors.index.json"),"utf8")) as {weight_map:Record<string,string>};
    names = [...new Set(Object.values(index.weight_map))].sort();
  }
  const weights: Record<string, string> = {};
  for (const name of names) weights[name] = await fileSha(join(dir, name));
  return { files, weights };
}

async function produce(main: string, targetDir: string, draftDir: string) {
  const [{ loadModelConfig }, { createModel }, { Weights }, { loadTokenizer }, { ChatTemplate }, gemma, ops, ffi, { QwenMtpProvider }, { bindMlxGateway }, { MlxPhaseDispatch }] =
    await Promise.all([
      load(main, "artifacts/config.ts"), load(main, "models/factory.ts"), load(main, "artifacts/weights.ts"), load(main, "input/tokenizer.ts"),
      load(main, "input/chat-template.ts"), load(main, "kernels/logits.ts"), import(join(main,"packages/mlx/src/ops.ts")), import(join(main,"packages/mlx/src/ffi.ts")),
      load(main, "generation/speculative/sources/qwen-mtp-source.ts"), load(main, "execution/gateway-binding.ts"),
      load(main,"generation/phase-dispatch.ts"),
    ]);
  const config = await loadModelConfig(targetDir);
  const tokenizer = await loadTokenizer(targetDir), template = await ChatTemplate.load(targetDir);
  const prompts = PROMPTS.map(text => {
    const ids: number[] = tokenizer.encode(template.render([{ role: "user", content: text }], { enableThinking: false }));
    return { text, ids: ids[0] === ids[1] && ids[0] === tokenizer.bosTokenId ? ids.slice(1) : ids };
  });
  const weights = await Weights.open(targetDir);
  const model = createModel(weights, config);
  const graph = (model as Loose).constructor.name as string;
  const dispatch = new MlxPhaseDispatch(model);

  const trajectories: unknown[] = [];
  for (const prompt of prompts) {
    const cache = model.makeCache();
    const greedy: number[] = [], logits: string[] = [];
    let finite = true, tokens = prompt.ids;
    try {
      for (let step = 0; step < TRAJECTORY_STEPS; step++) {
        const ids = ops.fromInt32(tokens,[1,tokens.length]);
        let out: Loose;
        try {
          const hidden = await dispatch.prompt(tokens.length===1 ? "decode" : "tail",1,tokens.length)(ids,cache,null);
          try { out = dispatch.logits(tokens.length)(hidden); } finally { hidden.dispose(); }
        } finally { ids.dispose(); }
        const last: Float32Array = gemma.lastPositionLogits(out);
        finite &&= last.every(Number.isFinite);
        logits.push(sha(new Uint8Array(last.buffer, last.byteOffset, last.byteLength)));
        const next: number = gemma.argmaxLastPosition(out);
        out.dispose();
        greedy.push(next);
        tokens = [next];
      }
    } finally { for (const c of cache) c.dispose(); ffi.clearCache(); }
    trajectories.push({ prompt: prompt.text, promptIds: prompt.ids, greedy, logitsSha256: logits, finite, vocab: config.text.vocabSize });
    console.log(`trajectory ${JSON.stringify(prompt.text.slice(0, 40))}: ${greedy.length} steps`);
  }

  // Native MTP: record every target forward's input ids, verify rows included.
  const provider = await QwenMtpProvider.load(draftDir);
  let forwards: number[][] = [];
  let heads: LogitsEvidence[][] = [];
  const view = observeTargetPhases(model,(ids: Loose)=>forwards.push(ids.toIntTokens()), (out: Loose) => {
    const all: Float32Array = out.toFloat32(), V: number = out.shape.at(-1), L = all.length / V;
    const rows = forwards.length === 1 ? [L - 1] : Array.from({length: L}, (_, i) => i);
    heads.push(rows.map(i => logitsEvidence(all.subarray(i * V, (i + 1) * V))));
  });
  const binding = bindMlxGateway(view, { provider, numDraftTokens: MTP_DEPTH });
  const runs: unknown[] = [];
  try {
    for (const prompt of prompts) {
      forwards = []; heads = [];
      const emitted: number[] = [];
      const options = { maxTokens: MTP_TOKENS, temperature: 0 };
      const plan = binding.plan({ hasVision: false, hasAdapters: false, hasRepetitionPenalty: false, userSeed: false,
        hasLogitsExtras: false, hasGrammar: false, wantsLogprobs: false, hasDraft: true }, options,
      { checkpoints: false });
      if (plan.method !== "speculative") throw new Error("main did not bind shared speculation");
      const method = binding.methodRequest(plan, options);
      if (!method) throw new Error("main's shared MTP method is unavailable");
      const group = binding.createBatchGroup({ maxBatch: 1 });
      let stats: Loose;
      try {
        stats = await group.submit({ method, promptIds: prompt.ids, maxTokens: MTP_TOKENS, eosTokenIds: [],
          onToken(token: number) { emitted.push(token); } });
      } finally { await group.close(); }
      const spec = stats.spec as Record<string, unknown>;
      validateNativeHeads(heads, forwards, emitted, config.text.vocabSize);
      const teacherforce: TeacherforceEvidence = {plain: [], windows: [], phases: []};
      const cache = model.makeCache();
      const { rollbackVerifyBlock } = await load(main, "state/rollback.ts");
      const plain = async (tokens: number[], phase: "prompt" | "decode") => {
        const before = cachePositions(cache);
        using ids = ops.fromInt32(tokens, [1, tokens.length]);
        using hidden = await dispatch.prompt(phase === "decode" ? "decode" : "tail", 1, tokens.length)(ids, cache, null);
        using out = dispatch.logits(tokens.length)(hidden);
        teacherforce.plain.push(logitsEvidence(gemma.lastPositionLogits(out)));
        teacherforce.phases.push({phase, ids: [...tokens], before, after: cachePositions(cache)});
      };
      try {
        await plain(prompt.ids, "prompt");
        for (const round of acceptedRounds(forwards, emitted)) {
          const before = cachePositions(cache);
          for (const c of cache) c.specRoundBegin?.();
          using ids = ops.fromInt32(round.window, [1, round.window.length]);
          using hidden = await dispatch.verify(1, round.window.length)(ids, cache, null);
          using out = dispatch.logits(round.window.length)(hidden);
          ops.evalAll([out]);
          const all: Float32Array = out.toFloat32(), V = config.text.vocabSize;
          for (let j = 0; j < round.kept; j++) teacherforce.windows.push(logitsEvidence(all.subarray(j * V, (j + 1) * V)));
          const after = cachePositions(cache);
          rollbackVerifyBlock(cache, 0, round.window.length);
          teacherforce.phases.push({phase: "verify", ids: round.window, before, after, afterRollback: cachePositions(cache)});
          for (let j = 0; j < round.kept; j++) await plain([emitted[round.start + j]!], "decode");
        }
        validateTeacherforce(teacherforce, forwards, emitted, config.text.vocabSize);
      } finally { for (const c of cache) c.dispose(); ffi.clearCache(); }
      runs.push({ prompt: prompt.text, tokens: emitted, forwards, heads, teacherforce, spec: Object.fromEntries(
        ["drafted", "accepted", "rejected", "targetCalls", "rounds", "acceptanceLengths", "draftedByPos", "acceptedByPos"].map(k => [k, spec[k]])) });
      console.log(`mtp ${JSON.stringify(prompt.text.slice(0, 40))}: ${emitted.length} tokens, ${String(spec.accepted)}/${String(spec.drafted)} accepted`);
    }
  } finally {
    model.dispose();
    provider.dispose();
    weights.dispose();
    ffi.clearCache();
  }
  return { graph, runtime: { mlx: ffi.MLX_VERSION, architecture: ffi.deviceArchitecture() }, prompts, trajectories, mtp: { execution: "shared-b1", depth: MTP_DEPTH, maxTokens: MTP_TOKENS, runs } };
}

const { values } = parseArgs({ options: { help: {type:"boolean"}, main: { type: "string" }, target: { type: "string" }, draft: { type: "string" }, out: { type: "string" } } });
if (values.help) {
  console.log("bun main-qwen-trellis-reference.ts --main <isolated monorepo checkout> --target <snapshot> --draft <snapshot> --out <external.json>");
  process.exit(0);
}
for (const name of ["target", "draft", "out"] as const) if (!values[name]) throw new Error(`--${name} is required (see the file header)`);
const main = resolve(values.main ?? process.cwd());
const [target, draft, out] = [resolve(values.target!), resolve(values.draft!), resolve(values.out!)];
for (const root of [main,resolve(import.meta.dir,"../../..")]) {
  const rel = relative(root,out);
  if (!rel || (!rel.startsWith("..") && !rel.startsWith("/"))) throw new Error("reference output must be outside source checkouts");
}
if (existsSync(out)) throw new Error("reference output already exists");
const artifacts = { target: await pin(target), draft: await pin(draft) };
const reference = await produce(main, target, draft);
writeFileSync(out, JSON.stringify({ producer: "main-qwen-trellis-reference", schema: 2,
  provenance: { producerScript: "packages/inference/scripts/main-qwen-trellis-reference.ts",
    producerSha256: await fileSha(import.meta.path), phaseObserverSha256: await fileSha(join(import.meta.dir,"reference-phases.ts")),
    teacherforceSha256: await fileSha(join(import.meta.dir,"reference-logits.ts")),
    producerRevision: Bun.spawnSync(["git","-C",resolve(import.meta.dir,"../../.."),"rev-parse","HEAD"]).stdout.toString().trim(),
    producerWorktreeDirty: Bun.spawnSync(["git","-C",resolve(import.meta.dir,"../../.."),"status","--porcelain","--untracked-files=no"]).stdout.length > 0,
    bun: Bun.version },
  mainRevision: Bun.spawnSync(["git", "-C", main, "rev-parse", "HEAD"]).stdout.toString().trim(), artifacts, ...reference }, null, 1) + "\n");
console.log(`wrote ${out}`);
