/** Teacher-forced decode KL through the app's one-shot `generate` path, the serving-decode
 * method of 02d723a:scripts/eval.ts `kl --decode` (src/eval/kl.ts evaluateKlServingDecode)
 * as the Trellis checks ran it. Every gate in PLAN that accepts a numerical change quotes it.
 *
 * capture: each of the 16 default prompts is tokenized raw and cut to 256 tokens;
 * prefillLen = max(1, L - 32); the head tokens[0:prefillLen] is the request's prompt and
 * the tail tokens[prefillLen:L] is fed back one token per decode step. The loaded graph's
 * head is wrapped before the engine is built (the `logits` width table where the graph has
 * one, else `logitsFromHidden`): each call's last-position logits are copied to the host as
 * float32 (exact for bf16), and the scheduler receives a one-hot array of the same shape and
 * dtype at the teacher token, so greedy sampling emits exactly the teacher token and the next
 * step feeds it. Nothing else on the path changes. Call k projects generated token k: call 0
 * is the prompt's last position, calls 1..n the decode steps that fed tail[0..n-1]. A graph
 * whose decode projects logits elsewhere (a compiled decode step) is refused by the
 * emitted-token check. The tree is imported by path (`--root`), one prompt per `generate`.
 *
 * compare: KL(reference ‖ candidate) per token with scoring/kl.ts `klPerToken`, aggregated
 * as 02d723a:src/eval/kl.ts did: the mean of per-prompt means over decode positions
 * (calls 1..n), median and p95 over the pooled per-token values; also the max, argmax
 * agreement, position 0 apart, and the max |logit difference|. */
import { appendFileSync, existsSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";
import { parseArgs } from "node:util";
import { cellFlags, REPOSITORY } from "./digest";

// 02d723a:src/eval/kl-prompts.ts DEFAULT_KL_PROMPTS, verbatim. The method tiles them to
// n=64 (each exactly four times); every forward is deterministic, so the 16 distinct
// prompts give the same per-prompt numbers and the same mean.
export const PROMPTS = [
  "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable type with the screw press. Explain how this changed the spread of information across Europe.",
  "Write a Python function that takes a list of integers and returns the longest strictly increasing contiguous subsequence. Include a docstring and handle the empty-list case.",
  "A train leaves Chicago travelling east at 60 miles per hour. Two hours later a second train leaves the same station travelling east at 80 miles per hour. How long after the second train departs does it catch the first?",
  "Summarize the central argument of the following claim and then give one objection: economic growth and ecological sustainability are fundamentally incompatible over the long run.",
  "Describe, step by step, how a TCP connection is established and torn down, including the purpose of each flag in the three-way handshake and the role of TIME_WAIT.",
  "In the kitchen, photosynthesis of ideas rarely occurs, yet a careful cook learns to balance acid, fat, salt, and heat. Continue this passage in the same reflective tone for several sentences.",
  "Given a 4x4 matrix of integers, write pseudocode to rotate it 90 degrees clockwise in place, and explain why the in-place approach avoids allocating a second matrix.",
  "Explain the difference between supervised, unsupervised, and reinforcement learning to a curious high-school student, using a single running example for all three.",
  "Translate the following sentence into formal written English and then into a casual spoken register: 'the meeting got moved cause half the team was out sick.'",
  "What are the trade-offs between depth-first and breadth-first search for finding the shortest path in an unweighted graph? When does each fail or waste work?",
  "A baker has 3 cups of flour and each loaf needs 3/4 of a cup. After making as many loaves as possible, how much flour is left, and how many loaves were made? Show the arithmetic.",
  "Outline the plot of a short story in which a lighthouse keeper discovers that the light has been guiding something other than ships. Keep it to four sentences.",
  "Explain what a hash collision is, why it matters for hash tables, and how separate chaining and open addressing each respond to one. Be precise about average-case behavior.",
  "Compare the causes commonly cited for the fall of the Western Roman Empire, and explain why historians treat single-cause explanations with suspicion.",
  "Refactor the idea of 'technical debt' for a non-engineering executive: define it, give a concrete example, and describe one healthy and one unhealthy way teams accumulate it.",
  "Derive the formula for the sum of the first n positive integers two different ways: once by pairing terms, and once by induction. State the result clearly.",
];
export const SEQ = 256, DECODE_STEPS = 32;
const SENTINEL = "\u0000teacher-forced-prompt\u0000";

export interface PromptRecord {
  index: number; graph: string; head: number[]; tail: number[]; steps: number; prefillLen: number;
  calls: number; emitted: number; shapes: number[][]; projection: "logits" | "logitsFromHidden";
}
export interface CaptureMeta {
  root: string; commit: string | null; modelDir: string; config: string; prefillChunk: number | null;
  vocab: number; seq: number; decodeSteps: number; prompts: PromptRecord[];
}
/** A capture: its metadata and the logits, prompt after prompt, each (steps + 1) × vocab float32. */
export interface Capture { meta: CaptureMeta; logits: Float32Array }

export function readCapture(prefix: string): Capture {
  const meta = JSON.parse(readFileSync(`${prefix}.json`, "utf8")) as CaptureMeta;
  const bytes = readFileSync(`${prefix}.f32`);
  const logits = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const expected = meta.prompts.reduce((sum, p) => sum + (p.steps + 1) * meta.vocab, 0);
  if (logits.length !== expected) throw new Error(`${prefix}.f32 holds ${logits.length} values, its metadata ${expected}`);
  return { meta, logits };
}

/** KL(p ‖ q) for each of T positions of V logits. */
export type KlPerToken = (p: Float32Array, q: Float32Array, positions: number, vocab: number) => number[];

export interface KlReport {
  label: string; reference: string; candidate: string;
  /** Decode positions (calls 1..n): mean of per-prompt means, pooled median, p95, max; argmax agreement. */
  mean: number; median: number; p95: number; max: number; agree: number; positions: number;
  /** Position 0 (the prompt's last position), apart. */
  first: { mean: number; max: number; agree: number; prompts: number };
  maxLogitDiff: number;
  perPrompt: Array<{ index: number; prefillLen: number; steps: number; mean: number; first: number }>;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
/** 02d723a's percentile: the sorted value at floor(pct/100 · n), clamped. */
export const percentile = (xs: number[], pct: number) => {
  if (!xs.length) return 0;
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor((pct / 100) * sorted.length)))]!;
};
const argmax = (a: Float32Array, from: number, vocab: number) => {
  let best = from;
  for (let i = from; i < from + vocab; i++) if (a[i]! > a[best]!) best = i;
  return best - from;
};

export function compareCaptures(ref: Capture, cand: Capture, kl: KlPerToken, names: { reference: string; candidate: string; label?: string }): KlReport {
  const V = ref.meta.vocab;
  if (cand.meta.vocab !== V || cand.meta.prompts.length !== ref.meta.prompts.length) throw new Error("the captures differ in vocabulary or prompt count");
  let offset = 0, agree = 0, positions = 0, firstAgree = 0, maxLogitDiff = 0;
  const promptMeans: number[] = [], pooled: number[] = [], first: number[] = [];
  const perPrompt: KlReport["perPrompt"] = [];
  for (let p = 0; p < ref.meta.prompts.length; p++) {
    const r = ref.meta.prompts[p]!, c = cand.meta.prompts[p]!;
    if (r.index !== c.index || r.steps !== c.steps || r.head.join() !== c.head.join() || r.tail.join() !== c.tail.join())
      throw new Error(`prompt ${r.index}: the teacher tokens differ between the captures`);
    const T = r.steps + 1, size = T * V;
    const refSlice = ref.logits.subarray(offset, offset + size), candSlice = cand.logits.subarray(offset, offset + size);
    for (let i = 0; i < size; i++) { const d = Math.abs(refSlice[i]! - candSlice[i]!); if (d > maxLogitDiff) maxLogitDiff = d; }
    const values = kl(refSlice, candSlice, T, V);
    first.push(values[0]!);
    if (argmax(refSlice, 0, V) === argmax(candSlice, 0, V)) firstAgree++;
    const decode = values.slice(1);
    promptMeans.push(mean(decode));
    pooled.push(...decode);
    perPrompt.push({ index: r.index, prefillLen: r.head.length, steps: r.steps, mean: mean(decode), first: values[0]! });
    for (let t = 1; t < T; t++) { positions++; if (argmax(refSlice, t * V, V) === argmax(candSlice, t * V, V)) agree++; }
    offset += size;
  }
  return { label: names.label ?? `${ref.meta.config} -> ${cand.meta.config}`, reference: names.reference, candidate: names.candidate,
    mean: mean(promptMeans), median: percentile(pooled, 50), p95: percentile(pooled, 95), max: pooled.length ? Math.max(...pooled) : 0,
    agree, positions, first: { mean: mean(first), max: first.length ? Math.max(...first) : 0, agree: firstAgree, prompts: first.length },
    maxLogitDiff, perPrompt };
}

const fmt = (x: number) => x.toExponential(3);
/** The Trellis checks' two lines (and, with perPrompt, one line per prompt first). */
export function formatReport(r: KlReport, ref: CaptureMeta, cand: CaptureMeta, perPrompt = false): string {
  return [
    ...(perPrompt ? r.perPrompt.map(p => `    prompt ${p.index} prefill=${p.prefillLen} decode=${p.steps} meanKL=${fmt(p.mean)} pos0=${fmt(p.first)}`) : []),
    `${r.label}: ref=${basename(r.reference)} (${ref.prompts[0]?.graph ?? "?"}) cand=${basename(r.candidate)} (${cand.prompts[0]?.graph ?? "?"})`,
    `  decode KL mean=${fmt(r.mean)} median=${fmt(r.median)} p95=${fmt(r.p95)} max=${fmt(r.max)} argmax=${r.agree}/${r.positions} | ` +
      `position0 KL mean=${fmt(r.first.mean)} max=${fmt(r.first.max)} argmax=${r.first.agree}/${r.first.prompts} | max|dlogit|=${r.maxLogitDiff} | prompts=${r.perPrompt.length}`,
  ].join("\n");
}

/** scoring/kl.ts `klPerToken` over [1, T, V] float32 arrays (this checkout's MLX). */
async function mlxKl(): Promise<KlPerToken> {
  const { MlxArray } = await import("@mlx-bun/mlx/array");
  const { klPerToken } = await import("../src/scoring/kl");
  return (p, q, T, V) => {
    const pa = MlxArray.fromFloat32(p.slice(), [1, T, V]), qa = MlxArray.fromFloat32(q.slice(), [1, T, V]);
    try { return Array.from(klPerToken(pa, qa)); } finally { pa.dispose(); qa.dispose(); }
  };
}

type Loose = any; // the measured tree's modules are imported by path at run time, so they have no static types here

const within = (path: string, root: string) => {
  const rel = relative(realpathSync(root), path);
  return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
};
/** Captures stay outside the measured tree and this checkout. */
function outputPrefix(prefix: string, roots: string[]): string {
  if (!isAbsolute(prefix)) throw new Error("--out must be an absolute path prefix");
  const dir = dirname(prefix);
  if (!existsSync(dir)) throw new Error(`${dir} does not exist`);
  const real = join(realpathSync(dir), basename(prefix));
  for (const root of roots) if (within(real, root)) throw new Error(`--out ${prefix} is inside ${root}; captures stay outside every tree`);
  return real;
}

export interface CaptureOptions { root: string; model: string; out: string; cell: string; prefillChunk: number | null; prompts: number[] }

async function capture(o: CaptureOptions): Promise<number> {
  if (o.prefillChunk !== null) process.env.MLX_BUN_RD_PREFILL_CHUNK = String(o.prefillChunk);
  process.chdir(o.root);
  const flags = cellFlags(o.cell);
  const { runInference } = await import(join(o.root, "apps/mlx-bun/src/cli/inference.ts"));
  const { createAppEngine } = await import(join(o.root, "apps/mlx-bun/src/engine/index.ts"));
  const { MlxArray } = await import(join(o.root, "packages/mlx/src/array.ts"));
  const head = Bun.spawnSync(["git", "-C", o.root, "rev-parse", "HEAD"]);
  writeFileSync(`${o.out}.f32`, new Uint8Array(0));
  const records: PromptRecord[] = [];
  let vocab = 0;
  for (const index of o.prompts) {
    let graph = "?", projection: PromptRecord["projection"] = "logits", headIds: number[] = [], tail: number[] = [];
    const shapes: number[][] = [], captured: Float32Array[] = [], emitted: number[] = [];
    // The method's 32 decodes + 1 for every prompt; a shorter tail keeps its first n + 1 calls.
    const values = { ...flags, raw: true, "max-tokens": String(DECODE_STEPS + 1) };
    await runInference("generate", { positionals: [o.model, SENTINEL], values }, {
      write: () => {},
      async engine(context: Loose, scheme: Loose) {
        graph = context.model?.constructor?.name ?? "?";
        const tokens = (context.tokenizer.encode(PROMPTS[index]!, true) as number[]).slice(0, SEQ);
        const prefillLen = Math.max(1, tokens.length - DECODE_STEPS);
        headIds = tokens.slice(0, prefillLen); tail = tokens.slice(prefillLen);
        const encode = context.tokenizer.encode.bind(context.tokenizer);
        context.tokenizer.encode = (text: string, special: boolean) => text === SENTINEL ? headIds.slice() : encode(text, special);
        const forced = (project: (hidden: Loose) => Loose) => (hidden: Loose) => {
          const logits = project(hidden);
          const shape = logits.shape as number[], dtype = logits.dtype;
          shapes.push(shape);
          const k = captured.length, V = shape[shape.length - 1]!;
          const all = logits.toFloat32() as Float32Array;
          logits.dispose();
          captured.push(all.slice(all.length - V));
          const teacher = k < tail.length ? tail[k]! : tail[tail.length - 1]!;
          const onehot = new Float32Array(all.length);
          for (let at = teacher; at < onehot.length; at += V) onehot[at] = 1;
          const f32 = MlxArray.fromFloat32(onehot, shape);
          const out = f32.astype(dtype);
          f32.dispose();
          return out;
        };
        const model = context.model;
        const table = model.logits;
        if (table && typeof table === "object" && typeof table.one === "function") {
          // The width table every phase projects through (one entry per row class).
          const wrapped = Object.freeze(Object.fromEntries(Object.keys(table).map(name => [name, forced(table[name])])));
          Object.defineProperty(model, "logits", { value: wrapped, configurable: true, writable: true, enumerable: true });
        } else {
          projection = "logitsFromHidden";
          model.logitsFromHidden = forced(model.logitsFromHidden.bind(model));
        }
        const engine: Loose = await createAppEngine(context, { capacity: 1, gateway: { kvScheme: scheme } });
        const completion = engine.completion;
        engine.completion = new Proxy(completion, {
          get(target, prop) {
            if (prop === "run") return (ids: number[], options: Loose, onToken: Loose, ...rest: Loose[]) =>
              target.run(ids, options, (token: number, lp: Loose) => { emitted.push(token); return onToken(token, lp); }, ...rest);
            const value = target[prop];
            return typeof value === "function" ? value.bind(target) : value;
          },
        });
        return engine;
      },
    });
    const n = tail.length;
    if (captured.length < n + 1) throw new Error(`prompt ${index}: ${captured.length} head calls for ${n + 1} tokens`);
    if (!emitted.slice(0, n).every((token, i) => token === tail[i]))
      throw new Error(`prompt ${index}: emitted ${emitted.slice(0, n)} is not the teacher ${tail}; the decode step does not project through the ${projection} it wrapped`);
    vocab = captured[0]!.length;
    for (let k = 0; k <= n; k++) appendFileSync(`${o.out}.f32`, new Uint8Array(captured[k]!.buffer, captured[k]!.byteOffset, captured[k]!.byteLength));
    records.push({ index, graph, head: headIds, tail, steps: n, prefillLen: headIds.length, calls: captured.length, emitted: emitted.length,
      shapes: shapes.slice(0, 3), projection });
    console.log(`[${o.cell}] prompt ${index}: graph=${graph} prefill=${headIds.length} decode=${n} headCalls=${captured.length} ` +
      `firstShapes=${JSON.stringify(shapes.slice(0, 2))} forced=true`);
  }
  const meta: CaptureMeta = { root: o.root, commit: head.exitCode === 0 ? head.stdout.toString().trim() : null, modelDir: o.model,
    config: o.cell, prefillChunk: o.prefillChunk, vocab, seq: SEQ, decodeSteps: DECODE_STEPS, prompts: records };
  writeFileSync(`${o.out}.json`, JSON.stringify(meta, null, 1));
  console.log(`capture → ${o.out}.f32, ${o.out}.json (${records.length} prompts, vocabulary ${vocab})`);
  return 0;
}

export const USAGE = `Teacher-forced decode KL through the app's generate path (02d723a's serving-decode method).

  bun packages/inference/scripts/teacher-forced-kl.ts capture --model /abs/snapshot --out /abs/prefix
      [--root /abs/tree] [--cell bf16|kv-quant=8,...] [--prefill-chunk N] [--prompts 0,1,...]
      The ${PROMPTS.length} default prompts, tokenized raw and cut to ${SEQ} tokens; the last ${DECODE_STEPS} are fed one per decode
      step with the graph's head forced to the teacher token, and every step's last-position logits are written
      to PREFIX.f32 (float32, prompt after prompt, (steps + 1) x vocabulary) with PREFIX.json. --root is the tree
      to measure, imported by path (default: this checkout; run it with that tree's native library). --cell is
      digest.ts's cell syntax (default bf16). --prefill-chunk sets MLX_BUN_RD_PREFILL_CHUNK. The output stays
      outside the measured tree and this checkout.
  bun packages/inference/scripts/teacher-forced-kl.ts compare --reference PREFIX --candidate PREFIX
      [--label TEXT] [--per-prompt]
      KL(reference || candidate) per token with scoring/kl.ts klPerToken. Prints the mean of per-prompt means over
      the decode positions, the pooled median, p95 and max, argmax agreement, position 0 (the prompt's last
      position) apart, and the largest |logit difference|. Both captures must hold the same teacher tokens.`;

const positiveInteger = (name: string, raw: string) => {
  const value = Number(raw);
  if (!/^\d+$/.test(raw) || value < 1) throw new Error(`--${name}: ${raw} is not a positive integer`);
  return value;
};
export function parseCaptureArgs(argv: string[]): CaptureOptions {
  const { values } = parseArgs({ args: argv, options: { root: { type: "string", default: REPOSITORY }, model: { type: "string" },
    out: { type: "string" }, cell: { type: "string", default: "bf16" }, "prefill-chunk": { type: "string" }, prompts: { type: "string" } } });
  if (!values.model || !isAbsolute(values.model)) throw new Error("--model must be an absolute snapshot directory");
  if (!isAbsolute(values.root!)) throw new Error("--root must be an absolute path");
  if (!values.out) throw new Error("--out is required");
  cellFlags(values.cell!);
  const prompts = values.prompts === undefined ? PROMPTS.map((_, i) => i) : values.prompts.split(",").map(raw => {
    const index = Number(raw);
    if (!/^\d+$/.test(raw) || index >= PROMPTS.length) throw new Error(`--prompts: ${raw} is not a prompt index below ${PROMPTS.length}`);
    return index;
  });
  return { root: resolve(values.root!), model: resolve(values.model), out: values.out, cell: values.cell!,
    prefillChunk: values["prefill-chunk"] === undefined ? null : positiveInteger("prefill-chunk", values["prefill-chunk"]), prompts };
}

async function main(argv: string[]): Promise<number> {
  const [command, ...rest] = argv;
  if (command === "capture") {
    const options = parseCaptureArgs(rest);
    if (!existsSync(join(options.model, "config.json"))) throw new Error(`${options.model} has no config.json`);
    if (!existsSync(join(options.root, "apps/mlx-bun/src/cli/inference.ts"))) throw new Error(`${options.root} is not an mlx-bun tree`);
    return capture({ ...options, out: outputPrefix(options.out, [options.root, REPOSITORY]) });
  }
  if (command === "compare") {
    const { values } = parseArgs({ args: rest, options: { reference: { type: "string" }, candidate: { type: "string" },
      label: { type: "string" }, "per-prompt": { type: "boolean", default: false } } });
    if (!values.reference || !values.candidate) throw new Error("compare needs --reference and --candidate");
    const ref = readCapture(values.reference), cand = readCapture(values.candidate);
    const report = compareCaptures(ref, cand, await mlxKl(), { reference: values.reference, candidate: values.candidate,
      ...(values.label ? { label: values.label } : {}) });
    console.log(formatReport(report, ref.meta, cand.meta, values["per-prompt"]));
    return 0;
  }
  console.log(USAGE);
  return command === "--help" || command === "-h" || command === "help" ? 0 : 2;
}

if (import.meta.main) process.exit(await main(process.argv.slice(2)));
