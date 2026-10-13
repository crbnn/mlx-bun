// The teacher-forced KL's statistics over synthetic captures: no model. The compare
// command runs scoring/kl.ts klPerToken on MLX; the statistics are checked against a
// float64 reference KL computed here.
import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { compareCaptures, formatReport, parseCaptureArgs, percentile, PROMPTS, readCapture, type Capture, type CaptureMeta,
  type KlPerToken, type PromptRecord } from "../../scripts/teacher-forced-kl";
import { REPOSITORY } from "../../scripts/digest";

const SCRIPT = resolve(import.meta.dir, "../../scripts/teacher-forced-kl.ts");
const scratch = mkdtempSync(join(tmpdir(), "tf-kl-test-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const V = 5;
/** KL(p ‖ q) in float64 from logits. */
const referenceKl: KlPerToken = (p, q, T, vocab) => Array.from({ length: T }, (_, t) => {
  const row = (a: Float32Array) => Array.from(a.subarray(t * vocab, (t + 1) * vocab));
  const logSoftmax = (xs: number[]) => { const m = Math.max(...xs), lse = m + Math.log(xs.reduce((s, x) => s + Math.exp(x - m), 0)); return xs.map(x => x - lse); };
  const lp = logSoftmax(row(p)), lq = logSoftmax(row(q));
  return lp.reduce((s, x, i) => s + Math.exp(x) * (x - lq[i]!), 0);
});

function prompt(index: number, steps: number): PromptRecord {
  return { index, graph: "FakeGraph", head: [1, 2, index], tail: Array.from({ length: steps }, (_, i) => 10 + i), steps, prefillLen: 3,
    calls: steps + 1, emitted: steps + 1, shapes: [[1, 1, V]], projection: "logits" };
}
/** Two prompts of 3 and 2 decode steps; the candidate perturbs chosen positions. */
function captures(): { ref: Capture; cand: Capture } {
  const prompts = [prompt(0, 3), prompt(5, 2)];
  const meta = (config: string): CaptureMeta => ({ root: "/tree", commit: null, modelDir: "/m", config, prefillChunk: null, vocab: V, seq: 256,
    decodeSteps: 32, prompts });
  const positions = prompts.reduce((s, p) => s + p.steps + 1, 0);
  const ref = new Float32Array(positions * V), cand = new Float32Array(positions * V);
  for (let i = 0; i < ref.length; i++) ref[i] = cand[i] = Math.sin(i * 0.7) * 3;
  // Prompt 0: position 0 shifted (reported apart); decode position 2 flipped to another argmax.
  cand[0 * V + 1] = cand[0 * V + 1]! + 0.5;
  cand[2 * V + 4] = 9;
  // Prompt 1 (offset 4 positions): decode position 1 nudged without changing its argmax.
  cand[(4 + 1) * V + 0] = cand[(4 + 1) * V + 0]! - 0.25;
  return { ref: { meta: meta("bf16"), logits: ref }, cand: { meta: meta("kv-quant=8"), logits: cand } };
}

test("mean of per-prompt means over decode positions, pooled median, p95 and max, position 0 apart", () => {
  const { ref, cand } = captures();
  const report = compareCaptures(ref, cand, referenceKl, { reference: "/x/ref", candidate: "/x/cand" });
  const kl0 = referenceKl(ref.logits.subarray(0, 4 * V), cand.logits.subarray(0, 4 * V), 4, V);
  const kl1 = referenceKl(ref.logits.subarray(4 * V), cand.logits.subarray(4 * V), 3, V);
  const decode0 = kl0.slice(1), decode1 = kl1.slice(1), pooled = [...decode0, ...decode1];
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  expect(report.label).toBe("bf16 -> kv-quant=8");
  expect(report.mean).toBeCloseTo((mean(decode0) + mean(decode1)) / 2, 12); // not the pooled mean
  expect(report.mean).not.toBeCloseTo(mean(pooled), 6);
  expect(report.median).toBe(percentile(pooled, 50));
  expect(report.p95).toBe(percentile(pooled, 95));
  expect(report.max).toBe(Math.max(...pooled));
  expect(report.positions).toBe(5);
  expect(report.agree).toBe(4); // the flipped decode position disagrees
  expect(report.first).toEqual({ mean: (kl0[0]! + kl1[0]!) / 2, max: Math.max(kl0[0]!, kl1[0]!), agree: 2, prompts: 2 });
  expect(report.maxLogitDiff).toBeCloseTo(9 - Math.sin((2 * V + 4) * 0.7) * 3, 5);
  expect(report.perPrompt.map(p => [p.index, p.steps, p.prefillLen])).toEqual([[0, 3, 3], [5, 2, 3]]);
  const text = formatReport(report, ref.meta, cand.meta, true);
  expect(text).toContain("bf16 -> kv-quant=8: ref=ref (FakeGraph) cand=cand (FakeGraph)");
  expect(text).toContain("argmax=4/5 | position0 KL");
  expect(text).toContain("prompts=2");
  expect(text).toContain("    prompt 5 prefill=3 decode=2 meanKL=");
});

test("identical captures are zero everywhere; the percentile is 02d723a's floor index", () => {
  const { ref } = captures();
  const report = compareCaptures(ref, ref, referenceKl, { reference: "a", candidate: "b", label: "control" });
  expect([report.mean, report.median, report.p95, report.max, report.first.mean, report.maxLogitDiff]).toEqual([0, 0, 0, 0, 0, 0]);
  expect([report.agree, report.positions]).toEqual([5, 5]);
  expect(percentile([4, 1, 3, 2], 50)).toBe(3);
  expect(percentile([4, 1, 3, 2], 95)).toBe(4);
  expect(percentile([7], 95)).toBe(7);
  expect(percentile([], 50)).toBe(0);
});

test("captures that teach different tokens, or differ in shape, are refused", () => {
  const { ref, cand } = captures();
  const other = structuredClone(cand.meta);
  other.prompts[1]!.tail[0] = 99;
  expect(() => compareCaptures(ref, { meta: other, logits: cand.logits }, referenceKl, { reference: "a", candidate: "b" }))
    .toThrow("prompt 5: the teacher tokens differ");
  expect(() => compareCaptures(ref, { meta: { ...cand.meta, vocab: V + 1 }, logits: cand.logits }, referenceKl, { reference: "a", candidate: "b" }))
    .toThrow("differ in vocabulary or prompt count");
  const prefix = join(scratch, "short");
  writeFileSync(`${prefix}.json`, JSON.stringify(ref.meta));
  writeFileSync(`${prefix}.f32`, new Uint8Array(ref.logits.buffer, 0, ref.logits.byteLength - 4));
  expect(() => readCapture(prefix)).toThrow(`holds ${ref.logits.length - 1} values, its metadata ${ref.logits.length}`);
});

test("compare: scoring/kl.ts klPerToken on the written captures agrees with the float64 reference", () => {
  const { ref, cand } = captures();
  for (const [name, capture] of [["ref", ref], ["cand", cand]] as const) {
    writeFileSync(join(scratch, `${name}.json`), JSON.stringify(capture.meta));
    writeFileSync(join(scratch, `${name}.f32`), new Uint8Array(capture.logits.buffer));
  }
  expect(readCapture(join(scratch, "ref")).logits).toEqual(ref.logits);
  const run = Bun.spawnSync([process.execPath, SCRIPT, "compare", "--reference", join(scratch, "ref"), "--candidate", join(scratch, "cand"),
    "--label", "synthetic"], { env: process.env });
  expect(run.stderr.toString()).toBe("");
  expect(run.exitCode).toBe(0);
  const out = run.stdout.toString();
  const expected = compareCaptures(ref, cand, referenceKl, { reference: "ref", candidate: "cand" });
  const field = (name: string) => Number(out.match(new RegExp(`decode KL[^|]* ${name}=([0-9.e+-]+)`))![1]);
  for (const [name, value] of [["mean", expected.mean], ["median", expected.median], ["p95", expected.p95], ["max", expected.max]] as const)
    expect(Math.abs(field(name) - value)).toBeLessThanOrEqual(Math.max(1e-6, value * 2e-3));
  expect(out).toContain("synthetic: ref=ref (FakeGraph) cand=cand (FakeGraph)");
  expect(out).toContain("argmax=4/5");
}, 60_000);

test("capture arguments: the cell syntax, prompt indices and the default tree", () => {
  const options = parseCaptureArgs(["--model", "/m", "--out", "/tmp/x"]);
  expect(options).toEqual({ root: REPOSITORY, model: "/m", out: "/tmp/x", cell: "bf16", prefillChunk: null, prompts: PROMPTS.map((_, i) => i) });
  expect(parseCaptureArgs(["--model", "/m", "--out", "/tmp/x", "--cell", "kv-quant=8", "--prefill-chunk", "2", "--prompts", "0,3"]))
    .toMatchObject({ cell: "kv-quant=8", prefillChunk: 2, prompts: [0, 3] });
  expect(PROMPTS).toHaveLength(16);
  expect(() => parseCaptureArgs(["--model", "/m"])).toThrow("--out is required");
  expect(() => parseCaptureArgs(["--model", "/m", "--out", "/x", "--prompts", "16"])).toThrow("not a prompt index below 16");
  expect(() => parseCaptureArgs(["--model", "/m", "--out", "/x", "--cell", "rows=4"])).toThrow("this tool's option");
  expect(() => parseCaptureArgs(["--model", "/m", "--out", "/x", "--prefill-chunk", "0"])).toThrow("--prefill-chunk: 0");
});

test("--help documents both commands and loads no native library", () => {
  const help = Bun.spawnSync([process.execPath, SCRIPT, "--help"], { env: { ...process.env, MLX_BUN_LIBMLXC: "/nonexistent" } });
  expect(help.exitCode).toBe(0);
  for (const part of ["capture --model /abs/snapshot --out /abs/prefix", "compare --reference PREFIX --candidate PREFIX", "klPerToken",
    "cut to 256 tokens", "the last 32"]) expect(help.stdout.toString()).toContain(part);
});
