import { expect, test } from "bun:test";
import { resolve } from "node:path";
import type { CellResult, RawRequest, ReqResult } from "./measure";
import { decide, EXIT, foreignProcesses, machineProblems, pairedDelta, summary, t975, type DecideInput, type MachineReadings,
  type PreflightRecord } from "./perf-check";
import { planCells, REQUIRED_PHASES, WORKLOAD, type Plan, type Tree } from "./plan";
import { qualification, type CellRecord, type MetricComparison, type RunRecord } from "./report";

// ---- a complete saved run, built by hand: no servers -------------------------------------
const LIB = { name: "libmlxc.dylib", path: "/native/libmlxc.dylib", bytes: 1, sha256: "a".repeat(64) };
/** Final-attempt requests per phase that qualification expects (main's workload). */
const REQUESTS = { warmup: 1, parity: 1, decode: WORKLOAD.decodeRuns, ttft1k: WORKLOAD.ttftRuns + 1, ctx: 3, restart: 1, agg: WORKLOAD.aggregateStreams };
type Samples = { decode: number[]; ttft: number[] };
const SAME: Samples = { decode: [30, 30.4, 29.8, 30.1, 30.2], ttft: [400, 404, 398] };

function plan(): Plan {
  const model = (id: string) => ({ id, label: `${id} artifact, candidate graph ${id}-graph`, path: `/models/${id}`, modelType: "fake",
    kvConfig: false, packedTrellis: false, files: [{ name: "model.safetensors", bytes: 1, sha256: "d".repeat(64) }] });
  const tree = (root: string, commit: string) => ({ root, commit: commit.repeat(40), command: ["bun", `${root}/apps/mlx-bun/bin/mlx-bun.mjs`, "serve"] });
  const p: Plan = { schema: 1, profile: "scoped", seed: "s", workload: { ...WORKLOAD, withContext: true },
    trees: { baseline: tree("/base", "b"), candidate: tree("/head", "c") }, references: [], configurations: ["default"],
    models: [model("qwen27b"), model("e4b")], native: { library: LIB.path, files: [LIB] }, cells: [] };
  return { ...p, cells: planCells(p) };
}

function cellResult(key: string, samples: Samples): CellResult {
  const probe = { text: "2, 3, 5, 7, 11, 13, 17, 19", promptTokens: 12, completionTokens: 24, finishReason: "length" };
  return { key, readyMs: 900, idleRssMB: 4000, parity: { completion: probe, chat: probe }, peakRssMB: 9000, rssByLeg: [], coldStartMs: 1200,
    restart: { readyMs: 900, ctxTtftMs: 300, cachedTokens: 16000, durability: null },
    decodeTps: samples.decode, decodeSpread: 1, decodeStable: true,
    ttft: { coldMs: samples.ttft, prefill1kTps: samples.ttft.map(ms => 1100_000 / ms), warmMs: 40, warmCachedTokens: 1090 },
    ctx: { promptTokens: 16000, prefillTps: 400, ttftMs: 40_000, decodeTps: [25, 25, 25], cachedRepeatTtftMs: 60 },
    agg: { tps: 90, perStream: 24, wallMs: 5600, arrivalMs: [0, 0, 0, 0], ttftMs: [90, 95, 99, 101], promptTokens: [20, 20, 20, 20],
      cachedTokens: [0, 0, 0, 0] },
    phaseFailures: [], measured: [...REQUIRED_PHASES], loadedLibraries: [{ path: LIB.path, sha256: LIB.sha256 }], processes: [] };
}

function requests(cell: string, samples: Samples): RawRequest[] {
  return Object.entries(REQUESTS).flatMap(([phase, count]) => Array.from({ length: count }, (_, index) => {
    const prompt = phase === "ttft1k" ? 1100 : phase === "ctx" || phase === "restart" ? 16000 : 20;
    const ttftMs = phase === "ttft1k" ? samples.ttft[index] ?? 40 : phase === "ctx" && index === 0 ? 40_000 : 90;
    const result = { decodeTps: phase === "decode" ? samples.decode[index]! : 25, ttftMs, wallMs: 7000, promptTokens: prompt,
      genTokens: 8, cachedTokens: 0, finishReason: "length" } as ReqResult;
    return { cell, phase, attempt: 0, index, requestSha256: `${phase}-${index}`, request: { content: "", maxTokens: 8, bodyExtra: {} }, result };
  }));
}

/** A complete run; `candidate` overrides the candidate's samples for one model. */
function runRecord(candidate: Partial<Record<string, Samples>> = {}): RunRecord {
  const p = plan(), samplesFor = (model: string, tree: Tree) => tree === "candidate" ? candidate[model] ?? SAME : SAME;
  const cells: CellRecord[] = p.cells.map(c => ({ key: c.key, model: c.model, kind: "tree", tree: c.tree!, configuration: c.configuration!,
    order: c.order, command: [], intendedLibrary: LIB.path, result: cellResult(c.key, samplesFor(c.model, c.tree!)) }));
  const pin = { root: "/head", head: "c".repeat(40), clean: true, sha256: "t".repeat(64) };
  return { schema: 1, tool: { start: pin, end: pin }, plan: p,
    machine: { chip: "Apple M1 Max", memoryBytes: 32 * 2 ** 30, loadAverage: "{ 1.0 1.0 1.0 }", host: "m1max", os: "27.0.1", bun: "1.4.2" },
    startedAt: "2026-10-11T10:00:00.000Z", finishedAt: "2026-10-11T10:40:00.000Z", pins: { start: [], end: [] }, cells,
    requests: cells.flatMap(c => requests(c.key, samplesFor(c.model, c.tree!))) };
}

const readings: MachineReadings = { swapUsedMB: 0, freePercent: 60, cpuSpeedLimit: -1, gpuWiredLimitMB: 0, loadAvg1m: 1.2, bigProcesses: [] };
function preflight(problems: string[] = []): PreflightRecord {
  return { ...readings, ok: problems.length === 0, problems, at: "2026-10-11T09:59:00.000Z", attempts: 1, waitedMs: 0,
    machine: { chip: "Apple M1 Max", memoryBytes: 32 * 2 ** 30, host: "m1max", os: "27.0.1" } };
}
const input = (run: RunRecord | null, before: PreflightRecord | null = preflight(), after: PreflightRecord | null = preflight()): DecideInput =>
  ({ before, after, run, configuration: "default" });

test("the synthetic run is complete by bench-serve's own qualification", () => {
  expect(qualification(runRecord()).problems).toEqual([]);
});

// ---- the paired delta and its interval -----------------------------------------------------
const comparison = (direction: MetricComparison["direction"], baseline: number[], candidate: number[]): MetricComparison =>
  ({ metric: "m", unit: direction === "higher-is-better" ? "tok/s" : "ms", direction, baselineMedian: null, candidateMedian: null,
    ratio: null, observation: "equal", unpaired: { baseline: [], candidate: [] }, excluded: [],
    pairs: baseline.map((b, i) => ({ id: String(i), baseline: b, candidate: candidate[i]! })) });

test("the interval is the 95% Student-t interval of the mean log ratio around the geometric-mean ratio", () => {
  expect(t975(2)).toBe(4.303);
  expect(t975(4)).toBe(2.776);
  expect(t975(400)).toBe(2.042); // above the table: the df-30 value, never narrower
  const d = pairedDelta(comparison("higher-is-better", [100, 100, 100], [90, 92, 88]), true);
  const logs = [0.9, 0.92, 0.88].map(Math.log), mean = logs.reduce((a, x) => a + x) / 3;
  const sd = Math.sqrt(logs.reduce((a, x) => a + (x - mean) ** 2, 0) / 2), half = 4.303 * sd / Math.sqrt(3);
  expect(d.ratio).toBeCloseTo(Math.exp(mean), 12);
  expect(d.interval![0]).toBeCloseTo(Math.exp(mean - half), 12);
  expect(d.interval![1]).toBeCloseTo(Math.exp(mean + half), 12);
  expect(d.verdict).toBe("regression");
});

test("a regression needs the whole interval on the worse side; direction decides which side", () => {
  // Noisy pairs straddling 1: within noise, even with a worse mean.
  expect(pairedDelta(comparison("higher-is-better", [100, 100, 100], [80, 101, 115]), true).verdict).toBe("within noise");
  expect(pairedDelta(comparison("higher-is-better", [100, 100, 100, 100, 100], [104, 105, 103, 106, 104]), true).verdict).toBe("improvement");
  // Lower is better: a consistently longer TTFT regresses, a shorter one improves.
  expect(pairedDelta(comparison("lower-is-better", [400, 400, 400], [440, 445, 438]), true).verdict).toBe("regression");
  expect(pairedDelta(comparison("lower-is-better", [400, 400, 400], [380, 378, 383]), true).verdict).toBe("improvement");
  // One pair has no interval; no pairs has no ratio; a non-positive sample is refused.
  expect(pairedDelta(comparison("higher-is-better", [100], [50]), false)).toMatchObject({ ratio: 0.5, interval: null, verdict: "one pair, not gated" });
  expect(pairedDelta(comparison("higher-is-better", [], []), true)).toMatchObject({ ratio: null, verdict: null });
  expect(() => pairedDelta(comparison("higher-is-better", [100, 100], [0, 100]), true)).toThrow("not a positive number");
});

// ---- the decision over saved records --------------------------------------------------------
test("a complete run within noise passes and the summary quotes every gated delta with its interval", () => {
  const run = runRecord({ qwen27b: { decode: [30.3, 29.9, 30.0, 30.4, 29.7], ttft: [401, 399, 405] } });
  const d = decide(input(run));
  expect(d).toMatchObject({ outcome: "pass", headline: "no regression outside the paired noise band" });
  expect(EXIT[d.outcome]).toBe(0);
  const qwen = d.compositions.find(c => c.model === "qwen27b")!;
  expect(qwen.label).toBe("qwen27b artifact, candidate graph qwen27b-graph");
  expect(qwen.deltas.filter(x => x.gated).map(x => [x.metric, x.pairs])).toEqual([["decode", 5], ["prefill (nominal 1k prompt)", 3],
    ["TTFT cold (nominal 1k prompt)", 3]]);
  expect(qwen.deltas.filter(x => !x.gated).every(x => x.pairs === 1 && x.interval === null)).toBe(true);
  const text = summary(d, input(run));
  expect(text).toContain("## Performance check: no regression outside the paired noise band");
  expect(text).toContain("### qwen27b · qwen27b artifact, candidate graph qwen27b-graph");
  expect(text).toMatch(/\| decode \| tok\/s ↑ \| 30\.1 \| 30\.0 \| 0\.99\d \| \[0\.9\d\d, 1\.0\d\d\] \| 5 \| within noise \|/);
  expect(text).toContain("| prefill @ctx | tok/s ↑ | 400.0 | 400.0 | 1.000 | — | 1 | context, not gated |");
  expect(text).toContain(`baseline: ${"b".repeat(40)} · candidate: ${"c".repeat(40)}`);
});

test("a consistent slowdown on one composition fails with the regressed metrics named", () => {
  const slower = { decode: SAME.decode.map(x => x * 0.93), ttft: SAME.ttft.map(x => x * 1.08) };
  const d = decide(input(runRecord({ e4b: slower })));
  expect(d.outcome).toBe("regression");
  expect(EXIT[d.outcome]).toBe(1);
  expect(d.headline).toBe("regression outside the paired noise band: e4b decode, e4b prefill (nominal 1k prompt), e4b TTFT cold (nominal 1k prompt)");
  expect(summary(d, input(runRecord({ e4b: slower })))).toContain("| **regression** |");
});

test("a failed preflight, a missing or incomplete run quote no numbers", () => {
  const busy = decide(input(runRecord(), preflight(["swap in use: 3000 MB (> 512 MB); reboot before benchmarking"]), null));
  expect(busy).toMatchObject({ outcome: "no measurement", headline: "preflight failed, no numbers" });
  expect(EXIT[busy.outcome]).toBe(2);
  const text = summary(busy, input(null, preflight(["swap in use: 3000 MB (> 512 MB); reboot before benchmarking"]), null));
  expect(text).toContain("## Performance check: preflight failed, no numbers");
  expect(text).toContain("- swap in use: 3000 MB");
  expect(text).not.toContain("tok/s");
  expect(decide(input(null, null, null)).headline).toBe("no preflight record, no numbers");
  expect(decide(input(null)).headline).toBe("no run record, no numbers");
  // bench-serve's qualification decides completeness: a lost decode request leaves the run incomplete.
  const incomplete = runRecord();
  incomplete.requests = incomplete.requests.filter(q => !(q.cell === "e4b/default/candidate" && q.phase === "decode" && q.index === 4));
  const d = decide(input(incomplete));
  expect(d.headline).toBe("run incomplete, no numbers");
  expect(d.reasons.join("\n")).toContain("e4b/default/candidate: decode has 4 successful");
  expect(summary(d, input(incomplete))).not.toContain("| decode |");
});

test("the preflight after the run is shown, and as on main the one before gates", () => {
  const d = decide(input(runRecord(), preflight(), preflight(["1-minute load 9.5 (> 8)"])));
  expect(d.outcome).toBe("pass");
  expect(d.notes).toEqual(["machine not clear after the run: 1-minute load 9.5 (> 8)"]);
  expect(summary(d, input(runRecord(), preflight(), preflight(["1-minute load 9.5 (> 8)"])))).toContain("- preflight after: NOT CLEAR");
});

// ---- the preflight's rules ----------------------------------------------------------------------
test("main's preflight thresholds", () => {
  expect(machineProblems(readings)).toEqual([]);
  expect(machineProblems({ ...readings, swapUsedMB: 513, freePercent: 34, cpuSpeedLimit: 80, loadAvg1m: 8.5 })).toEqual([
    "swap in use: 513 MB (> 512 MB); reboot before benchmarking", "free memory 34% (< 35%); close apps or reboot",
    "thermal throttle active: CPU speed limit 80%", "1-minute load 8.5 (> 8)"]);
  expect(machineProblems({ ...readings, swapUsedMB: -1, freePercent: -1 })).toEqual(["could not read vm.swapusage", "could not read memory_pressure"]);
});

test("a resident model server or a training run is foreign, bun or Python; the check's own ancestry is not", () => {
  const listing = [
    "    1     0   0.0    20000 /sbin/launchd",
    "  500     1   1.0   200000 /Users/runner/actions-runner/bin/Runner.Listener run",
    "  600   500  60.0   300000 /Users/runner/actions-runner/bin/Runner.Worker spawnclient",
    "  700   600   0.0     4000 /bin/bash -e /tmp/step.sh",
    "  800   700  95.0    60000 /Users/runner/.bun/bin/bun scripts/bench/perf-check.ts preflight",
    "  900     1   5.0 12000000 /Users/josh/.bun/bin/bun /Users/josh/mlx-bun/apps/mlx-bun/bin/mlx-bun.mjs serve",
    "  901     1  99.0   900000 /Users/josh/.venv/bin/python -m mlx_lm.lora --train",
    "  902     1  10.0  3000000 /Applications/Chrome.app/Contents/MacOS/Chrome",
    "  903     1   3.0   500000 /Applications/Slack.app/Contents/MacOS/Slack",
    "  904     0  70.0  4000000 kernel_task",
    "  905     1  80.0  3000000 /System/Library/Frameworks/CoreServices.framework/mds_stores",
  ].join("\n");
  expect(foreignProcesses(listing, 800).map(p => [p.pid, p.rssMB])).toEqual([[900, 11719], [901, 879], [902, 2930]]);
  expect(machineProblems({ ...readings, bigProcesses: foreignProcesses(listing, 800) })[1])
    .toBe("foreign process 901: 879 MB, 99% CPU: /Users/josh/.venv/bin/python -m mlx_lm.lora --train");
});

test("--help documents every subcommand and exits 0", () => {
  const help = Bun.spawnSync([process.execPath, resolve(import.meta.dir, "perf-check.ts"), "--help"]);
  expect(help.exitCode).toBe(0);
  for (const part of ["preflight --out", "graph --model", "decide --preflight-before", "95% Student-t", "TTFT cold (nominal 1k prompt)"])
    expect(help.stdout.toString()).toContain(part);
});
