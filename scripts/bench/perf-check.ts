// Automatic performance-regression check (PLAN, Phase F) around the existing
// paired serve benchmark. bench-serve.ts measures, unchanged; this file adds
// what a CI gate needs around it: main's machine preflight before measuring
// (src/preflight.ts at 02d723a, ported), the label of the graph the candidate's
// table selects, and the decision over the saved records. `decide` is pure: it
// reads saved JSON and never touches the machine. Consumer: .github/workflows/perf.yml.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { hostname, release } from "node:os";
import { parseArgs } from "node:util";
import { CONFIGURATIONS, isContext, type Configuration } from "./plan";
import { pooledComparison, qualification, type MetricComparison, type RunRecord } from "./report";

// ---- preflight (main's src/preflight.ts, checkMachine) ------------------------------
export const PREFLIGHT_LIMITS = { maxSwapUsedMB: 512, minFreePercent: 35, bigRssMB: 2048, busyCpuPercent: 50, maxLoad1m: 8 } as const;

export interface ForeignProcess { pid: number; rssMB: number; cpuPercent: number; command: string }
export interface MachineReadings {
  /** -1 when unreadable. */
  swapUsedMB: number;
  /** memory_pressure's free percentage; -1 when unreadable. */
  freePercent: number;
  /** pmset CPU_Speed_Limit (100 = no throttle); -1 where the platform does not report it. */
  cpuSpeedLimit: number;
  /** iogpu.wired_limit_mb (0 = the macOS default). Recorded, not judged: runs under different ceilings do not compare. */
  gpuWiredLimitMB: number;
  loadAvg1m: number;
  bigProcesses: ForeignProcess[];
}
export interface MachineState extends MachineReadings { ok: boolean; problems: string[]; at: string }
export interface PreflightRecord extends MachineState {
  attempts: number; waitedMs: number;
  machine: { chip: string; memoryBytes: number; host: string; os: string };
}

/** Processes over the resident or CPU limit, from `ps -axo pid=,ppid=,pcpu=,rss=,command=`.
 * Main exempted bun and Python by name because its harness's own servers could be live; here no
 * benchmark process exists while the check runs, so a resident model server or a training run
 * blocks the measurement. Exempt: the checking process and its ancestors (the job's shell and the
 * runner), and the macOS processes main exempted (kernel_task, WindowServer, launchd, Spotlight). */
export function foreignProcesses(listing: string, ownPid: number): ForeignProcess[] {
  const rows = listing.split("\n").flatMap(line => {
    const m = line.match(/^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(.*)$/);
    return m ? [{ pid: Number(m[1]), parentPid: Number(m[2]), cpuPercent: Number(m[3]), rssMB: Number(m[4]) / 1024, command: m[5]! }] : [];
  });
  const parent = new Map(rows.map(row => [row.pid, row.parentPid]));
  const exempt = new Set<number>();
  for (let pid: number | undefined = ownPid; pid !== undefined && pid > 0 && !exempt.has(pid); pid = parent.get(pid)) exempt.add(pid);
  return rows.filter(row => !exempt.has(row.pid) && !/(kernel_task|WindowServer|launchd|mds|spotlight)/i.test(row.command) &&
    (row.rssMB >= PREFLIGHT_LIMITS.bigRssMB || row.cpuPercent >= PREFLIGHT_LIMITS.busyCpuPercent))
    .map(row => ({ pid: row.pid, rssMB: Math.round(row.rssMB), cpuPercent: row.cpuPercent, command: row.command.slice(0, 100) }));
}

/** Main's rules and thresholds over one set of readings. */
export function machineProblems(r: MachineReadings): string[] {
  const l = PREFLIGHT_LIMITS, problems: string[] = [];
  if (r.swapUsedMB < 0) problems.push("could not read vm.swapusage");
  else if (r.swapUsedMB > l.maxSwapUsedMB) problems.push(`swap in use: ${r.swapUsedMB.toFixed(0)} MB (> ${l.maxSwapUsedMB} MB); reboot before benchmarking`);
  if (r.freePercent < 0) problems.push("could not read memory_pressure");
  else if (r.freePercent < l.minFreePercent) problems.push(`free memory ${r.freePercent}% (< ${l.minFreePercent}%); close apps or reboot`);
  if (r.cpuSpeedLimit !== -1 && r.cpuSpeedLimit < 100) problems.push(`thermal throttle active: CPU speed limit ${r.cpuSpeedLimit}%`);
  if (r.loadAvg1m > l.maxLoad1m) problems.push(`1-minute load ${r.loadAvg1m.toFixed(1)} (> ${l.maxLoad1m})`);
  for (const p of r.bigProcesses) problems.push(`foreign process ${p.pid}: ${p.rssMB} MB, ${p.cpuPercent.toFixed(0)}% CPU: ${p.command}`);
  return problems;
}

const sh = (cmd: string[]) => { const r = Bun.spawnSync(cmd, { stdout: "pipe", stderr: "pipe" }); return r.exitCode === 0 ? r.stdout.toString() : ""; };

export function checkMachine(ownPid = process.pid): MachineState {
  const speed = sh(["pmset", "-g", "therm"]).match(/CPU_Speed_Limit\s*=\s*(\d+)/);
  const readings: MachineReadings = {
    swapUsedMB: Number(sh(["sysctl", "vm.swapusage"]).match(/used = ([\d.]+)M/)?.[1] ?? -1),
    freePercent: Number(sh(["memory_pressure", "-Q"]).match(/free percentage: (\d+)/)?.[1] ?? -1),
    cpuSpeedLimit: speed ? Number(speed[1]) : -1,
    gpuWiredLimitMB: Number(sh(["sysctl", "-n", "iogpu.wired_limit_mb"]).trim() || 0) || 0,
    loadAvg1m: Number(sh(["sysctl", "-n", "vm.loadavg"]).match(/([\d.]+)/)?.[1] ?? -1),
    bigProcesses: foreignProcesses(sh(["ps", "-axo", "pid=,ppid=,pcpu=,rss=,command="]), ownPid),
  };
  const problems = machineProblems(readings);
  return { ...readings, ok: problems.length === 0, problems, at: new Date().toISOString() };
}

const stateLine = (s: MachineState) => `swap ${s.swapUsedMB.toFixed(0)} MB, free ${s.freePercent}%, cpu limit ` +
  `${s.cpuSpeedLimit === -1 ? "n/a" : `${s.cpuSpeedLimit}%`}, gpu ceiling ${s.gpuWiredLimitMB ? `${s.gpuWiredLimitMB} MB (raised)` : "default"}, ` +
  `load ${s.loadAvg1m.toFixed(1)}, ${s.bigProcesses.length} foreign process(es): ${s.ok ? "clear" : "NOT CLEAR"}`;

async function preflight(out: string, waitMinutes: number): Promise<number> {
  const started = performance.now(), deadline = started + waitMinutes * 60_000;
  let attempts = 0, state: MachineState;
  for (;;) {
    state = checkMachine();
    attempts++;
    console.log(`preflight ${attempts}: ${stateLine(state)}`);
    for (const problem of state.problems) console.log(`  ${problem}`);
    if (state.ok || performance.now() + 60_000 > deadline) break;
    await Bun.sleep(60_000);
  }
  const memoryBytes = Number(sh(["sysctl", "-n", "hw.memsize"]).trim());
  const record: PreflightRecord = { ...state, attempts, waitedMs: Math.round(performance.now() - started),
    machine: { chip: sh(["sysctl", "-n", "machdep.cpu.brand_string"]).trim(), memoryBytes, host: hostname(), os: release() } };
  writeFileSync(out, JSON.stringify(record, null, 1));
  return state.ok ? 0 : 1;
}

// ---- the candidate's graph label ------------------------------------------------------
async function graphLabel(modelDir: string): Promise<string> {
  const { loadModelConfig } = await import("@mlx-bun/inference/artifacts");
  const { familyOf, MLX_GRAPH_TABLES, resolveComposition } = await import("@mlx-bun/inference/models");
  const config = await loadModelConfig(modelDir);
  const family = familyOf(config);
  if (!family) throw new Error(`no model family serves ${modelDir}`);
  const table = MLX_GRAPH_TABLES[family.graph];
  return table ? table.select(config, resolveComposition(config)).id : family.graph;
}

// ---- the decision -----------------------------------------------------------------------
/** The report's paired request metrics that gate (PLAN: prefill, decode, TTFT). */
export const GATED = ["decode", "prefill (nominal 1k prompt)", "TTFT cold (nominal 1k prompt)"] as const;
/** Shown beside them, never gated: one pair per session. Memory is the worker's MLX allocator and the
 * physical footprint, never `ps` RSS (measure.ts `MemoryReading` says why). */
export const CONTEXT = ["prefill @ctx", "TTFT @ctx", "aggregate ×4", "peak MLX active", "peak physical footprint"] as const;

/** Two-sided 95% Student-t quantiles for 1..30 degrees of freedom; above 30 the df-30 value (wider, never narrower). */
const T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
  2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042];
export const t975 = (df: number): number => T975[Math.min(Math.max(df, 1), T975.length) - 1]!;

export type Verdict = "regression" | "improvement" | "within noise" | "one pair, not gated";
export interface PairedDelta {
  metric: string; unit: string; direction: MetricComparison["direction"]; gated: boolean; pairs: number;
  baselineMedian: number | null; candidateMedian: number | null;
  /** Geometric mean of the paired candidate/baseline ratios. */
  ratio: number | null;
  /** 95% Student-t interval of that ratio, computed on the log ratios; null below two pairs. */
  interval: [number, number] | null;
  verdict: Verdict | null;
  /** Each session's own medians over its pairs (one entry when the plan runs one session). */
  sessions?: Array<{ session: number; baselineMedian: number | null; candidateMedian: number | null }>;
}

/** The paired delta of one report metric. Throws on a non-positive sample: a complete run has none. */
export function pairedDelta(c: MetricComparison, gated: boolean): PairedDelta {
  const base = { metric: c.metric, unit: c.unit, direction: c.direction, gated, pairs: c.pairs.length,
    baselineMedian: c.baselineMedian, candidateMedian: c.candidateMedian };
  if (!c.pairs.length) return { ...base, ratio: null, interval: null, verdict: null };
  const logs = c.pairs.map(p => Math.log(p.candidate / p.baseline));
  if (logs.some(x => !Number.isFinite(x))) throw new Error(`${c.metric}: a paired sample is not a positive number`);
  const n = logs.length, mean = logs.reduce((a, x) => a + x, 0) / n;
  if (n < 2) return { ...base, ratio: Math.exp(mean), interval: null, verdict: "one pair, not gated" };
  const sd = Math.sqrt(logs.reduce((a, x) => a + (x - mean) ** 2, 0) / (n - 1)), half = t975(n - 1) * sd / Math.sqrt(n);
  const interval: [number, number] = [Math.exp(mean - half), Math.exp(mean + half)];
  const worse = c.direction === "higher-is-better" ? interval[1] < 1 : interval[0] > 1;
  const better = c.direction === "higher-is-better" ? interval[0] > 1 : interval[1] < 1;
  return { ...base, ratio: Math.exp(mean), interval, verdict: worse ? "regression" : better ? "improvement" : "within noise" };
}

export interface Composition { model: string; configuration: Configuration; label: string; deltas: PairedDelta[] }
export interface Decision {
  outcome: "pass" | "regression" | "no measurement";
  headline: string;
  /** Why there are no numbers (preflight problems or the run's qualification problems). */
  reasons: string[];
  notes: string[];
  /** Every applicable model and configuration of the plan, in plan order. */
  compositions: Composition[];
}
export interface DecideInput { before: PreflightRecord | null; after: PreflightRecord | null; run: RunRecord | null }

export function decide({ before, after, run }: DecideInput): Decision {
  const none = (headline: string, reasons: string[]): Decision => ({ outcome: "no measurement", headline, reasons, notes: [], compositions: [] });
  if (!before) return none("no preflight record, no numbers", ["the job stopped before the preflight ran"]);
  if (!before.ok) return none("preflight failed, no numbers", before.problems);
  if (!run) return none("no run record, no numbers", ["the preflight was clear, but bench-serve saved no run record"]);
  const q = qualification(run);
  if (!q.complete) return none("run incomplete, no numbers", q.problems);
  const notes = [after ? after.ok ? "" : `machine not clear after the run: ${after.problems.join("; ")}` : "no preflight after the run"].filter(Boolean);
  // What qualification reported, not required, about a context configuration's cells and pairs.
  const contexts = run.plan.configurations.filter(isContext);
  notes.push(...q.notes.filter(note => !note.includes(": not applicable") && !note.includes("not observed mapped") &&
    contexts.some(name => note.includes(`/${name}/`) || note.includes(`/${name} `))).map(note => `context: ${note}`));
  const reasons: string[] = [], compositions: Composition[] = [];
  for (const model of run.plan.models) for (const configuration of run.plan.configurations) {
    const cells = run.cells.filter(cell => cell.kind === "tree" && cell.model === model.id && cell.configuration === configuration);
    if (cells.length && cells.every(cell => cell.skipped)) continue; // not applicable, as the plan predeclared
    // The trees ran different settings: every metric is shown beside the gate, none in it, and a missing
    // measurement is a note.
    const context = isContext(configuration);
    const missing = context ? notes : reasons;
    const comparison = pooledComparison(run, model.id, configuration);
    if (!comparison) { missing.push(`${model.id}/${configuration}: no paired measurement`); continue; }
    const deltas = [...GATED, ...CONTEXT].flatMap(name => {
      const metric = comparison.pooled.find(c => c.metric === name);
      if (!metric) { missing.push(`${model.id}/${configuration}: no ${name} comparison`); return []; }
      try {
        return [{ ...pairedDelta(metric, !context && (GATED as readonly string[]).includes(name)),
          sessions: comparison.sessions.map(({ session, metrics }) => {
            const own = metrics.find(c => c.metric === name);
            return { session, baselineMedian: own?.baselineMedian ?? null, candidateMedian: own?.candidateMedian ?? null };
          }) }];
      } catch (error) { missing.push(`${model.id}/${configuration}: ${(error as Error).message}`); return []; }
    });
    compositions.push({ model: model.id, configuration, label: model.label, deltas });
  }
  if (reasons.length) return none("run incomplete, no numbers", reasons);
  const regressions = compositions.flatMap(c => c.deltas.filter(d => d.gated && d.verdict === "regression")
    .map(d => `${c.model}/${c.configuration} ${d.metric}`));
  return regressions.length
    ? { outcome: "regression", headline: `regression outside the paired noise band: ${regressions.join(", ")}`, reasons, notes, compositions }
    : { outcome: "pass", headline: "no regression outside the paired noise band", reasons, notes, compositions };
}

const num = (x: number | null, digits: number) => x === null ? "—" : x.toFixed(digits);
const digitsFor = (unit: string) => unit === "ms" || unit === "MB" ? 0 : 1;
const machineLine = (m: { chip: string; memoryBytes: number; host: string }) => `${m.chip} · ${(m.memoryBytes / 2 ** 30).toFixed(0)} GB · ${m.host}`;
const preflightLine = (label: string, p: PreflightRecord | null) => !p ? `- preflight ${label}: not run` :
  `- preflight ${label}: ${p.ok ? "clear" : "NOT CLEAR"} at ${p.at} (${p.attempts} check${p.attempts === 1 ? "" : "s"} over ` +
  `${Math.round(p.waitedMs / 60_000)} min); swap ${p.swapUsedMB.toFixed(0)} MB, free ${p.freePercent}%, load ${p.loadAvg1m.toFixed(1)}`;
/** What each tree of a configuration ran: the shared flags, then a tree's own (a context configuration's). */
const treeFlags = (name: Configuration) => {
  const spec = CONFIGURATIONS[name];
  return "treeArgs" in spec ? `; baseline ${spec.treeArgs.baseline.join(" ")}, candidate ${spec.treeArgs.candidate.join(" ")}` : "";
};
const configurationLine = (name: Configuration) => `${name} (${[...CONFIGURATIONS[name].args, ...("dflash2" in CONFIGURATIONS[name] ? ["--draft-model <DFlash 2 drafter>"] : []),
  ...("draftDepth" in CONFIGURATIONS[name] ? [`--draft-kind mtp --num-draft-tokens ${CONFIGURATIONS[name].draftDepth}`] : [])].join(" ") || "CLI defaults"}` +
  `${treeFlags(name)})`;

/** The job summary: headline, provenance, and per model and configuration the paired deltas with their intervals. */
export function summary(d: Decision, input: DecideInput): string {
  const { before, after, run } = input, lines = [`## Performance check: ${d.headline}`, ""];
  const machine = run?.machine ?? before?.machine;
  const sessions = run?.plan.sessions ?? 1;
  if (machine) lines.push(`machine: ${machineLine(machine)}`);
  if (run) lines.push(`baseline: ${run.plan.trees.baseline.commit} · candidate: ${run.plan.trees.candidate.commit}`,
    `run: ${run.startedAt} to ${run.finishedAt ?? "unfinished"}; configurations ${run.plan.configurations.map(configurationLine).join(", ")}; ` +
    `${sessions} server session${sessions === 1 ? "" : "s"} per tree, alternating trees; per session ${run.plan.workload.decodeRuns} decode samples of ` +
    `${run.plan.workload.decodeTokens} tokens, ${run.plan.workload.ttftRuns} cold TTFT samples (nominal 1k prompt)`);
  lines.push("", preflightLine("before", before), preflightLine("after", after), "");
  if (d.outcome === "no measurement") {
    lines.push("No numbers are quoted:", ...d.reasons.map(reason => `- ${reason}`), "");
    return lines.join("\n");
  }
  lines.push("Paired delta: geometric mean of the per-request candidate/baseline ratios, each session paired with its counterpart and " +
    "the pairs pooled across sessions. Interval: 95% Student-t, on the log ratios; " + (sessions > 1
      ? "the sessions alternate trees, so it covers session-to-session drift as well as request noise. "
      : "with one session per tree it covers request-to-request noise, not drift between sessions. ") +
    "A gated metric fails when its whole interval is on the worse side of 1. Memory: the worker's MLX allocator from /stats and the " +
    "physical footprint, never ps RSS.", "");
  for (const note of d.notes) lines.push(`- note: ${note}`);
  if (d.notes.length) lines.push("");
  for (const c of d.compositions) {
    lines.push(`### ${c.model} · ${c.configuration} · ${c.label}`, "");
    if (isContext(c.configuration))
      lines.push(`Context, not gated: the trees ran different settings (${treeFlags(c.configuration).slice(2)}), so no metric here ` +
        "decides the check.", "");
    lines.push(
      "| metric | unit | baseline median | candidate median | candidate/baseline | 95% interval | pairs | per-session medians | verdict |",
      "|---|---|---|---|---|---|---|---|---|");
    for (const x of c.deltas) {
      const digits = digitsFor(x.unit), arrow = x.direction === "higher-is-better" ? "↑" : "↓";
      const verdict = !x.gated ? "context, not gated" : x.verdict === "regression" ? "**regression**" : x.verdict ?? "—";
      const perSession = (x.sessions ?? []).map(s => `s${s.session} ${num(s.baselineMedian, digits)} → ${num(s.candidateMedian, digits)}`).join("; ") || "—";
      lines.push(`| ${x.metric} | ${x.unit} ${arrow} | ${num(x.baselineMedian, digits)} | ${num(x.candidateMedian, digits)} | ${num(x.ratio, 3)} | ` +
        `${x.interval ? `[${x.interval[0].toFixed(3)}, ${x.interval[1].toFixed(3)}]` : "—"} | ${x.pairs} | ${perSession} | ${verdict} |`);
    }
    lines.push("");
  }
  lines.push("Full paired report: report.md in the run artifact.", "");
  return lines.join("\n");
}

export const EXIT = { pass: 0, regression: 1, "no measurement": 2 } as const;

const USAGE = `Performance-regression check around the paired serve benchmark (scripts/bench-serve.ts).

  bun scripts/bench/perf-check.ts preflight --out /abs/preflight.json [--wait-minutes N]
      Main's machine preflight: swap, free memory, thermal limit, 1-minute load, and any foreign
      process over 2048 MB resident or 50% CPU (a resident model server or a training run, bun or
      Python, counts). Rechecks every minute for up to N minutes (default 0). Exits 0 when clear.
  bun scripts/bench/perf-check.ts graph --model /abs/snapshot
      Prints the graph this tree's table selects for the artifact on this machine under the default
      serve composition (bf16 KV, no drafter, 8 rows). Needs MLX_BUN_LIBMLXC or staged native files.
  bun scripts/bench/perf-check.ts decide --preflight-before FILE [--preflight-after FILE] [--run FILE]
      Prints the markdown summary for a saved bench-serve run, one table per applicable model and
      configuration, and exits 0 (no regression), 1 (a regression outside the paired noise band) or 2
      (no qualified measurement: preflight failed, or the run is missing or incomplete; no numbers
      are quoted).

Decision: numbers count only when the preflight before the run was clear and bench-serve's own
qualification says the run is complete (every phase paired, decode stable, provenance verified,
probes equal). For each model and configuration, ${GATED.map(name => `"${name}"`).join(", ")} gate.
Each session pairs with its counterpart (bench-serve --sessions N alternates trees, A B A B) and the
pairs pool across sessions; the paired delta is the geometric mean of the per-request
candidate/baseline ratios and its interval the 95% Student-t interval of the mean log ratio, so with
several sessions it covers session-to-session drift. A metric regresses when the whole interval is
on the worse side of 1. Per-session medians are shown beside the pooled ones. One-pair-per-session
metrics (${CONTEXT.join(", ")}) are shown, never gated; memory is the worker's MLX allocator from
/stats and the physical footprint, not ps RSS. A context configuration, whose trees run different
settings (dflash2-d7-baseline: the baseline at depth 7, the candidate adaptive), is shown and never
gates; its probes and request pairing are notes. The preflight after the run is recorded and shown;
as on main, the one before the run gates.`;

const readJson = <T>(path: string | undefined): T | null => path && existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) as T : null;

async function main(argv: string[]): Promise<number> {
  const [command, ...rest] = argv;
  if (command === "preflight") {
    const { values } = parseArgs({ args: rest, options: { out: { type: "string" }, "wait-minutes": { type: "string", default: "0" } } });
    const wait = Number(values["wait-minutes"]);
    if (!values.out || !Number.isFinite(wait) || wait < 0) throw new Error("preflight needs --out FILE and a nonnegative --wait-minutes");
    return preflight(values.out, wait);
  }
  if (command === "graph") {
    const { values } = parseArgs({ args: rest, options: { model: { type: "string" } } });
    if (!values.model) throw new Error("graph needs --model DIR");
    console.log(await graphLabel(values.model));
    return 0;
  }
  if (command === "decide") {
    const { values } = parseArgs({ args: rest, options: { "preflight-before": { type: "string" }, "preflight-after": { type: "string" },
      run: { type: "string" } } });
    const input: DecideInput = { before: readJson(values["preflight-before"]), after: readJson(values["preflight-after"]), run: readJson(values.run) };
    const decision = decide(input);
    console.log(summary(decision, input));
    return EXIT[decision.outcome];
  }
  console.log(USAGE);
  return command === "--help" || command === "help" || command === "-h" ? 0 : 2;
}

if (import.meta.main) process.exit(await main(process.argv.slice(2)));
