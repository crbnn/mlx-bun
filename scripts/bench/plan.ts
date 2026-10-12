// Paired serve benchmark plan: which trees, reference servers and artifacts
// are compared, pinned before any server starts and verified again at the end.
// Metadata only: nothing here loads MLX, models or Python.
import { createHash } from "node:crypto";
import { closeSync, existsSync, openSync, readdirSync, readFileSync, readSync, realpathSync, statSync } from "node:fs";
import { basename, isAbsolute, join, relative, resolve } from "node:path";

export type Tree = "baseline" | "candidate";
export const TREES: readonly Tree[] = ["baseline", "candidate"];

/** Server configurations measured on both trees, as main's matrix measured its own arms. */
export const CONFIGURATIONS = {
  default: { args: [] as string[], note: "real CLI defaults" },
  /** Capacity-1 control: main's `--batch 1` serial path versus the candidate's
   * continuous scheduler at capacity 1. Not a candidate serial lane. */
  serial: { args: ["--batch", "1"], note: "capacity-1 control (baseline serial, candidate continuous at capacity 1)" },
  mixed: { args: ["--kv-quant", "config"], note: "artifact kv_config.json", requiresKvConfig: true },
  /** Native MTP companion (`--draft ID=PATH`) at a fixed verify depth; models without one are not applicable. */
  mtp2: { args: [] as string[], note: "MTP companion, depth 2", draftDepth: 2 },
  mtp3: { args: [] as string[], note: "MTP companion, depth 3", draftDepth: 3 },
  /** The served DFlash 2 composition: the model's DFlash 2 drafter (`--dflash2-draft ID=PATH[=BASIS]`),
   * adaptive depth over 4-bit KV. Models without a drafter are not applicable. */
  dflash2: { args: ["--kv-quant", "4", "--num-draft-tokens", "adaptive"], note: "DFlash 2 drafter, adaptive depth, 4-bit KV", dflash2: true },
} as const;
export type Configuration = keyof typeof CONFIGURATIONS;

/** Main's canonical `all` matrix: every model, every configuration, the stock reference. */
export const CANONICAL = {
  models: ["cpm5", "e4b", "12B", "qwen27b"],
  configurations: ["default", "serial", "mixed"] as Configuration[],
  /** The refactor gate's drafter configurations, which a qualifying matrix may add after main's configurations. */
  optionalConfigurations: ["mtp2", "mtp3", "dflash2"] as Configuration[],
  references: ["mlx-lm"],
  withContext: true,
} as const;

/** Main's workload, unchanged: sample counts, lengths and the stability guard. */
export const WORKLOAD = {
  decodeTokens: 192, contextTokens: 16384, decodeRuns: 5, ttftRuns: 3,
  aggregateStreams: 4, aggregateTokens: 128, aggregateContext: 0, aggregateStaggerMs: 0,
  /** Stability guard on one arm's own decode samples, never a comparison tolerance. */
  spreadLimit: 1.15,
} as const;

export const REQUIRED_PHASES = ["warmup", "parity", "decode", "ttft1k", "ctx", "restart", "agg"] as const;
export type Phase = (typeof REQUIRED_PHASES)[number];

export interface FileRecord { name: string; bytes: number; sha256: string; path?: string }
export interface TreeSpec {
  root: string;
  commit: string;
  /** Exact argv prefix that starts this tree's server; serve arguments follow. */
  command: string[];
}
export interface ReferenceSpec {
  /** Labeled external server, e.g. "mlx-lm". Its environment stays outside this repo. */
  label: string;
  command: string[];
  /** Used instead for architectures the stock server cannot load (main: gemma4_unified). */
  registerCommand?: string[];
  version?: string;
  /** Bearer key the reference server requires, if any. */
  apiKey?: string;
}
export interface ModelSpec {
  id: string;
  label: string;
  path: string;
  modelType: string;
  kvConfig: boolean;
  packedTrellis: boolean;
  files: FileRecord[];
  /** MTP companion for the mtp configurations, pinned like the model. */
  draft?: { path: string; files: FileRecord[] };
  /** DFlash 2 drafter for the dflash2 configuration, pinned like the model. `basis` is the target's
   * rotation basis file: main's DFlash 2 provider reads it from MLX_BUN_DFLASH2_TARGET_BASIS, the
   * refactor declares it from the artifact's sidecar and ignores the variable; both servers get it. */
  dflash2?: { path: string; files: FileRecord[]; basis?: FileRecord & { path: string } };
}
export interface Plan {
  schema: 1;
  profile: "all" | "scoped";
  seed: string;
  /** Scoped plans may shrink the workload; profile all requires main's exact values. */
  workload: { -readonly [K in keyof typeof WORKLOAD]: number } & { withContext: boolean };
  trees: Record<Tree, TreeSpec>;
  references: ReferenceSpec[];
  configurations: Configuration[];
  /** Server sessions per tree for each model and configuration (absent: one). Sessions alternate
   * trees, so pairs pooled across them carry session-to-session drift. */
  sessions?: number;
  models: ModelSpec[];
  /** Intended MLX library for both trees; children must prove they loaded it.
   * files[0] is the exact resolved library, the rest its bundled runtime. */
  native: { library: string; files: FileRecord[] };
  /** Predeclared applicability: every cell, with main's N/A reasons. Recomputed at run time. */
  cells: PlannedCell[];
}

export const sha256 = (bytes: Uint8Array | string) => createHash("sha256").update(bytes).digest("hex");
/** Streamed: weight shards are many gigabytes. */
export function fileSha(path: string): string {
  const hash = createHash("sha256"), fd = openSync(path, "r"), buffer = Buffer.allocUnsafe(8 << 20);
  try {
    for (let read; (read = readSync(fd, buffer, 0, buffer.length, null)) > 0;) hash.update(buffer.subarray(0, read));
  } finally { closeSync(fd); }
  return hash.digest("hex");
}

/** Every regular file of a snapshot, followed through the hub's symlinks. */
export function artifactFiles(dir: string): FileRecord[] {
  return readdirSync(dir).sort().flatMap(name => {
    const path = join(dir, name);
    if (!statSync(path).isFile()) return [];
    return [{ name, bytes: statSync(path).size, sha256: fileSha(path) }];
  });
}

export function describeModel(id: string, path: string, label?: string): ModelSpec {
  const resolved = realpathSync(path);
  const config = JSON.parse(readFileSync(join(resolved, "config.json"), "utf8"));
  const quant = config.quantization ?? config.quantization_config ?? config.text_config?.quantization ?? {};
  const files = artifactFiles(resolved);
  if (!files.some(file => file.name.endsWith(".safetensors") && file.bytes > 0)) throw new Error(`${id}: no weights in ${resolved}`);
  return {
    id, label: label || basename(resolved), path: resolved, modelType: config.model_type,
    kvConfig: existsSync(join(resolved, "kv_config.json")),
    packedTrellis: quant.mode === "trellis" || Object.values(quant).some(v =>
      v !== null && typeof v === "object" && (v as { mode?: string }).mode === "trellis"),
    files,
  };
}

/** The exact library supplied (resolved through symlinks) first, then every
 * runtime file bundled beside it (MLX, JACCL, the metallib, and any others). */
export function nativeFiles(library: string): FileRecord[] {
  const chosen = realpathSync(library), dir = resolve(chosen, "..");
  const record = (path: string): FileRecord => ({ name: basename(path), path, bytes: statSync(path).size, sha256: fileSha(path) });
  const bundled = readdirSync(dir).sort().map(name => join(dir, name))
    .filter(path => path !== chosen && /\.(dylib|metallib)$/.test(path) && statSync(path).isFile());
  return [record(chosen), ...bundled.map(record)];
}

const within = (path: string, root: string) => {
  const rel = relative(realpathSync(root), path);
  return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
};
/** Captures stay outside every source tree the run measures (and this tooling's own). */
export function outsideTrees(path: string, roots: string[]): string {
  if (!isAbsolute(path)) throw new Error(`${path} must be an absolute path`);
  let probe = path;
  while (!existsSync(probe)) probe = resolve(probe, "..");
  const real = join(realpathSync(probe), relative(probe, path));
  for (const root of roots) if (within(real, root)) throw new Error(`${path} is inside source tree ${root}`);
  return real;
}
export function checkOutputDirectory(out: string, roots: string[]): string {
  const real = outsideTrees(out, roots);
  if (existsSync(out) && readdirSync(out).length) throw new Error(`--out ${out} must be empty or absent`);
  return real;
}

/** A plan labeled `all` must be main's full matrix; anything narrower is scoped. */
export function profileProblems(plan: Plan): string[] {
  if (plan.profile !== "all") return [];
  const problems: string[] = [];
  const ids = plan.models.map(model => model.id);
  if (JSON.stringify(ids) !== JSON.stringify(CANONICAL.models)) problems.push(`models ${ids} are not ${CANONICAL.models}`);
  const base = plan.configurations.slice(0, CANONICAL.configurations.length), extra = plan.configurations.slice(base.length);
  if (JSON.stringify(base) !== JSON.stringify(CANONICAL.configurations) ||
      JSON.stringify(extra) !== JSON.stringify(CANONICAL.optionalConfigurations.filter(name => extra.includes(name))))
    problems.push(`configurations ${plan.configurations} are not ${CANONICAL.configurations} optionally followed by ${CANONICAL.optionalConfigurations}`);
  if (JSON.stringify(plan.references.map(ref => ref.label)) !== JSON.stringify(CANONICAL.references))
    problems.push(`references ${plan.references.map(ref => ref.label)} are not ${CANONICAL.references}`);
  if (!plan.workload.withContext) problems.push("long-context, restart phases are skipped");
  for (const model of plan.models) if (needsRegister(model))
    for (const ref of plan.references) if (!ref.registerCommand) problems.push(`${ref.label} has no register command for ${model.id}`);
  for (const [key, value] of Object.entries(WORKLOAD))
    if ((plan.workload as Record<string, unknown>)[key] !== value) problems.push(`workload ${key} differs from main`);
  return problems;
}

export function validatePlan(value: unknown): Plan {
  const plan = value as Plan;
  const fail = (message: string) => { throw new Error(`invalid plan: ${message}`); };
  if (!plan || plan.schema !== 1) fail("schema must be 1");
  if (plan.profile !== "all" && plan.profile !== "scoped") fail("profile must be all or scoped");
  if (typeof plan.seed !== "string" || !plan.seed) fail("seed is required");
  for (const tree of TREES) {
    const spec = plan.trees?.[tree];
    if (!spec || !isAbsolute(spec.root) || !/^[0-9a-f]{40}$/.test(spec.commit) || !Array.isArray(spec.command) ||
        !spec.command.length || spec.command.some(part => typeof part !== "string" || !part))
      fail(`${tree} needs an absolute root, a full commit and an explicit command`);
  }
  if (plan.trees.baseline.root === plan.trees.candidate.root) fail("baseline and candidate must be different trees");
  if (!Array.isArray(plan.references) || plan.references.some(ref => !ref.label || !Array.isArray(ref.command) || !ref.command.length))
    fail("each reference needs a label and an explicit command");
  if (!Array.isArray(plan.configurations) || !plan.configurations.length ||
      plan.configurations.some(name => !(name in CONFIGURATIONS))) fail("unknown configuration");
  if (!Array.isArray(plan.models) || !plan.models.length) fail("at least one model is required");
  for (const model of plan.models) {
    if (!model.id || !isAbsolute(model.path) || !Array.isArray(model.files) || !model.files.length) fail(`model ${model.id} is not pinned`);
    if (model.draft && (!isAbsolute(model.draft.path) || !Array.isArray(model.draft.files) || !model.draft.files.length))
      fail(`model ${model.id} draft is not pinned`);
    const dflash2 = model.dflash2;
    if (dflash2 && (!isAbsolute(dflash2.path) || !Array.isArray(dflash2.files) || !dflash2.files.length ||
        (dflash2.basis && (!isAbsolute(dflash2.basis.path) || !/^[0-9a-f]{64}$/.test(dflash2.basis.sha256)))))
      fail(`model ${model.id} DFlash 2 drafter is not pinned`);
  }
  if (plan.sessions !== undefined && (!Number.isSafeInteger(plan.sessions) || plan.sessions < 2))
    fail("sessions, when present, must be an integer of at least 2");
  if (!plan.native?.library || !isAbsolute(plan.native.library) || !plan.native.files?.length) fail("native library is not pinned");
  const problems = profileProblems(plan);
  if (problems.length) fail(`profile all is not main's full matrix: ${problems.join("; ")}`);
  if (JSON.stringify(plan.cells) !== JSON.stringify(planCells(plan))) fail("predeclared cells do not match the plan's matrix");
  return plan;
}

/** Tracked and untracked, non-ignored files of a tree: the measured source identity. */
export function sourceSnapshot(root: string): { head: string; clean: boolean; sha256: string; files: number } {
  const git = (...args: string[]) => {
    const result = Bun.spawnSync(["git", "-C", root, ...args]);
    if (result.exitCode !== 0) throw new Error(`git ${args[0]} failed in ${root}`);
    return result.stdout.toString();
  };
  const head = git("rev-parse", "HEAD").trim(), clean = !git("status", "--porcelain").trim();
  const names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0").filter(Boolean).sort();
  const files = names.filter(name => existsSync(join(root, name)) && statSync(join(root, name)).isFile())
    .map(name => [name, fileSha(join(root, name))]);
  return { head, clean, sha256: sha256(JSON.stringify(files)), files: files.length };
}

/** Pins that must hold at start and end: trees, artifacts and the intended native library. */
export function pinProblems(plan: Plan): string[] {
  const problems: string[] = [];
  for (const tree of TREES) {
    const snapshot = sourceSnapshot(plan.trees[tree].root);
    if (snapshot.head !== plan.trees[tree].commit) problems.push(`${tree} head ${snapshot.head} is not ${plan.trees[tree].commit}`);
    if (!snapshot.clean) problems.push(`${tree} tree is not clean`);
  }
  for (const model of plan.models) {
    const actual = artifactFiles(model.path);
    if (JSON.stringify(actual) !== JSON.stringify(model.files)) problems.push(`model ${model.id} files changed`);
    if (model.draft && JSON.stringify(artifactFiles(model.draft.path)) !== JSON.stringify(model.draft.files))
      problems.push(`model ${model.id} draft files changed`);
    if (model.dflash2 && JSON.stringify(artifactFiles(model.dflash2.path)) !== JSON.stringify(model.dflash2.files))
      problems.push(`model ${model.id} DFlash 2 drafter files changed`);
    const basis = model.dflash2?.basis;
    if (basis && (!existsSync(basis.path) || fileSha(basis.path) !== basis.sha256)) problems.push(`model ${model.id} DFlash 2 basis changed`);
  }
  if (JSON.stringify(nativeFiles(plan.native.library)) !== JSON.stringify(plan.native.files)) problems.push("native library files changed");
  return problems;
}


/** Main: plain mlx-lm cannot load gemma4_unified; the reference registers it through optiq (bf16 KV). */
export const needsRegister = (model: ModelSpec) => model.modelType === "gemma4_unified";

export interface PlannedCell {
  key: string; model: string; kind: "tree" | "reference"; order: number;
  tree?: Tree; configuration?: Configuration; reference?: string; skipped?: string;
  /** 1-based server session of a multi-session plan; absent when the plan runs one session per tree. */
  session?: number;
}
/** The sessions every tree runs per model and configuration, 1-based. */
export const sessionsOf = (plan: Pick<Plan, "sessions">): number[] => Array.from({ length: plan.sessions ?? 1 }, (_, i) => i + 1);
/** Every cell of the matrix in run order. Per model, each configuration runs on
 * both trees back to back, alternating which tree goes first; with several sessions
 * the pair repeats (A B A B), each session a fresh server. References follow. */
export function planCells(plan: Plan): PlannedCell[] {
  const cells: PlannedCell[] = [];
  const multi = (plan.sessions ?? 1) > 1;
  plan.models.forEach((model, m) => {
    plan.configurations.forEach((configuration, c) => {
      const trees = (m + c) % 2 ? [...TREES].reverse() : TREES;
      const spec = CONFIGURATIONS[configuration];
      const skipped = "requiresKvConfig" in spec && !model.kvConfig ? "no kv_config.json in the artifact"
        : "draftDepth" in spec && !model.draft ? "no MTP companion (--draft) for this model"
        : "dflash2" in spec && !model.dflash2 ? "no DFlash 2 drafter (--dflash2-draft) for this model" : undefined;
      for (const session of sessionsOf(plan)) for (const tree of trees)
        cells.push({ key: `${model.id}/${configuration}/${tree}${multi ? `/s${session}` : ""}`, model: model.id, kind: "tree", tree,
          configuration, order: cells.length, ...(multi ? { session } : {}), ...(skipped ? { skipped } : {}) });
    });
    for (const ref of plan.references) {
      const skipped = model.packedTrellis ? "packed trellis has no stock reference loader; an expanded carrier is a different artifact"
        : needsRegister(model) && !ref.registerCommand ? `${ref.label} cannot load ${model.modelType} without a register command` : undefined;
      cells.push({ key: `${model.id}/${ref.label}`, model: model.id, kind: "reference", reference: ref.label,
        order: cells.length, ...(skipped ? { skipped } : {}) });
    }
  });
  return cells;
}

/** The exact argv for one cell: the explicit command, then serve arguments. */
export function cellArgs(plan: Plan, cell: PlannedCell, port: number, ssdDir: string): string[] {
  const model = plan.models.find(m => m.id === cell.model)!;
  if (cell.kind === "tree") {
    const spec = CONFIGURATIONS[cell.configuration!];
    const draft = "draftDepth" in spec
      ? ["--draft-kind", "mtp", "--draft-model", model.draft!.path, "--num-draft-tokens", String(spec.draftDepth)]
      : "dflash2" in spec ? ["--draft-model", model.dflash2!.path] : [];
    return [...plan.trees[cell.tree!].command, "--model", model.path, "--port", String(port), "--no-open",
      "--ssd-cache", ssdDir, ...spec.args, ...draft];
  }
  const ref = plan.references.find(r => r.label === cell.reference)!;
  return [...(needsRegister(model) ? ref.registerCommand! : ref.command), "--model", model.path, "--port", String(port)];
}

/** Environment a cell's configuration adds, identical for both trees: the DFlash 2 basis file
 * when one is pinned (main reads it; the refactor ignores the variable). */
export function cellExtraEnvironment(plan: Plan, cell: PlannedCell): Record<string, string> {
  if (cell.kind !== "tree" || !("dflash2" in CONFIGURATIONS[cell.configuration!])) return {};
  const basis = plan.models.find(m => m.id === cell.model)?.dflash2?.basis;
  return basis ? { MLX_BUN_DFLASH2_TARGET_BASIS: basis.path } : {};
}
