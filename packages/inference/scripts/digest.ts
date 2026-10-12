/** App-path digest: the token ids and top-8 logprobs of N greedy tokens (default 32)
 * through the app's one-shot `generate` path (`runInference`: the command's KV flags,
 * the loaded composition, `createAppEngine`, the scheduler), hashed. Every gate in
 * PLAN compares trees by these hashes; the perf workflow's numerical check runs it
 * on both trees. The tree is imported by path (`--root`, default this checkout), so
 * one tool measures main and the candidate alike; nothing here imports the
 * workspace statically, and `--help` loads no native library.
 *
 * Reconstructed from the Trellis checks' digest.ts (the E1+D2+E3 merge's final form),
 * so a one-row cell hashes as the plan records (Trellis kv4 `69e8cc07…` / `ee58a136…`):
 * tokens = sha256 of the ids joined by ",", logprobs = sha256 of one JSON line per
 * token joined by "\n", both cut to 16 hex digits.
 *
 * A cell is "bf16" or a comma list of generate flags ("kv-quant=4,fused-sdpa=on"),
 * crossed with `--rows` and `--prefill-chunk`. Each cell runs in its own process
 * (the chunk override is `MLX_BUN_RD_PREFILL_CHUNK`, which the tree's runtime reads
 * once when it loads). R rows run on an engine of capacity R, composed for R rows,
 * admitted together under one execution lease: row 0 is the one-row cell's request,
 * row r drops the last 24·r prompt tokens so the rows sit at different positions.
 * An R-row cell hashes its rows in order (ids joined "\n", logprob lines "\n\n") and
 * lists each row's own pair, so row 0 compares directly with the one-row cell. */
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { isAbsolute, join, resolve } from "node:path";
import { parseArgs } from "node:util";

/** This checkout: the default tree. */
export const REPOSITORY = resolve(import.meta.dir, "../../..");
export const TOP_LOGPROBS = 8;
export const ROW_TRIM_TOKENS = 24;
/** The saved digest's default cells: bf16, TurboQuant, then affine 4, 8 and the
 * artifact's kv_config, each with `--fused-sdpa` unset, on and off. */
export const DEFAULT_CELLS = ["bf16", "kv-quant=turbo",
  ...["4", "8", "config"].flatMap(kv => ["", ",fused-sdpa=on", ",fused-sdpa=off"].map(fused => `kv-quant=${kv}${fused}`))];
const PARAGRAPH = "The history of computing is a history of abstraction layers. Each generation of engineers builds on the " +
  "tools of the last, hiding detail behind interfaces so that the next problem can be stated more simply. ";
export const PROMPT = `${PARAGRAPH.repeat(12)}\nSummarize the passage above in three short sentences, then list two open questions it raises.`;

export const sha16 = (text: string) => createHash("sha256").update(text).digest("hex").slice(0, 16);

/** A cell's generate flags; "bf16" is the command's default (no KV flag). */
export function cellFlags(spec: string): Record<string, string> {
  if (!spec.trim()) throw new Error("empty cell");
  if (spec === "bf16") return {};
  const flags: Record<string, string> = {};
  for (const pair of spec.split(",")) {
    const at = pair.indexOf("=");
    if (at <= 0 || at === pair.length - 1) throw new Error(`cell ${spec}: "${pair}" is not flag=value`);
    const name = pair.slice(0, at);
    if (name === "max-tokens" || name === "rows" || name === "prefill-chunk")
      throw new Error(`cell ${spec}: ${name} is this tool's option (--tokens, --rows, --prefill-chunk)`);
    flags[name] = pair.slice(at + 1);
  }
  return flags;
}

export interface RowCapture { tokens: number[]; logprobs: string[] }
export interface Digest { tokens: string; logprobs: string; rows: Array<{ tokens: string; logprobs: string; n: number }> }
/** The cell's two hashes over its rows in order, and each row's own pair. */
export function digestRows(rows: readonly RowCapture[]): Digest {
  return {
    tokens: sha16(rows.map(row => row.tokens.join(",")).join("\n")),
    logprobs: sha16(rows.map(row => row.logprobs.join("\n")).join("\n\n")),
    rows: rows.map(row => ({ tokens: sha16(row.tokens.join(",")), logprobs: sha16(row.logprobs.join("\n")), n: row.tokens.length })),
  };
}

export interface PlannedCell { spec: string; rows: number; prefillChunk: number | null; label: string }
export const cellLabel = (spec: string, rows: number, prefillChunk: number | null) =>
  `${spec}${prefillChunk === null ? "" : `,prefill-chunk=${prefillChunk}`}${rows > 1 ? `,rows=${rows}` : ""}`;

export interface DigestOptions { root: string; model: string; tokens: number; cells: PlannedCell[] }
const positive = (name: string, raw: string): number => {
  const value = Number(raw);
  if (!/^\d+$/.test(raw) || !Number.isSafeInteger(value) || value < 1) throw new Error(`--${name}: ${raw} is not a positive integer`);
  return value;
};
/** The options, or null for --help. Cells cross every row count and chunk, in that order. */
export function parseDigestArgs(argv: string[]): DigestOptions | null {
  const { values, positionals } = parseArgs({ args: argv, allowPositionals: true, options: {
    help: { type: "boolean", short: "h" }, root: { type: "string", default: REPOSITORY }, model: { type: "string" },
    tokens: { type: "string", default: "32" }, rows: { type: "string", default: "1" }, "prefill-chunk": { type: "string", default: "default" },
  } });
  if (values.help) return null;
  if (!values.model || !isAbsolute(values.model)) throw new Error("--model must be an absolute snapshot directory");
  if (!isAbsolute(values.root!)) throw new Error("--root must be an absolute path");
  const rows = values.rows!.split(",").map(raw => positive("rows", raw));
  const chunks = values["prefill-chunk"]!.split(",").map(raw => raw === "default" ? null : positive("prefill-chunk", raw));
  const specs = positionals.length ? positionals : DEFAULT_CELLS;
  for (const spec of specs) cellFlags(spec);
  return { root: resolve(values.root!), model: resolve(values.model), tokens: positive("tokens", values.tokens!),
    cells: specs.flatMap(spec => rows.flatMap(r => chunks.map(chunk => ({ spec, rows: r, prefillChunk: chunk, label: cellLabel(spec, r, chunk) })))) };
}

export const USAGE = `App-path digest: token ids and top-${TOP_LOGPROBS} logprobs of N greedy tokens through the app's generate path.

  bun packages/inference/scripts/digest.ts --model /abs/snapshot [--root /abs/tree] [--tokens 32]
      [--rows 1[,R...]] [--prefill-chunk default[,N...]] [CELL ...]

  CELL      "bf16" or a comma list of generate flags, e.g. kv-quant=4,fused-sdpa=on or kv-quant=turbo.
            Default: ${DEFAULT_CELLS.join(" ")}
  --root    The tree to measure, imported by path (default: this checkout). Run it with that tree's native
            library (MLX_BUN_LIBMLXC or its staged files).
  --tokens  Greedy tokens per row (default 32).
  --rows    Rows per cell (default 1). R rows share one engine of capacity R, composed for R rows and admitted
            together; row 0 is the one-row request, row r drops the last ${ROW_TRIM_TOKENS}·r prompt tokens.
  --prefill-chunk  Prefill chunk override (MLX_BUN_RD_PREFILL_CHUNK); "default" leaves the environment's.

Every cell (each cell × row count × chunk) runs in its own process and prints one line:
  CELL: tokens <16 hex> logprobs <16 hex> n=<tokens per row> [| rows <tokens>/<logprobs> ...] | graph=<class> | "<text>"
tokens = sha256 of the ids joined by ","; logprobs = sha256 of one JSON line per token (the engine's logprob
record, top ${TOP_LOGPROBS}) joined by "\\n"; an R-row cell hashes its rows in order and lists each row's pair. A cell that
fails prints "CELL: ERROR <message>". Exits 0 when every cell produced its digest.`;

type Loose = any; // the measured tree's modules are imported by path at run time, so they have no static types here

/** One cell in this process: the chunk override must be in the environment before the tree loads. */
async function measureCell(o: DigestOptions, cell: PlannedCell): Promise<boolean> {
  if (cell.prefillChunk !== null) process.env.MLX_BUN_RD_PREFILL_CHUNK = String(cell.prefillChunk);
  process.chdir(o.root);
  const app = (path: string): Promise<Loose> => import(join(o.root, "apps/mlx-bun/src", path));
  const { runInference } = await app("cli/inference.ts");
  const { createAppEngine } = await app("engine/index.ts");
  const { planRequest, RequestOwnership } = await app("server/request-plan.ts");
  const captures: RowCapture[] = Array.from({ length: cell.rows }, () => ({ tokens: [], logprobs: [] }));
  let graph = "?", text = "";
  const record = (row: number, onToken?: (token: number, lp: unknown) => unknown) => (token: number, lp: unknown) => {
    captures[row]!.tokens.push(token);
    captures[row]!.logprobs.push(JSON.stringify(lp ?? null));
    return onToken?.(token, lp);
  };
  const deps: Record<string, unknown> = {
    write: (chunk: string) => { text += chunk; },
    async engine(context: Loose, scheme: Loose) {
      graph = context.model?.constructor?.name ?? "?";
      const engine: Loose = await createAppEngine(context, { capacity: cell.rows, gateway: { kvScheme: scheme } });
      const completion = engine.completion;
      const shapes = new WeakMap<object, object>();
      const shapeOf = (shape: Loose) => {
        let wide = shapes.get(shape);
        if (!wide) shapes.set(shape, wide = Object.freeze({ ...shape, wantsLogprobs: true }));
        return wide;
      };
      const opts = (options: Loose) => ({ ...options, logprobs: true, topLogprobs: TOP_LOGPROBS });
      engine.completion = new Proxy(completion, {
        get(target, prop) {
          if (prop === "place") return (shape: Loose, options: Loose) => target.place(shapeOf(shape), options && opts(options));
          if (prop === "run") return (ids: number[], options: Loose, onToken: Loose, vision: Loose, shape: Loose, placement: Loose, signal: Loose, trace: Loose) => {
            const primary = () => target.run(ids, opts(options), record(0, onToken), vision, shapeOf(shape), placement, signal, trace);
            if (cell.rows === 1) return primary();
            if (ids.length - ROW_TRIM_TOKENS * (cell.rows - 1) < ROW_TRIM_TOKENS)
              throw new Error(`the prompt (${ids.length} tokens) is too short for ${cell.rows} rows`);
            const extra = async (row: number) => {
              const promptIds = ids.slice(0, ids.length - ROW_TRIM_TOKENS * row);
              const plan = planRequest({ promptIds, options: { ...options, stopSequences: [] }, requestedMaxTokens: options.maxTokens,
                contextLimit: null, stream: false, wantLogprobs: true, topLogprobs: TOP_LOGPROBS, adapterIds: [], hasVision: false,
                userSeed: false, hasGrammar: false, hasDraft: false, ownership: new RequestOwnership() });
              if (!plan.ok) { plan.dispose(); throw new Error(plan.error.message); }
              try {
                const placed = target.place(plan.shape, plan.options);
                plan.transferOwnership();
                await target.run(plan.promptIds, plan.options, record(row), undefined, plan.shape, placed, signal);
              } finally { plan.dispose(); }
            };
            return together(engine.gateway, cell.rows, [primary, ...Array.from({ length: cell.rows - 1 }, (_, i) => () => extra(i + 1))]);
          };
          const value = target[prop];
          return typeof value === "function" ? value.bind(target) : value;
        },
      });
      return engine;
    },
  };
  // R rows need a composition built for R rows; the one-row cell keeps generate's own load.
  if (cell.rows > 1) deps.load = async (selected: Loose, maxTokens: number | undefined, composition: Loose) => {
    const install = join(o.root, "apps/mlx-bun/src/cli/ane-install.ts");
    if (existsSync(install)) await (await import(install)).ensureAneInstall();
    const { loadContext } = await app("engine/model-host.ts");
    return loadContext(selected.path, selected.repoId, { requireChatTemplate: false,
      runtime: { nativeDraft: false, maxGenerationTokens: maxTokens }, ...composition, maxRows: cell.rows });
  };
  try {
    await runInference("generate", { positionals: [o.model, PROMPT], values: { "max-tokens": String(o.tokens), ...cellFlags(cell.spec) } }, deps);
  } catch (error) {
    console.log(`${cell.label}: ERROR ${String((error as Error)?.message ?? error).split("\n")[0]} | graph=${graph}`);
    return false;
  }
  const digest = digestRows(captures);
  const rows = cell.rows > 1 ? ` | rows ${digest.rows.map(row => `${row.tokens}/${row.logprobs}`).join(" ")}` : "";
  console.log(`${cell.label}: tokens ${digest.tokens} logprobs ${digest.logprobs} n=${digest.rows.map(row => row.n).join(",")}${rows}` +
    ` | graph=${graph} | ${JSON.stringify(text.slice(0, 48))}`);
  return true;
}

/** Start every row while the lease holds the scheduler, release once all are pending, return row 0's result. */
async function together(gateway: Loose, rows: number, starts: Array<() => Promise<unknown>>): Promise<unknown> {
  const lease = await gateway.acquireExecutionLease();
  let held = true;
  const release = () => { if (held) { held = false; lease.dispose(); } };
  try {
    let rejected = false;
    const settled = Promise.allSettled(starts.map(async start => {
      try { return await start(); } catch (error) { rejected = true; throw error; }
    }));
    for (const begun = performance.now(); gateway.pendingRows < rows && !rejected; await Bun.sleep(5))
      if (performance.now() - begun > 120_000) throw new Error(`only ${gateway.pendingRows} of ${rows} rows were admitted`);
    release();
    const results = await settled;
    const failed = results.find(result => result.status === "rejected");
    if (failed) throw (failed as PromiseRejectedResult).reason;
    return (results[0] as PromiseFulfilledResult<unknown>).value;
  } finally { release(); }
}

async function main(argv: string[]): Promise<number> {
  const options = parseDigestArgs(argv);
  if (!options) { console.log(USAGE); return 0; }
  if (!existsSync(join(options.model, "config.json"))) throw new Error(`${options.model} has no config.json`);
  if (!existsSync(join(options.root, "apps/mlx-bun/src/cli/inference.ts"))) throw new Error(`${options.root} is not an mlx-bun tree`);
  if (options.cells.length === 1) return await measureCell(options, options.cells[0]!) ? 0 : 1;
  let failed = 0;
  for (const cell of options.cells) {
    const child = Bun.spawn([process.execPath, "--no-env-file", import.meta.path, "--root", options.root, "--model", options.model,
      "--tokens", String(options.tokens), "--rows", String(cell.rows), "--prefill-chunk", String(cell.prefillChunk ?? "default"), cell.spec],
    { cwd: options.root, stdout: "inherit", stderr: "inherit", env: process.env });
    if (await child.exited !== 0) failed++;
  }
  return failed ? 1 : 0;
}

if (import.meta.main) process.exit(await main(process.argv.slice(2)));
