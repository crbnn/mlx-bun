// The Neural Engine install step as the app runs it. `mlx-bun compile-ane` compiles every Neural Engine program the
// shipped compositions load on this Mac into the system's compile cache (`compileAnePrograms` from the inference
// library) and records the app version that ran it. The installer runs the verb after placing a new bundle; for an
// install that runs nothing (Homebrew, npm), the first model load of a version that has no record runs it as a child
// process first: the app's first launch. A model load never compiles: a program missing from the cache fails the
// load with AneBridgeError naming this verb. The cache is per code-signing identity, so the child is this executable.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { executablePath } from "../jobs/executable";
import { storagePath } from "../storage/paths";
import type { CommandArgs } from "./args";
import pkg from "../../package.json" with { type: "json" };

/** What `compile-ane` records once it has compiled (or found) every program. */
interface InstallRecord {
  readonly version: string;
  readonly device: string;
  readonly prefillChunkTokens: number;
  readonly programs: readonly string[];
}

/** The app version whose install step ran here, if any. */
function recordedVersion(path: string): string | undefined {
  if (!existsSync(path)) return undefined;
  try { return (JSON.parse(readFileSync(path, "utf8")) as Partial<InstallRecord>).version; }
  catch { return undefined; }
}

const seconds = (ms: number) => ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(ms < 10 ? 1 : 0)} ms`;

/** `mlx-bun compile-ane`: compile (or, with `--check`, only list) this Mac's Neural Engine programs. Returns the exit
 * code: 0, or 1 when `--check` finds a program the cache lacks. A refused compile throws `AneBridgeError`. */
export async function runCompileAne(args: CommandArgs, log: (line: string) => void = console.log): Promise<number> {
  const { aneInstallPrograms, aneInstallStatus, compileAnePrograms, machineComposition } = await import("@mlx-bun/inference/models");
  const composition = machineComposition();
  const where = `${composition.device}, ${composition.prefillChunkTokens}-token prompt chunks`;
  const used = (graphs: readonly string[]) => `(${graphs.join(", ")})`;
  const programs = aneInstallPrograms({ composition });
  if (!programs.length) log(`No model mlx-bun ships for this Mac uses the Neural Engine (${where}); nothing to compile.`);
  else if (args.values.check) {
    const status = aneInstallStatus({ composition });
    log(`Neural Engine programs on this Mac (${where}):`);
    for (const program of status) log(`  ${program.cached ? "compiled" : "missing "}  ${program.spec.label} ${used(program.graphs)}`);
    if (!status.every(program => program.cached)) {
      log("Run mlx-bun compile-ane to compile the missing programs.");
      return 1;
    }
  } else {
    log(`Compiling the Neural Engine programs this Mac's models use (${where}):`);
    compileAnePrograms({ composition, onProgram: result =>
      log(`  ${result.status === "compiled" ? "compiled" : "cached  "}  ${result.spec.label} in ${seconds(result.milliseconds)} ${used(result.graphs)}`) });
  }
  if (!args.values.check) record({ version: pkg.version, device: composition.device,
    prefillChunkTokens: composition.prefillChunkTokens, programs: programs.map(program => program.spec.label) });
  return 0;
}

function record(value: InstallRecord): void {
  const path = storagePath("aneInstall");
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n");
}

export interface AneInstallDependencies {
  /** Runs `mlx-bun compile-ane` in a child process of this executable and resolves its exit code. */
  run(): Promise<number>;
  error(message: string): void;
}

const defaults: AneInstallDependencies = {
  async run() {
    const entry = fileURLToPath(new URL("./main.ts", import.meta.url));
    const command = entry.includes("$bunfs") ? [executablePath, "compile-ane"] : [executablePath, entry, "compile-ane"];
    return await Bun.spawn(command, { stdin: "ignore", stdout: "inherit", stderr: "inherit" }).exited;
  },
  error: message => console.error(message),
};

/** The app's first launch: before the first model load of a version with no install record on this machine, runs
 * the install step as a child process. A failed step is reported and the launch goes on; a model whose composition
 * loads Neural Engine programs then fails its load, naming the verb. */
export async function ensureAneInstall(supplied: Partial<AneInstallDependencies> = {}): Promise<void> {
  if (recordedVersion(storagePath("aneInstall")) === pkg.version) return;
  const deps = { ...defaults, ...supplied };
  const code = await deps.run();
  if (code !== 0)
    deps.error(`mlx-bun: the Neural Engine install step failed (exit ${code}); a model that uses the Neural Engine will not load until \`mlx-bun compile-ane\` succeeds`);
}
