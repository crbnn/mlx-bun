// The Neural Engine install step as the app runs it. `mlx-bun compile-ane` checks every Neural Engine program the
// shipped compositions load on this Mac against the system's compile cache and compiles the ones it lacks
// (`compileAnePrograms` from the inference library). The installer runs the verb after placing a new bundle, and the
// app runs it quietly before every model load (`ensureAneInstall`), in a child process of this executable: a cached
// program costs one cache check, and a program the system removed from its cache, or one no installer compiled
// (Homebrew, npm), is compiled again before the model loads. The model load itself never compiles: a program missing
// there fails it with AneBridgeError naming this verb, which only a load that skipped this step (a library host
// through `mlx-bun/engine`) can meet. The cache is per code-signing identity, so the child is this executable, and the
// check runs in the child so a serving parent that holds no model loads no native library.
import { fileURLToPath } from "node:url";
import { executablePath } from "../jobs/executable";
import type { CommandArgs } from "./args";

const duration = (ms: number) => ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(ms < 10 ? 1 : 0)} ms`;

/** `mlx-bun compile-ane`: compile what the cache lacks (or, with `--check`, only list). `--quiet` prints only the
 * programs it compiles. Returns the exit code: 0, or 1 when `--check` finds a program the cache lacks. A refused
 * compile throws `AneBridgeError`. */
export async function runCompileAne(args: CommandArgs, log: (line: string) => void = console.log): Promise<number> {
  const { aneInstallPrograms, aneInstallStatus, compileAnePrograms, machineComposition } = await import("@mlx-bun/inference/models");
  const composition = machineComposition();
  const where = `${composition.device}, ${composition.prefillChunkTokens}-token prompt chunks`;
  const used = (graphs: readonly string[]) => `(${graphs.join(", ")})`;
  const quiet = !!args.values.quiet;
  if (!aneInstallPrograms({ composition }).length) {
    if (!quiet) log(`No model mlx-bun ships for this Mac uses the Neural Engine (${where}); nothing to compile.`);
    return 0;
  }
  if (args.values.check) {
    const status = aneInstallStatus({ composition });
    log(`Neural Engine programs on this Mac (${where}):`);
    for (const program of status) log(`  ${program.cached ? "compiled" : "missing "}  ${program.spec.label} ${used(program.graphs)}`);
    if (status.every(program => program.cached)) return 0;
    log("Run mlx-bun compile-ane to compile the missing programs.");
    return 1;
  }
  let announced = false;
  const announce = () => {
    if (!announced) log(`Compiling the Neural Engine programs this Mac's models use (${where}):`);
    announced = true;
  };
  if (!quiet) announce();
  compileAnePrograms({ composition, onProgram: result => {
    if (quiet && result.status === "cached") return;
    announce();
    log(`  ${result.status === "compiled" ? "compiled" : "cached  "}  ${result.spec.label} in ${duration(result.milliseconds)} ${used(result.graphs)}`);
  } });
  return 0;
}

export interface AneInstallDependencies {
  /** Runs `mlx-bun compile-ane --quiet` in a child process of this executable and resolves its exit code. */
  run(): Promise<number>;
  error(message: string): void;
}

const defaults: AneInstallDependencies = {
  async run() {
    const entry = fileURLToPath(new URL("./main.ts", import.meta.url));
    const verb = ["compile-ane", "--quiet"];
    const command = entry.includes("$bunfs") ? [executablePath, ...verb] : [executablePath, entry, ...verb];
    return await Bun.spawn(command, { stdin: "ignore", stdout: "inherit", stderr: "inherit" }).exited;
  },
  error: message => console.error(message),
};

/** The app's startup step, before every model load: checks each Neural Engine program this Mac's compositions load
 * and compiles the ones the cache lacks, in a child process (silent when every program is cached). A failed step is
 * reported and the launch goes on; a model whose composition loads a missing program then fails its load, naming the
 * verb. */
export async function ensureAneInstall(supplied: Partial<AneInstallDependencies> = {}): Promise<void> {
  const deps = { ...defaults, ...supplied };
  const code = await deps.run();
  if (code !== 0)
    deps.error(`mlx-bun: the Neural Engine install step failed (exit ${code}); a model that uses the Neural Engine will not load until \`mlx-bun compile-ane\` succeeds`);
}
