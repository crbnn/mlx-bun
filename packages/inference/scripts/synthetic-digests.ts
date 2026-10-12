/** Synthetic-weight digests: record or check the hashes the digest suites hold
 * (tests/fixtures/synthetic-digests/<suite>.json, written by
 * tests/models/synthetic-digest.ts). Each suite is one test file over synthetic
 * weights for a family no artifact holds; `record` runs it with the record
 * switch set, which rewrites this GPU architecture's entry and leaves the
 * others; `check` runs it as the test suite does. `--help` loads no native
 * library. Record only after a change whose numbers are meant to move, and say
 * why in the commit. */
import { resolve } from "node:path";
import { parseArgs } from "node:util";

const PACKAGE = resolve(import.meta.dir, "..");
/** Each suite: the test file that holds its cases and the opt-in that file needs to run them. */
export const SUITES: Readonly<Record<string, { readonly file: string; readonly env?: Readonly<Record<string, string>> }>> = {
  glm52: { file: "tests/models/glm52/glm52-digest.test.ts" },
  "diffusion-gemma": { file: "tests/models/diffusion-gemma-digest.test.ts" },
  "dense-family": { file: "tests/models/dense-family-digest.test.ts" },
  gemma2: { file: "tests/models/universal/gemma2-batching.test.ts", env: { MLX_BUN_GEMMA2_NATIVE: "1" } },
};
const RECORD_ENV = "MLX_BUN_RECORD_SYNTHETIC_DIGESTS";

const USAGE = `usage: bun packages/inference/scripts/synthetic-digests.ts <record|check> [suite...]

  record   run each suite's test file with ${RECORD_ENV}=1: its digest cases write
           this GPU architecture's hashes into tests/fixtures/synthetic-digests/<suite>.json
  check    run each suite's test file as the test suite does

suites (default: all): ${Object.keys(SUITES).join(", ")}
The native library is the one the tests load (MLX_BUN_LIBMLXC when set).`;

const { positionals, values } = parseArgs({ allowPositionals: true, options: { help: { type: "boolean", short: "h" } } });
if (values.help || positionals.length === 0) {
  console.log(USAGE);
  process.exit(values.help ? 0 : 1);
}
const [command, ...named] = positionals;
if (command !== "record" && command !== "check") {
  console.error(`unknown command ${command}\n${USAGE}`);
  process.exit(1);
}
const unknown = named.filter(suite => !(suite in SUITES));
if (unknown.length) {
  console.error(`unknown suite ${unknown.join(", ")}; suites: ${Object.keys(SUITES).join(", ")}`);
  process.exit(1);
}
const suites = named.length ? named : Object.keys(SUITES);
let failed = false;
for (const suite of suites) {
  const { file, env: optIn } = SUITES[suite]!;
  const env = { ...process.env, ...optIn, [RECORD_ENV]: command === "record" ? "1" : undefined };
  // Only the digest cases (their names end in "(digest)") run when recording.
  const filter = command === "record" ? ["--test-name-pattern", "\\(digest\\)$"] : [];
  const child = Bun.spawn([process.execPath, "test", file, ...filter], { cwd: PACKAGE, env, stdout: "inherit", stderr: "inherit" });
  const code = await child.exited;
  console.log(`synthetic-digests ${command} ${suite}: exit ${code}`);
  failed ||= code !== 0;
}
process.exit(failed ? 1 : 0);
