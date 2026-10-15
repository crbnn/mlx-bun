/** CPU-only: LEAN=/path/to/lean bun proofs/check.ts [--self-test]. No MLX import. */
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const root = new URL("../", import.meta.url);
const sourcePath = new URL("packages/inference/src/kernels/trellis/codebook.ts", root);
const proofPath = new URL("Trellis.lean", import.meta.url);
const lean = process.env.LEAN ?? "lean";
const expected = `static inline int trellis_y(uint s) {
  const uint x = s * 34038481u + 76625530u;
  const uint p = (x & 0x00FF00FFu) + ((x >> 8) & 0x00FF00FFu);
  return (int)((p & 0xFFFFu) + (p >> 16)) - 510;
}`;

function checkSource(source: string): void {
  const definitions = [...source.matchAll(/static\s+inline\s+int\s+trellis_y\s*\([^)]*\)\s*\{[^}]*\}/g)];
  const normalize = (text: string) => text.replace(/\s+/g, "").trim();
  if (definitions.length !== 1 || normalize(definitions[0]![0]) !== normalize(expected)) {
    throw new Error("trellis_y changed: review the Metal-to-Lean translation and update its proof before updating this source pin.");
  }
}

async function run(args: string[]) {
  const child = Bun.spawn([lean, ...args], { cwd: root.pathname, stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, code] = await Promise.all([
    new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited,
  ]);
  return { output: stdout + stderr, code };
}

const version = await run(["--version"]);
if (version.code !== 0 || !version.output.includes("version 4.35.0,")) {
  throw new Error(`Expected Lean 4.35.0; got ${version.output}`);
}
const source = await Bun.file(sourcePath).text();
checkSource(source);
const result = await run([proofPath.pathname]);
process.stdout.write(result.output);
if (result.code !== 0 || result.output.includes("sorryAx")) {
  throw new Error("Lean did not establish the decoder contract without proof holes.");
}

if (process.argv.includes("--self-test")) {
  for (const [before, after] of [["34038481u", "34038483u"], ["x >> 8", "x >> 9"], ["- 510;", "- 511;"]]) {
    let rejected = false;
    try { checkSource(source.replace(before!, after!)); } catch { rejected = true; }
    if (!rejected) throw new Error(`Source mutation was accepted: ${before}`);
  }
  const dir = await mkdtemp(join(tmpdir(), "mlx-bun-proof-negative-"));
  try {
    const proof = await Bun.file(proofPath).text();
    const wrongClaim = proof.replace("implementation s = specification s :=", "implementation s = specification s + 1 :=");
    if (proof === wrongClaim) throw new Error("Negative proof test did not alter its target.");
    const path = join(dir, "Incorrect.lean");
    await Bun.write(path, wrongClaim);
    if ((await run([path])).code === 0) throw new Error("Lean accepted the deliberately false contract.");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
  console.log("Negative checks passed: changed arithmetic and a false theorem were rejected.");
}
console.log("Proved trellis_y byte-sum equality and [-510, 510] bounds for all 32-bit states.");
console.log("Scope: source-pinned manual translation; native LRAT checking trusts Lean's compiler. Floating-point decoding and the M1 Max graph remain unproved.");
