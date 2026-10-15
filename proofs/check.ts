/** CPU-only: LEAN=/path/to/lean bun proofs/check.ts [--self-test].
 * Lake resolves the pinned FloatLib/mathlib proof dependencies in proofs/.
 * No MLX import. */
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, dirname, join } from "node:path";

const root = new URL("../", import.meta.url);
const lean = process.env.LEAN ?? "lean";
const lake = process.env.LAKE ?? join(dirname(lean), "lake");
let dependencyPath = "";
// Reviewed manual translations, not a verified Metal/TypeScript frontend. Pins
// prevent source drift; changing a hash requires reviewing the affected proofs.
const sourcePins = {
  "kernels/trellis/codebook.ts": "0ba8624952767d077cdecc531fd9f49a159545b95ae4fc4ddd213ed13fa1e276",
  "kernels/trellis/down-codebook.ts": "5649124f12f6828241d88bb86f9e9f6bf965af2b3b5a298ddd8a2973277eec55",
  "kernels/trellis/gate-up-codebook.ts": "c0aec9b9f951c5fd94270a52f4bf4da43e7fb7d0ecda6ccd65e20571e8931f45",
  "kernels/trellis/launch.ts": "da0a19922ccc93935c5cab869de040d4b759271df08496c88c7371c386724052",
  "layers/trellis-codebook-gate-up.ts": "28d27e84cb669c18a4532723e8dbc66786e386a4390454121640a3e14e5526f5",
  "layers/trellis-codebook-down.ts": "48752c429a3ab558131f95c6d361b15b6795c3bb6ee161f8afdca71609437898",
  "models/qwen/written-layers.ts": "fc298c473e61a1e851443ff8263b65c0c15c27172b9db6615f73f486414bfb38",
};
const proofFiles = [
  "contracts/mlx/M1Max.lean",
  "contracts/mlx/Memory.lean",
  "contracts/mlx/Numerics.lean",
  "contracts/mlx/FlushToZero.lean",
  "contracts/mlx/BFloat16.lean",
  "contracts/mlx/BFloat16Arithmetic.lean",
  "contracts/mlx/Reduction.lean",
  "contracts/mlx/DotProduct.lean",
  "contracts/mlx/FlushedDotProduct.lean",
  "contracts/mlx/SquaredSum.lean",
  "contracts/mlx/NormalizationSchedule.lean",
  "contracts/mlx/Normalization.lean",
  "contracts/mlx/NormalizationExecution.lean",
  "kernels/trellis/Codebook.lean",
  "kernels/trellis/CodebookLoad.lean",
  "kernels/trellis/CodebookFlushToZero.lean",
  "kernels/trellis/CodebookAccuracy.lean",
  "kernels/trellis/CodebookInitialization.lean",
  "kernels/trellis/DownCodebook.lean",
  "kernels/trellis/GateUpCodebook.lean",
  "kernels/trellis/DownExecution.lean",
  "kernels/trellis/DownMemory.lean",
  "kernels/trellis/DownAccuracy.lean",
  "kernels/trellis/DownReduction.lean",
  "kernels/trellis/GateUpExecution.lean",
  "kernels/trellis/GateUpAccuracy.lean",
  "kernels/trellis/GateUpFlushToZero.lean",
  "kernels/trellis/DownFlushToZero.lean",
  "kernels/trellis/GateUpTail.lean",
  "kernels/trellis/SwiGLUStability.lean",
  "kernels/trellis/GateUpTailFlushToZero.lean",
  "kernels/trellis/GateUpSplitTail.lean",
  "layers/TrellisCodebookGateUp.lean",
  "layers/TrellisCodebookDown.lean",
  "models/qwen/WrittenLayers.lean",
];

const discoveredProofs = await Array.fromAsync(new Bun.Glob("**/*.lean").scan({
  cwd: new URL("./", import.meta.url).pathname, onlyFiles: true, dot: false,
}));
for (const file of discoveredProofs) {
  if (!proofFiles.includes(file)) throw new Error(`Proof module is missing from the checked build: ${file}`);
}

function checkSource(name: keyof typeof sourcePins, source: string): void {
  if (new Bun.CryptoHasher("sha256").update(source).digest("hex") !== sourcePins[name]) {
    throw new Error(`${name} changed: review its translation and proofs before updating the source pin.`);
  }
}

// Audit every theorem in the current module, including private helpers and newly
// added theorems. collectAxioms walks transitive dependencies in the kernel env.
const audit = `
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let names := env.constants.foldStage2 (fun ns name info =>
    if info.isTheorem then ns.push name else ns) (#[] : Array Name)
  if names.isEmpty then throwError "No proof declarations found"
  for name in names do
    let axioms ← collectAxioms name
    for axiomName in axioms do
      unless #[\`propext, \`Classical.choice, \`Quot.sound].contains axiomName do
        throwError "Forbidden axiom {axiomName} in {name}"
  logInfo m!"Axiom audit passed: {names.size} theorems"
`;

async function run(args: string[], leanPath?: string) {
  const child = Bun.spawn([lean, ...args], {
    cwd: root.pathname, stdout: "pipe", stderr: "pipe",
    env: { ...process.env, ...(leanPath ? { LEAN_PATH: [leanPath, dependencyPath].filter(Boolean).join(delimiter) } : {}) },
  });
  const [stdout, stderr, code] = await Promise.all([
    new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited,
  ]);
  return { output: stdout + stderr, code };
}

const version = await run(["--version"]);
if (version.code !== 0 || !version.output.includes("version 4.35.0,")) {
  throw new Error(`Expected Lean 4.35.0; got ${version.output}`);
}
async function runLake(args: string[]) {
  const child = Bun.spawn([lake, ...args], {
    cwd: new URL("./", import.meta.url).pathname, stdout: "pipe", stderr: "pipe",
    env: { ...process.env, MATHLIB_NO_CACHE_ON_UPDATE: "1" },
  });
  const [stdout, stderr, code] = await Promise.all([
    new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited,
  ]);
  if (code !== 0) throw new Error(`Lake failed: ${stdout}${stderr}`);
  return stdout;
}
console.log("Building pinned numerical proof dependencies...");
process.stdout.write(await runLake(["build",
  "FloatLib.Floats.Formats.BinaryInterchange.Analysis.StandardModel",
  "FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.LeanModel.MulDiv",
  "FloatLib.Floats.Formats.IEEE754.Native.Model",
  "FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Proof",
  "FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.Constants",
  "FloatLib.Floats.Formats.BinaryInterchange.Operations.Classification.Semantics",
  "FloatLib.Floats.Formats.BinaryInterchange.Operations.Compare.Proof",
  "Mathlib.Analysis.SpecialFunctions.Exp",
  "Mathlib.Analysis.SpecialFunctions.ExpDeriv",
  "Mathlib.Analysis.Calculus.MeanValue",
  "Mathlib.Analysis.Calculus.Deriv.Inv",
  "Mathlib.Analysis.Real.Sqrt",
]));
dependencyPath = (await runLake(["env", "printenv", "LEAN_PATH"])).trim();
const sources = new Map<keyof typeof sourcePins, string>();
for (const name of Object.keys(sourcePins) as (keyof typeof sourcePins)[]) {
  const source = await Bun.file(new URL(`packages/inference/src/${name}`, root)).text();
  checkSource(name, source);
  sources.set(name, source);
}

const dir = await mkdtemp(join(tmpdir(), "mlx-bun-proof-"));
async function checkProof(name: string, text: string, build = false) {
  const path = join(dir, name);
  await mkdir(dirname(path), { recursive: true });
  await Bun.write(path, `import Lean\n${text}\n${audit}`);
  return run(["-R", dir, ...(build ? ["-o", path.replace(/\.lean$/, ".olean")] : []), path], dir);
}

try {
  for (const file of proofFiles) {
    console.log(`Checking ${file}...`);
    const result = await checkProof(file, await Bun.file(new URL(file, import.meta.url)).text(), true);
    process.stdout.write(result.output);
    if (result.code !== 0 || result.output.includes("sorryAx") || !result.output.includes("Axiom audit passed:")) {
      throw new Error(`Lean did not establish ${file} without forbidden assumptions.`);
    }
  }

  if (process.argv.includes("--self-test")) {
    for (const [name, source] of sources) {
      let rejected = false;
      try { checkSource(name, source + "\n// source mutation\n"); } catch { rejected = true; }
      if (!rejected) throw new Error(`Source mutation was accepted: ${name}`);
    }
    // Small prefix retains the arithmetic contract and its dependencies, avoiding
    // repeated exhaustive floating-point checking for deliberately broken proofs.
    const source = await Bun.file(new URL("kernels/trellis/Codebook.lean", import.meta.url)).text();
    const proof = source.split("namespace PackedState")[0]! + "\nend Inference.Kernels.Trellis.Codebook\n";
    const wrongClaim = proof.replace("implementation s = specification s :=", "implementation s = specification s + 1 :=");
    const assumedClaim = proof.replace(
      /theorem correct[\s\S]*?(?=\ntheorem range)/,
      "axiom injected : ∀ s, implementation s = specification s\n" +
      "theorem correct (s : BitVec 32) : implementation s = specification s := injected s\n",
    );
    const nativeProof = proof.replace("import Std\n", "import Std.Tactic.BVDecide\n").replace(
      /theorem packedSum_bound[\s\S]*?(?=\ntheorem correct)/,
      "theorem packedSum_bound (x : BitVec 32) : packedSum x ≤ 1020 := by\n  unfold packedSum\n  bv_decide\n",
    );
    const uncheckedTheorem = proof + "\nnamespace Inference\naxiom newAssumption : False\ntheorem newClaim : False := newAssumption\nend Inference\n";
    const proofHole = proof + "\nnamespace Inference\ntheorem missingProof : False := by sorry\nend Inference\n";
    for (const [name, text, axiomCheck] of [
      ["FalseContract", wrongClaim, false],
      ["AssumedContract", assumedClaim, true],
      ["NativeComputation", nativeProof, true],
      ["NewUncheckedTheorem", uncheckedTheorem, true],
      ["ProofHole", proofHole, true],
    ] as const) {
      if (proof === text) throw new Error(`Negative proof test did not alter its target: ${name}`);
      const rejected = await checkProof(`${name}.lean`, text);
      if (rejected.code === 0) throw new Error(`Lean accepted ${name}.`);
      if (axiomCheck && !rejected.output.includes("Forbidden axiom")) {
        throw new Error(`${name} failed before the axiom audit:\n${rejected.output}`);
      }
    }
    console.log("Negative checks passed: source drift, false claims, custom axioms, native computation, unaudited additions, and proof holes rejected.");
  }
} finally {
  await rm(dir, { recursive: true, force: true });
}
console.log(`Checked ${proofFiles.length} proof modules: decoding, packed windows, conditional M1 loads, initialization and scratch stores, flushed binary32/bfloat16 arithmetic, both SwiGLU tails including exponential overflow, single-row projection error propagation, and conditional RMS arithmetic bounds.`);
console.log("Permitted axioms: propext, Classical.choice, Quot.sound. No native proof evaluation.");
console.log("Scope: these are conditional numerical bounds for reviewed source models. Hardware/compiler conformance, model-wide range bounds, native memory/dispatch, delta adapters, remaining kernels, and the complete inference graph remain unproved.");
