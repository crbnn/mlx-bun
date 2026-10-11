// The generated Gemma 4 graphs are compiled from a composition by
// scripts/gen-gemma4.ts. This regenerates every composition from its committed
// artifact inputs (tests/fixtures/gemma4-graphs: config.json, kv_config.json and
// the index names the generator reads; no weights) into a temp directory and
// fails when a committed file differs, so a change to the generator, or to the
// blocks it composes, cannot leave a stale graph. Whether a regenerated graph
// computes the same bits is tests/models/gemma4-generated-identity.test.ts
// (synthetic weights) and tests/parity/gemma4-generated.test.ts (real weights).
import { expect, test } from "bun:test";
import { cpSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GEMMA4_COMPOSITIONS, GENERATED_DIR } from "../../scripts/gen-gemma4";
import { loadModelConfig } from "../../src/artifacts/config";
import { GENERATED_GEMMA_FINGERPRINTS } from "../../src/models/gemma4/family";
import { GEMMA4_GENERATED_GRAPHS } from "../../src/models/gemma4/generated";
import { resolveKvScheme } from "../../src/state/kv-scheme";
import { kvStorageOf } from "../../src/state/kv-storage";

const FIXTURES = join(import.meta.dir, "../fixtures/gemma4-graphs");
const SCRIPT = join(import.meta.dir, "../../scripts/gen-gemma4.ts");
const stems = Object.keys(GEMMA4_COMPOSITIONS);
const REPO: Readonly<Record<string, string>> = {
  "gemma4-12b": "gemma-4-12B-it-OptiQ-4bit", "gemma4-e4b": "gemma-4-e4b-it-OptiQ-4bit", "gemma4-26b": "gemma-4-26B-A4B-it-OptiQ-4bit",
};

/** The hub-layout snapshot directory holding a composition's inputs. */
function snapshot(stem: string, root = FIXTURES): string {
  const repo = join(root, `models--mlx-community--${REPO[stem]}`, "snapshots");
  const [hash] = readdirSync(repo);
  return join(repo, hash!);
}

async function run(...args: string[]) {
  const child = Bun.spawn([process.execPath, SCRIPT, ...args], { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  return { stdout, stderr, code };
}

test("every generated Gemma 4 graph equals what the generator emits from its committed inputs", async () => {
  const out = mkdtempSync(join(tmpdir(), "gen-gemma4-"));
  try {
    for (const stem of stems) {
      const result = await run(snapshot(stem), stem, "--out", out);
      expect({ stem, code: result.code, stderr: result.stderr }).toEqual({ stem, code: 0, stderr: "" });
      const committed = readFileSync(join(GENERATED_DIR, `${stem}.ts`), "utf8");
      const regenerated = readFileSync(join(out, `${stem}.ts`), "utf8");
      // A difference here means: run `bun packages/inference/scripts/gen-gemma4.ts <snapshot> ${stem}` (the
      // fixture snapshot works) and re-run tests/models/gemma4-generated-identity.test.ts and, on real weights,
      // tests/parity/gemma4-generated.test.ts.
      expect({ stem, current: regenerated === committed }).toEqual({ stem, current: true });
    }
  } finally { rmSync(out, { recursive: true, force: true }); }
});

test("the index lists each generated graph with the composition the loader keys on", async () => {
  expect(readdirSync(GENERATED_DIR).filter(file => file !== "index.ts").sort()).toEqual(stems.map(stem => `${stem}.ts`).sort());
  expect(GEMMA4_GENERATED_GRAPHS.map(entry => entry.stem as string).sort()).toEqual([...stems].sort());
  for (const entry of GEMMA4_GENERATED_GRAPHS) {
    const module = await import(`../../src/models/gemma4/generated/${entry.stem}`);
    expect(entry.Graph).toBe(module.GeneratedGemma4);
    expect(entry.composition).toBe(module.COMPOSITION);
    // An entire graph: it extends nothing.
    expect(Object.getPrototypeOf(entry.Graph.prototype)).toBe(Object.prototype);
    expect(entry.composition.fingerprint).toBe(GENERATED_GEMMA_FINGERPRINTS[GEMMA4_COMPOSITIONS[entry.stem]!.key]);
    // The KV scheme is the one `--kv-quant config` resolves for the artifact.
    const config = await loadModelConfig(snapshot(entry.stem));
    expect(entry.composition.kv).toEqual(kvStorageOf(resolveKvScheme({ override: "config", config: config.kvQuant })));
  }
});

test("a source whose fingerprint is not the registered one is refused", async () => {
  const root = mkdtempSync(join(tmpdir(), "gen-gemma4-drift-"));
  try {
    cpSync(FIXTURES, root, { recursive: true });
    const dir = snapshot("gemma4-e4b", root);
    const kv = JSON.parse(readFileSync(join(dir, "kv_config.json"), "utf8"));
    kv[0].bits = kv[0].bits === 4 ? 8 : 4;
    writeFileSync(join(dir, "kv_config.json"), JSON.stringify(kv));
    const result = await run(dir, "gemma4-e4b", "--out", root);
    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain("is not the registered");
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("--help prints usage without a model; unknown compositions and missing arguments exit non-zero", async () => {
  const help = await run("--help");
  expect(help.code).toBe(0);
  expect(help.stdout).toContain("Usage: bun packages/inference/scripts/gen-gemma4.ts <model-dir>");
  for (const args of [[], [snapshot("gemma4-12b")], [snapshot("gemma4-12b"), "gemma4-nope"]]) {
    const result = await run(...args);
    expect({ args: args.length, code: result.code }).toEqual({ args: args.length, code: 1 });
    expect(result.stderr).toContain("Usage:");
  }
});
