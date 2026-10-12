// The Qwen3.8 Trellis graphs are compiled from a composition by scripts/gen-qwen38.ts. This regenerates every
// composition from the committed artifact description (tests/fixtures/qwen38-graphs: the quant table and stored
// layouts --describe read from the artifact; no weights) into a temp directory and fails when a committed file
// differs, so a change to the generator, or to the blocks and layers it composes, cannot leave a stale graph.
// Whether the regenerated graph computes the same bits is tests/models/qwen38-m4pro-identity.test.ts and
// tests/models/qwen38-m1max-identity.test.ts.
import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GENERATED_DIR, QWEN38_COMPOSITIONS, emitQwen38Graph, type Qwen38Description } from "../../scripts/gen-qwen38";

const DESCRIPTION = join(import.meta.dir, "../fixtures/qwen38-graphs/Qwen3.8-27B-Trellis-3.2bpw.json");
const SCRIPT = join(import.meta.dir, "../../scripts/gen-qwen38.ts");

async function run(...args: string[]) {
  const child = Bun.spawn([process.execPath, SCRIPT, ...args], { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  return { stdout, stderr, code };
}

test("every generated Qwen3.8 graph equals what the generator emits from its committed description", async () => {
  const out = mkdtempSync(join(tmpdir(), "gen-qwen38-"));
  try {
    for (const stem of Object.keys(QWEN38_COMPOSITIONS)) {
      const result = await run(DESCRIPTION, stem, "--out", out);
      expect({ stem, code: result.code, stderr: result.stderr }).toEqual({ stem, code: 0, stderr: "" });
      // A difference here means: run `bun packages/inference/scripts/gen-qwen38.ts <description or snapshot> ${stem}`
      // and re-run tests/models/qwen38-m4pro-identity.test.ts (and the real-weight comparison).
      const committed = readFileSync(join(GENERATED_DIR, `${stem}.ts`), "utf8");
      expect({ stem, current: readFileSync(join(out, `${stem}.ts`), "utf8") === committed }).toEqual({ stem, current: true });
    }
  } finally { rmSync(out, { recursive: true, force: true }); }
});

test("a description whose fingerprint is not the composition's is refused", async () => {
  const root = mkdtempSync(join(tmpdir(), "gen-qwen38-drift-"));
  try {
    const description = JSON.parse(readFileSync(DESCRIPTION, "utf8")) as Qwen38Description;
    const other = join(root, "other.json");
    writeFileSync(other, JSON.stringify({ ...description, fingerprint: "0000000000000000" }));
    const result = await run(other, "qwen38-27b-trellis-m4pro", "--out", root);
    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain("is not the composition's");
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("the M1 Max composition's mixedGateUp chooses the mixed-width layer the generator writes; other devices refuse it", () => {
  const description = JSON.parse(readFileSync(DESCRIPTION, "utf8")) as Qwen38Description;
  const stem = "qwen38-27b-trellis-m1max", base = QWEN38_COMPOSITIONS[stem]!;
  const emit = (mixedGateUp: "split" | "fused" | "off") => emitQwen38Graph(description, stem, { [stem]: { ...base, mixedGateUp } });
  // The committed default: the split tail, bit-identical to the two projections plus a separate SwiGLU.
  expect(base.mixedGateUp).toBe("split");
  expect(emit("split")).toBe(readFileSync(join(GENERATED_DIR, `${stem}.ts`), "utf8"));
  expect(emit("fused")).toContain(`new ReduceMixedGateUp(gate, up, "fused")`);
  // The A/B control: those layers keep the generic blocks at one to four rows, and the mixed kernel is not imported.
  const off = emit("off");
  expect(off).toContain("if (spec.gate !== spec.up) return genericStep(layer);");
  expect(off).not.toContain("ReduceMixedGateUp");
  const m4pro = "qwen38-27b-trellis-m4pro";
  expect(() => emitQwen38Graph(description, m4pro, { [m4pro]: { ...QWEN38_COMPOSITIONS[m4pro]!, mixedGateUp: "split" } }))
    .toThrow("does not name");
});

test("--help prints usage without a model; unknown compositions and missing arguments exit non-zero", async () => {
  const help = await run("--help");
  expect(help.code).toBe(0);
  expect(help.stdout).toContain("Usage: bun packages/inference/scripts/gen-qwen38.ts <source>");
  for (const args of [[], [DESCRIPTION], [DESCRIPTION, "qwen38-nope"]]) expect((await run(...args)).code).not.toBe(0);
});
