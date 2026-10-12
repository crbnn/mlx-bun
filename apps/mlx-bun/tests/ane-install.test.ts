// The Neural Engine install step as the app runs it (src/cli/ane-install.ts): before every model load the app runs
// `mlx-bun compile-ane --quiet` in a child process, which checks each program against the system cache and compiles
// what it lacks; no record decides whether it runs. A failed step is reported, not fatal. The verb's help needs no
// native library. Opt-in: MLX_BUN_TEST_ANE=1 spawns the real verb (it compiles this Mac's programs, or finds them in
// the cache).
import { expect, test } from "bun:test";
import { mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { ensureAneInstall } from "../src/cli/ane-install";

const entry = resolve(import.meta.dir, "../src/cli/main.ts");

async function cli(home: string, env: Record<string, string>, ...args: string[]) {
  const child = Bun.spawn([process.execPath, "--no-env-file", entry, ...args], { stdout: "pipe", stderr: "pipe",
    env: { ...process.env, MLX_BUN_HOME: home, HF_HUB_OFFLINE: "1", NO_COLOR: "1", ...env } });
  const [out, err, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  return { out, err, code };
}

test("every launch runs the step: no record skips it", async () => {
  let runs = 0;
  const errors: string[] = [];
  for (let launch = 0; launch < 3; launch++) await ensureAneInstall({ run: async () => { runs++; return 0; }, error: message => errors.push(message) });
  expect(runs).toBe(3);
  expect(errors).toEqual([]);
});

test("a failed step is reported, naming the verb, and the launch goes on", async () => {
  const errors: string[] = [];
  await ensureAneInstall({ run: async () => 3, error: message => errors.push(message) });
  expect(errors).toEqual(["mlx-bun: the Neural Engine install step failed (exit 3); a model that uses the Neural Engine will not load until `mlx-bun compile-ane` succeeds"]);
});

test("compile-ane --help documents the step without the native library", async () => {
  const home = mkdtempSync(join(tmpdir(), "mlx-ane-install-"));
  try {
    const noMlx = { MLX_BUN_LIBMLXC: "/nonexistent/libmlxc.dylib" };
    const help = await cli(home, noMlx, "compile-ane", "--help");
    expect(help.code).toBe(0);
    expect(help.out).toStartWith("mlx-bun compile-ane — Compile the Neural Engine programs this Mac's models use that the system cache lacks (the install step)\n\nUsage: mlx-bun compile-ane [options]\n");
    for (const text of ["--check", "--quiet", "before\nevery model load", "never compiles", "MLX_BUN_RD_PREFILL_CHUNK"]) expect(help.out).toContain(text);
    expect((await cli(home, noMlx, "--help")).out).toMatch(/compile-ane\s+Compile the Neural Engine programs/);
    expect((await cli(home, noMlx, "compile-ane", "extra")).code).toBe(1);
  } finally { rmSync(home, { recursive: true, force: true }); }
});

test.skipIf(!process.env.MLX_BUN_TEST_ANE)("the real verb compiles (or finds) every program and writes nothing; --quiet then prints nothing and --check finds them all", async () => {
  const home = mkdtempSync(join(tmpdir(), "mlx-ane-install-"));
  try {
    const compiled = await cli(home, {}, "compile-ane");
    expect({ code: compiled.code, err: compiled.err }).toEqual({ code: 0, err: "" });
    expect(compiled.out.split("\n").filter(line => /^ {2}(compiled|cached) {2}/.test(line)).length).toBeGreaterThan(0);
    expect(readdirSync(home)).toEqual([]);
    const quiet = await cli(home, {}, "compile-ane", "--quiet");
    expect(quiet).toEqual({ out: "", err: "", code: 0 });
    const check = await cli(home, {}, "compile-ane", "--check");
    expect(check.code).toBe(0);
    expect(check.out.split("\n").filter(line => line.startsWith("  missing"))).toEqual([]);
    console.error(compiled.out.trim());
  } finally { rmSync(home, { recursive: true, force: true }); }
}, 300_000);
