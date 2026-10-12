// The Neural Engine install step as the app runs it (src/cli/ane-install.ts): the first launch of a version runs
// `mlx-bun compile-ane` once, from the version record the verb leaves under MLX_BUN_HOME, and a failed step is
// reported, not fatal. The verb's help needs no native library. Opt-in: MLX_BUN_TEST_ANE=1 spawns the real verb
// (it compiles this Mac's programs, or finds them in the cache) under a temporary MLX_BUN_HOME.
import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { configureRuntime } from "@mlx-bun/inference/runtime/config";
import { ensureAneInstall } from "../src/cli/ane-install";
import { storagePath } from "../src/storage/paths";
import pkg from "../package.json" with { type: "json" };

const entry = resolve(import.meta.dir, "../src/cli/main.ts");
let home = "", restore = () => {};
beforeEach(() => { home = mkdtempSync(join(tmpdir(), "mlx-ane-install-")); restore = configureRuntime({ MLX_BUN_HOME: home }); });
afterEach(() => { restore(); rmSync(home, { recursive: true, force: true }); });

async function cli(env: Record<string, string>, ...args: string[]) {
  const child = Bun.spawn([process.execPath, "--no-env-file", entry, ...args], { stdout: "pipe", stderr: "pipe",
    env: { ...process.env, MLX_BUN_HOME: home, HF_HUB_OFFLINE: "1", NO_COLOR: "1", ...env } });
  const [out, err, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  return { out, err, code };
}

test("the first launch of a version runs the install step once; a version's record skips it; another version runs it again", async () => {
  const runs: string[] = [], errors: string[] = [];
  const run = async () => { runs.push("compile-ane"); return 0; };
  await ensureAneInstall({ run, error: message => errors.push(message) });
  expect(runs).toEqual(["compile-ane"]);
  const record = storagePath("aneInstall");
  expect(record).toBe(join(home, "cache/ane-install.json"));
  mkdirSync(dirname(record), { recursive: true });
  writeFileSync(record, JSON.stringify({ version: pkg.version, device: "applegpu_g16s", prefillChunkTokens: 2048, programs: [] }));
  await ensureAneInstall({ run, error: message => errors.push(message) });
  expect(runs).toEqual(["compile-ane"]);
  writeFileSync(record, JSON.stringify({ version: "0.0.0-older" }));
  await ensureAneInstall({ run, error: message => errors.push(message) });
  writeFileSync(record, "not json");
  await ensureAneInstall({ run, error: message => errors.push(message) });
  expect(runs).toEqual(["compile-ane", "compile-ane", "compile-ane"]);
  expect(errors).toEqual([]);
});

test("a failed install step is reported, naming the verb, and the launch goes on", async () => {
  const errors: string[] = [];
  await ensureAneInstall({ run: async () => 3, error: message => errors.push(message) });
  expect(errors).toEqual(["mlx-bun: the Neural Engine install step failed (exit 3); a model that uses the Neural Engine will not load until `mlx-bun compile-ane` succeeds"]);
  expect(existsSync(storagePath("aneInstall"))).toBe(false);
});

test("compile-ane --help documents the step without the native library", async () => {
  const help = await cli({ MLX_BUN_LIBMLXC: "/nonexistent/libmlxc.dylib" }, "compile-ane", "--help");
  expect(help.code).toBe(0);
  expect(help.out).toStartWith("mlx-bun compile-ane — Compile the Neural Engine programs this Mac's models use, once (the install step)\n\nUsage: mlx-bun compile-ane [options]\n");
  for (const text of ["--check", "never", "compiles", "MLX_BUN_RD_PREFILL_CHUNK"]) expect(help.out).toContain(text);
  expect((await cli({ MLX_BUN_LIBMLXC: "/nonexistent/libmlxc.dylib" }, "--help")).out).toMatch(/compile-ane\s+Compile the Neural Engine programs/);
  expect((await cli({}, "compile-ane", "extra")).code).toBe(1);
});

test.skipIf(!process.env.MLX_BUN_TEST_ANE)("the real verb compiles (or finds) every program, records the version, and --check then finds them all", async () => {
  const compiled = await cli({}, "compile-ane");
  expect({ code: compiled.code, err: compiled.err }).toEqual({ code: 0, err: "" });
  const record = JSON.parse(readFileSync(join(home, "cache/ane-install.json"), "utf8")) as { version: string; programs: string[] };
  expect(record.version).toBe(pkg.version);
  const listed = compiled.out.split("\n").filter(line => /^ {2}(compiled|cached) {2}/.test(line));
  expect(listed).toHaveLength(record.programs.length);
  const check = await cli({}, "compile-ane", "--check");
  expect(check.code).toBe(0);
  expect(check.out.split("\n").filter(line => line.startsWith("  missing"))).toEqual([]);
  console.error(compiled.out.trim());
}, 300_000);
