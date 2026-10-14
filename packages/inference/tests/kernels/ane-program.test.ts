// ANE programs (kernels/ane/linear): compiling is the install step and creating only loads
// (PLAN C3), and creation in a process that has created, run and kept many programs of
// different shapes never blocks. The sequence runs in a child process, killed when it does not
// finish in time, because a blocked creation is a native call this process could not
// interrupt. Opt-in: set MLX_BUN_TEST_ANE=1 on a real Apple Silicon machine (every M-series
// chip has a Neural Engine; a virtual machine such as a hosted CI runner loads the framework
// but cannot compile). Unset skips, which is not evidence of a pass.
import { expect, test } from "bun:test";
import { resolve } from "node:path";
import { Dtype } from "@mlx-bun/mlx";
import { AneBridgeError, AneLinear, AneProgram, int8LinearProgram, streamedMatmulProgram } from "../../src/kernels/ane/linear";

const ANE = !!process.env.MLX_BUN_TEST_ANE;
// About 10 s on an M4 Pro with every program already in the system compile cache; the rest
// allows a fresh machine its first compiles. The failure this guards blocked for over 20 minutes.
const LIMIT_MS = 240_000;

/** The ANE driver's direct-path clients held on the whole machine (one per loaded program). */
function clients(): number {
  const listing = Bun.spawnSync(["ioreg", "-r", "-c", "H1xANELoadBalancerDirectPathClient", "-d", "1"]).stdout.toString();
  return listing.split("\n").filter(line => line.startsWith("+-o")).length;
}

test.skipIf(!ANE)("create only loads: a program missing from the compile cache fails naming the install step and compiles nothing; compiled once, it loads", () => {
  const cin = 384, cout = 192, seq = 64, spec = streamedMatmulProgram(cin, cout, seq);
  AneProgram.purge(spec);
  expect(AneProgram.cached(spec)).toBe(false);
  expect(() => AneProgram.create(spec)).toThrow(AneBridgeError);
  expect(() => AneProgram.create(spec)).toThrow(/is not compiled on this Mac; run the install step, `mlx-bun compile-ane`, once/);
  expect(AneProgram.cached(spec)).toBe(false);
  let started = performance.now();
  expect(AneProgram.compile(spec)).toBe("compiled");
  const compileMs = performance.now() - started;
  expect(AneProgram.cached(spec)).toBe(true);
  started = performance.now();
  expect(AneProgram.compile(spec)).toBe("cached");
  const cachedMs = performance.now() - started;
  started = performance.now();
  using program = AneProgram.create(spec);
  const loadMs = performance.now() - started;
  // w and x all ones: every output is the reduction width.
  new Uint16Array(program.inputView(0).buffer, program.inputView(0).byteOffset, cout * cin).fill(0x3c00);
  new Uint16Array(program.inputView(1).buffer, program.inputView(1).byteOffset, seq * cin).fill(0x3c00);
  program.evalAsync();
  program.wait();
  using out = program.output([seq, cout]);
  using wide = out.astype(Dtype.float32);
  expect(new Set(wide.toFloat32())).toEqual(new Set([cin]));
  console.error(`[ane] ${spec.label}: compile ${compileMs.toFixed(1)} ms, cache check ${cachedMs.toFixed(2)} ms, load ${loadMs.toFixed(1)} ms`);
  AneProgram.purge(spec);
  expect(AneProgram.cached(spec)).toBe(false);
}, 60_000);

test.skipIf(!ANE)("compiling holds none of the ANE driver's clients", () => {
  const specs = Array.from({ length: 16 }, (_, i) => streamedMatmulProgram(256, 64, 32 * (i + 1)));
  for (const spec of specs) AneProgram.purge(spec);
  const before = clients();
  for (const spec of specs) expect(AneProgram.compile(spec)).toBe("compiled");
  const after = clients();
  for (const spec of specs) AneProgram.purge(spec);
  // Machine-wide, so another process may load or release programs meanwhile; one client per compile would add 16.
  expect(after - before).toBeLessThan(specs.length / 2);
}, 120_000);

test.skipIf(!ANE)("programs of different shapes created, run and held in sequence in one process never block the next creation", async () => {
  const child = Bun.spawn([process.execPath, resolve(import.meta.dir, "ane-program-sequence.ts")],
    { stdout: "pipe", stderr: "pipe", timeout: LIMIT_MS, killSignal: "SIGKILL" });
  const [code, out, log] = await Promise.all([child.exited, new Response(child.stdout).text(), new Response(child.stderr).text()]);
  expect(child.signalCode, `program creation blocked; the child's last progress:\n${log.slice(-1500)}`).toBeNull();
  expect(code, log).toBe(0);
  expect(out.trim()).toBe("done");
}, LIMIT_MS + 30_000);

test.skipIf(!ANE)("a shared program with the same MIL but other weights stays its own program", () => {
  // INT8 linears [64, 64] over 32 columns of ones: output row 0 is the weight row 0 sum,
  // -3 for `three` and -2 for `two`. `same` repeats `three` exactly, so it runs on its program.
  const cin = 64, cout = 64, seq = 32, ones = new Uint8Array(cin * seq * 2);
  new Uint16Array(ones.buffer).fill(0x3c00);
  const scales = new Uint16Array(cout).fill(0x3c00);
  const rows = (period: number) => Int8Array.from({ length: cout * cin }, (_, i) => i % cin % period - (period >> 1));
  for (const period of [7, 5]) AneProgram.compile(int8LinearProgram(cin, cout, seq, rows(period), scales));
  using three = AneLinear.create(cin, cout, seq, rows(7), scales);
  using two = AneLinear.create(cin, cout, seq, rows(5), scales, three);
  using same = AneLinear.create(cin, cout, seq, rows(7), scales, three);
  const first = (linear: AneLinear) => {
    linear.writeInput(ones);
    linear.evalAsync();
    linear.wait();
    using out = linear.output();
    using wide = out.astype(Dtype.float32);
    return wide.toFloat32()[0];
  };
  expect([three, two, same, two, three].map(first)).toEqual([-3, -2, -3, -2, -3]);
}, 60_000);

test.skipIf(!ANE)("programs on one set of buffers share one wrap per buffer, made once, which outlives a disposed lender", () => {
  // y = x · wᵀ over ones: every output is the reduction width.
  const cin = 256, cout = 128, wide = streamedMatmulProgram(cin, cout, 64), narrow = streamedMatmulProgram(cin, cout, 32);
  for (const spec of [wide, narrow]) AneProgram.compile(spec);
  const lender = AneProgram.create(wide);
  using borrower = AneProgram.create(narrow, lender);
  const weights = lender.inputArray(0), input = lender.inputArray(1);
  expect(borrower.inputArray(0)).toBe(weights);
  expect(borrower.inputArray(1)).toBe(input);
  expect(lender.inputArray(0)).toBe(weights);
  expect(input.size * 2).toBeGreaterThanOrEqual(wide.inputBytes[1]!);
  lender.dispose();
  new Uint16Array(borrower.inputView(0).buffer, borrower.inputView(0).byteOffset, cout * cin).fill(0x3c00);
  new Uint16Array(borrower.inputView(1).buffer, borrower.inputView(1).byteOffset, 32 * cin).fill(0x3c00);
  borrower.evalAsync();
  borrower.wait();
  using out = borrower.output([32, cout]);
  using f32 = out.astype(Dtype.float32);
  expect(new Set(f32.toFloat32())).toEqual(new Set([cin]));
  expect(borrower.inputArray(1)).toBe(input);
}, 60_000);
