// ANE program creation in a process that has created, run and kept many programs of
// different shapes (PLAN C3). The sequence runs in a child process, killed when it does not
// finish in time, because a blocked creation is a native call this process could not
// interrupt. Opt-in: set MLX_BUN_TEST_ANE=1 on a real Apple Silicon machine (every M-series
// chip has a Neural Engine; a virtual machine such as a hosted CI runner loads the framework
// but cannot compile). Unset skips, which is not evidence of a pass.
import { expect, test } from "bun:test";
import { resolve } from "node:path";
import { Dtype } from "@mlx-bun/mlx";
import { AneLinear } from "../../src/kernels/ane/linear";

// About 10 s on an M4 Pro with every program already in the system compile cache; the rest
// allows a fresh machine its first compiles. The failure this guards blocked for over 20 minutes.
const LIMIT_MS = 240_000;

test.skipIf(!process.env.MLX_BUN_TEST_ANE)("programs of different shapes created, run and held in sequence in one process never block the next creation", async () => {
  const child = Bun.spawn([process.execPath, resolve(import.meta.dir, "ane-program-sequence.ts")],
    { stdout: "pipe", stderr: "pipe", timeout: LIMIT_MS, killSignal: "SIGKILL" });
  const [code, out, log] = await Promise.all([child.exited, new Response(child.stdout).text(), new Response(child.stderr).text()]);
  expect(child.signalCode, `program creation blocked; the child's last progress:\n${log.slice(-1500)}`).toBeNull();
  expect(code, log).toBe(0);
  expect(out.trim()).toBe("done");
}, LIMIT_MS + 30_000);

test.skipIf(!process.env.MLX_BUN_TEST_ANE)("a shared program with the same MIL but other weights stays its own program", () => {
  // INT8 linears [64, 64] over 32 columns of ones: output row 0 is the weight row 0 sum,
  // -3 for `three` and -2 for `two`. `same` repeats `three` exactly, so it runs on its program.
  const cin = 64, cout = 64, seq = 32, ones = new Uint8Array(cin * seq * 2);
  new Uint16Array(ones.buffer).fill(0x3c00);
  const scales = new Uint16Array(cout).fill(0x3c00);
  const rows = (period: number) => Int8Array.from({ length: cout * cin }, (_, i) => i % cin % period - (period >> 1));
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
