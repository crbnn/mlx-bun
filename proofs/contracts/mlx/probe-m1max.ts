/** Finite hardware observations, not a proof of the universal chip contract.
 * Run with Bun; the outer process acquires Josh's GPU lock before loading MLX.
 * MLX_BUN_LIBMLXC may select an existing native build. */
import { homedir } from "node:os";
import { join } from "node:path";

if (!process.argv.includes("--under-gpu-lock")) {
  const child = Bun.spawn([
    join(homedir(), "mlx-bun-perf/gpu.sh"), process.execPath, import.meta.path, "--under-gpu-lock",
  ], { stdout: "inherit", stderr: "inherit", stdin: "ignore" });
  process.exit(await child.exited);
}

const chip = Bun.spawnSync(["sysctl", "-n", "machdep.cpu.brand_string"]);
if (chip.exitCode !== 0 || chip.stdout.toString().trim() !== "Apple M1 Max") {
  throw new Error("This probe's recorded expectations target Apple M1 Max.");
}
const { MlxArray } = await import("../../../packages/mlx/src/array.ts");
const { MetalKernel } = await import("../../../packages/mlx/src/metal-kernel.ts");
const { Dtype } = await import("../../../packages/mlx/src/ffi.ts");

// Input triples; expected multiply, add, fma, and bf16-cast/widen output bits.
const cases = [
  { input: [0x00800000, 0x3f000000, 0], expected: [0, 0x3f000000, 0, 0x00800000] },
  { input: [1, 0x40000000, 0], expected: [0, 0x40000000, 0, 0] },
  { input: [0x007fffff, 1, 0], expected: [0, 0, 0, 0x00800000] },
  { input: [0x80800000, 0x3f000000, 0], expected: [0x80000000, 0x3f000000, 0x80000000, 0x80800000] },
  { input: [0x3f800001, 0x3f7ffffe, 0xbf800000], expected: [0x3f800000, 0x40000000, 0xa8800000, 0x3f800000] },
  { input: [0x00010000, 0x3f800000, 0], expected: [0, 0x3f800000, 0, 0x00010000] },
] as const;
function floats(column: 0 | 1 | 2): Float32Array {
  return new Float32Array(Uint32Array.from(cases, row => row.input[column]).buffer);
}
using a = MlxArray.fromFloat32(floats(0), [cases.length]);
using b = MlxArray.fromFloat32(floats(1), [cases.length]);
using c = MlxArray.fromFloat32(floats(2), [cases.length]);
const kernel = new MetalKernel({
  name: "proof_m1max_float_contract_probe_v1",
  inputNames: ["a", "b", "c"], outputNames: ["out"],
  source: String.raw`
    uint i = thread_position_in_grid.x;
    out[4*i] = as_type<uint>(a[i] * b[i]);
    out[4*i+1] = as_type<uint>(a[i] + b[i]);
    out[4*i+2] = as_type<uint>(metal::fma(a[i], b[i], c[i]));
    out[4*i+3] = as_type<uint>(float(T(a[i])));
  `,
});
try {
  const [output] = kernel.apply([a, b, c], {
    outputs: [{ shape: [cases.length * 4], dtype: Dtype.uint32 }],
    grid: [cases.length, 1, 1], threadGroup: [32, 1, 1], templateDtypes: { T: Dtype.bfloat16 },
  });
  using out = output!;
  const values = out.toIntTokens();
  for (const [i, row] of cases.entries()) {
    for (const [j, expected] of row.expected.entries()) {
      const actual = values[i * 4 + j]! >>> 0;
      if (actual !== expected) {
        throw new Error(`Case ${i}, operation ${j}: 0x${actual.toString(16)} != 0x${expected.toString(16)}`);
      }
    }
  }
  console.log("M1 Max observations matched: arithmetic flushing, fused cancellation, and preserved bf16 subnormal cast.");
} finally {
  kernel.dispose();
}
