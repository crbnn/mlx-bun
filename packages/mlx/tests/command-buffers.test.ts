import { expect, test } from "bun:test";
import { COMMAND_BUFFER_LIMITS, commandBufferEnvironment } from "@mlx-bun/mlx/command-buffers";

const LIMITS = ["MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER"] as const;

/** Load the binding in a fresh process with `environment` and read back the native variables MLX reads, and the
 *  architecture MLX then reports. */
function loaded(environment: Record<string, string>): { values: (string | null)[]; architecture: string } {
  const inherited = Object.fromEntries(Object.entries(process.env)
    .filter(([name]) => ![...LIMITS, "MLX_METAL_GPU_ARCH"].includes(name)));
  const child = Bun.spawnSync([process.execPath, "--no-env-file", "-e", `
    import { CString, dlopen } from "bun:ffi";
    import { deviceArchitecture } from ${JSON.stringify(Bun.resolveSync("@mlx-bun/mlx/ffi", import.meta.dir))};
    const { getenv } = dlopen("/usr/lib/libSystem.B.dylib", { getenv: { args: ["ptr"], returns: "ptr" } }).symbols;
    const values = ${JSON.stringify(LIMITS)}.map(name => {
      const value = getenv(Buffer.from(name + "\\0"));
      return value ? new CString(value).toString() : null;
    });
    console.log(JSON.stringify({ values, architecture: deviceArchitecture() }));
  `], { env: { ...inherited, ...environment }, stdout: "pipe", stderr: "pipe" });
  expect(child.stderr.toString()).toBe("");
  expect(child.exitCode).toBe(0);
  return JSON.parse(child.stdout.toString());
}

test("the M1 Max takes its measured command-buffer limits; every other architecture keeps MLX's own", () => {
  expect(Object.keys(COMMAND_BUFFER_LIMITS)).toEqual(["applegpu_g13s"]);
  expect(commandBufferEnvironment("applegpu_g13s")).toEqual({ MLX_MAX_OPS_PER_BUFFER: "400", MLX_MAX_MB_PER_BUFFER: "100" });
  for (const architecture of ["applegpu_g16s", "applegpu_g13g", "applegpu_g15s", ""])
    expect(commandBufferEnvironment(architecture)).toEqual({});
});

test("loading the binding sets this machine's limits from the architecture MLX then reports", () => {
  const { values, architecture } = loaded({});
  const expected = commandBufferEnvironment(architecture);
  expect(values).toEqual(LIMITS.map(name => expected[name] ?? null));
});

test("an architecture MLX is told to use decides the limits, and a variable the user set wins", () => {
  expect(loaded({ MLX_METAL_GPU_ARCH: "applegpu_g13s" })).toEqual({ values: ["400", "100"], architecture: "applegpu_g13s" });
  expect(loaded({ MLX_METAL_GPU_ARCH: "applegpu_g16s" })).toEqual({ values: [null, null], architecture: "applegpu_g16s" });
  expect(loaded({ MLX_METAL_GPU_ARCH: "applegpu_g13s", MLX_MAX_OPS_PER_BUFFER: "123" }).values).toEqual(["123", "100"]);
  expect(loaded({ MLX_METAL_GPU_ARCH: "applegpu_g13s", MLX_MAX_OPS_PER_BUFFER: "50", MLX_MAX_MB_PER_BUFFER: "50" }).values)
    .toEqual(["50", "50"]);
});
