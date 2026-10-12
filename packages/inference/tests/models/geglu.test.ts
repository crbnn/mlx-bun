// geglu is gelu_approx(gate) * up whether its owner composed the compiled
// closure (the default) or the spelled-out ops (the composition's `compiledGeglu`
// off, read by the graph's constructor), and it consumes both inputs.
import { expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { geglu } from "../../src/layers/geglu";

const bytes = (a: MlxArray): Uint8Array => {
  const c = ops.contiguous(a);
  try { return Uint8Array.from(c.rawBytes()); } finally { c.dispose(); }
};
/** Deterministic values spread over [-4, 4]. */
const ramp = (n: number, phase: number) => Float32Array.from({ length: n }, (_, i) => 4 * Math.sin(i * 0.37 + phase));
const bf16 = (data: Float32Array, shape: number[]) => {
  const wide = MlxArray.fromFloat32(data, shape);
  try { return wide.astype(Dtype.bfloat16); } finally { wide.dispose(); }
};

test("geglu equals gelu_approx(gate) * up compiled and spelled out, and consumes its inputs", () => {
  const spelled = (g: MlxArray, u: MlxArray) => {
    const act = ops.geluApprox(g);
    try { return ops.mul(act, u); } finally { act.dispose(); }
  };
  const gate = bf16(ramp(4096, 0), [4, 1024]), up = bf16(ramp(4096, 1.3), [4, 1024]);
  const expected = spelled(gate, up);
  try {
    for (const compiled of [true, false]) {
      const g = gate.astype(Dtype.bfloat16), u = up.astype(Dtype.bfloat16);
      const out = geglu(g, u, compiled);
      try {
        expect(out.dtype).toBe(Dtype.bfloat16);
        expect(Buffer.compare(bytes(out), bytes(expected))).toBe(0);
        // both inputs were handed over: using a consumed handle throws
        expect(() => g.shape).toThrow();
        expect(() => u.shape).toThrow();
      } finally { out.dispose(); }
    }
  } finally { for (const a of [gate, up, expected]) a.dispose(); }
});
