// The Neural Engine install step (models/ane-install): the programs the loader's graph table entries load on a
// device, read from the entries as a function of the composition alone (no model, no artifact files), one program
// per distinct MIL and weights; and, opted in, their compile into the system cache and the load that follows.
// The table tests read data only. Opt-in compile and load: MLX_BUN_TEST_ANE=1 on a real Apple Silicon machine
// (unset skips, which is not evidence of a pass).
import { describe, expect, test } from "bun:test";
import type { Composition } from "../../src/contracts/portable/composition";
import { AneProgram, aneProgramKey, mlpSliceProgram, streamedMatmulProgram } from "../../src/kernels/ane/linear";
import { aneInstallPrograms, compileAnePrograms } from "../../src/models/ane-install";
import { machineComposition } from "../../src/models/composition";
import { GraphTable, type GraphEntry } from "../../src/models/implementation";
import { ANE_WIDTHS as M1_MAX_ANE_WIDTHS, QWEN38_TRELLIS_M1MAX_GRAPH } from "../../src/models/qwen/qwen38-27b-trellis-m1max";
import { ANE_WIDTHS, QWEN38_TRELLIS_M4PRO_GRAPH } from "../../src/models/qwen/qwen38-27b-trellis-m4pro";

const M4_PRO = "applegpu_g16s", M1_MAX = "applegpu_g13s";
const record = (fields: Partial<Composition> = {}): Composition => ({ ...machineComposition(), device: M4_PRO, ...fields });
const labels = (composition: Composition) => aneInstallPrograms({ composition }).map(program => [program.spec.label, program.graphs]);

describe("the install table", () => {
  test("on an M4 Pro the shipped table lists the Qwen3.8 graph's three programs for each of its Neural Engine row ranges, from the composition alone", () => {
    expect(machineComposition().prefillChunkTokens).toBe(2048);
    expect(labels(record())).toEqual([1024, 1536, 2048].flatMap(rows => [
      [`MLP slice 5120×9600, ${rows} rows`, [QWEN38_TRELLIS_M4PRO_GRAPH]],
      [`projection 5120→3392, ${rows} rows`, [QWEN38_TRELLIS_M4PRO_GRAPH]],
      [`projection 6144→2816, ${rows} rows`, [QWEN38_TRELLIS_M4PRO_GRAPH]],
    ]));
    // The artifact's independent projection widths.
    expect(ANE_WIDTHS).toEqual({ hidden: 5120, intermediate: 17408, projections: [[5120, 6144], [6144, 5120]] });
  });

  test("on an M1 Max the shipped table lists the Qwen3.8 graph's MLP slice for each of its Neural Engine row ranges, and from 768 rows its projection splits", () => {
    const programs = [256, 320, 384, 448, 512, 576, 640, 704, 768, 896, 1024, 1280, 1536, 1792, 2048];
    expect(labels(record({ device: M1_MAX }))).toEqual(programs.flatMap(rows => [
      [`MLP slice 5120×${rows <= 768 ? 6144 : 6656}, ${rows} rows, 4 slices`, [QWEN38_TRELLIS_M1MAX_GRAPH]],
      ...(rows < 768 ? [] : [[5120, 3840], [5120, 2304], [6144, 1920], [5120, 4608]].map(([inputs, outputs]) =>
        [`projection ${inputs}→${outputs}, ${rows} rows, 4 slices`, [QWEN38_TRELLIS_M1MAX_GRAPH]])),
    ]));
    expect(M1_MAX_ANE_WIDTHS).toEqual({ hidden: 5120, intermediate: 17408,
      projections: [[5120, 10240], [5120, 6144], [6144, 5120], [5120, 12288]] });
  });

  test("the install list follows the default device and includes its fixed programs regardless of chunk size", () => {
    expect(labels(record({ prefillChunkTokens: 128 }))).toEqual(labels(record()));
    expect(labels(record({ device: "applegpu_g15s" }))).toEqual([]);
  });

  test("entries constrained to another device are skipped; a program two entries load is one program naming both", () => {
    const shared = streamedMatmulProgram(64, 32, 32), own = mlpSliceProgram(64, 32, 32);
    const entry = (id: string, device: string | undefined, programs: () => typeof shared[]): GraphEntry<null, string> =>
      ({ id, serves: device ? { device } : {}, create: () => id, anePrograms: programs });
    const tables = {
      a: new GraphTable([entry("a-m4", M4_PRO, () => [shared, own]), entry("a-generic", undefined, () => [])]),
      b: new GraphTable([entry("b-m4-pro", M4_PRO, () => [shared]), entry("b-m1", M1_MAX, () => [own]), entry("b", undefined, () => [])]),
    };
    const listed = aneInstallPrograms({ composition: record(), tables });
    expect(listed.map(program => [program.spec.label, program.graphs]))
      .toEqual([[shared.label, ["a-m4", "b-m4-pro"]], [own.label, ["a-m4"]]]);
    expect(aneInstallPrograms({ composition: record({ device: M1_MAX }), tables }).map(program => program.graphs)).toEqual([["b-m1"]]);
  });

  test("one program is one key: the same MIL and weights, not the same label", () => {
    const a = streamedMatmulProgram(64, 32, 32);
    expect(aneProgramKey({ ...a, label: "other" })).toBe(aneProgramKey(a));
    expect(aneProgramKey(streamedMatmulProgram(64, 32, 64))).not.toBe(aneProgramKey(a));
    const blob = (byte: number) => ({ "weight_data.bin": new Uint8Array([byte, 1, 2]) });
    expect(aneProgramKey({ ...a, weights: blob(0) })).not.toBe(aneProgramKey({ ...a, weights: blob(9) }));
    expect(aneProgramKey({ ...a, weights: blob(0) })).toBe(aneProgramKey({ ...a, weights: blob(0) }));
  });
});

test.skipIf(!process.env.MLX_BUN_TEST_ANE)("the install step compiles each distinct program once, holds nothing, and every program then loads from the cache", () => {
  const programs = [mlpSliceProgram(512, 192, 160), streamedMatmulProgram(512, 192, 160), streamedMatmulProgram(320, 128, 160)];
  const entry = (id: string): GraphEntry<null, string> => ({ id, serves: { device: M4_PRO }, create: () => id, anePrograms: () => programs });
  const tables = { synthetic: new GraphTable([entry("synthetic-m4"), { id: "generic", serves: {}, create: () => "generic" }]) };
  for (const spec of programs) AneProgram.purge(spec);
  const heard: string[] = [];
  const first = compileAnePrograms({ composition: record(), tables, onProgram: result => heard.push(result.status) });
  expect(first.map(result => result.status)).toEqual(["compiled", "compiled", "compiled"]);
  expect(heard).toEqual(["compiled", "compiled", "compiled"]);
  expect(programs.every(spec => AneProgram.cached(spec))).toBe(true);
  const again = compileAnePrograms({ composition: record(), tables });
  expect(again.map(result => result.status)).toEqual(["cached", "cached", "cached"]);
  const loads = programs.map(spec => {
    const started = performance.now();
    AneProgram.create(spec).dispose();
    return performance.now() - started;
  });
  console.error(`[ane] install step: compile ${first.map(result => result.milliseconds.toFixed(0)).join(", ")} ms; ` +
    `cached ${again.map(result => result.milliseconds.toFixed(2)).join(", ")} ms; load ${loads.map(ms => ms.toFixed(1)).join(", ")} ms`);
  for (const spec of programs) AneProgram.purge(spec);
}, 120_000);
