import { afterAll, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TURBOQUANT_FOLD_FILE, readTurboQuantFold } from "../../src/artifacts/turboquant-fold";

const root = mkdtempSync(join(tmpdir(), "mlx-bun-turboquant-fold-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

/** The fields every record carries, as the Trellis quantizer writes them. */
const RECORD = { source: "src", generatedAt: "2026-10-11T00:00:00.000Z", seed: 42, r1: true, r2: false, hiddenSize: 4,
  gamma_convention: "stored+1 folded in; folded norms written as zeros" };

function artifact(name: string, record?: Record<string, unknown>): string {
  const dir = join(root, name);
  mkdirSync(dir);
  if (record) writeFileSync(join(dir, TURBOQUANT_FOLD_FILE), JSON.stringify(record));
  return dir;
}

test("a record with final_gain reads as its seed, rotation, width and the gain in float32", () => {
  const gain = [1.9609375, 0.71484375, 2.71875, Math.fround(1.0001)];
  const fold = readTurboQuantFold(artifact("gain", { ...RECORD, final_gain: gain, final_gain_convention: "c", final_gain_source: "s" }));
  expect(fold).toEqual({ seed: 42, r1: true, hiddenSize: 4, finalGain: Float32Array.from(gain) });
  expect(fold!.finalGain).toBeInstanceOf(Float32Array);
});

test("a record without final_gain reads without it; an artifact without the file has no record", () => {
  const fold = readTurboQuantFold(artifact("no-gain", RECORD));
  expect(fold).toEqual({ seed: 42, r1: true, hiddenSize: 4 });
  expect(fold && "finalGain" in fold).toBe(false);
  expect(readTurboQuantFold(artifact("none"))).toBeNull();
});

test("a record missing a field, or whose gain does not span the hidden width, is refused", () => {
  const { seed: _seed, ...noSeed } = RECORD;
  expect(() => readTurboQuantFold(artifact("no-seed", noSeed))).toThrow("seed must be an integer");
  expect(() => readTurboQuantFold(artifact("no-r1", { ...RECORD, r1: undefined }))).toThrow("r1 must be a boolean");
  expect(() => readTurboQuantFold(artifact("short", { ...RECORD, final_gain: [1, 1, 1] }))).toThrow("final_gain must be 4 finite numbers");
  expect(() => readTurboQuantFold(artifact("text", { ...RECORD, final_gain: [1, 1, 1, "1"] }))).toThrow("final_gain must be 4 finite numbers");
});
