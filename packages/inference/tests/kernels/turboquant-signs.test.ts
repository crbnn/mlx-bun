// TurboQuant's sign tables are generated (scripts/gen-turboquant-signs.ts) from
// the reference's construction. This regenerates them and fails when the
// committed src/kernels/turboquant/signs.ts differs, so the tables cannot drift
// from the construction, and the reference's four sizes stay its bits.
import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { HEAD_DIMS, SIGNS_FILE, definedHeadDim, render } from "../../scripts/gen-turboquant-signs";

test("the committed sign tables equal what the generator emits from the construction", async () => {
  expect(await render([...HEAD_DIMS])).toBe(readFileSync(SIGNS_FILE, "utf8"));
});

test("the construction defines a rotation for 32-element groups at MLX's Hadamard sizes only", () => {
  expect([32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512].every(definedHeadDim)).toBe(true);
  // 72 and 80 split no 32-element groups; 288 = 9·32 is no MLX Hadamard size.
  expect([72, 80, 288, 0, 31].some(definedHeadDim)).toBe(false);
});
