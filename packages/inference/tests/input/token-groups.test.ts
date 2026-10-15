import { expect, test } from "bun:test";
import { mapPackedTokens, mapTokenGroups } from "../../src/input/token-groups";

import * as ops from "@mlx-bun/mlx/ops";

test("tokenwise packing preserves unequal lengths and B>1 without padding", () => {
  using decode = ops.fromInt32([1, 2], [2, 1, 1]);
  using prefill = ops.fromInt32([3, 4, 5, 6, 7, 8], [2, 3, 1]);
  let shape: readonly number[] = [];
  const output = mapPackedTokens([decode, prefill], packed => {
    shape = packed.shape;
    return ops.add(packed, packed);
  });
  try {
    expect(shape).toEqual([1, 8, 1]);
    expect(output.map(array => array.shape)).toEqual([[2, 1, 1], [2, 3, 1]]);
    expect(output.map(array => array.toIntTokens())).toEqual([[2, 4], [6, 8, 10, 12, 14, 16]]);
    expect(decode.toIntTokens()).toEqual([1, 2]);
  } finally { for (const array of output) array.dispose(); }
});

test("a lone token group preserves the operation's original geometry", () => {
  using input = ops.fromInt32([1, 2, 3, 4], [2, 2, 1]);
  const output = mapPackedTokens([input], value => {
    expect(value).toBe(input);
    return ops.copyOf(value);
  });
  try { expect(output[0]!.shape).toEqual([2, 2, 1]); }
  finally { for (const array of output) array.dispose(); }
});

test("verification keeps its row geometry while other token groups share a packed operation", () => {
  using verify = ops.fromInt32([1, 2, 3, 4], [2, 2, 1]);
  using first = ops.fromInt32([5, 6], [1, 2, 1]);
  using second = ops.fromInt32([7], [1, 1, 1]);
  const shapes: number[][] = [], widths: number[] = [];
  const inputs = [verify, first, second];
  const groups = inputs.map((ids, row) => ({ phase: row === 0 ? "verify" as const : "window" as const,
    rows: (ids.size), ids, state: [], capture: null }));
  // The packed call runs at the class the caller names for the packed rows; the verify group at its own.
  const result = mapTokenGroups(groups, inputs, (hidden, width) => {
    shapes.push([...hidden.shape]); widths.push(width); return ops.copyOf(hidden);
  }, 2);
  try {
    expect(shapes).toEqual([[1, 3, 1], [2, 2, 1]]);
    expect(widths).toEqual([2, 4]);
    expect(result.map(hidden => hidden.toIntTokens())).toEqual([[1, 2, 3, 4], [5, 6], [7]]);
  } finally { for (const hidden of result) hidden.dispose(); }
});
