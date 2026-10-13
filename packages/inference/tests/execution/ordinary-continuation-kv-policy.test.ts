import { expect, test } from "bun:test";
import { resolveKvScheme } from "../../src/state/kv-scheme";
import { generationCheckpointKey } from "../../src/generation/checkpoint-identity";
import { snapshotGenerationPolicy } from "../../src/execution/request-policy";

// The KV scheme is the composition's: a checkpoint store is keyed by its
// `cacheKey` (the store's configuration fingerprint), so two schemes never share
// a resume identity even where the request policy is the same.
test("optional server threshold preserves omitted policy and separates scheme identities", () => {
  const config = [{ layerIdx: 0, bits: 4, groupSize: 64 }, { layerIdx: 1, bits: 8, groupSize: 64 }];
  const arms = [{ override: 4 }, { override: 8 }, { turboQuant: { kBits: 8, vBits: 3 } },
    { override: "config" as const, config }];
  for (const input of arms) {
    const original = resolveKvScheme(input);
    expect(resolveKvScheme({ ...input, quantizedKvStart: undefined }).storage).toEqual(original.storage);
    const delayed = resolveKvScheme({ ...input, quantizedKvStart: 14 });
    expect(delayed.storage).toMatchObject({ quantizedKvStart: 14 });
    expect(delayed.cacheKey).not.toBe(original.cacheKey);
  }
  for (const quantizedKvStart of [-1, 0.5, NaN, Infinity])
    expect(() => resolveKvScheme({ override: 4, quantizedKvStart })).toThrow("nonnegative integer");
});

test("continuation policy snapshots nested recipes before key construction", () => {
  const source = { adapters: ["upper"], logitBias: { 5: -100 } as Record<number, number> };
  const captured = snapshotGenerationPolicy(source);
  const key = generationCheckpointKey([1, 2], captured);
  source.adapters[0] = "lower";
  source.logitBias[5] = 3;
  expect(generationCheckpointKey([1, 2], captured)).toBe(key);
  expect(generationCheckpointKey([1, 2], source)).not.toBe(key);
});
