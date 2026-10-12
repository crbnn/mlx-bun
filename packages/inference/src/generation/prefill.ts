import type { PrefillStep } from "../contracts/portable/prefill";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { Cache } from "../contracts/mlx/cache";
import * as ops from "@mlx-bun/mlx/ops";
import { clearCache } from "@mlx-bun/mlx/ffi";
import { leaseCacheStates } from "../state/leases";
import { withResource } from "../runtime/resources";
import type { MlxPhaseDispatch } from "./phase-dispatch";

/** State views with explicit ownership are released even on deferred errors.
 * Owned views precede borrowed state, preserving the serial eval ordering. */
export function evalCacheState(cache: Cache[]): void {
  withResource(leaseCacheStates(cache), (state) => { if (state.length) ops.evalAll([...state]); });
}

/** Forward one prompt step of one sequence through the operation its planned
 * phase names. Drains evaluate state (the composed caches converted inside
 * their own appends), then clear the allocator cache. Finals return owned
 * hidden state without forcing eval. */
export async function executeMlxPrefillStep(
  phases: MlxPhaseDispatch, cache: Cache[], prompt: readonly number[], step: PrefillStep,
): Promise<MlxArray | null> {
  const chunk = prompt.slice(step.start, step.end);
  const ids = ops.fromInt32(chunk, [1, chunk.length]);
  let hidden: MlxArray;
  try { hidden = await phases.prompt(step.phase, 1, chunk.length)(ids, cache, null); }
  finally { ids.dispose(); }
  if (step.kind === "final") return hidden;
  hidden.dispose();
  evalCacheState(cache);
  clearCache();
  return null;
}
