import type { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxPhaseGroup } from "../contracts/mlx/graph";
import type { MlxForwardStep, MlxForwardWork } from "../contracts/mlx/forward-work";

/** Join two method-owned work producers at their forward boundary: the first
 * forward preparation offers and the first forward decode makes run as one
 * mixed forward (`mixed`, the composition's), each group in its own phase;
 * every other forward runs on its own (`step.own`). Decode finishes
 * sampling/publication/retirement before preparation can join rows. The model
 * owns packing; this coordinator never merges caches or samples. A
 * restored/cancelled preparation need not produce a forward at all. */
export async function runMixedTokenIteration(options: {
  prepare(forward: MlxForwardWork): Promise<void>;
  decode(forward: MlxForwardWork): Promise<void>;
  mixed(groups: readonly MlxPhaseGroup[], rows: number): MlxArray[];
}): Promise<boolean> {
  let notify!: () => void;
  const offered = new Promise<void>(resolve => { notify = resolve; });
  let pending: { group: MlxPhaseGroup; step: MlxForwardStep; resolve(hidden: MlxArray): void; reject(error: unknown): void } | undefined;
  let captured = false;
  let preparationError: unknown;
  let preparationFailed = false;
  const groupOf = (ids: MlxArray, state: MlxPhaseGroup["state"], step: MlxForwardStep): MlxPhaseGroup =>
    ({ phase: step.phase, rows: step.rows, ids, state, capture: step.capture });
  const preparing = options.prepare(async (ids, cache, step) => {
    if (captured) return step.own();
    captured = true;
    return new Promise<MlxArray>((resolve, reject) => {
      pending = { group: groupOf(ids, cache, step), step, resolve, reject }; notify();
    });
  }).catch(error => { preparationFailed = true; preparationError = error; }).finally(notify);
  await offered;
  if (!pending) {
    await preparing;
    if (preparationFailed) throw preparationError;
    return false;
  }
  const held = pending;
  let preparedHidden: MlxArray | undefined;
  let mixed = false;
  try {
    await options.decode(async (ids, cache, step) => {
      if (mixed) return step.own();
      const outputs = options.mixed([groupOf(ids, cache, step), held.group], step.rows + held.step.rows);
      mixed = true; preparedHidden = outputs[1]!;
      return outputs[0]!;
    });
    // A final unread token can retire decode without another model call.
    preparedHidden ??= await held.step.own();
    held.resolve(preparedHidden); preparedHidden = undefined;
    await preparing;
    if (preparationFailed) throw preparationError;
    return true;
  } catch (error) {
    preparedHidden?.dispose(); held.reject(error);
    await preparing;
    throw error;
  }
}

export { type MlxForwardWork, type MlxPreparationWork } from "../contracts/mlx/forward-work";
