import { rowTable } from "../../src/models/graph";
import { MlxPhaseDispatch } from "../../src/generation/phase-dispatch";
// A test's forwards through a graph's named phases. The test is the scheduler
// here: it names the phase of each call from the tokens it hands over, as the
// planner does, one position per sequence a decode and more a prompt tail, and
// indexes each table by the rows it is loading.
import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { Cache } from "../../src/contracts/mlx/cache";
import { type MlxLayerCapture, type MlxTokenPhases, type MlxRowTable } from "../../src/contracts/mlx/graph";
import { argmaxLastPosition } from "../../src/kernels/logits";

/** A forward the test knows runs synchronously (every resident graph's operations do). */
function settled(result: MlxArray | Promise<MlxArray>): MlxArray {
  if (result instanceof Promise) throw new Error("phase-forward: this graph's operation is asynchronous");
  return result;
}

/** `ids` [sequences, tokens] appended to `caches`: one token per sequence through
 *  `decode`, more through `prefillTail`, each at the class of its rows. */
export function phaseHidden(model: Pick<MlxTokenPhases, "decode" | "prompt" | "multiSequencePrompt">, ids: MlxArray, caches: Cache[],
  capture: MlxLayerCapture | null = null): MlxArray {
  const [sequences, tokens] = ids.shape as [number, number];
  return settled(tokens === 1 ? model.decode[(sequences)]!(ids, caches, capture)
    : new MlxPhaseDispatch(model as MlxTokenPhases).prompt("tail", sequences, tokens)(ids, caches, capture));
}

/** Logits [1, L, vocabulary] for `tokens` appended to `caches` (`phaseHidden`, then
 *  the head for every position at the class of those rows). */
export function phaseLogits(model: Pick<MlxTokenPhases, "decode" | "prompt" | "multiSequencePrompt" | "logits">,
  tokens: readonly number[] | MlxArray, caches: Cache[]): MlxArray {
  const ids = Array.isArray(tokens) ? ops.fromInt32([...tokens], [1, tokens.length]) : tokens as MlxArray;
  try {
    using hidden = phaseHidden(model, ids, caches);
    const [sequences, length] = ids.shape as [number, number];
    return model.logits[(sequences * length)]!(hidden);
  } finally { if (Array.isArray(tokens)) ids.dispose(); }
}

/** A stand-in graph's phase tables over its one forward and head: a fixture whose forward serves every phase
 *  and every width (the forward takes the capture a phase hands it, which a fixture may ignore). */
export function stubPhases(forward: (ids: MlxArray, caches: Cache[], capture: MlxLayerCapture | null) => MlxArray | Promise<MlxArray>,
  head: (hidden: MlxArray) => MlxArray): MlxTokenPhases {
  const every = rowTable;
  const run = (ids: MlxArray, caches: Cache[], capture: MlxLayerCapture | null = null) => forward(ids, caches, capture);
  return { prompt: [{from:1,to:Infinity,operation:run}], multiSequencePrompt: [{from:1,to:Infinity,operation:run}], decode: every(run), verify: every(run),
    appendKnown: every((ids: MlxArray, caches: Cache[]) => run(ids, caches)), logits: every(head) };
}

/** A stand-in recurrent MTP block whose one forward serves its decode table and its window. */
export function mtpModule<Forward>(forward: Forward): { readonly decode: MlxRowTable<Forward>; readonly window: Forward } {
  return { decode: rowTable(forward),
    window: forward };
}

export { observedPhases, type ObservedPhase, type PhaseObserver } from "./phase-observer";

/** Greedy tokens after `prompt` on fresh caches, stopping at an EOS id. */
export function phaseGreedy(model: Pick<MlxTokenPhases, "decode" | "prompt" | "multiSequencePrompt" | "logits"> & { makeCache(): Cache[] },
  prompt: readonly number[], maxTokens: number, eosIds: readonly number[] = []): number[] {
  const caches = model.makeCache();
  const out: number[] = [];
  try {
    let tokens: readonly number[] = prompt;
    for (let step = 0; step < maxTokens; step++) {
      const logits = phaseLogits(model, tokens, caches);
      const next = argmaxLastPosition(logits);
      logits.dispose();
      if (eosIds.includes(next)) break;
      out.push(next);
      tokens = [next];
    }
  } finally { for (const cache of caches) cache.dispose(); }
  return out;
}
