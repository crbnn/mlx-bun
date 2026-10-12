// A test's forwards through a graph's named phases. The test is the scheduler
// here: it names the phase of each call from the tokens it hands over, as the
// planner does, one position per sequence a decode and more a prompt tail, and
// indexes each table by the rows it is loading.
import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { Cache } from "../../src/contracts/mlx/cache";
import { widthClass, type MlxLayerCapture, type MlxTokenPhases, type MlxWidthTable } from "../../src/contracts/mlx/graph";
import { argmaxLastPosition } from "../../src/kernels/logits";

/** A forward the test knows runs synchronously (every resident graph's operations do). */
function settled(result: MlxArray | Promise<MlxArray>): MlxArray {
  if (result instanceof Promise) throw new Error("phase-forward: this graph's operation is asynchronous");
  return result;
}

/** `ids` [sequences, tokens] appended to `caches`: one token per sequence through
 *  `decode`, more through `prefillTail`, each at the class of its rows. */
export function phaseHidden(model: Pick<MlxTokenPhases, "decode" | "prefillTail">, ids: MlxArray, caches: Cache[],
  capture: MlxLayerCapture | null = null): MlxArray {
  const [sequences, tokens] = ids.shape as [number, number];
  return settled(tokens === 1 ? model.decode[widthClass(sequences)](ids, caches, capture)
    : model.prefillTail[widthClass(sequences * tokens)](ids, caches, capture));
}

/** Logits [1, L, vocabulary] for `tokens` appended to `caches` (`phaseHidden`, then
 *  the head for every position at the class of those rows). */
export function phaseLogits(model: Pick<MlxTokenPhases, "decode" | "prefillTail" | "logits">,
  tokens: readonly number[] | MlxArray, caches: Cache[]): MlxArray {
  const ids = Array.isArray(tokens) ? ops.fromInt32([...tokens], [1, tokens.length]) : tokens as MlxArray;
  try {
    using hidden = phaseHidden(model, ids, caches);
    const [sequences, length] = ids.shape as [number, number];
    return model.logits[widthClass(sequences * length)](hidden);
  } finally { if (Array.isArray(tokens)) ids.dispose(); }
}

/** A stand-in graph's phase tables over its one forward and head: a fixture whose forward serves every phase
 *  and every width (the forward takes the capture a phase hands it, which a fixture may ignore). */
export function stubPhases(forward: (ids: MlxArray, caches: Cache[], capture: MlxLayerCapture | null) => MlxArray | Promise<MlxArray>,
  head: (hidden: MlxArray) => MlxArray): MlxTokenPhases {
  const every = <Operation>(operation: Operation) =>
    Object.freeze({ one: operation, twoToThree: operation, four: operation, fiveToEight: operation, wider: operation });
  const run = (ids: MlxArray, caches: Cache[], capture: MlxLayerCapture | null = null) => forward(ids, caches, capture);
  return { prefillChunk: Object.freeze({ gpu: run }), prefillTail: every(run), decode: every(run), verify: every(run),
    appendKnown: every((ids: MlxArray, caches: Cache[]) => run(ids, caches)), logits: every(head) };
}

/** A stand-in recurrent MTP block whose one forward serves its decode table and its window. */
export function mtpModule<Forward>(forward: Forward): { readonly decode: MlxWidthTable<Forward>; readonly window: Forward } {
  return { decode: Object.freeze({ one: forward, twoToThree: forward, four: forward, fiveToEight: forward, wider: forward }),
    window: forward };
}

/** What a test observes of the phases a scheduler calls. */
export type ObservedPhase = "prefillChunk" | "prefillTail" | "decode" | "verify" | "appendKnown";
export interface PhaseObserver {
  /** Before each forward runs: its input, caches and the phase the scheduler named. */
  forward?(ids: MlxArray, caches: Cache[], phase: ObservedPhase): void;
  /** After each forward has appended: its input, caches and hidden (still the scheduler's). */
  forwarded?(ids: MlxArray, caches: Cache[], hidden: MlxArray, phase: ObservedPhase): void;
  /** After each head projection: the hidden it took and the logits it returns (both still owned by the scheduler). */
  head?(hidden: MlxArray, logits: MlxArray): void;
}

/** `model` with its phase operations observed: a view whose prompt, decode, verify and known-token tables and
 *  head call `observer` around the graph's own operation; every other member is the model's own, bound to it.
 *  Bind the scheduler to the view, not the model. */
export function observedPhases<Model extends object>(model: Model, observer: PhaseObserver): Model {
  const phases = model as unknown as MlxTokenPhases;
  type Forward = (ids: MlxArray, caches: Cache[], capture: MlxLayerCapture | null) => MlxArray | Promise<MlxArray>;
  type Head = (hidden: MlxArray) => MlxArray;
  const watch = (name: ObservedPhase, run: Forward): Forward => (ids, caches, capture) => {
    observer.forward?.(ids, caches, name);
    const out = run(ids, caches, capture);
    const after = (hidden: MlxArray) => {
      try { observer.forwarded?.(ids, caches, hidden, name); } catch (error) { hidden.dispose(); throw error; }
      return hidden;
    };
    return out instanceof Promise ? out.then(after) : after(out);
  };
  const forwards = (entries: MlxWidthTable<Forward>, name: ObservedPhase) =>
    Object.freeze(Object.fromEntries(Object.entries(entries).map(([width, run]) => [width, watch(name, run as Forward)])));
  const ane = phases.prefillChunk.ane;
  const overrides: Record<string, unknown> = {
    prefillChunk: Object.freeze({ gpu: watch("prefillChunk", phases.prefillChunk.gpu),
      ...(ane ? { ane: watch("prefillChunk", ane) } : {}) }),
    prefillTail: forwards(phases.prefillTail, "prefillTail"),
    decode: forwards(phases.decode, "decode"),
    verify: forwards(phases.verify, "verify"),
    appendKnown: forwards(phases.appendKnown as MlxWidthTable<Forward>, "appendKnown"),
    logits: Object.freeze(Object.fromEntries(Object.entries(phases.logits).map(([width, head]) =>
      [width, (hidden: MlxArray) => {
        const out = (head as Head)(hidden);
        try { observer.head?.(hidden, out); } catch (error) { out.dispose(); throw error; }
        return out;
      }]))),
  };
  return new Proxy(model, {
    get(target, key) {
      if (typeof key === "string" && key in overrides) return overrides[key];
      const value = Reflect.get(target, key, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}

/** Greedy tokens after `prompt` on fresh caches, stopping at an EOS id. */
export function phaseGreedy(model: Pick<MlxTokenPhases, "decode" | "prefillTail" | "logits"> & { makeCache(): Cache[] },
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
