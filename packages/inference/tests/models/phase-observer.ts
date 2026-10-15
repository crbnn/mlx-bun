import type { MlxArray } from "@mlx-bun/mlx/array";
import type { Cache } from "../../src/contracts/mlx/cache";
import type { MlxLayerCapture, MlxRowTable, MlxTokenPhases } from "../../src/contracts/mlx/graph";
import { mapOperations } from "../../src/models/speculative/dflash2-operations";

/** What a test observes of the phases a scheduler calls. */
export type ObservedPhase = "prefillChunk" | "prompt" | "multiSequencePrompt" | "decode" | "verify" | "appendKnown";
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
  const forwards = (entries: MlxRowTable<Forward>, name: ObservedPhase) =>
    mapOperations(entries, run => watch(name, run));
  const overrides: Record<string, unknown> = {
    prompt: phases.prompt.map(range => ({...range, operation: watch("prompt", range.operation)})),
    multiSequencePrompt: phases.multiSequencePrompt.map(range => ({...range, operation: watch("prompt", range.operation)})),
    ...(phases.promptChunk ? { promptChunk: phases.promptChunk.map(range => ({...range, operation: watch("prefillChunk", range.operation)})) } : {}),
    ...(phases.multiSequencePromptChunk ? { multiSequencePromptChunk: phases.multiSequencePromptChunk.map(range => ({...range, operation: watch("prefillChunk", range.operation)})) } : {}),
    decode: forwards(phases.decode, "decode"),
    verify: forwards(phases.verify, "verify"),
    appendKnown: Array.isArray(phases.appendKnown)
      ? phases.appendKnown.map(range=>({...range,operation: watch("appendKnown",range.operation)}))
      : forwards(phases.appendKnown as MlxRowTable<Forward>, "appendKnown"),
    logits: mapOperations(phases.logits, head => (hidden: MlxArray) => {
        const out = (head as Head)(hidden);
        try { observer.head?.(hidden, out); } catch (error) { out.dispose(); throw error; }
        return out;
      }),
  };
  return new Proxy(model, {
    get(target, key) {
      if (typeof key === "string" && key in overrides) return overrides[key];
      const value = Reflect.get(target, key, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}
