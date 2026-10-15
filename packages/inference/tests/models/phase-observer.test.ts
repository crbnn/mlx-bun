import { expect, test } from "bun:test";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxTokenPhases } from "../../src/contracts/mlx/graph";
import { MlxPhaseDispatch } from "../../src/generation/phase-dispatch";
import { rowOperationFor, rowTableWithDefault } from "../../src/models/row-table";
import { observedPhases } from "./phase-observer";

const tensor = (name: string, disposed: string[] = []): MlxArray =>
  ({ dispose() { disposed.push(name); } }) as MlxArray;
function fixture() {
  const input = tensor("input"), hidden = tensor("hidden"), logits = tensor("logits");
  const forward = () => hidden, head = () => logits;
  const table = rowTableWithDefault({ 1: forward }, forward);
  const phases: MlxTokenPhases = {
    decode: table, verify: table, appendKnown: table,
    prompt: [{ from: 1, to: Infinity, operation: forward }],
    multiSequencePrompt: [{ from: 1, to: Infinity, operation: forward }],
    logits: rowTableWithDefault({ 1: head }, head),
  };
  return { input, hidden, logits, phases };
}

test("phase observers retain numeric defaults and return the graph's owned outputs", () => {
  const f = fixture(), seen: string[] = [];
  const view = observedPhases(f.phases, {
    forward(_ids, _state, phase) { seen.push(phase); }, head() { seen.push("head"); },
  });
  const dispatch = new MlxPhaseDispatch(view);
  expect(dispatch.decode(65)(f.input, [], null)).toBe(f.hidden);
  expect(dispatch.verify(8, 11)(f.input, [], null)).toBe(f.hidden);
  expect(dispatch.known(262144)(f.input, [])).toBe(f.hidden);
  expect(dispatch.logits(65)(f.hidden)).toBe(f.logits);
  expect(dispatch.decode(65)).toBe(dispatch.decode(65));
  expect(Object.keys(view.decode)).toEqual(["1", "default"]);
  expect(Reflect.get(view.decode, "then")).toBeUndefined();
  expect(seen).toEqual(["decode", "verify", "appendKnown", "head"]);
});

test("chunk observers cover single and multiple sequences without changing the chosen operation", () => {
  const f = fixture(), called: string[] = [], seen: string[] = [];
  const operation = (label: string) => () => { called.push(label); return f.hidden; };
  f.phases = { ...f.phases,
    prompt: [{ from: 1, to: Infinity, operation: operation("tail") }],
    promptChunk: [{ from: 1, to: Infinity, unit: "gpu", operation: operation("chunk") }],
    multiSequencePromptChunk: [{ from: 1, to: Infinity, unit: "gpu", operation: operation("multi chunk") }],
  };
  const view = observedPhases(f.phases, { forward(_ids, _state, phase) { seen.push(phase); } });
  const dispatch = new MlxPhaseDispatch(view);
  dispatch.prompt("chunk", 1, 4)(f.input, [], null);
  dispatch.prompt("chunk", 2, 2)(f.input, [], null);
  dispatch.prompt("tail", 1, 4)(f.input, [], null);
  expect(called).toEqual(["chunk", "multi chunk", "tail"]);
  expect(seen).toEqual(["prefillChunk", "prefillChunk", "prompt"]);
  expect(view.promptChunk![0]!.unit).toBe("gpu");
  expect(view.multiSequencePromptChunk![0]!.to).toBe(Infinity);
});

test("an observer failure releases only its returned output, including asynchronous forwards", async () => {
  const disposed: string[] = [], input = tensor("input", disposed), hidden = tensor("hidden", disposed);
  const f = fixture();
  const failure = () => { throw new Error("observer failed"); };
  const sync = observedPhases<MlxTokenPhases>({ ...f.phases, decode: rowTableWithDefault({}, () => hidden) }, { forwarded: failure });
  expect(() => rowOperationFor(sync.decode,65)(input, [], null)).toThrow("observer failed");
  expect(disposed).toEqual(["hidden"]);
  const output = tensor("logits", disposed);
  const head = observedPhases<MlxTokenPhases>({ ...f.phases, logits: rowTableWithDefault({}, () => output) }, { head: failure });
  expect(() => rowOperationFor(head.logits,65)(input)).toThrow("observer failed");
  expect(disposed).toEqual(["hidden", "logits"]);
  const asynchronous = observedPhases<MlxTokenPhases>({ ...f.phases,
    verify: rowTableWithDefault({}, () => Promise.resolve(tensor("async hidden", disposed))),
  }, { forwarded: failure });
  await expect(rowOperationFor(asynchronous.verify,65)(input, [], null)).rejects.toThrow("observer failed");
  expect(disposed).toEqual(["hidden", "logits", "async hidden"]);
});
