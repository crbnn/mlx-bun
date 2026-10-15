import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxTokenPhases } from "../../src/contracts/mlx/graph";
import { MlxPhaseDispatch } from "../../src/generation/phase-dispatch";

// The shared execution fixtures import MLX. Run the actual stand-in declaration
// over opaque handles here so missing chunk bindings also fail on CPU-only CI.
function standIn(forward: (...args: unknown[]) => MlxArray): MlxTokenPhases {
  const source = readFileSync(new URL("../models/phase-forward.ts", import.meta.url), "utf8");
  const declaration = source.slice(source.indexOf("export function stubPhases("),
    source.indexOf("/** A stand-in recurrent MTP block"));
  const code = new Bun.Transpiler({loader:"ts",target:"bun"}).transformSync(
    declaration.replace("export function", "function") + "\nreturn stubPhases;");
  const stub = new Function("rowTable", code)((operation: unknown) => Object.freeze({1:operation}));
  return stub(forward, () => ({}) as MlxArray);
}

test("the shared execution stand-in explicitly declares single and multi-sequence chunks", () => {
  const input={} as MlxArray,output={} as MlxArray,calls:unknown[][]=[];
  const phases=standIn((...args)=>{calls.push(args);return output;});
  const dispatch=new MlxPhaseDispatch(phases);
  expect(phases.promptChunk).toBe(phases.prompt);
  expect(phases.multiSequencePromptChunk).toBe(phases.multiSequencePrompt);
  expect(dispatch.prompt("chunk",1,8)(input,[],null)).toBe(output);
  expect(dispatch.prompt("chunk",2,4)(input,[],null)).toBe(output);
  expect(calls).toEqual([[input,[],null],[input,[],null]]);
});

test("mutable forward overrides reach the intended declared chunk operation", () => {
  const model={forward:()=>({}) as MlxArray};
  const phases=standIn(()=>model.forward()),dispatch=new MlxPhaseDispatch(phases);
  model.forward=()=>{throw new Error("observed first chunk");};
  const input={} as MlxArray;
  expect(()=>dispatch.prompt("chunk",1,3)(input,[],null)).toThrow("observed first chunk");
  expect(()=>dispatch.prompt("chunk",2,3)(input,[],null)).toThrow("observed first chunk");
});
