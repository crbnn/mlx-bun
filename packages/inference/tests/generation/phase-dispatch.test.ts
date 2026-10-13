// The scheduler's dispatch of prompt pieces (generation/phase-dispatch): which operation a chunk, a tail or a
// one-position step runs, given the Neural Engine operations a graph declares. Model-free: the operations are
// labelled stand-ins the dispatch returns and never calls.
import { describe, expect, test } from "bun:test";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxCapturingForward, MlxPhaseForward, MlxTokenPhases, MlxWidthClass } from "../../src/contracts/mlx/graph";
import { MlxPhaseDispatch } from "../../src/generation/phase-dispatch";
import { atEveryWidth } from "../../src/models/graph";

const labelled = new Map<MlxCapturingForward, string>();
const op = (label: string): MlxCapturingForward => {
  const forward: MlxCapturingForward = () => { throw new Error(`${label} is a stand-in`); };
  labelled.set(forward, label);
  return forward;
};
const table = (phase: string) => Object.freeze(Object.fromEntries((["one", "twoToThree", "four", "fiveToEight", "wider"] as const)
  .map(width => [width, op(`${phase}.${width}`)])) as Record<MlxWidthClass, MlxCapturingForward>);

function graph(ane?: readonly { minRows: number; rows: number }[]): MlxTokenPhases {
  return {
    prefillChunk: Object.freeze({ gpu: op("chunk.gpu"),
      ...(ane ? { ane: Object.freeze(ane.map(range => Object.freeze({ ...range, forward: op(`ane.${range.rows}`) }))) } : {}) }),
    prefillTail: table("tail"), decode: table("decode"), verify: table("verify"),
    appendKnown: atEveryWidth<MlxPhaseForward>(() => { throw new Error("known is a stand-in"); }), logits: atEveryWidth((x: MlxArray) => x),
  };
}
const ranges = [{ minRows: 640, rows: 1024 }, { minRows: 1025, rows: 1536 }, { minRows: 1537, rows: 2048 }];

describe("prompt pieces", () => {
  test("one sequence's chunk or tail runs on the Neural Engine operation whose range holds its positions", () => {
    const dispatch = new MlxPhaseDispatch(graph(ranges));
    const run = (phase: "chunk" | "tail", tokens: number) => labelled.get(dispatch.prompt(phase, 1, tokens));
    expect(dispatch.neuralEngine).toBe(true);
    expect([run("tail", 640), run("tail", 757), run("tail", 1024), run("tail", 1025), run("tail", 1536), run("tail", 1537)])
      .toEqual(["ane.1024", "ane.1024", "ane.1024", "ane.1536", "ane.1536", "ane.2048"]);
    expect([run("chunk", 2048), run("chunk", 1000), run("chunk", 512)]).toEqual(["ane.2048", "ane.1024", "chunk.gpu"]);
    expect([run("tail", 639), run("tail", 8), run("tail", 4)]).toEqual(["tail.wider", "tail.fiveToEight", "tail.four"]);
    expect([run("chunk", 4096), run("tail", 2049)]).toEqual(["chunk.gpu", "tail.wider"]);
  });

  test("pieces of several sequences and one-position steps never run on the Neural Engine", () => {
    const dispatch = new MlxPhaseDispatch(graph(ranges));
    expect(labelled.get(dispatch.prompt("chunk", 2, 1024))).toBe("chunk.gpu");
    expect(labelled.get(dispatch.prompt("tail", 4, 700))).toBe("tail.wider");
    expect(labelled.get(dispatch.prompt("decode", 1, 1))).toBe("decode.one");
  });

  test("a graph that declares no Neural Engine operation runs every piece on the GPU", () => {
    const dispatch = new MlxPhaseDispatch(graph());
    expect(dispatch.neuralEngine).toBe(false);
    expect([labelled.get(dispatch.prompt("chunk", 1, 2048)), labelled.get(dispatch.prompt("tail", 1, 757))])
      .toEqual(["chunk.gpu", "tail.wider"]);
  });
});
