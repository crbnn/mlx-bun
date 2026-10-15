import { expect, test } from "bun:test";
import type { MlxCapturingForward, MlxTokenPhases } from "../../src/contracts/mlx/graph";
import { MlxPhaseDispatch } from "../../src/generation/phase-dispatch";
import { rowTableWithDefault } from "../../src/models/row-table";

const op = (_label: string): MlxCapturingForward => () => { throw new Error("forward must not run"); };
function graph(): MlxTokenPhases {
  const forward=op("forward"),head=()=>{throw new Error("head must not run");};
  return {prompt:[],multiSequencePrompt:[],appendKnown:[],
    decode:{1:forward},verify:{1:forward},logits:{1:head,3:()=>{throw new Error("narrow head must not run");}}};
}

test("missing custom graph rows fail at the dispatch boundary with their actual count",()=>{
  const f=graph(),dispatch=new MlxPhaseDispatch({...f,decode:{1:f.decode[1]!},verify:{1:f.verify[1]!},logits:{1:f.logits[1]!}});
  expect(dispatch.decode(1)).toBe(f.decode[1]!);
  expect(()=>dispatch.decode(2)).toThrow("no declared operation for 2 rows");
  expect(()=>dispatch.prompt("decode",3,1)).toThrow("no declared operation for 3 rows");
  expect(()=>dispatch.verify(4,7)).toThrow("no declared operation for 28 rows");
  expect(()=>dispatch.logits(65)).toThrow("no declared operation for 65 rows");
});

test("phase dispatch retains explicit heads and unbounded positive defaults",()=>{
  const f=graph(),forward=op("default"),head=f.logits[1]!;
  const dispatch=new MlxPhaseDispatch({...f,decode:rowTableWithDefault({1:f.decode[1]!},forward),
    verify:rowTableWithDefault({},forward),logits:rowTableWithDefault({3:f.logits[3]!},head)});
  expect(dispatch.decode(1)).toBe(f.decode[1]!);
  expect(dispatch.decode(262144)).toBe(forward);
  expect(dispatch.prompt("decode",262144,1)).toBe(forward);
  expect(dispatch.verify(16,2048)).toBe(forward);
  expect(dispatch.logits(3)).toBe(f.logits[3]!);
  expect(dispatch.logits(262144)).toBe(head);
  for (const rows of [0,-1,1.5,Infinity,NaN]) {
    expect(()=>dispatch.decode(rows)).toThrow("no declared operation");
    expect(()=>dispatch.verify(1,rows)).toThrow("no declared operation");
    expect(()=>dispatch.logits(rows)).toThrow("no declared operation");
  }
});
