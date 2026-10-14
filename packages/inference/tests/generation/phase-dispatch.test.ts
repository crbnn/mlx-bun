import { describe, expect, test } from "bun:test";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { Cache } from "../../src/contracts/mlx/cache";
import type { ModelConfig } from "../../src/artifacts/config";
import type { MlxTokenGraph } from "../../src/models/graph";
import { declareGraph } from "../../src/models/capabilities";
import { bindMlxAutoregressiveGraph } from "../../src/generation/bindings/autoregressive";
import type { MlxCapturingForward, MlxTokenPhases } from "../../src/contracts/mlx/graph";
import { MlxPhaseDispatch, promptWorkStep, verifyWorkStep } from "../../src/generation/phase-dispatch";

const labels = new Map<Function, string>();
const op = (label: string): MlxCapturingForward => {
  const forward = () => { throw new Error(`${label} is a stand-in`); };
  labels.set(forward, label); return forward;
};
const table = (phase: string) => Object.fromEntries(Array.from({length:64}, (_,i)=>[i+1,op(`${phase}.${i+1}`)]));
function graph(ane = true): MlxTokenPhases {
  const gpu = op("gpu");
  const prompt = ane ? [{from:1,to:639,operation:gpu},
    {from:640,to:1024,operation:op("ane.1024"),unit:"ane" as const},
    {from:1025,to:1536,operation:op("ane.1536"),unit:"ane" as const},
    {from:1537,to:2048,operation:op("ane.2048"),unit:"ane" as const},
    {from:2049,to:Infinity,operation:gpu}] : [{from:1,to:Infinity,operation:gpu}];
  return {prompt, multiSequencePrompt:[{from:1,to:4,operation:op("multi.small")},
    {from:5,to:Infinity,operation:op("multi.large")}], decode:table("decode"),verify:table("verify"),
    appendKnown:table("known") as unknown as MlxTokenPhases["appendKnown"], logits:Object.fromEntries(Array.from({length:64},(_,i)=>{
      const head = (_hidden:MlxArray):MlxArray => {throw new Error("stand-in");};
      labels.set(head,`logits.${i+1}`);return [i+1,head];
    }))};
}
const name = (operation:Function)=>labels.get(operation);

describe("prompt ranges",()=>{
  test("chunks and tails share the operation for their positions, including every boundary",()=>{
    const dispatch=new MlxPhaseDispatch(graph());
    expect(dispatch.neuralEngine).toBe(true);
    for(const phase of ["chunk","tail"] as const)
      expect([639,640,1024,1025,1536,1537,2048,2049].map(tokens=>name(dispatch.prompt(phase,1,tokens))))
        .toEqual(["gpu","ane.1024","ane.1024","ane.1536","ane.1536","ane.2048","ane.2048","gpu"]);
  });
  test("multi-sequence pieces use total rows and one-position steps use exact decode rows",()=>{
    const dispatch=new MlxPhaseDispatch(graph());
    expect(name(dispatch.prompt("tail",2,2))).toBe("multi.small");
    expect(name(dispatch.prompt("chunk",2,1024))).toBe("multi.large");
    expect(name(dispatch.prompt("decode",3,1))).toBe("decode.3");
  });
  test("a GPU-only graph has no Neural Engine operation",()=>{
    const dispatch=new MlxPhaseDispatch(graph(false));
    expect(dispatch.neuralEngine).toBe(false);
    expect(name(dispatch.prompt("tail",1,1000))).toBe("gpu");
  });
});
test("every token phase indexes the exact count, without shared row classes",()=>{
  const dispatch=new MlxPhaseDispatch(graph());
  expect([2,3,4,5,8].map(rows=>name(dispatch.decode(rows)))).toEqual(["decode.2","decode.3","decode.4","decode.5","decode.8"]);
  expect(name(dispatch.verify(3,7))).toBe("verify.21");
  expect(name(dispatch.known(6))).toBe("known.6");
  expect(name(dispatch.logits(7))).toBe("logits.7");
});
test("mixed-work metadata carries real rows",()=>{
  const own=async()=>{throw new Error("stand-in");};
  expect(promptWorkStep({phase:"tail",sequences:3,tokens:5},null,own)).toEqual({phase:"window",rows:15,capture:null,own});
  expect(verifyWorkStep({sequences:2,positions:4},null,own)).toEqual({phase:"verify",rows:8,capture:null,own});
});

test("known-span ranges preserve a whole grammar span beyond decode and verify caps",()=>{
  const narrow=()=>{throw new Error("narrow operation must not run");};
  const calls:{ids:MlxArray;state:unknown}[]=[];
  const expanded=(ids:MlxArray,state:unknown)=>{calls.push({ids,state});return ids;};
  const dispatch=new MlxPhaseDispatch({...graph(),appendKnown:[
    {from:1,to:4,operation:narrow},{from:5,to:Infinity,operation:expanded},
  ]});
  const input={} as MlxArray,state:Cache[]=[];
  expect(dispatch.known(80)).toBe(expanded);
  expect(dispatch.known(262144)).toBe(expanded);
  expect(dispatch.known(80)(input,state)).toBe(input);
  expect(calls).toEqual([{ids:input,state}]);
});

test("prepared embedding dispatch selects total rows and preserves every borrowed argument",()=>{
  const embeddings={} as MlxArray,ids={} as MlxArray,imageMask={} as MlxArray,multimodalMask={} as MlxArray,state:Cache[]=[];
  const calls:unknown[][]=[];
  const narrow=(...args:unknown[])=>{calls.push(["narrow",...args]);return embeddings;};
  const wide=(...args:unknown[])=>{calls.push(["wide",...args]);return embeddings;};
  const model={...graph(false),weightsBytes:0,makeCache:()=>[],dispose:()=>{},
    config:{modelType:"test",eosTokenIds:[]} as unknown as ModelConfig,
    graphCapabilities:declareGraph({prefill:{boundedWorkspace:false}}),
    embeddingPrompt:[{from:1,to:4,operation:narrow},{from:5,to:Infinity,operation:wide}],
  } satisfies MlxTokenGraph;
  const binding=bindMlxAutoregressiveGraph(model);
  expect(binding.prefillEmbeddings!(embeddings,state,imageMask,ids,multimodalMask,4)).toBe(embeddings);
  expect(binding.prefillEmbeddings!(embeddings,state,imageMask,ids,multimodalMask,40)).toBe(embeddings);
  expect(calls).toEqual([["narrow",embeddings,state,imageMask,ids,multimodalMask],["wide",embeddings,state,imageMask,ids,multimodalMask]]);
});

test("literal row tables can reuse one operation, alternate operations, or declare eight distinct operations",()=>{
  const shared=op("shared"),odd=op("odd"),even=op("even");
  const one=op("one"),two=op("two"),three=op("three"),four=op("four"),five=op("five"),six=op("six"),seven=op("seven"),eight=op("eight");
  const uniform={1:shared,2:shared,3:shared,4:shared,5:shared,6:shared,7:shared,8:shared};
  const alternating={1:odd,2:even,3:odd,4:even,5:odd,6:even,7:odd,8:even};
  const distinct={1:one,2:two,3:three,4:four,5:five,6:six,7:seven,8:eight};
  const rows=[1,2,3,4,5,6,7,8];
  const repeated=new MlxPhaseDispatch({...graph(),decode:uniform});
  const parity=new MlxPhaseDispatch({...graph(),decode:alternating});
  const individual=new MlxPhaseDispatch({...graph(),decode:distinct});
  expect(rows.map(row=>repeated.decode(row))).toEqual([shared,shared,shared,shared,shared,shared,shared,shared]);
  expect(rows.map(row=>parity.decode(row))).toEqual([odd,even,odd,even,odd,even,odd,even]);
  expect(rows.map(row=>individual.decode(row))).toEqual([one,two,three,four,five,six,seven,eight]);
  expect(parity.decode(3)).toBe(parity.decode(7));
  expect(parity.decode(2)).toBe(parity.decode(8));
  expect(individual.decode(2)).not.toBe(individual.decode(3));
});
