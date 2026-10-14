import { expect,test } from "bun:test";
import { mkdtempSync,rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { originalOraclePython,ORIGINAL_REVISION,ORIGINAL_SCRIPT,externalOutput } from "../../scripts/gemma4-oracle-reference";
import { observeTargetPhases } from "../../scripts/reference-phases";

test("Gemma references use the unchanged pinned producer calculation",()=>{
  const original=Bun.spawnSync(["git","show",`${ORIGINAL_REVISION}:${ORIGINAL_SCRIPT}`]);
  expect(original.exitCode).toBe(0);
  const python=originalOraclePython(original.stdout.toString());
  expect(python).toContain('cache = make_prompt_cache(model)');
  expect(python).toContain('last = logits[0, -1, :].astype(mx.float32)');
  expect(()=>originalOraclePython(original.stdout.toString()+"\n")).toThrow("differs from its Git pin");
});

test("a reference output cannot become data inside the source checkout",()=>{
  const root=mkdtempSync(join(tmpdir(),"reference-paths-"));
  try {
    expect(()=>externalOutput(root,join(root,"goldens"))).toThrow("outside source");
    expect(()=>externalOutput(root,"relative")).toThrow("absolute external");
    expect(externalOutput(root,`${root}-external`)).toBe(`${root}-external`);
  } finally {rmSync(root,{recursive:true,force:true});}
});

test("phase observers preserve actual named-width GPU and ANE calls",()=>{
  const calls:number[][]=[],seen:number[][]=[];
  const forward=(ids:number[])=>{calls.push(ids);return ids;};
  class Model {
    #value=7;
    get value(){return this.#value;}
    method(){return this.#value;}
    decode={one:forward}; verify={twoToFour:forward}; appendKnown={any:forward}; prefillTail={any:forward};
    prefillChunk={gpu:forward,ane:[{minRows:640,rows:1024,forward}]};
  }
  const base=new Model(),view=observeTargetPhases(base,ids=>seen.push(ids));
  expect(view.value).toBe(7);expect(view.method()).toBe(7);
  expect(view.prefillChunk.ane[0]!.minRows).toBe(640);
  for(const [run,ids] of [[view.prefillChunk.gpu,[1,2]], [view.prefillChunk.ane[0]!.forward,[3,4]],
    [view.verify.twoToFour,[5,6,7]], [view.decode.one,[8]]] as const) expect(run([...ids])).toEqual([...ids]);
  expect(seen).toEqual(calls);expect(seen).toEqual([[1,2],[3,4],[5,6,7],[8]]);
});

test("numeric and range phase observers keep operation ordering and row facts",()=>{
  const seen:number[][]=[];
  const forward=(ids:number[])=>ids;
  const base={decode:{1:forward},verify:{3:forward},appendKnown:[{from:1,to:Infinity,operation:forward}],
    prompt:[{from:640,to:1024,unit:"ane",operation:forward}],multiSequencePrompt:[{from:1,to:Infinity,operation:forward}]};
  const view=observeTargetPhases(base,ids=>seen.push(ids));
  expect(view.prompt[0]!.unit).toBe("ane");expect(view.prompt[0]!.from).toBe(640);
  view.prompt[0]!.operation([1,2]);view.verify[3]([3,4,5]);view.appendKnown[0]!.operation([6]);view.decode[1]([7]);
  expect(seen).toEqual([[1,2],[3,4,5],[6],[7]]);
});

test("producer help runs with unavailable native libraries",()=>{
  for(const file of ["gemma4-oracle-reference.ts","main-qwen-trellis-reference.ts"]) {
    const result=Bun.spawnSync([process.execPath,"--no-env-file",join(import.meta.dir,"../../scripts",file),"--help"],
      {env:{...process.env,MLX_BUN_LIBMLXC:"/unavailable/native"}});
    expect(result.exitCode).toBe(0);expect(result.stdout.toString()).toContain("--out");
  }
});
