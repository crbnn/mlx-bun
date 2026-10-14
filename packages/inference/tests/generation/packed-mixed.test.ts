import { expect,test } from "bun:test";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { Cache } from "../../src/contracts/mlx/cache";
import type { MlxPhaseGroup } from "../../src/contracts/mlx/graph";
import { PackedMixedScope } from "../../src/input/packed-mixed";

function handle() {
  let disposals=0;
  return {value:{dispose:()=>{disposals++;}} as unknown as MlxArray,disposals:()=>disposals};
}
const group=(ids:MlxArray,capture:MlxPhaseGroup["capture"]):MlxPhaseGroup=>({phase:"decode",rows:1,ids,state:[{} as Cache],capture});

test("a failed layer capture releases the intermediate and new hidden exactly once",()=>{
  const input=handle(),hidden=handle(),mid=handle(),output=handle();
  expect(()=>{
    using scope=new PackedMixedScope([group(input.value,()=>{throw new Error("capture failed");})],()=>hidden.value);
    scope.accept(0,(value,cache,phase)=>{
      expect(value).toBe(hidden.value);expect(phase).toBe("decode");expect(cache).toBeDefined();return mid.value;
    },value=>{expect(value).toBe(mid.value);return output.value;});
  }).toThrow("capture failed");
  expect(hidden.disposals()).toBe(1);expect(mid.disposals()).toBe(1);expect(output.disposals()).toBe(1);expect(input.disposals()).toBe(0);
});

test("a failed final capture releases its result and the scope's hidden",()=>{
  const input=handle(),hidden=handle(),output=handle();
  expect(()=>{
    using scope=new PackedMixedScope([group(input.value,()=>{throw new Error("final capture failed");})],()=>hidden.value);
    scope.finish(64,value=>{expect(value).toBe(hidden.value);return output.value;});
  }).toThrow("final capture failed");
  expect(hidden.disposals()).toBe(1);expect(output.disposals()).toBe(1);expect(input.disposals()).toBe(0);
});

test("a partially encoded group list releases its owned embeddings on failure",()=>{
  const input=handle(),hidden=handle();let calls=0;
  expect(()=>new PackedMixedScope([group(input.value,null),group(input.value,null)],()=>{
    if (++calls===2) throw new Error("encode failed");return hidden.value;
  })).toThrow("encode failed");
  expect(hidden.disposals()).toBe(1);expect(input.disposals()).toBe(0);
});
