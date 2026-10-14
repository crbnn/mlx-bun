// Fast, tiny-tensor coverage for adapter interchange and save durability.
// No model weights are loaded: AdapterManager only needs a model-shaped target
// map, and the safetensors fixtures contain one 3x4 LoRA delta.

import { afterAll, describe, expect, test } from "bun:test";
import {
  existsSync,
  mkdtempSync,
  mkdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AdapterManager, type AdapterTarget } from "../../src/adapters/manager";
import { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";

import { LoraDelta, LoraState, LoraOperation, type LoraWeights } from "../../src/layers/lora";


const root = mkdtempSync(join(tmpdir(), "mlx-bun-lora-format-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

const MODULE = "model.layers.0.self_attn.q_proj";

test("adapter cache identity survives remounts and changes with weights or scale", async () => {
  const dir = join(root, "cache-identity");
  const manager = new AdapterManager(stubModel().model);
  const a = { shape: [2, 3], values: [1, 2, 3, 4, 5, 6] };
  const b = { shape: [4, 2], values: [1, 2, 3, 4, 5, 6, 7, 8] };
  writePeftAdapter(dir, a, b);
  try {
    await manager.mount("same-name", dir);
    const original = manager.cacheNamespace(["same-name"]);
    manager.unmount("same-name");
    expect(() => manager.cacheNamespace(["same-name"])).toThrow("no longer mounted");
    await manager.mount("same-name", dir);
    expect(manager.cacheNamespace(["same-name"])).toBe(original);
    manager.unmount("same-name");
    writePeftAdapter(dir, a, b, { alpha: 8 });
    await manager.mount("same-name", dir);
    expect(manager.cacheNamespace(["same-name"])).not.toBe(original);
    const scaled = manager.cacheNamespace(["same-name"]);
    manager.unmount("same-name");
    writePeftAdapter(dir, a, { ...b, values: b.values.map((v) => v + 1) }, { alpha: 8 });
    await manager.mount("same-name", dir);
    expect(manager.cacheNamespace(["same-name"])).not.toBe(scaled);
    expect(manager.cacheNamespace([])).toBe("");
  } finally { manager.unmount("same-name"); }
});

interface StubLinear {
  inFeatures: number;
  outFeatures: number;
  adapters: Map<string, LoraWeights> | null;
  loraState: LoraState | null;
}

function stubModel(): {
  model: AdapterTarget;
  linear: StubLinear;
} {
  const state = new LoraState();
  const linear: StubLinear = {
    inFeatures: 3,
    outFeatures: 4,
    adapters: null,
    loraState: null,
  };
  const targets = new Map([[MODULE, linear]]);
  const model = {
    prefixBase: "model",
    loraState: state,
    loraTargets: () => targets,
  } satisfies AdapterTarget;
  return { model, linear };
}

function writeF32Safetensors(
  file: string,
  tensors: Record<string, { shape: number[]; values: number[] }>,
): void {
  let offset = 0;
  const header: Record<string, unknown> = { __metadata__: { format: "pt" } };
  const parts: Uint8Array[] = [];
  for (const [name, tensor] of Object.entries(tensors)) {
    const values = new Float32Array(tensor.values);
    const bytes = new Uint8Array(values.buffer);
    header[name] = {
      dtype: "F32",
      shape: tensor.shape,
      data_offsets: [offset, offset + bytes.byteLength],
    };
    parts.push(bytes);
    offset += bytes.byteLength;
  }

  const json = new TextEncoder().encode(JSON.stringify(header));
  const paddedLength = Math.ceil(json.byteLength / 8) * 8;
  const out = new Uint8Array(8 + paddedLength + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(paddedLength), true);
  out.fill(0x20, 8, 8 + paddedLength);
  out.set(json, 8);
  let dataOffset = 8 + paddedLength;
  for (const part of parts) {
    out.set(part, dataOffset);
    dataOffset += part.byteLength;
  }
  writeFileSync(file, out);
}

function writePeftAdapter(
  dir: string,
  a: { shape: number[]; values: number[] },
  b: { shape: number[]; values: number[] },
  options: { alpha?: number; rank?: number; useRslora?: boolean } = {},
): void {
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "adapter_config.json"), JSON.stringify({
    base_model_name_or_path: "example/base",
    lora_alpha: options.alpha ?? 4,
    peft_type: "LORA",
    r: options.rank ?? 2,
    target_modules: ["q_proj"],
    use_rslora: options.useRslora ?? false,
  }));
  writeF32Safetensors(join(dir, "adapter_model.safetensors"), {
    [`base_model.model.${MODULE}.lora_A.weight`]: a,
    [`base_model.model.${MODULE}.lora_B.weight`]: b,
  });
}

describe("AdapterManager PEFT layout", () => {
  test("mounts standard PEFT names/orientation and produces the expected delta", async () => {
    const dir = join(root, "peft-valid");
    // PEFT/PyTorch layout: A [rank,in], B [out,rank].
    writePeftAdapter(
      dir,
      { shape: [2, 3], values: [1, 2, 3, 4, 5, 6] },
      { shape: [4, 2], values: [7, 8, 9, 10, 11, 12, 13, 14] },
    );
    const { model, linear } = stubModel();
    const manager = new AdapterManager(model);

    const info = await manager.mount("peft", dir);
    expect(info.mountedLayers).toBe(1);
    expect(info.rank).toBe(2);
    expect(info.scale).toBe(2);

    const mounted = linear.adapters!.get("peft")!;
    expect(mounted.a.shape).toEqual([3, 2]);
    expect(mounted.b.shape).toEqual([2, 4]);

    const x = MlxArray.fromFloat32(new Float32Array([1, 0, -1]), [1, 3]);
    const xa = ops.matmul(x, mounted.a);
    const delta = ops.matmul(xa, mounted.b);
    const scaled = ops.mulScalar(delta, mounted.scale);
    expect([...scaled.toFloat32()]).toEqual([-60, -76, -92, -108]);
    x.dispose();
    xa.dispose();
    delta.dispose();
    scaled.dispose();
    expect(manager.unmount("peft")).toBe(1);
  });

  test("mounts PEFT use_rslora with alpha/sqrt(rank) applied exactly once", async () => {
    const dir = join(root, "peft-rslora");
    writePeftAdapter(
      dir,
      { shape: [2, 3], values: [1, 2, 3, 4, 5, 6] },
      { shape: [4, 2], values: [7, 8, 9, 10, 11, 12, 13, 14] },
      { alpha: 4, rank: 2, useRslora: true },
    );
    const { model, linear } = stubModel();
    const manager = new AdapterManager(model);

    const info = await manager.mount("peft-rslora", dir);
    const mounted = linear.adapters!.get("peft-rslora")!;
    expect(info.rank).toBe(2);
    expect(info.scale).toBe(4);
    expect(mounted.scale).toBeCloseTo(4 / Math.sqrt(2), 6);

    const x = MlxArray.fromFloat32(new Float32Array([1, 0, -1]), [1, 3]);
    const xa = ops.matmul(x, mounted.a);
    const delta = ops.matmul(xa, mounted.b);
    const scaled = ops.mulScalar(delta, mounted.scale);
    const expected = [-30, -38, -46, -54].map((v) => v * 4 / Math.sqrt(2));
    const actual = [...scaled.toFloat32()];
    for (let i = 0; i < expected.length; i++)
      expect(actual[i]).toBeCloseTo(expected[i]!, 4);
    x.dispose();
    xa.dispose();
    delta.dispose();
    scaled.dispose();
    expect(manager.unmount("peft-rslora")).toBe(1);
  });

  test("rejects tensors matching neither mlx-lm nor PEFT orientation", async () => {
    const dir = join(root, "peft-mismatch");
    writePeftAdapter(
      dir,
      { shape: [2, 5], values: new Array(10).fill(1) },
      { shape: [4, 2], values: new Array(8).fill(1) },
    );
    const { model } = stubModel();
    const manager = new AdapterManager(model);

    await expect(manager.mount("bad", dir)).rejects.toThrow(
      "Expected mlx-lm [in, rank] + [rank, out] or PEFT [rank, in] + [out, rank]",
    );
    expect(manager.list()).toEqual([]);
  });
});


test("mounted adapters apply after an arbitrary projection and follow request selection", async () => {
  const dir = join(root, "projection-delta");
  writePeftAdapter(dir,
    { shape: [2, 3], values: [1, 2, 3, 4, 5, 6] },
    { shape: [4, 2], values: [7, 8, 9, 10, 11, 12, 13, 14] });
  const delta = new LoraDelta(3, 4), state = new LoraState();
  const manager = new AdapterManager({ prefixBase: "model", loraState: state,
    loraTargets: () => new Map([[MODULE, delta]]) });
  using x = MlxArray.fromFloat32(new Float32Array([1, 0, -1]), [1, 3]);
  using weight = MlxArray.fromFloat32(new Float32Array([1, 2, 3, 4, 0, 0, 0, 0, 4, 3, 2, 1]), [3, 4]);
  const project = () => delta.forward(x, ops.matmul(x, weight));
  try {
    expect((await manager.mount("request-a", dir)).mountedLayers).toBe(1);
    state.active = ["request-a"];
    using adapted = project();
    expect([...adapted.toFloat32()]).toEqual([-63, -77, -91, -105]);
    state.active = [];
    using base = project();
    expect([...base.toFloat32()]).toEqual([-3, -1, 1, 3]);
  } finally { manager.unmount("request-a"); }
});


test("fused Trellis gate/up/down adapters follow B1/B4 request contexts and unmount", async () => {
  const { TrellisLinear, fusedGateUpSwiglu } = await import("../../src/layers/trellis-linear");
  const { TrellisFusedMLP, TrellisReducePrompt, TrellisScatterPrompt } = await import("../../src/layers/trellis-prompt");
  const { bindMlxGateway } = await import("../../src/execution/gateway-binding");
  const { declareGraph } = await import("../../src/models/capabilities");
  const { stubPhases } = await import("../models/phase-forward");
  const { compiledSwiglu } = await import("../../src/layers/swiglu");
  const { Dtype } = await import("@mlx-bun/mlx/ffi");
  const DIM = 512, RANK = 2;
  const owned: MlxArray[] = [];
  const packed = (axis: 0 | 1) => {
    using codes = ops.fromInt32(Array.from({length:DIM*64}, (_,i)=>(i*7919+17)|0), [DIM,64]);
    using scaleValues = MlxArray.fromFloat32(new Float32Array(DIM).fill(0.01), [DIM]);
    const ints = codes.astype(Dtype.uint32), scales = scaleValues.astype(Dtype.bfloat16);
    owned.push(ints, scales);
    return new TrellisLinear(ints, scales, {mode:"trellis",bits:4,groupSize:256,
      trellis:{L:12,code:"1mad",axis}});
  };
  const gate = packed(1), up = packed(1), down = packed(0), state = new LoraState();
  const projections = [gate,up,down];
  const targets = new Map(projections.map((projection,index)=>[`model.layers.0.mlp.${["gate_proj","up_proj","down_proj"][index]}`,projection]));
  const never=()=>{throw new Error("request context never forwards");};
  const graph = {prefixBase:"model",loraState:state,loraTargets:()=>targets, weightsBytes:0, dispose(){}, makeCache:()=>[],
    config:{modelType:"adapter-test",eosTokenIds:[],text:{numHiddenLayers:0}},
    graphCapabilities:declareGraph(), ...stubPhases(never,never)};
  const manager = new AdapterManager(graph);
  const binding = bindMlxGateway(graph as unknown as import("../../src/models/graph").MlxTokenGraph);
  const fused = new TrellisFusedMLP(gate,up,new TrellisScatterPrompt(down),state);
  // Custom graphs may construct the fused layer before their manager binds its targets.
  const inferred = new TrellisFusedMLP(gate,up,new TrellisScatterPrompt(down));

  const makeAdapter = (name:string,alpha:number) => {
    const dir=join(root,name);mkdirSync(dir,{recursive:true});
    writeFileSync(join(dir,"adapter_config.json"),JSON.stringify({lora_alpha:alpha,r:RANK}));
    const tensors:Parameters<typeof writeF32Safetensors>[1]={};
    for(const path of targets.keys()) {
      tensors[`${path}.lora_a`]={shape:[DIM,RANK],values:Array.from({length:DIM*RANK},(_,i)=>Math.sin(i*0.31)*0.01)};
      tensors[`${path}.lora_b`]={shape:[RANK,DIM],values:Array.from({length:DIM*RANK},(_,i)=>Math.cos(i*0.17)*0.01)};
    }
    writeF32Safetensors(join(dir,"adapters.safetensors"),tensors);return dir;
  };
  const bytesEqual=(a:MlxArray,b:MlxArray)=>expect(Buffer.from(a.rawBytesView())).toEqual(Buffer.from(b.rawBytesView()));
  const residual=(x:MlxArray,target:typeof gate,id:string,base:MlxArray)=>{
    const lw=target.adapters!.get(id)!;
    using xa=ops.matmul(x,lw.a), z=ops.matmul(xa,lw.b), zs=ops.mulScalar(z,lw.scale), cast=zs.astype(x.dtype);
    return ops.add(base,cast);
  };
  try {
    expect((await manager.mount("A",makeAdapter("trellis-a",2))).mountedLayers).toBe(3);
    expect((await manager.mount("B",makeAdapter("trellis-b",4))).mountedLayers).toBe(3);
    for(const rows of [1,4]) {
      using raw=MlxArray.fromFloat32(Float32Array.from({length:rows*DIM},(_,i)=>Math.sin(i*0.07)*0.2),[rows,1,DIM]);
      using x=raw.astype(Dtype.bfloat16);
      for(const id of ["A","B"]) {
        const restore=binding.bindAdapterContext!([id],id).enter();
        let actual:MlxArray;
        try {actual=fused.forward(x); using automatic=inferred.forward(x); bytesEqual(automatic,actual);} finally {restore();}

        using candidate=actual;
        expect(state.active).toEqual([]);
        using plainGate=new TrellisReducePrompt(gate).forward(x),plainUp=new TrellisReducePrompt(up).forward(x);
        using g=residual(x,gate,id,plainGate),u=residual(x,up,id,plainUp),hidden=compiledSwiglu(g,u);
        using plainDown=new TrellisScatterPrompt(down).forward(hidden),expected=residual(hidden,down,id,plainDown);
        bytesEqual(candidate,expected);
      }
      using originalHidden=fusedGateUpSwiglu(x,gate,up),original=new TrellisScatterPrompt(down).forward(originalHidden);
      using recovered=fused.forward(x),automaticRecovery=inferred.forward(x);bytesEqual(recovered,original);bytesEqual(automaticRecovery,original);

      const restore=binding.bindAdapterContext!(["A"],"cancelled").enter();
      try {throw new Error("cancelled request");} catch {} finally {restore();}
      using afterCancel=fused.forward(x);bytesEqual(afterCancel,original);
    }
    expect(manager.unmount("A")).toBe(3);expect(manager.unmount("B")).toBe(3);
    expect(manager.list()).toEqual([]);expect(state.active).toEqual([]);
  } finally {manager.unmount("A");manager.unmount("B");fused.dispose();inferred.dispose();for(const array of owned)array.dispose();}

});

test("optimized affine projections add the mounted delta after their B1/B4 kernel", async () => {
  const { QuantizedLinear } = await import("../../src/layers/quantized-linear");
  const { Affine3RowsLinear, Affine3MmaLinear, Affine4RowsLinear, Affine4MmaLinear } = await import("../../src/layers/affine-verify-linear");
  const { Dtype } = await import("@mlx-bun/mlx/ffi");
  const DIM = 512;
  using weightWide = MlxArray.fromFloat32(Float32Array.from({length:DIM*DIM},(_,i)=>Math.sin(i*0.11)*0.02),[DIM,DIM]);
  using weight = weightWide.astype(Dtype.bfloat16);
  for(const bits of [3,4]) {
    const q=ops.quantize(weight,64,bits),state=new LoraState();
    const linear=new QuantizedLinear(q.packed,q.scales,q.biases,{mode:"affine",bits,groupSize:64});
    const manager=new AdapterManager({prefixBase:"model",loraState:state,loraTargets:()=>new Map([[MODULE,linear]])});
    const dir=join(root,`optimized-affine-${bits}`);
    writePeftAdapter(dir,{shape:[2,DIM],values:Array.from({length:2*DIM},(_,i)=>Math.sin(i*0.3)*0.01)},
      {shape:[DIM,2],values:Array.from({length:2*DIM},(_,i)=>Math.cos(i*0.2)*0.01)});
    try {
      await manager.mount("adapter",dir);
      for(const rows of [1,4]) {
        using xWide=MlxArray.fromFloat32(Float32Array.from({length:rows*DIM},(_,i)=>Math.sin(i*0.07)*0.2),[rows,1,DIM]);
        using x=xWide.astype(Dtype.bfloat16);
        const layers=bits===3?[new Affine3MmaLinear(linear),...(rows>1?[new Affine3RowsLinear(linear)]:[])]:
          [new Affine4MmaLinear(linear),...(rows>1?[new Affine4RowsLinear(linear)]:[])];
        for(const layer of layers) {
          state.active=[];using base=layer.forward(x);
          state.active=["adapter"];using actual=layer.forward(x);
          const lw=linear.adapters!.get("adapter")!;
          using xa=ops.matmul(x,lw.a),z=ops.matmul(xa,lw.b),zs=ops.mulScalar(z,lw.scale),cast=zs.astype(x.dtype),expected=ops.add(base,cast);
          expect(Buffer.compare(Buffer.from(actual.rawBytes()),Buffer.from(expected.rawBytes()))).toBe(0);
        }
      }
    } finally {manager.unmount("adapter");q.packed.dispose();q.scales.dispose();q.biases?.dispose();}
  }
});
