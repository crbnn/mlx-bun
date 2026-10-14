import { expect, test } from "bun:test";
import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { parseQuantization, type ModelConfig } from "../../../src/artifacts/config";
import type { Weights } from "../../../src/artifacts/weights";
import { UniversalDenseModel } from "../../../src/models/universal/dense";
import { UniversalDenseWideModel } from "../../../src/models/universal/dense-wide";
import { TrellisLinear } from "../../../src/layers/trellis-linear";
import { sequenceHiddenOperation, sequenceLogitsOperation } from "../../../src/scoring/full-sequence";
import { composeGraphCaches } from "../../../src/models/cache-composition";

function fixture(attentionType: Dtype) {
  const width=512, tensors=new Map<string,MlxArray>(), owned:MlxArray[]=[];
  const store=(name:string,value:MlxArray)=>{tensors.set(name,value);owned.push(value);};
  const dense=(name:string,shape:number[],dtype=Dtype.bfloat16,shift=0)=>{
    using values=MlxArray.fromFloat32(Float32Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>shift+Math.sin(i*0.31+name.length)*0.02),shape);
    store(name,values.astype(dtype));
  };
  const quant:Record<string,unknown>={bits:4,group_size:64,mode:"affine"};
  const packed=(path:string,axis:0|1)=>{
    using values=ops.fromInt32(Array.from({length:width*64},(_,i)=>(i*7919+23)|0),[width,64]);
    store(`${path}.weight`,values.astype(Dtype.uint32));
    dense(`${path}.scales`,[width],Dtype.bfloat16,0.03);
    quant[path]={bits:4,group_size:256,mode:"trellis",trellis:{L:12,code:"1mad",axis}};
  };
  dense("model.embed_tokens.weight",[width,width]);dense("model.norm.weight",[width],Dtype.bfloat16,1);
  const layer="model.layers.0";
  dense(`${layer}.input_layernorm.weight`,[width],Dtype.bfloat16,1);dense(`${layer}.post_attention_layernorm.weight`,[width],Dtype.bfloat16,1);
  dense(`${layer}.self_attn.q_proj.weight`,[width,width],attentionType);
  dense(`${layer}.self_attn.k_proj.weight`,[256,width],attentionType);dense(`${layer}.self_attn.v_proj.weight`,[256,width],attentionType);
  dense(`${layer}.self_attn.o_proj.weight`,[width,width],attentionType);
  packed(`${layer}.mlp.gate_proj`,1);packed(`${layer}.mlp.up_proj`,1);packed(`${layer}.mlp.down_proj`,0);packed("lm_head",1);
  const raw={model_type:"llama",hidden_size:width,num_hidden_layers:1,num_attention_heads:4,num_key_value_heads:2,head_dim:128,
    intermediate_size:width,vocab_size:width,rms_norm_eps:1e-6,max_position_embeddings:512,rope_theta:10000,tie_word_embeddings:false};
  const config={modelDir:"",modelType:"llama",architectures:[],dtype:"bfloat16",kvQuant:null,hasVisionSidecar:false,eosTokenIds:[],raw,
    quantization:parseQuantization(quant),text:{hiddenSize:width,numHiddenLayers:1,numAttentionHeads:4,numKvHeads:2,headDim:128,
      intermediateSize:width,vocabSize:width,rmsNormEps:1e-6,maxPositionEmbeddings:512,tieWordEmbeddings:false}} as unknown as ModelConfig;
  const weights={tensorNames:[...tensors.keys()],has:(name:string)=>tensors.has(name),tensor:(name:string)=>tensors.get(name)!,
    shards:{files:new Map()},info:(name:string)=>({shape:tensors.get(name)!.shape})} as unknown as Weights;
  return {config,weights,dispose(){for(const array of owned)array.dispose();}};
}

for(const Model of [UniversalDenseModel,UniversalDenseWideModel]) for(const attentionType of [Dtype.bfloat16,Dtype.float32]) {
  test(`${Model.name} fixes Trellis projections and unbounded heads after ${attentionType===Dtype.float32?"promoted":"bf16"} attention`,()=>{
    const source=fixture(attentionType), model=new Model(source.weights,source.config,undefined,8);
    const cache=()=>composeGraphCaches(model,{kv:{kind:"bf16"},pagedKv:null});
    try {
      expect([...model.loraTargets().keys()].sort()).toEqual(["model.layers.0.mlp.down_proj","model.layers.0.mlp.gate_proj","model.layers.0.mlp.up_proj"]);
      expect(model.logitsFromHidden).toBeUndefined();
      for(const rows of [1,4,5,8,9,16,32,33,129]) {
        using ids=ops.fromInt32(Array.from({length:rows},(_,i)=>(i*19+7)%512),[1,rows]);
        const oldCache=cache(),fixedCache=cache();
        try {
          let hidden=model.embed.encode(ids);
          for(const [index,layer] of model.layers.entries()){const next=layer.forward(hidden,oldCache[index]!,"window");hidden.dispose();hidden=next;}
          using oldHidden=hidden, expected=model.finalNorm.forward(oldHidden), oldLogits=model.lmHead!.forward(expected);
          // Owning carriers remain available to tools; serving must use their fixed borrowed ports.
          const carriers=[model.layers[0]!.mlp.gate!,model.layers[0]!.mlp.up!,model.layers[0]!.mlp.down,model.lmHead!]
            .filter((projection):projection is TrellisLinear=>projection instanceof TrellisLinear);
          const forwards=carriers.map(carrier=>carrier.forward);
          try {
            for(const carrier of carriers)carrier.forward=()=>{throw new Error("dynamic projection called");};
            using actual=sequenceHiddenOperation(model,ids,fixedCache), actualLogits=sequenceLogitsOperation(model,actual);
            expect(actual.dtype).toBe(expected.dtype);
            expect(Buffer.from(actual.rawBytesView())).toEqual(Buffer.from(expected.rawBytesView()));
            expect(Buffer.from(actualLogits.rawBytesView())).toEqual(Buffer.from(oldLogits.rawBytesView()));
          } finally {carriers.forEach((carrier,index)=>{carrier.forward=forwards[index]!;});}
        } finally {for(const state of [...oldCache,...fixedCache])state.dispose();}
      }
    } finally {model.dispose();source.dispose();}
  });
}
