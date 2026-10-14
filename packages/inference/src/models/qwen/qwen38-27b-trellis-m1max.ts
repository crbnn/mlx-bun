import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { ModelConfig } from "../../artifacts/config";
import type { Weights } from "../../artifacts/weights";
import type { Cache } from "../../contracts/mlx/cache";
import type { ResidualBasis, TargetView } from "../../contracts/mlx/draft-target";
import type { MlxDeclaredGraph, MlxTokenPhases, MlxLayerCapture, MlxHeadForward, MlxRowTable, MlxPromptRange, MlxCacheLayer, MlxPhaseForward, MlxSequenceRange, MlxMixedRange, MlxPhaseGroup, MlxMixedGroupRange, MlxSeparateMixed, MlxKnownRange } from "../../contracts/mlx/graph";
import type { MediaEncoders, MlxPromptInput, Vision, MlxPositionedForward, MlxPositionedRange, MlxPositionedEmbeddingsRange } from "../../contracts/mlx/media";
import type { MropeForwardState } from "../../contracts/mlx/positions";
import type { GraphCapabilities } from "../../contracts/portable/graph";
import { LoraState } from "../../layers/lora";
import { RMSNorm } from "../../layers/normalization";
import { QuantizedEmbedding } from "../../layers/quantized-embedding";
import { QuantizedLinear, QuantizedLinearPlain as Plain } from "../../layers/quantized-linear";
import { mropeInvFreq } from "../../layers/qwen-mrope";
import { ReduceGateUpRow as GateRow, ReduceGateUpRows as GateRows, ReduceSplitMixedGateUp as MixedGate } from "../../layers/trellis-reduce-gate-up";
import { ScatterDownK3iRow as DownK3iRow, ScatterDownK3iRows as DownK3iRows, ScatterDownRow as DownRow, ScatterDownRows as DownRows } from "../../layers/trellis-scatter-down";
import { TrellisFusedMLP, TrellisUnfusedMLP, TrellisScatterPrompt as Scatter, TrellisSplitKPrompt as SplitK, TrellisTiledPrompt as Tiled, TrellisExpandReducePrompt as ExpandGate, TrellisExpandScatterPrompt as ExpandDown } from "../../layers/trellis-prompt";
import { GraphForwardScope, BoundedGraphForwardScope } from "../../runtime/graph-forward";
import { disposeResources } from "../../runtime/resources";
import { declareGraph } from "../capabilities";
import type { ReadPhase } from "../graph";
import { QwenDeltaLayer as Delta, QwenAttentionLayer as Attention } from "./blocks";
import { bindQwenMediaInput, qwenDraftTarget, qwenMediaEncoders, qwenResidualBasis } from "./media-input";
import { DeltaWeights, AttentionWeights, GateDownMLP, canonicalQwenWeights, type QwenProjectionTarget } from "./written-layers";
import { PackedMixedScope } from "../../input/packed-mixed";

export const QWEN38_TRELLIS_M1MAX_GRAPH="qwen3.8-27b-trellis-3.2bpw-m1max";
export const DECODE_ROWS=8;
export const VERIFY_ROWS=64;
const PREFIX="language_model.model";
const REPRESENTATIONS={gateUpCodebook:true,mixedExactBits:true,scatter:{bits:true,genericBits:true,floatCodebook:true},widePrefill:false};
type Layers=readonly [
  Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,
  Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,
  Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,
  Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,
];
export type RowsForward=(ids:MlxArray,state:Cache[],capture:MlxLayerCapture|null)=>MlxArray;

/** The M1 Max arrangement, with its own reduce/scatter decode operations and
 * GPU prompt operations. The loader recommends it; callers may choose it. */
export class Qwen38TrellisM1Max implements MlxDeclaredGraph,MlxTokenPhases {
  readonly graph=QWEN38_TRELLIS_M1MAX_GRAPH;
  readonly prefixBase=PREFIX;
  readonly loraState=new LoraState();
  readonly faIdx=3;
  readonly weightsBytes:number;
  readonly embed:QuantizedEmbedding;
  readonly finalNorm:RMSNorm;
  readonly lmHead:QuantizedLinear;
  readonly graphCapabilities:GraphCapabilities=declareGraph({media:{input:"embeddings+positions",video:true},hiddenLayerTaps:true,prefill:{boundedWorkspace:true}});
  readonly verifyRoundCosts=new Map([[1,1],[2,1.89],[3,2.35],[4,2.97],[5,12.22],[6,13.16],[7,13.02],[8,12.88]]);
  readonly cacheLayers:readonly MlxCacheLayer[]=[
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense"},
  ];
  readonly prompt:readonly MlxPromptRange[];
  readonly multiSequencePrompt:readonly MlxPromptRange[];
  readonly decode:MlxRowTable<RowsForward>;
  readonly verify:MlxRowTable<RowsForward>;
  readonly appendKnown:readonly MlxKnownRange[];
  readonly logits:MlxRowTable<MlxHeadForward>;
  readonly sequence:readonly MlxSequenceRange[];
  readonly mediaPrefill:readonly MlxPositionedEmbeddingsRange[];
  readonly mediaPrompt:readonly MlxPositionedRange[];
  readonly mediaDecode:MlxRowTable<MlxPositionedForward>;
  readonly mixed:readonly MlxMixedRange[];
  readonly mixedSeparate:MlxSeparateMixed;
  readonly #head:Plain;
  readonly #targets:Map<string,QwenProjectionTarget>;
  readonly #owned:{dispose():void}[];
  readonly #residualBasis:ResidualBasis|undefined;
  readonly #mrope:{invFreq:MlxArray;rotaryDims:number};
  readonly #one:Layers;
  readonly #rows:Layers;
  readonly #stockNarrow:Layers;
  readonly #fiveToEight:Layers;
  readonly #tiled:Layers;
  readonly #expanded:Layers;

  constructor(weights:Weights,readonly config:ModelConfig) {
    const w=canonicalQwenWeights(weights),c=config,t=config.text;
    this.weightsBytes=[...weights.shards.files.values()].reduce((bytes,file)=>bytes+file.mmap.size,0);
    this.#residualBasis=qwenResidualBasis(this.graph,config.modelDir);
    this.embed=QuantizedEmbedding.load(w,`${PREFIX}.embed_tokens`,c);
    this.finalNorm=new RMSNorm(w.tensor(`${PREFIX}.norm.weight`),t.rmsNormEps);
    this.lmHead=QuantizedLinear.load(w,"language_model.lm_head",c);
    this.#head=new Plain(this.lmHead);
    const invScale=Math.pow(t.linearKeyHeadDim,-0.5),qk={q:ops.filledBf16(invScale*invScale,t.linearKeyHeadDim),k:ops.filledBf16(invScale,t.linearKeyHeadDim)};
    const rotaryDims=Math.trunc(t.headDim*t.partialRotaryFactor);
    this.#mrope={invFreq:mropeInvFreq(rotaryDims,t.ropeParameters.full_attention!.ropeTheta),rotaryDims};
    this.#owned=[qk.q,qk.k,this.#mrope.invFreq];
    const state=this.loraState;
    const d0=new DeltaWeights(w,c,`${PREFIX}.layers.0`,REPRESENTATIONS,state);
    const d1=new DeltaWeights(w,c,`${PREFIX}.layers.1`,REPRESENTATIONS,state);
    const d2=new DeltaWeights(w,c,`${PREFIX}.layers.2`,REPRESENTATIONS,state);
    const a3=new AttentionWeights(w,c,`${PREFIX}.layers.3`,REPRESENTATIONS,state);
    const d4=new DeltaWeights(w,c,`${PREFIX}.layers.4`,REPRESENTATIONS,state);
    const d5=new DeltaWeights(w,c,`${PREFIX}.layers.5`,REPRESENTATIONS,state);
    const d6=new DeltaWeights(w,c,`${PREFIX}.layers.6`,REPRESENTATIONS,state);
    const a7=new AttentionWeights(w,c,`${PREFIX}.layers.7`,REPRESENTATIONS,state);
    const d8=new DeltaWeights(w,c,`${PREFIX}.layers.8`,REPRESENTATIONS,state);
    const d9=new DeltaWeights(w,c,`${PREFIX}.layers.9`,REPRESENTATIONS,state);
    const d10=new DeltaWeights(w,c,`${PREFIX}.layers.10`,REPRESENTATIONS,state);
    const a11=new AttentionWeights(w,c,`${PREFIX}.layers.11`,REPRESENTATIONS,state);
    const d12=new DeltaWeights(w,c,`${PREFIX}.layers.12`,REPRESENTATIONS,state);
    const d13=new DeltaWeights(w,c,`${PREFIX}.layers.13`,REPRESENTATIONS,state);
    const d14=new DeltaWeights(w,c,`${PREFIX}.layers.14`,REPRESENTATIONS,state);
    const a15=new AttentionWeights(w,c,`${PREFIX}.layers.15`,REPRESENTATIONS,state);
    const d16=new DeltaWeights(w,c,`${PREFIX}.layers.16`,REPRESENTATIONS,state);
    const d17=new DeltaWeights(w,c,`${PREFIX}.layers.17`,REPRESENTATIONS,state);
    const d18=new DeltaWeights(w,c,`${PREFIX}.layers.18`,REPRESENTATIONS,state);
    const a19=new AttentionWeights(w,c,`${PREFIX}.layers.19`,REPRESENTATIONS,state);
    const d20=new DeltaWeights(w,c,`${PREFIX}.layers.20`,REPRESENTATIONS,state);
    const d21=new DeltaWeights(w,c,`${PREFIX}.layers.21`,REPRESENTATIONS,state);
    const d22=new DeltaWeights(w,c,`${PREFIX}.layers.22`,REPRESENTATIONS,state);
    const a23=new AttentionWeights(w,c,`${PREFIX}.layers.23`,REPRESENTATIONS,state);
    const d24=new DeltaWeights(w,c,`${PREFIX}.layers.24`,REPRESENTATIONS,state);
    const d25=new DeltaWeights(w,c,`${PREFIX}.layers.25`,REPRESENTATIONS,state);
    const d26=new DeltaWeights(w,c,`${PREFIX}.layers.26`,REPRESENTATIONS,state);
    const a27=new AttentionWeights(w,c,`${PREFIX}.layers.27`,REPRESENTATIONS,state);
    const d28=new DeltaWeights(w,c,`${PREFIX}.layers.28`,REPRESENTATIONS,state);
    const d29=new DeltaWeights(w,c,`${PREFIX}.layers.29`,REPRESENTATIONS,state);
    const d30=new DeltaWeights(w,c,`${PREFIX}.layers.30`,REPRESENTATIONS,state);
    const a31=new AttentionWeights(w,c,`${PREFIX}.layers.31`,REPRESENTATIONS,state);
    const d32=new DeltaWeights(w,c,`${PREFIX}.layers.32`,REPRESENTATIONS,state);
    const d33=new DeltaWeights(w,c,`${PREFIX}.layers.33`,REPRESENTATIONS,state);
    const d34=new DeltaWeights(w,c,`${PREFIX}.layers.34`,REPRESENTATIONS,state);
    const a35=new AttentionWeights(w,c,`${PREFIX}.layers.35`,REPRESENTATIONS,state);
    const d36=new DeltaWeights(w,c,`${PREFIX}.layers.36`,REPRESENTATIONS,state);
    const d37=new DeltaWeights(w,c,`${PREFIX}.layers.37`,REPRESENTATIONS,state);
    const d38=new DeltaWeights(w,c,`${PREFIX}.layers.38`,REPRESENTATIONS,state);
    const a39=new AttentionWeights(w,c,`${PREFIX}.layers.39`,REPRESENTATIONS,state);
    const d40=new DeltaWeights(w,c,`${PREFIX}.layers.40`,REPRESENTATIONS,state);
    const d41=new DeltaWeights(w,c,`${PREFIX}.layers.41`,REPRESENTATIONS,state);
    const d42=new DeltaWeights(w,c,`${PREFIX}.layers.42`,REPRESENTATIONS,state);
    const a43=new AttentionWeights(w,c,`${PREFIX}.layers.43`,REPRESENTATIONS,state);
    const d44=new DeltaWeights(w,c,`${PREFIX}.layers.44`,REPRESENTATIONS,state);
    const d45=new DeltaWeights(w,c,`${PREFIX}.layers.45`,REPRESENTATIONS,state);
    const d46=new DeltaWeights(w,c,`${PREFIX}.layers.46`,REPRESENTATIONS,state);
    const a47=new AttentionWeights(w,c,`${PREFIX}.layers.47`,REPRESENTATIONS,state);
    const d48=new DeltaWeights(w,c,`${PREFIX}.layers.48`,REPRESENTATIONS,state);
    const d49=new DeltaWeights(w,c,`${PREFIX}.layers.49`,REPRESENTATIONS,state);
    const d50=new DeltaWeights(w,c,`${PREFIX}.layers.50`,REPRESENTATIONS,state);
    const a51=new AttentionWeights(w,c,`${PREFIX}.layers.51`,REPRESENTATIONS,state);
    const d52=new DeltaWeights(w,c,`${PREFIX}.layers.52`,REPRESENTATIONS,state);
    const d53=new DeltaWeights(w,c,`${PREFIX}.layers.53`,REPRESENTATIONS,state);
    const d54=new DeltaWeights(w,c,`${PREFIX}.layers.54`,REPRESENTATIONS,state);
    const a55=new AttentionWeights(w,c,`${PREFIX}.layers.55`,REPRESENTATIONS,state);
    const d56=new DeltaWeights(w,c,`${PREFIX}.layers.56`,REPRESENTATIONS,state);
    const d57=new DeltaWeights(w,c,`${PREFIX}.layers.57`,REPRESENTATIONS,state);
    const d58=new DeltaWeights(w,c,`${PREFIX}.layers.58`,REPRESENTATIONS,state);
    const a59=new AttentionWeights(w,c,`${PREFIX}.layers.59`,REPRESENTATIONS,state);
    const d60=new DeltaWeights(w,c,`${PREFIX}.layers.60`,REPRESENTATIONS,state);
    const d61=new DeltaWeights(w,c,`${PREFIX}.layers.61`,REPRESENTATIONS,state);
    const d62=new DeltaWeights(w,c,`${PREFIX}.layers.62`,REPRESENTATIONS,state);
    const a63=new AttentionWeights(w,c,`${PREFIX}.layers.63`,REPRESENTATIONS,state);
    this.#owned.push(d0.trellis,d1.trellis,d2.trellis,a3.trellis,d4.trellis,d5.trellis,d6.trellis,a7.trellis,
      d8.trellis,d9.trellis,d10.trellis,a11.trellis,d12.trellis,d13.trellis,d14.trellis,a15.trellis,
      d16.trellis,d17.trellis,d18.trellis,a19.trellis,d20.trellis,d21.trellis,d22.trellis,a23.trellis,
      d24.trellis,d25.trellis,d26.trellis,a27.trellis,d28.trellis,d29.trellis,d30.trellis,a31.trellis,
      d32.trellis,d33.trellis,d34.trellis,a35.trellis,d36.trellis,d37.trellis,d38.trellis,a39.trellis,
      d40.trellis,d41.trellis,d42.trellis,a43.trellis,d44.trellis,d45.trellis,d46.trellis,a47.trellis,
      d48.trellis,d49.trellis,d50.trellis,a51.trellis,d52.trellis,d53.trellis,d54.trellis,a55.trellis,
      d56.trellis,d57.trellis,d58.trellis,a59.trellis,d60.trellis,d61.trellis,d62.trellis,a63.trellis);
    this.#targets=new Map([...d0.targets(),...d1.targets(),...d2.targets(),...a3.targets(),...d4.targets(),...d5.targets(),...d6.targets(),...a7.targets(),
      ...d8.targets(),...d9.targets(),...d10.targets(),...a11.targets(),...d12.targets(),...d13.targets(),...d14.targets(),...a15.targets(),
      ...d16.targets(),...d17.targets(),...d18.targets(),...a19.targets(),...d20.targets(),...d21.targets(),...d22.targets(),...a23.targets(),
      ...d24.targets(),...d25.targets(),...d26.targets(),...a27.targets(),...d28.targets(),...d29.targets(),...d30.targets(),...a31.targets(),
      ...d32.targets(),...d33.targets(),...d34.targets(),...a35.targets(),...d36.targets(),...d37.targets(),...d38.targets(),...a39.targets(),
      ...d40.targets(),...d41.targets(),...d42.targets(),...a43.targets(),...d44.targets(),...d45.targets(),...d46.targets(),...a47.targets(),
      ...d48.targets(),...d49.targets(),...d50.targets(),...a51.targets(),...d52.targets(),...d53.targets(),...d54.targets(),...a55.targets(),
      ...d56.targets(),...d57.targets(),...d58.targets(),...a59.targets(),...d60.targets(),...d61.targets(),...d62.targets(),...a63.targets()]);
    this.#one=[
      d0.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d1.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d2.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      a3.stage(w,c,Attention,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain),
      d4.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d5.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d6.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      a7.stage(w,c,Attention,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain),
      d8.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d9.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d10.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      a11.stage(w,c,Attention,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain),
      d12.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d13.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d14.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      a15.stage(w,c,Attention,GateDownMLP,MixedGate,DownRow,Plain,Plain,Plain,Plain),
      d16.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d17.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d18.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      a19.stage(w,c,Attention,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain),
      d20.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d21.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d22.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      a23.stage(w,c,Attention,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain),
      d24.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d25.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d26.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      a27.stage(w,c,Attention,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain),
      d28.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d29.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d30.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      a31.stage(w,c,Attention,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain),
      d32.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d33.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d34.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      a35.stage(w,c,Attention,GateDownMLP,MixedGate,DownK3iRow,Plain,Plain,Plain,Plain),
      d36.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d37.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d38.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      a39.stage(w,c,Attention,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain),
      d40.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d41.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d42.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      a43.stage(w,c,Attention,GateDownMLP,MixedGate,DownK3iRow,Plain,Plain,Plain,Plain),
      d44.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d45.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d46.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      a47.stage(w,c,Attention,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain),
      d48.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d49.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d50.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      a51.stage(w,c,Attention,GateDownMLP,MixedGate,DownK3iRow,Plain,Plain,Plain,Plain),
      d52.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d53.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d54.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      a55.stage(w,c,Attention,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain),
      d56.stage(w,c,Delta,GateDownMLP,MixedGate,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d57.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d58.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      a59.stage(w,c,Attention,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain),
      d60.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d61.stage(w,c,Delta,GateDownMLP,GateRow,DownRow,Plain,Plain,Plain,Plain,Plain,qk),
      d62.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      a63.stage(w,c,Attention,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain),
    ];
    this.#rows=[
      d0.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d1.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d2.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      a3.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain),
      d4.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d5.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d6.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      a7.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain),
      d8.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d9.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d10.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      a11.stage(w,c,Attention,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain),
      d12.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d13.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d14.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      a15.stage(w,c,Attention,GateDownMLP,MixedGate,DownRows,Plain,Plain,Plain,Plain),
      d16.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d17.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d18.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      a19.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain),
      d20.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d21.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d22.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      a23.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain),
      d24.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d25.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d26.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      a27.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain),
      d28.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d29.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d30.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      a31.stage(w,c,Attention,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain),
      d32.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d33.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d34.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      a35.stage(w,c,Attention,GateDownMLP,MixedGate,DownK3iRows,Plain,Plain,Plain,Plain),
      d36.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d37.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d38.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      a39.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain),
      d40.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d41.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d42.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      a43.stage(w,c,Attention,GateDownMLP,MixedGate,DownK3iRows,Plain,Plain,Plain,Plain),
      d44.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d45.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d46.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      a47.stage(w,c,Attention,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain),
      d48.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d49.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d50.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      a51.stage(w,c,Attention,GateDownMLP,MixedGate,DownK3iRows,Plain,Plain,Plain,Plain),
      d52.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d53.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      d54.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      a55.stage(w,c,Attention,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain),
      d56.stage(w,c,Delta,GateDownMLP,MixedGate,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d57.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d58.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      a59.stage(w,c,Attention,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain),
      d60.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d61.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Plain,Plain,Plain,Plain,Plain,qk),
      d62.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRows,Plain,Plain,Plain,Plain,Plain,qk),
      a63.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Plain,Plain,Plain,Plain),
    ];
    this.#stockNarrow=[
      d0.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d1.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d2.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a3.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d4.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d5.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d6.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a7.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d8.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d9.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d10.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a11.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d12.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d13.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d14.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a15.stage(w,c,Attention,GateDownMLP,MixedGate,Scatter,Plain,Plain,Plain,Plain),
      d16.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d17.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d18.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a19.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d20.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d21.stage(w,c,Delta,GateDownMLP,MixedGate,Scatter,Plain,Plain,Plain,Plain,Plain,qk),
      d22.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a23.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d24.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d25.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d26.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a27.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d28.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d29.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d30.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a31.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d32.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d33.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d34.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a35.stage(w,c,Attention,GateDownMLP,MixedGate,Scatter,Plain,Plain,Plain,Plain),
      d36.stage(w,c,Delta,GateDownMLP,MixedGate,Scatter,Plain,Plain,Plain,Plain,Plain,qk),
      d37.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d38.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a39.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d40.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d41.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d42.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a43.stage(w,c,Attention,GateDownMLP,MixedGate,Scatter,Plain,Plain,Plain,Plain),
      d44.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d45.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d46.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a47.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d48.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d49.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d50.stage(w,c,Delta,GateDownMLP,MixedGate,Scatter,Plain,Plain,Plain,Plain,Plain,qk),
      a51.stage(w,c,Attention,GateDownMLP,MixedGate,Scatter,Plain,Plain,Plain,Plain),
      d52.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d53.stage(w,c,Delta,GateDownMLP,MixedGate,Scatter,Plain,Plain,Plain,Plain,Plain,qk),
      d54.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a55.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d56.stage(w,c,Delta,GateDownMLP,MixedGate,Scatter,Plain,Plain,Plain,Plain,Plain,qk),
      d57.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d58.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      a59.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d60.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d61.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),
      d62.stage(w,c,Delta,GateDownMLP,MixedGate,Scatter,Plain,Plain,Plain,Plain,Plain,qk),
      a63.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
    ];
    this.#fiveToEight=[
      d0.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d1.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d2.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a3.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d4.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d5.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d6.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a7.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d8.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d9.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d10.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a11.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d12.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d13.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d14.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a15.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d16.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d17.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d18.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a19.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d20.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d21.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d22.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a23.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d24.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d25.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d26.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a27.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d28.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d29.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d30.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a31.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d32.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d33.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d34.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a35.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d36.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d37.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d38.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a39.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d40.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d41.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d42.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a43.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d44.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d45.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d46.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a47.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d48.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d49.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d50.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a51.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d52.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d53.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d54.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a55.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d56.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d57.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d58.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a59.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
      d60.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d61.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      d62.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,SplitK,Plain,qk),
      a63.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,SplitK,Plain),
    ];
    this.#tiled=[
      d0.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d1.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d2.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a3.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d4.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d5.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d6.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a7.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d8.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d9.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d10.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a11.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d12.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d13.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d14.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a15.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d16.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d17.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d18.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a19.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d20.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d21.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d22.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a23.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d24.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d25.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d26.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a27.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d28.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d29.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d30.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a31.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d32.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d33.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d34.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a35.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d36.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d37.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d38.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a39.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d40.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d41.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d42.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a43.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d44.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d45.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d46.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a47.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d48.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d49.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d50.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a51.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d52.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d53.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d54.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a55.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d56.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d57.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d58.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a59.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d60.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d61.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      d62.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),
      a63.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
    ];
    this.#expanded=[
      d0.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d1.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d2.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a3.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d4.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d5.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d6.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a7.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d8.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d9.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d10.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a11.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d12.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d13.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d14.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a15.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d16.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d17.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d18.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a19.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d20.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d21.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d22.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a23.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d24.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d25.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d26.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a27.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d28.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d29.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d30.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a31.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d32.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d33.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d34.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a35.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d36.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d37.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d38.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a39.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d40.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d41.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d42.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a43.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d44.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d45.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d46.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a47.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d48.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d49.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d50.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a51.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d52.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d53.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d54.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a55.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d56.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d57.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d58.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a59.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d60.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d61.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      d62.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),
      a63.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
    ];
    const stock=this.#stockNarrow,five=this.#fiveToEight,tiled=this.#tiled,expanded=this.#expanded;
    const mediaStock:Layers=[
      stock[0],stock[1],stock[2],a3.mediaLayer(w,c,Attention,stock[3].mlp,Plain),
      stock[4],stock[5],stock[6],a7.mediaLayer(w,c,Attention,stock[7].mlp,Plain),
      stock[8],stock[9],stock[10],a11.mediaLayer(w,c,Attention,stock[11].mlp,Plain),
      stock[12],stock[13],stock[14],a15.mediaLayer(w,c,Attention,stock[15].mlp,Plain),
      stock[16],stock[17],stock[18],a19.mediaLayer(w,c,Attention,stock[19].mlp,Plain),
      stock[20],stock[21],stock[22],a23.mediaLayer(w,c,Attention,stock[23].mlp,Plain),
      stock[24],stock[25],stock[26],a27.mediaLayer(w,c,Attention,stock[27].mlp,Plain),
      stock[28],stock[29],stock[30],a31.mediaLayer(w,c,Attention,stock[31].mlp,Plain),
      stock[32],stock[33],stock[34],a35.mediaLayer(w,c,Attention,stock[35].mlp,Plain),
      stock[36],stock[37],stock[38],a39.mediaLayer(w,c,Attention,stock[39].mlp,Plain),
      stock[40],stock[41],stock[42],a43.mediaLayer(w,c,Attention,stock[43].mlp,Plain),
      stock[44],stock[45],stock[46],a47.mediaLayer(w,c,Attention,stock[47].mlp,Plain),
      stock[48],stock[49],stock[50],a51.mediaLayer(w,c,Attention,stock[51].mlp,Plain),
      stock[52],stock[53],stock[54],a55.mediaLayer(w,c,Attention,stock[55].mlp,Plain),
      stock[56],stock[57],stock[58],a59.mediaLayer(w,c,Attention,stock[59].mlp,Plain),
      stock[60],stock[61],stock[62],a63.mediaLayer(w,c,Attention,stock[63].mlp,Plain),
    ];
    const mediaFive:Layers=[
      five[0],five[1],five[2],a3.mediaLayer(w,c,Attention,five[3].mlp,Plain),
      five[4],five[5],five[6],a7.mediaLayer(w,c,Attention,five[7].mlp,Plain),
      five[8],five[9],five[10],a11.mediaLayer(w,c,Attention,five[11].mlp,Plain),
      five[12],five[13],five[14],a15.mediaLayer(w,c,Attention,five[15].mlp,Plain),
      five[16],five[17],five[18],a19.mediaLayer(w,c,Attention,five[19].mlp,Plain),
      five[20],five[21],five[22],a23.mediaLayer(w,c,Attention,five[23].mlp,Plain),
      five[24],five[25],five[26],a27.mediaLayer(w,c,Attention,five[27].mlp,Plain),
      five[28],five[29],five[30],a31.mediaLayer(w,c,Attention,five[31].mlp,Plain),
      five[32],five[33],five[34],a35.mediaLayer(w,c,Attention,five[35].mlp,Plain),
      five[36],five[37],five[38],a39.mediaLayer(w,c,Attention,five[39].mlp,Plain),
      five[40],five[41],five[42],a43.mediaLayer(w,c,Attention,five[43].mlp,Plain),
      five[44],five[45],five[46],a47.mediaLayer(w,c,Attention,five[47].mlp,Plain),
      five[48],five[49],five[50],a51.mediaLayer(w,c,Attention,five[51].mlp,Plain),
      five[52],five[53],five[54],a55.mediaLayer(w,c,Attention,five[55].mlp,Plain),
      five[56],five[57],five[58],a59.mediaLayer(w,c,Attention,five[59].mlp,Plain),
      five[60],five[61],five[62],a63.mediaLayer(w,c,Attention,five[63].mlp,Plain),
    ];
    const mediaTiled:Layers=[
      tiled[0],tiled[1],tiled[2],a3.mediaLayer(w,c,Attention,tiled[3].mlp,Plain),
      tiled[4],tiled[5],tiled[6],a7.mediaLayer(w,c,Attention,tiled[7].mlp,Plain),
      tiled[8],tiled[9],tiled[10],a11.mediaLayer(w,c,Attention,tiled[11].mlp,Plain),
      tiled[12],tiled[13],tiled[14],a15.mediaLayer(w,c,Attention,tiled[15].mlp,Plain),
      tiled[16],tiled[17],tiled[18],a19.mediaLayer(w,c,Attention,tiled[19].mlp,Plain),
      tiled[20],tiled[21],tiled[22],a23.mediaLayer(w,c,Attention,tiled[23].mlp,Plain),
      tiled[24],tiled[25],tiled[26],a27.mediaLayer(w,c,Attention,tiled[27].mlp,Plain),
      tiled[28],tiled[29],tiled[30],a31.mediaLayer(w,c,Attention,tiled[31].mlp,Plain),
      tiled[32],tiled[33],tiled[34],a35.mediaLayer(w,c,Attention,tiled[35].mlp,Plain),
      tiled[36],tiled[37],tiled[38],a39.mediaLayer(w,c,Attention,tiled[39].mlp,Plain),
      tiled[40],tiled[41],tiled[42],a43.mediaLayer(w,c,Attention,tiled[43].mlp,Plain),
      tiled[44],tiled[45],tiled[46],a47.mediaLayer(w,c,Attention,tiled[47].mlp,Plain),
      tiled[48],tiled[49],tiled[50],a51.mediaLayer(w,c,Attention,tiled[51].mlp,Plain),
      tiled[52],tiled[53],tiled[54],a55.mediaLayer(w,c,Attention,tiled[55].mlp,Plain),
      tiled[56],tiled[57],tiled[58],a59.mediaLayer(w,c,Attention,tiled[59].mlp,Plain),
      tiled[60],tiled[61],tiled[62],a63.mediaLayer(w,c,Attention,tiled[63].mlp,Plain),
    ];
    const mediaExpanded:Layers=[
      expanded[0],expanded[1],expanded[2],a3.mediaLayer(w,c,Attention,expanded[3].mlp,Plain),
      expanded[4],expanded[5],expanded[6],a7.mediaLayer(w,c,Attention,expanded[7].mlp,Plain),
      expanded[8],expanded[9],expanded[10],a11.mediaLayer(w,c,Attention,expanded[11].mlp,Plain),
      expanded[12],expanded[13],expanded[14],a15.mediaLayer(w,c,Attention,expanded[15].mlp,Plain),
      expanded[16],expanded[17],expanded[18],a19.mediaLayer(w,c,Attention,expanded[19].mlp,Plain),
      expanded[20],expanded[21],expanded[22],a23.mediaLayer(w,c,Attention,expanded[23].mlp,Plain),
      expanded[24],expanded[25],expanded[26],a27.mediaLayer(w,c,Attention,expanded[27].mlp,Plain),
      expanded[28],expanded[29],expanded[30],a31.mediaLayer(w,c,Attention,expanded[31].mlp,Plain),
      expanded[32],expanded[33],expanded[34],a35.mediaLayer(w,c,Attention,expanded[35].mlp,Plain),
      expanded[36],expanded[37],expanded[38],a39.mediaLayer(w,c,Attention,expanded[39].mlp,Plain),
      expanded[40],expanded[41],expanded[42],a43.mediaLayer(w,c,Attention,expanded[43].mlp,Plain),
      expanded[44],expanded[45],expanded[46],a47.mediaLayer(w,c,Attention,expanded[47].mlp,Plain),
      expanded[48],expanded[49],expanded[50],a51.mediaLayer(w,c,Attention,expanded[51].mlp,Plain),
      expanded[52],expanded[53],expanded[54],a55.mediaLayer(w,c,Attention,expanded[55].mlp,Plain),
      expanded[56],expanded[57],expanded[58],a59.mediaLayer(w,c,Attention,expanded[59].mlp,Plain),
      expanded[60],expanded[61],expanded[62],a63.mediaLayer(w,c,Attention,expanded[63].mlp,Plain),
    ];
    const one=this.#one,rows=this.#rows;
    const decodeOne:RowsForward=(ids,cache,capture)=>this.#lazy(one,ids,cache,"decode",capture);
    const decodeRows:RowsForward=(ids,cache,capture)=>this.#lazy(rows,ids,cache,"decode",capture);
    const decodeFive:RowsForward=(ids,cache,capture)=>this.#lazy(five,ids,cache,"decode",capture);
    this.decode={1:decodeOne,2:decodeRows,3:decodeRows,4:decodeRows,5:decodeFive,6:decodeFive,7:decodeFive,8:decodeFive};
    const windowOne:RowsForward=(ids,cache,capture)=>this.#lazy(one,ids,cache,"window",capture);
    const windowRows:RowsForward=(ids,cache,capture)=>this.#lazy(rows,ids,cache,"window",capture);
    const windowFive:RowsForward=(ids,cache,capture)=>this.#lazy(five,ids,cache,"window",capture);
    const windowTiled:RowsForward=(ids,cache,capture)=>this.#lazy(tiled,ids,cache,"window",capture);
    const windowExpanded:RowsForward=(ids,cache,capture)=>this.#lazy(expanded,ids,cache,"window",capture);
    this.verify={
      1:windowOne,2:windowRows,3:windowRows,4:windowRows,5:windowFive,6:windowFive,7:windowFive,8:windowFive,
      9:windowTiled,10:windowTiled,11:windowTiled,12:windowTiled,13:windowTiled,14:windowTiled,15:windowTiled,16:windowTiled,
      17:windowTiled,18:windowTiled,19:windowTiled,20:windowTiled,21:windowTiled,22:windowTiled,23:windowTiled,24:windowTiled,
      25:windowTiled,26:windowTiled,27:windowTiled,28:windowTiled,29:windowTiled,30:windowTiled,31:windowTiled,32:windowTiled,
      33:windowExpanded,34:windowExpanded,35:windowExpanded,36:windowExpanded,37:windowExpanded,38:windowExpanded,39:windowExpanded,40:windowExpanded,
      41:windowExpanded,42:windowExpanded,43:windowExpanded,44:windowExpanded,45:windowExpanded,46:windowExpanded,47:windowExpanded,48:windowExpanded,
      49:windowExpanded,50:windowExpanded,51:windowExpanded,52:windowExpanded,53:windowExpanded,54:windowExpanded,55:windowExpanded,56:windowExpanded,
      57:windowExpanded,58:windowExpanded,59:windowExpanded,60:windowExpanded,61:windowExpanded,62:windowExpanded,63:windowExpanded,64:windowExpanded,
    };
    const knownOne:MlxPhaseForward=(ids,cache)=>windowOne(ids,cache,null);
    const knownRows:MlxPhaseForward=(ids,cache)=>windowRows(ids,cache,null);
    const knownFive:MlxPhaseForward=(ids,cache)=>windowFive(ids,cache,null);
    const knownTiled:MlxPhaseForward=(ids,cache)=>windowTiled(ids,cache,null);
    const knownExpanded:MlxPhaseForward=(ids,cache)=>windowExpanded(ids,cache,null);
    this.appendKnown=[{from:1,to:1,operation:knownOne},{from:2,to:4,operation:knownRows},
      {from:5,to:8,operation:knownFive},{from:9,to:32,operation:knownTiled},{from:33,to:Infinity,operation:knownExpanded}];
    const head:MlxHeadForward=hidden=>this.#head.forward(hidden);
    this.logits={
      1:head,2:head,3:head,4:head,5:head,6:head,7:head,8:head,9:head,10:head,11:head,12:head,13:head,14:head,15:head,16:head,
      17:head,18:head,19:head,20:head,21:head,22:head,23:head,24:head,25:head,26:head,27:head,28:head,29:head,30:head,31:head,32:head,
      33:head,34:head,35:head,36:head,37:head,38:head,39:head,40:head,41:head,42:head,43:head,44:head,45:head,46:head,47:head,48:head,
      49:head,50:head,51:head,52:head,53:head,54:head,55:head,56:head,57:head,58:head,59:head,60:head,61:head,62:head,63:head,64:head,
    };
    const promptFive:RowsForward=(ids,cache,capture)=>this.#bounded(five,ids,cache,"window",capture);
    const promptTiled:RowsForward=(ids,cache,capture)=>this.#bounded(tiled,ids,cache,"window",capture);
    const promptExpanded:RowsForward=(ids,cache,capture)=>this.#bounded(expanded,ids,cache,"window",capture);
    this.prompt=[{from:1,to:1,operation:windowOne,unit:"gpu"},{from:2,to:4,operation:windowRows,unit:"gpu"},
      {from:5,to:8,operation:promptFive,unit:"gpu"},{from:9,to:32,operation:promptTiled,unit:"gpu"},{from:33,to:Infinity,operation:promptExpanded,unit:"gpu"}];
    this.multiSequencePrompt=[{from:1,to:1,operation:windowOne},{from:2,to:4,operation:windowRows},
      {from:5,to:8,operation:promptFive},{from:9,to:32,operation:promptTiled},{from:33,to:Infinity,operation:promptExpanded}];
    this.sequence=[{from:1,to:1,operation:(ids,cache)=>windowOne(ids,cache,null)},{from:2,to:4,operation:(ids,cache)=>windowRows(ids,cache,null)},
      {from:5,to:8,operation:(ids,cache)=>promptFive(ids,cache,null)},{from:9,to:32,operation:(ids,cache)=>promptTiled(ids,cache,null)},
      {from:33,to:Infinity,operation:(ids,cache)=>promptExpanded(ids,cache,null)}];
    this.mediaPrefill=[
      {from:1,to:4,operation:(hidden,cache,positions)=>this.#positioned(mediaStock,ops.contiguous(hidden),cache,"window",positions)},
      {from:5,to:8,operation:(hidden,cache,positions)=>this.#positionedBounded(mediaFive,ops.contiguous(hidden),cache,positions)},
      {from:9,to:32,operation:(hidden,cache,positions)=>this.#positionedBounded(mediaTiled,ops.contiguous(hidden),cache,positions)},
      {from:33,to:Infinity,operation:(hidden,cache,positions)=>this.#positionedBounded(mediaExpanded,ops.contiguous(hidden),cache,positions)},
    ];
    this.mediaPrompt=[
      {from:1,to:4,operation:(ids,cache,positions)=>this.#positioned(mediaStock,this.embed.encode(ids),cache,"window",positions)},
      {from:5,to:8,operation:(ids,cache,positions)=>this.#positionedBounded(mediaFive,this.embed.encode(ids),cache,positions)},
      {from:9,to:32,operation:(ids,cache,positions)=>this.#positionedBounded(mediaTiled,this.embed.encode(ids),cache,positions)},
      {from:33,to:Infinity,operation:(ids,cache,positions)=>this.#positionedBounded(mediaExpanded,this.embed.encode(ids),cache,positions)},
    ];
    const mediaDecodeNarrow:MlxPositionedForward=(ids,cache,positions)=>this.#positioned(mediaStock,this.embed.encode(ids),cache,"decode",positions);
    const mediaDecodeFive:MlxPositionedForward=(ids,cache,positions)=>this.#positioned(mediaFive,this.embed.encode(ids),cache,"decode",positions);
    this.mediaDecode={1:mediaDecodeNarrow,2:mediaDecodeNarrow,3:mediaDecodeNarrow,4:mediaDecodeNarrow,5:mediaDecodeFive,6:mediaDecodeFive,7:mediaDecodeFive,8:mediaDecodeFive};
    this.mixed=[{from:1,to:4,operation:groups=>this.#packed(stock,groups)},
      {from:5,to:8,operation:groups=>this.#packed(five,groups)},
      {from:9,to:32,operation:groups=>this.#packed(tiled,groups)},
      {from:33,to:Infinity,operation:groups=>this.#packed(expanded,groups)}];
    const separateWindow:readonly MlxMixedGroupRange[]=[
      {from:1,to:4,operation:group=>this.#lazy(stock,group.ids,group.state,"window",group.capture)},
      {from:5,to:8,operation:group=>windowFive(group.ids,group.state,group.capture)},
      {from:9,to:32,operation:group=>windowTiled(group.ids,group.state,group.capture)},
      {from:33,to:Infinity,operation:group=>windowExpanded(group.ids,group.state,group.capture)},
    ];
    this.mixedSeparate={decode:[
      {from:1,to:4,operation:group=>this.#lazy(stock,group.ids,group.state,"decode",group.capture)},
      {from:5,to:8,operation:group=>decodeFive(group.ids,group.state,group.capture)},
    ],window:separateWindow,verify:separateWindow};
    console.error(`[graph] ${this.graph}`);
  }

  dispose():void { disposeResources(this.#owned.splice(0)); }
  loraTargets():Map<string,QwenProjectionTarget> { return new Map(this.#targets); }
  bindMediaInput(input:Vision):MlxPromptInput { return bindQwenMediaInput(this,input.embeddings,input.mrope!); }
  mediaEncoders(modelDir:string):Promise<MediaEncoders> { return qwenMediaEncoders(this,modelDir); }
  draftTarget(_state:Cache[]):TargetView { return qwenDraftTarget(this,this.#residualBasis); }
  logitsFromHidden(hidden:MlxArray):MlxArray { return this.#head.forward(hidden); }
  #lazy(layers:Layers,ids:MlxArray,state:Cache[],phase:ReadPhase,capture:MlxLayerCapture|null):MlxArray {
    using scope=new GraphForwardScope(this.embed.encode(ids),capture);
    return this.#run(layers,state,phase,scope);
  }
  #bounded(layers:Layers,ids:MlxArray,state:Cache[],phase:ReadPhase,capture:MlxLayerCapture|null):MlxArray {
    using scope=new BoundedGraphForwardScope(this.embed.encode(ids),capture,state);
    return this.#run(layers,state,phase,scope);
  }
  #positioned(layers:Layers,hidden:MlxArray,state:Cache[],phase:ReadPhase,positions:MlxArray):MlxArray {
    using scope=new GraphForwardScope(hidden,null);
    return this.#run(layers,state,phase,scope,{posIds:positions,...this.#mrope});
  }
  #positionedBounded(layers:Layers,hidden:MlxArray,state:Cache[],positions:MlxArray):MlxArray {
    using scope=new BoundedGraphForwardScope(hidden,null,state);
    return this.#run(layers,state,"window",scope,{posIds:positions,...this.#mrope});
  }
  #run(layers:Layers,state:Cache[],phase:ReadPhase,scope:GraphForwardScope,positions:MropeForwardState|null=null):MlxArray {
    scope.accept(0,layers[0].forward(scope.hidden,state[0]!,phase));
    scope.accept(1,layers[1].forward(scope.hidden,state[1]!,phase));
    scope.accept(2,layers[2].forward(scope.hidden,state[2]!,phase));
    scope.accept(3,layers[3].forward(scope.hidden,state[3]!,phase,positions));
    scope.accept(4,layers[4].forward(scope.hidden,state[4]!,phase));
    scope.accept(5,layers[5].forward(scope.hidden,state[5]!,phase));
    scope.accept(6,layers[6].forward(scope.hidden,state[6]!,phase));
    scope.accept(7,layers[7].forward(scope.hidden,state[7]!,phase,positions));
    scope.accept(8,layers[8].forward(scope.hidden,state[8]!,phase));
    scope.accept(9,layers[9].forward(scope.hidden,state[9]!,phase));
    scope.accept(10,layers[10].forward(scope.hidden,state[10]!,phase));
    scope.accept(11,layers[11].forward(scope.hidden,state[11]!,phase,positions));
    scope.accept(12,layers[12].forward(scope.hidden,state[12]!,phase));
    scope.accept(13,layers[13].forward(scope.hidden,state[13]!,phase));
    scope.accept(14,layers[14].forward(scope.hidden,state[14]!,phase));
    scope.accept(15,layers[15].forward(scope.hidden,state[15]!,phase,positions));
    scope.accept(16,layers[16].forward(scope.hidden,state[16]!,phase));
    scope.accept(17,layers[17].forward(scope.hidden,state[17]!,phase));
    scope.accept(18,layers[18].forward(scope.hidden,state[18]!,phase));
    scope.accept(19,layers[19].forward(scope.hidden,state[19]!,phase,positions));
    scope.accept(20,layers[20].forward(scope.hidden,state[20]!,phase));
    scope.accept(21,layers[21].forward(scope.hidden,state[21]!,phase));
    scope.accept(22,layers[22].forward(scope.hidden,state[22]!,phase));
    scope.accept(23,layers[23].forward(scope.hidden,state[23]!,phase,positions));
    scope.accept(24,layers[24].forward(scope.hidden,state[24]!,phase));
    scope.accept(25,layers[25].forward(scope.hidden,state[25]!,phase));
    scope.accept(26,layers[26].forward(scope.hidden,state[26]!,phase));
    scope.accept(27,layers[27].forward(scope.hidden,state[27]!,phase,positions));
    scope.accept(28,layers[28].forward(scope.hidden,state[28]!,phase));
    scope.accept(29,layers[29].forward(scope.hidden,state[29]!,phase));
    scope.accept(30,layers[30].forward(scope.hidden,state[30]!,phase));
    scope.accept(31,layers[31].forward(scope.hidden,state[31]!,phase,positions));
    scope.accept(32,layers[32].forward(scope.hidden,state[32]!,phase));
    scope.accept(33,layers[33].forward(scope.hidden,state[33]!,phase));
    scope.accept(34,layers[34].forward(scope.hidden,state[34]!,phase));
    scope.accept(35,layers[35].forward(scope.hidden,state[35]!,phase,positions));
    scope.accept(36,layers[36].forward(scope.hidden,state[36]!,phase));
    scope.accept(37,layers[37].forward(scope.hidden,state[37]!,phase));
    scope.accept(38,layers[38].forward(scope.hidden,state[38]!,phase));
    scope.accept(39,layers[39].forward(scope.hidden,state[39]!,phase,positions));
    scope.accept(40,layers[40].forward(scope.hidden,state[40]!,phase));
    scope.accept(41,layers[41].forward(scope.hidden,state[41]!,phase));
    scope.accept(42,layers[42].forward(scope.hidden,state[42]!,phase));
    scope.accept(43,layers[43].forward(scope.hidden,state[43]!,phase,positions));
    scope.accept(44,layers[44].forward(scope.hidden,state[44]!,phase));
    scope.accept(45,layers[45].forward(scope.hidden,state[45]!,phase));
    scope.accept(46,layers[46].forward(scope.hidden,state[46]!,phase));
    scope.accept(47,layers[47].forward(scope.hidden,state[47]!,phase,positions));
    scope.accept(48,layers[48].forward(scope.hidden,state[48]!,phase));
    scope.accept(49,layers[49].forward(scope.hidden,state[49]!,phase));
    scope.accept(50,layers[50].forward(scope.hidden,state[50]!,phase));
    scope.accept(51,layers[51].forward(scope.hidden,state[51]!,phase,positions));
    scope.accept(52,layers[52].forward(scope.hidden,state[52]!,phase));
    scope.accept(53,layers[53].forward(scope.hidden,state[53]!,phase));
    scope.accept(54,layers[54].forward(scope.hidden,state[54]!,phase));
    scope.accept(55,layers[55].forward(scope.hidden,state[55]!,phase,positions));
    scope.accept(56,layers[56].forward(scope.hidden,state[56]!,phase));
    scope.accept(57,layers[57].forward(scope.hidden,state[57]!,phase));
    scope.accept(58,layers[58].forward(scope.hidden,state[58]!,phase));
    scope.accept(59,layers[59].forward(scope.hidden,state[59]!,phase,positions));
    scope.accept(60,layers[60].forward(scope.hidden,state[60]!,phase));
    scope.accept(61,layers[61].forward(scope.hidden,state[61]!,phase));
    scope.accept(62,layers[62].forward(scope.hidden,state[62]!,phase));
    scope.accept(63,layers[63].forward(scope.hidden,state[63]!,phase,positions));
    return scope.finish(64,this.finalNorm.forward(scope.hidden));
  }
  #packed(layers:Layers,groups:readonly MlxPhaseGroup[]):MlxArray[] {
    using scope=new PackedMixedScope(groups,ids=>this.embed.encode(ids));
    scope.accept(0,(hidden,cache,phase)=>layers[0].forwardAttn(hidden,cache,phase),hidden=>layers[0].forwardMlp(hidden));
    scope.accept(1,(hidden,cache,phase)=>layers[1].forwardAttn(hidden,cache,phase),hidden=>layers[1].forwardMlp(hidden));
    scope.accept(2,(hidden,cache,phase)=>layers[2].forwardAttn(hidden,cache,phase),hidden=>layers[2].forwardMlp(hidden));
    scope.accept(3,(hidden,cache,phase)=>layers[3].forwardAttn(hidden,cache,phase),hidden=>layers[3].forwardMlp(hidden));
    scope.accept(4,(hidden,cache,phase)=>layers[4].forwardAttn(hidden,cache,phase),hidden=>layers[4].forwardMlp(hidden));
    scope.accept(5,(hidden,cache,phase)=>layers[5].forwardAttn(hidden,cache,phase),hidden=>layers[5].forwardMlp(hidden));
    scope.accept(6,(hidden,cache,phase)=>layers[6].forwardAttn(hidden,cache,phase),hidden=>layers[6].forwardMlp(hidden));
    scope.accept(7,(hidden,cache,phase)=>layers[7].forwardAttn(hidden,cache,phase),hidden=>layers[7].forwardMlp(hidden));
    scope.accept(8,(hidden,cache,phase)=>layers[8].forwardAttn(hidden,cache,phase),hidden=>layers[8].forwardMlp(hidden));
    scope.accept(9,(hidden,cache,phase)=>layers[9].forwardAttn(hidden,cache,phase),hidden=>layers[9].forwardMlp(hidden));
    scope.accept(10,(hidden,cache,phase)=>layers[10].forwardAttn(hidden,cache,phase),hidden=>layers[10].forwardMlp(hidden));
    scope.accept(11,(hidden,cache,phase)=>layers[11].forwardAttn(hidden,cache,phase),hidden=>layers[11].forwardMlp(hidden));
    scope.accept(12,(hidden,cache,phase)=>layers[12].forwardAttn(hidden,cache,phase),hidden=>layers[12].forwardMlp(hidden));
    scope.accept(13,(hidden,cache,phase)=>layers[13].forwardAttn(hidden,cache,phase),hidden=>layers[13].forwardMlp(hidden));
    scope.accept(14,(hidden,cache,phase)=>layers[14].forwardAttn(hidden,cache,phase),hidden=>layers[14].forwardMlp(hidden));
    scope.accept(15,(hidden,cache,phase)=>layers[15].forwardAttn(hidden,cache,phase),hidden=>layers[15].forwardMlp(hidden));
    scope.accept(16,(hidden,cache,phase)=>layers[16].forwardAttn(hidden,cache,phase),hidden=>layers[16].forwardMlp(hidden));
    scope.accept(17,(hidden,cache,phase)=>layers[17].forwardAttn(hidden,cache,phase),hidden=>layers[17].forwardMlp(hidden));
    scope.accept(18,(hidden,cache,phase)=>layers[18].forwardAttn(hidden,cache,phase),hidden=>layers[18].forwardMlp(hidden));
    scope.accept(19,(hidden,cache,phase)=>layers[19].forwardAttn(hidden,cache,phase),hidden=>layers[19].forwardMlp(hidden));
    scope.accept(20,(hidden,cache,phase)=>layers[20].forwardAttn(hidden,cache,phase),hidden=>layers[20].forwardMlp(hidden));
    scope.accept(21,(hidden,cache,phase)=>layers[21].forwardAttn(hidden,cache,phase),hidden=>layers[21].forwardMlp(hidden));
    scope.accept(22,(hidden,cache,phase)=>layers[22].forwardAttn(hidden,cache,phase),hidden=>layers[22].forwardMlp(hidden));
    scope.accept(23,(hidden,cache,phase)=>layers[23].forwardAttn(hidden,cache,phase),hidden=>layers[23].forwardMlp(hidden));
    scope.accept(24,(hidden,cache,phase)=>layers[24].forwardAttn(hidden,cache,phase),hidden=>layers[24].forwardMlp(hidden));
    scope.accept(25,(hidden,cache,phase)=>layers[25].forwardAttn(hidden,cache,phase),hidden=>layers[25].forwardMlp(hidden));
    scope.accept(26,(hidden,cache,phase)=>layers[26].forwardAttn(hidden,cache,phase),hidden=>layers[26].forwardMlp(hidden));
    scope.accept(27,(hidden,cache,phase)=>layers[27].forwardAttn(hidden,cache,phase),hidden=>layers[27].forwardMlp(hidden));
    scope.accept(28,(hidden,cache,phase)=>layers[28].forwardAttn(hidden,cache,phase),hidden=>layers[28].forwardMlp(hidden));
    scope.accept(29,(hidden,cache,phase)=>layers[29].forwardAttn(hidden,cache,phase),hidden=>layers[29].forwardMlp(hidden));
    scope.accept(30,(hidden,cache,phase)=>layers[30].forwardAttn(hidden,cache,phase),hidden=>layers[30].forwardMlp(hidden));
    scope.accept(31,(hidden,cache,phase)=>layers[31].forwardAttn(hidden,cache,phase),hidden=>layers[31].forwardMlp(hidden));
    scope.accept(32,(hidden,cache,phase)=>layers[32].forwardAttn(hidden,cache,phase),hidden=>layers[32].forwardMlp(hidden));
    scope.accept(33,(hidden,cache,phase)=>layers[33].forwardAttn(hidden,cache,phase),hidden=>layers[33].forwardMlp(hidden));
    scope.accept(34,(hidden,cache,phase)=>layers[34].forwardAttn(hidden,cache,phase),hidden=>layers[34].forwardMlp(hidden));
    scope.accept(35,(hidden,cache,phase)=>layers[35].forwardAttn(hidden,cache,phase),hidden=>layers[35].forwardMlp(hidden));
    scope.accept(36,(hidden,cache,phase)=>layers[36].forwardAttn(hidden,cache,phase),hidden=>layers[36].forwardMlp(hidden));
    scope.accept(37,(hidden,cache,phase)=>layers[37].forwardAttn(hidden,cache,phase),hidden=>layers[37].forwardMlp(hidden));
    scope.accept(38,(hidden,cache,phase)=>layers[38].forwardAttn(hidden,cache,phase),hidden=>layers[38].forwardMlp(hidden));
    scope.accept(39,(hidden,cache,phase)=>layers[39].forwardAttn(hidden,cache,phase),hidden=>layers[39].forwardMlp(hidden));
    scope.accept(40,(hidden,cache,phase)=>layers[40].forwardAttn(hidden,cache,phase),hidden=>layers[40].forwardMlp(hidden));
    scope.accept(41,(hidden,cache,phase)=>layers[41].forwardAttn(hidden,cache,phase),hidden=>layers[41].forwardMlp(hidden));
    scope.accept(42,(hidden,cache,phase)=>layers[42].forwardAttn(hidden,cache,phase),hidden=>layers[42].forwardMlp(hidden));
    scope.accept(43,(hidden,cache,phase)=>layers[43].forwardAttn(hidden,cache,phase),hidden=>layers[43].forwardMlp(hidden));
    scope.accept(44,(hidden,cache,phase)=>layers[44].forwardAttn(hidden,cache,phase),hidden=>layers[44].forwardMlp(hidden));
    scope.accept(45,(hidden,cache,phase)=>layers[45].forwardAttn(hidden,cache,phase),hidden=>layers[45].forwardMlp(hidden));
    scope.accept(46,(hidden,cache,phase)=>layers[46].forwardAttn(hidden,cache,phase),hidden=>layers[46].forwardMlp(hidden));
    scope.accept(47,(hidden,cache,phase)=>layers[47].forwardAttn(hidden,cache,phase),hidden=>layers[47].forwardMlp(hidden));
    scope.accept(48,(hidden,cache,phase)=>layers[48].forwardAttn(hidden,cache,phase),hidden=>layers[48].forwardMlp(hidden));
    scope.accept(49,(hidden,cache,phase)=>layers[49].forwardAttn(hidden,cache,phase),hidden=>layers[49].forwardMlp(hidden));
    scope.accept(50,(hidden,cache,phase)=>layers[50].forwardAttn(hidden,cache,phase),hidden=>layers[50].forwardMlp(hidden));
    scope.accept(51,(hidden,cache,phase)=>layers[51].forwardAttn(hidden,cache,phase),hidden=>layers[51].forwardMlp(hidden));
    scope.accept(52,(hidden,cache,phase)=>layers[52].forwardAttn(hidden,cache,phase),hidden=>layers[52].forwardMlp(hidden));
    scope.accept(53,(hidden,cache,phase)=>layers[53].forwardAttn(hidden,cache,phase),hidden=>layers[53].forwardMlp(hidden));
    scope.accept(54,(hidden,cache,phase)=>layers[54].forwardAttn(hidden,cache,phase),hidden=>layers[54].forwardMlp(hidden));
    scope.accept(55,(hidden,cache,phase)=>layers[55].forwardAttn(hidden,cache,phase),hidden=>layers[55].forwardMlp(hidden));
    scope.accept(56,(hidden,cache,phase)=>layers[56].forwardAttn(hidden,cache,phase),hidden=>layers[56].forwardMlp(hidden));
    scope.accept(57,(hidden,cache,phase)=>layers[57].forwardAttn(hidden,cache,phase),hidden=>layers[57].forwardMlp(hidden));
    scope.accept(58,(hidden,cache,phase)=>layers[58].forwardAttn(hidden,cache,phase),hidden=>layers[58].forwardMlp(hidden));
    scope.accept(59,(hidden,cache,phase)=>layers[59].forwardAttn(hidden,cache,phase),hidden=>layers[59].forwardMlp(hidden));
    scope.accept(60,(hidden,cache,phase)=>layers[60].forwardAttn(hidden,cache,phase),hidden=>layers[60].forwardMlp(hidden));
    scope.accept(61,(hidden,cache,phase)=>layers[61].forwardAttn(hidden,cache,phase),hidden=>layers[61].forwardMlp(hidden));
    scope.accept(62,(hidden,cache,phase)=>layers[62].forwardAttn(hidden,cache,phase),hidden=>layers[62].forwardMlp(hidden));
    scope.accept(63,(hidden,cache,phase)=>layers[63].forwardAttn(hidden,cache,phase),hidden=>layers[63].forwardMlp(hidden));
    return scope.finish(64,hidden=>this.finalNorm.forward(hidden));
  }
}
