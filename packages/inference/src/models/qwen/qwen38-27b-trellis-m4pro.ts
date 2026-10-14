import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { ModelConfig } from "../../artifacts/config";
import type { Weights } from "../../artifacts/weights";
import type { Cache } from "../../contracts/mlx/cache";
import type { ResidualBasis, TargetView } from "../../contracts/mlx/draft-target";
import type { MlxDeclaredGraph, MlxTokenPhases, MlxLayerCapture, MlxHeadForward, MlxRowTable, MlxPromptRange } from "../../contracts/mlx/graph";
import type { MediaEncoders, MlxPromptInput, Vision } from "../../contracts/mlx/media";
import type { MropeForwardState } from "../../contracts/mlx/positions";
import type { GraphCapabilities } from "../../contracts/portable/graph";
import type { GraphCacheLayer } from "../cache-composition";
import type { AneProgramSpec } from "../../kernels/ane/linear";
import { Affine3MmaLinear as Affine3Mma, Affine3RowsLinear as Affine3Rows, Affine4MmaLinear as Affine4Mma, Affine4RowsLinear as Affine4Rows } from "../../layers/affine-verify-linear";
import { AneAffineSplit, AneTrellisMlpSplit as AneMlp, AneTrellisMlpSplitK3i as AneMlpK3i, aneAffineSplitProgram, aneMlpSplitProgram } from "../../layers/ane-prefill-split";
import { LoraState } from "../../layers/lora";
import { RMSNorm } from "../../layers/normalization";
import { QuantizedEmbedding } from "../../layers/quantized-embedding";
import { QuantizedLinear, QuantizedLinearPlain as Plain } from "../../layers/quantized-linear";
import { qwenAppendChunkSize } from "../../layers/qwen-append";
import { mropeInvFreq } from "../../layers/qwen-mrope";
import { FactoredDownK3iRow as DownK3iRow, FactoredDownK3iRows as DownK3iRows, FactoredDownRow as DownRow, FactoredDownRows as DownRows, MmaDown, MmaDownK3i } from "../../layers/trellis-down";
import { FactoredGateUpRow as GateRow, FactoredGateUpRows as GateRows, FactoredMixedGateUp as MixedGate, MmaGateUp as MmaGate } from "../../layers/trellis-gate-up";
import { TrellisUnfusedMLP, TrellisWidePrompt as Wide, TrellisTiledPrompt as Tiled, TrellisExpandReducePrompt as ExpandGate, TrellisExpandScatterPrompt as ExpandDown } from "../../layers/trellis-prompt";
import { GraphForwardScope, BoundedGraphForwardScope, SubmittedGraphForwardScope } from "../../runtime/graph-forward";
import { disposeResources } from "../../runtime/resources";
import { declareGraph } from "../capabilities";
import type { ReadPhase } from "../graph";
import { QwenDeltaLayer as Delta, QwenAttentionLayer as Attention } from "./blocks";
import { bindQwenMediaInput, qwenDraftTarget, qwenMediaEncoders, qwenResidualBasis } from "./media-input";
import { DeltaWeights, AttentionWeights, GateDownMLP, AneBoundMlp, AneWeightShares } from "./written-layers";

export const QWEN38_TRELLIS_M4PRO_GRAPH = "qwen3.8-27b-trellis-3.2bpw-m4pro";
export const DECODE_ROWS = 8;
/** Eight request rows, each verifying a qualified depth-seven round. */
export const VERIFY_ROWS = 64;
const PREFIX = "language_model.model";
const REPRESENTATIONS = { gateUpCodebook: false, mixedExactBits: false,
  scatter: { bits: false, genericBits: false, floatCodebook: false }, widePrefill: true };
const ANE_FRACTION = 0.55;
export interface AneWidths { readonly hidden:number; readonly intermediate:number; readonly projections:readonly [readonly [number,number],readonly [number,number]] }
export const ANE_WIDTHS: AneWidths = {hidden:5120,intermediate:17408,projections:[[5120,6144],[6144,5120]]};
/** The install step fills these programs' cache; this file only loads them. */
export function anePrograms(widths:AneWidths=ANE_WIDTHS):AneProgramSpec[] {
  const [[zi,zo],[oi,oo]]=widths.projections;
  return [
    aneMlpSplitProgram(widths.hidden,widths.intermediate,ANE_FRACTION,1024),aneAffineSplitProgram(zi,zo,ANE_FRACTION,1024),aneAffineSplitProgram(oi,oo,ANE_FRACTION,1024),
    aneMlpSplitProgram(widths.hidden,widths.intermediate,ANE_FRACTION,1536),aneAffineSplitProgram(zi,zo,ANE_FRACTION,1536),aneAffineSplitProgram(oi,oo,ANE_FRACTION,1536),
    aneMlpSplitProgram(widths.hidden,widths.intermediate,ANE_FRACTION,2048),aneAffineSplitProgram(zi,zo,ANE_FRACTION,2048),aneAffineSplitProgram(oi,oo,ANE_FRACTION,2048),
  ];
}
type Layers=readonly [
  Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,
  Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,
  Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,
  Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,Delta,Delta,Delta,Attention,
];
export type RowsForward=(ids:MlxArray,state:Cache[],capture:MlxLayerCapture|null)=>MlxArray;

/** A written arrangement qualified on M4 Pro. The loader recommends it;
 * explicit selection remains the caller's choice on any machine. */
export class Qwen38TrellisM4Pro implements MlxDeclaredGraph,MlxTokenPhases {
  readonly graph=QWEN38_TRELLIS_M4PRO_GRAPH;
  readonly prefixBase=PREFIX;
  readonly loraState=new LoraState();
  readonly faIdx=3;
  readonly weightsBytes:number;
  readonly embed:QuantizedEmbedding;
  readonly finalNorm:RMSNorm;
  readonly lmHead:QuantizedLinear;
  readonly graphCapabilities:GraphCapabilities=declareGraph({media:{input:"embeddings+positions",video:true},hiddenLayerTaps:true,prefill:{boundedWorkspace:true}});
  readonly verifyRoundCosts=new Map([[1,1],[2,1.3],[3,1.46],[4,1.68],[5,1.82],[6,1.92],[7,1.93],[8,1.94]]);
  readonly knownSpans={maxChunkSize:(state:readonly Cache[])=>qwenAppendChunkSize(state[0]!.offset)};
  readonly cacheLayers:readonly GraphCacheLayer[]=[
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
    {kind:"recurrent"},{kind:"recurrent"},{kind:"recurrent"},{kind:"full",turboQuant:"dense",kv4Head256:true},
  ];
  readonly prompt:readonly MlxPromptRange[];
  readonly multiSequencePrompt:readonly MlxPromptRange[];
  readonly decode:MlxRowTable<RowsForward>;
  readonly verify:MlxRowTable<RowsForward>;
  readonly appendKnown:MlxRowTable<RowsForward>;
  readonly logits:MlxRowTable<MlxHeadForward>;
  readonly #targets:Map<string,QuantizedLinear>;
  readonly #owned:{dispose():void}[];
  readonly #residualBasis:ResidualBasis|undefined;
  readonly #mrope:{invFreq:MlxArray;rotaryDims:number};
  readonly #one:Layers;
  readonly #rows:Layers;
  readonly #four:Layers;
  readonly #mma:Layers;
  readonly #wide:Layers;
  readonly #tiled:Layers;
  readonly #expanded:Layers;
  readonly #ane1024:Layers;
  readonly #ane1536:Layers;
  readonly #ane2048:Layers;

  constructor(weights:Weights,readonly config:ModelConfig) {
    const w=weights,c=config,t=config.text;
    this.weightsBytes=[...weights.shards.files.values()].reduce((bytes,file)=>bytes+file.mmap.size,0);
    this.#residualBasis=qwenResidualBasis(this.graph,config.modelDir);
    this.embed=QuantizedEmbedding.load(w,`${PREFIX}.embed_tokens`,c);
    this.finalNorm=new RMSNorm(w.tensor(`${PREFIX}.norm.weight`),t.rmsNormEps);
    this.lmHead=QuantizedLinear.load(w,"language_model.lm_head",c);
    const invScale=Math.pow(t.linearKeyHeadDim,-0.5),qk={q:ops.filledBf16(invScale*invScale,t.linearKeyHeadDim),k:ops.filledBf16(invScale,t.linearKeyHeadDim)};
    const rotaryDims=Math.trunc(t.headDim*t.partialRotaryFactor);
    this.#mrope={invFreq:mropeInvFreq(rotaryDims,t.ropeParameters.full_attention!.ropeTheta),rotaryDims};
    const shares=new AneWeightShares();
    this.#owned=[qk.q,qk.k,this.#mrope.invFreq,{dispose:()=>shares[Symbol.dispose]()}];
    const d0=new DeltaWeights(w,c,`${PREFIX}.layers.0`,REPRESENTATIONS);
    const d1=new DeltaWeights(w,c,`${PREFIX}.layers.1`,REPRESENTATIONS);
    const d2=new DeltaWeights(w,c,`${PREFIX}.layers.2`,REPRESENTATIONS);
    const a3=new AttentionWeights(w,c,`${PREFIX}.layers.3`,REPRESENTATIONS);
    const d4=new DeltaWeights(w,c,`${PREFIX}.layers.4`,REPRESENTATIONS);
    const d5=new DeltaWeights(w,c,`${PREFIX}.layers.5`,REPRESENTATIONS);
    const d6=new DeltaWeights(w,c,`${PREFIX}.layers.6`,REPRESENTATIONS);
    const a7=new AttentionWeights(w,c,`${PREFIX}.layers.7`,REPRESENTATIONS);
    const d8=new DeltaWeights(w,c,`${PREFIX}.layers.8`,REPRESENTATIONS);
    const d9=new DeltaWeights(w,c,`${PREFIX}.layers.9`,REPRESENTATIONS);
    const d10=new DeltaWeights(w,c,`${PREFIX}.layers.10`,REPRESENTATIONS);
    const a11=new AttentionWeights(w,c,`${PREFIX}.layers.11`,REPRESENTATIONS);
    const d12=new DeltaWeights(w,c,`${PREFIX}.layers.12`,REPRESENTATIONS);
    const d13=new DeltaWeights(w,c,`${PREFIX}.layers.13`,REPRESENTATIONS);
    const d14=new DeltaWeights(w,c,`${PREFIX}.layers.14`,REPRESENTATIONS);
    const a15=new AttentionWeights(w,c,`${PREFIX}.layers.15`,REPRESENTATIONS);
    const d16=new DeltaWeights(w,c,`${PREFIX}.layers.16`,REPRESENTATIONS);
    const d17=new DeltaWeights(w,c,`${PREFIX}.layers.17`,REPRESENTATIONS);
    const d18=new DeltaWeights(w,c,`${PREFIX}.layers.18`,REPRESENTATIONS);
    const a19=new AttentionWeights(w,c,`${PREFIX}.layers.19`,REPRESENTATIONS);
    const d20=new DeltaWeights(w,c,`${PREFIX}.layers.20`,REPRESENTATIONS);
    const d21=new DeltaWeights(w,c,`${PREFIX}.layers.21`,REPRESENTATIONS);
    const d22=new DeltaWeights(w,c,`${PREFIX}.layers.22`,REPRESENTATIONS);
    const a23=new AttentionWeights(w,c,`${PREFIX}.layers.23`,REPRESENTATIONS);
    const d24=new DeltaWeights(w,c,`${PREFIX}.layers.24`,REPRESENTATIONS);
    const d25=new DeltaWeights(w,c,`${PREFIX}.layers.25`,REPRESENTATIONS);
    const d26=new DeltaWeights(w,c,`${PREFIX}.layers.26`,REPRESENTATIONS);
    const a27=new AttentionWeights(w,c,`${PREFIX}.layers.27`,REPRESENTATIONS);
    const d28=new DeltaWeights(w,c,`${PREFIX}.layers.28`,REPRESENTATIONS);
    const d29=new DeltaWeights(w,c,`${PREFIX}.layers.29`,REPRESENTATIONS);
    const d30=new DeltaWeights(w,c,`${PREFIX}.layers.30`,REPRESENTATIONS);
    const a31=new AttentionWeights(w,c,`${PREFIX}.layers.31`,REPRESENTATIONS);
    const d32=new DeltaWeights(w,c,`${PREFIX}.layers.32`,REPRESENTATIONS);
    const d33=new DeltaWeights(w,c,`${PREFIX}.layers.33`,REPRESENTATIONS);
    const d34=new DeltaWeights(w,c,`${PREFIX}.layers.34`,REPRESENTATIONS);
    const a35=new AttentionWeights(w,c,`${PREFIX}.layers.35`,REPRESENTATIONS);
    const d36=new DeltaWeights(w,c,`${PREFIX}.layers.36`,REPRESENTATIONS);
    const d37=new DeltaWeights(w,c,`${PREFIX}.layers.37`,REPRESENTATIONS);
    const d38=new DeltaWeights(w,c,`${PREFIX}.layers.38`,REPRESENTATIONS);
    const a39=new AttentionWeights(w,c,`${PREFIX}.layers.39`,REPRESENTATIONS);
    const d40=new DeltaWeights(w,c,`${PREFIX}.layers.40`,REPRESENTATIONS);
    const d41=new DeltaWeights(w,c,`${PREFIX}.layers.41`,REPRESENTATIONS);
    const d42=new DeltaWeights(w,c,`${PREFIX}.layers.42`,REPRESENTATIONS);
    const a43=new AttentionWeights(w,c,`${PREFIX}.layers.43`,REPRESENTATIONS);
    const d44=new DeltaWeights(w,c,`${PREFIX}.layers.44`,REPRESENTATIONS);
    const d45=new DeltaWeights(w,c,`${PREFIX}.layers.45`,REPRESENTATIONS);
    const d46=new DeltaWeights(w,c,`${PREFIX}.layers.46`,REPRESENTATIONS);
    const a47=new AttentionWeights(w,c,`${PREFIX}.layers.47`,REPRESENTATIONS);
    const d48=new DeltaWeights(w,c,`${PREFIX}.layers.48`,REPRESENTATIONS);
    const d49=new DeltaWeights(w,c,`${PREFIX}.layers.49`,REPRESENTATIONS);
    const d50=new DeltaWeights(w,c,`${PREFIX}.layers.50`,REPRESENTATIONS);
    const a51=new AttentionWeights(w,c,`${PREFIX}.layers.51`,REPRESENTATIONS);
    const d52=new DeltaWeights(w,c,`${PREFIX}.layers.52`,REPRESENTATIONS);
    const d53=new DeltaWeights(w,c,`${PREFIX}.layers.53`,REPRESENTATIONS);
    const d54=new DeltaWeights(w,c,`${PREFIX}.layers.54`,REPRESENTATIONS);
    const a55=new AttentionWeights(w,c,`${PREFIX}.layers.55`,REPRESENTATIONS);
    const d56=new DeltaWeights(w,c,`${PREFIX}.layers.56`,REPRESENTATIONS);
    const d57=new DeltaWeights(w,c,`${PREFIX}.layers.57`,REPRESENTATIONS);
    const d58=new DeltaWeights(w,c,`${PREFIX}.layers.58`,REPRESENTATIONS);
    const a59=new AttentionWeights(w,c,`${PREFIX}.layers.59`,REPRESENTATIONS);
    const d60=new DeltaWeights(w,c,`${PREFIX}.layers.60`,REPRESENTATIONS);
    const d61=new DeltaWeights(w,c,`${PREFIX}.layers.61`,REPRESENTATIONS);
    const d62=new DeltaWeights(w,c,`${PREFIX}.layers.62`,REPRESENTATIONS);
    const a63=new AttentionWeights(w,c,`${PREFIX}.layers.63`,REPRESENTATIONS);
    // One row: factored gate/up, with the literal mixed-width and down layout.
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
      d32.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
      d33.stage(w,c,Delta,GateDownMLP,GateRow,DownK3iRow,Plain,Plain,Plain,Plain,Plain,qk),
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
    // Two and three rows share factored gate/up and down, with affine row reads.
    this.#rows=[
      d0.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d1.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d2.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      a3.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d4.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d5.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d6.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a7.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d8.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d9.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d10.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a11.stage(w,c,Attention,GateDownMLP,GateRows,DownRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d12.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d13.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d14.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a15.stage(w,c,Attention,GateDownMLP,MixedGate,DownRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d16.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d17.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d18.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a19.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d20.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d21.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d22.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a23.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d24.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d25.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d26.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a27.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d28.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d29.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d30.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a31.stage(w,c,Attention,GateDownMLP,GateRows,DownRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d32.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d33.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d34.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a35.stage(w,c,Attention,GateDownMLP,MixedGate,DownK3iRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d36.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d37.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d38.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a39.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d40.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d41.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d42.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a43.stage(w,c,Attention,GateDownMLP,MixedGate,DownK3iRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d44.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d45.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d46.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a47.stage(w,c,Attention,GateDownMLP,GateRows,DownRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d48.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d49.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d50.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a51.stage(w,c,Attention,GateDownMLP,MixedGate,DownK3iRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d52.stage(w,c,Delta,GateDownMLP,GateRows,DownK3iRows,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d53.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRows,Affine3Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d54.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      a55.stage(w,c,Attention,GateDownMLP,GateRows,DownRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d56.stage(w,c,Delta,GateDownMLP,MixedGate,DownRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d57.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d58.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine4Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a59.stage(w,c,Attention,GateDownMLP,GateRows,DownRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d60.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d61.stage(w,c,Delta,GateDownMLP,GateRows,DownRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d62.stage(w,c,Delta,GateDownMLP,MixedGate,DownK3iRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a63.stage(w,c,Attention,GateDownMLP,GateRows,DownK3iRows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
    ];
    // Four rows: the row projections and gate/up, and the matrix down kernel.
    this.#four=[
      d0.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d1.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d2.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      a3.stage(w,c,Attention,GateDownMLP,GateRows,MmaDownK3i,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d4.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d5.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d6.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a7.stage(w,c,Attention,GateDownMLP,GateRows,MmaDownK3i,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d8.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d9.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d10.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a11.stage(w,c,Attention,GateDownMLP,GateRows,MmaDown,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d12.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d13.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d14.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a15.stage(w,c,Attention,GateDownMLP,MixedGate,MmaDown,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d16.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d17.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d18.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a19.stage(w,c,Attention,GateDownMLP,GateRows,MmaDownK3i,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d20.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d21.stage(w,c,Delta,GateDownMLP,MixedGate,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d22.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a23.stage(w,c,Attention,GateDownMLP,GateRows,MmaDownK3i,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d24.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d25.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d26.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a27.stage(w,c,Attention,GateDownMLP,GateRows,MmaDownK3i,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d28.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d29.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d30.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a31.stage(w,c,Attention,GateDownMLP,GateRows,MmaDown,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d32.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d33.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d34.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a35.stage(w,c,Attention,GateDownMLP,MixedGate,MmaDownK3i,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d36.stage(w,c,Delta,GateDownMLP,MixedGate,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d37.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d38.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a39.stage(w,c,Attention,GateDownMLP,GateRows,MmaDownK3i,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d40.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d41.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d42.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a43.stage(w,c,Attention,GateDownMLP,MixedGate,MmaDownK3i,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d44.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d45.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d46.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a47.stage(w,c,Attention,GateDownMLP,GateRows,MmaDown,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d48.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d49.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d50.stage(w,c,Delta,GateDownMLP,MixedGate,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a51.stage(w,c,Attention,GateDownMLP,MixedGate,MmaDownK3i,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d52.stage(w,c,Delta,GateDownMLP,GateRows,MmaDownK3i,Affine3Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d53.stage(w,c,Delta,GateDownMLP,MixedGate,MmaDownK3i,Affine3Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d54.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      a55.stage(w,c,Attention,GateDownMLP,GateRows,MmaDown,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d56.stage(w,c,Delta,GateDownMLP,MixedGate,MmaDown,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d57.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d58.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine4Rows,Affine3Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a59.stage(w,c,Attention,GateDownMLP,GateRows,MmaDown,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
      d60.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      d61.stage(w,c,Delta,GateDownMLP,GateRows,MmaDown,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,qk),
      d62.stage(w,c,Delta,GateDownMLP,MixedGate,MmaDownK3i,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows,Affine3Rows,qk),
      a63.stage(w,c,Attention,GateDownMLP,GateRows,MmaDownK3i,Affine4Rows,Affine4Rows,Affine4Rows,Affine4Rows),
    ];
    // Five through eight rows: matrix affine projections and Trellis gate/up/down.
    this.#mma=[
      d0.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,qk),
      d1.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine4Mma,qk),
      d2.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine4Mma,qk),
      a3.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDownK3i,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d4.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,qk),
      d5.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,qk),
      d6.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a7.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDownK3i,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d8.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d9.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d10.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a11.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d12.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d13.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d14.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a15.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d16.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d17.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d18.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a19.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDownK3i,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d20.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d21.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d22.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a23.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDownK3i,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d24.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d25.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine4Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d26.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a27.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDownK3i,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d28.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d29.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d30.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a31.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d32.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d33.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d34.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a35.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDownK3i,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d36.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d37.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d38.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a39.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDownK3i,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d40.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d41.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d42.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a43.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDownK3i,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d44.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d45.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d46.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a47.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d48.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d49.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d50.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a51.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDownK3i,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d52.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d53.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine3Mma,Affine4Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d54.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,qk),
      a55.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d56.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,qk),
      d57.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,qk),
      d58.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine3Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a59.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
      d60.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      d61.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDown,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,qk),
      d62.stage(w,c,Delta,GateDownMLP,MmaGate,MmaDownK3i,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma,Affine3Mma,qk),
      a63.stage(w,c,Attention,GateDownMLP,MmaGate,MmaDownK3i,Affine4Mma,Affine4Mma,Affine4Mma,Affine4Mma),
    ];
    // Native GemvWide gate/up, followed by the evaluated expanded down.
    this.#wide=[
      d0.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d1.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d2.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a3.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d4.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d5.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d6.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a7.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d8.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d9.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d10.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a11.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d12.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d13.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d14.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a15.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d16.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d17.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d18.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a19.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d20.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d21.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d22.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a23.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d24.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d25.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d26.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a27.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d28.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d29.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d30.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a31.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d32.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d33.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d34.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a35.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d36.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d37.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d38.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a39.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d40.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d41.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d42.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a43.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d44.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d45.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d46.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a47.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d48.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d49.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d50.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a51.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d52.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d53.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d54.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a55.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d56.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d57.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d58.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a59.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
      d60.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d61.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),d62.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,ExpandDown,Plain,qk),a63.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,ExpandDown,Plain),
    ];
    this.#tiled=[
      d0.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d1.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d2.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a3.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d4.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d5.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d6.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a7.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d8.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d9.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d10.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a11.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d12.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d13.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d14.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a15.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d16.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d17.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d18.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a19.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d20.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d21.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d22.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a23.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d24.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d25.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d26.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a27.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d28.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d29.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d30.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a31.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d32.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d33.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d34.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a35.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d36.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d37.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d38.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a39.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d40.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d41.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d42.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a43.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d44.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d45.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d46.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a47.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d48.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d49.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d50.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a51.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d52.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d53.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d54.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a55.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d56.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d57.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d58.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a59.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
      d60.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d61.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),d62.prompt(w,c,Delta,TrellisUnfusedMLP,Tiled,ExpandDown,Plain,qk),a63.prompt(w,c,Attention,TrellisUnfusedMLP,Tiled,ExpandDown,Plain),
    ];
    this.#expanded=[
      d0.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d1.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d2.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a3.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d4.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d5.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d6.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a7.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d8.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d9.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d10.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a11.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d12.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d13.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d14.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a15.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d16.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d17.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d18.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a19.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d20.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d21.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d22.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a23.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d24.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d25.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d26.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a27.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d28.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d29.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d30.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a31.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d32.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d33.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d34.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a35.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d36.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d37.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d38.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a39.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d40.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d41.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d42.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a43.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d44.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d45.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d46.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a47.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d48.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d49.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d50.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a51.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d52.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d53.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d54.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a55.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d56.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d57.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d58.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a59.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
      d60.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d61.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),d62.prompt(w,c,Delta,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain,qk),a63.prompt(w,c,Attention,TrellisUnfusedMLP,ExpandGate,ExpandDown,Plain),
    ];
    // Largest programs first: later ranges reuse their buffers. Each layer's
    // MLP and each eligible affine projection names its ANE split explicitly.
    this.#ane2048=[
      d0.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d1.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d2.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a3.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d4.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d5.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d6.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a7.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d8.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d9.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d10.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a11.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d12.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d13.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d14.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a15.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d16.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d17.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d18.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a19.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d20.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d21.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d22.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a23.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d24.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d25.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d26.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a27.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d28.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d29.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d30.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a31.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d32.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d33.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d34.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a35.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d36.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d37.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d38.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a39.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d40.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d41.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d42.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a43.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d44.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d45.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d46.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a47.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d48.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d49.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d50.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a51.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d52.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d53.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d54.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a55.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d56.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d57.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d58.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a59.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
      d60.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d61.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),d62.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares,qk),a63.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,2048,shares),
    ];
    this.#ane1536=[
      d0.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d1.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d2.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a3.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d4.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d5.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d6.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a7.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d8.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d9.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d10.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a11.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d12.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d13.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d14.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a15.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d16.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d17.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d18.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a19.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d20.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d21.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d22.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a23.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d24.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d25.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d26.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a27.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d28.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d29.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d30.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a31.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d32.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d33.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d34.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a35.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d36.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d37.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d38.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a39.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d40.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d41.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d42.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a43.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d44.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d45.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d46.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a47.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d48.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d49.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d50.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a51.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d52.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d53.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d54.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a55.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d56.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d57.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d58.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a59.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
      d60.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d61.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),d62.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares,qk),a63.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1536,shares),
    ];
    this.#ane1024=[
      d0.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d1.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d2.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a3.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d4.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d5.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d6.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a7.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d8.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d9.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d10.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a11.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d12.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d13.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d14.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a15.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d16.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d17.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d18.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a19.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d20.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d21.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d22.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a23.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d24.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d25.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d26.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a27.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d28.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d29.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d30.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a31.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d32.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d33.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d34.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a35.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d36.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d37.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d38.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a39.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d40.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d41.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d42.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a43.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d44.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d45.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d46.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a47.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d48.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d49.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d50.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a51.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d52.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d53.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d54.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a55.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d56.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d57.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d58.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a59.aneStage(w,c,Attention,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
      d60.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d61.aneStage(w,c,Delta,AneBoundMlp,AneMlp,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),d62.aneStage(w,c,Delta,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares,qk),a63.aneStage(w,c,Attention,AneBoundMlp,AneMlpK3i,AneAffineSplit,Plain,ANE_FRACTION,1024,shares),
    ];
    this.#targets=new Map([
      ...d0.targets(),...d1.targets(),...d2.targets(),...a3.targets(),...d4.targets(),...d5.targets(),...d6.targets(),...a7.targets(),
      ...d8.targets(),...d9.targets(),...d10.targets(),...a11.targets(),...d12.targets(),...d13.targets(),...d14.targets(),...a15.targets(),
      ...d16.targets(),...d17.targets(),...d18.targets(),...a19.targets(),...d20.targets(),...d21.targets(),...d22.targets(),...a23.targets(),
      ...d24.targets(),...d25.targets(),...d26.targets(),...a27.targets(),...d28.targets(),...d29.targets(),...d30.targets(),...a31.targets(),
      ...d32.targets(),...d33.targets(),...d34.targets(),...a35.targets(),...d36.targets(),...d37.targets(),...d38.targets(),...a39.targets(),
      ...d40.targets(),...d41.targets(),...d42.targets(),...a43.targets(),...d44.targets(),...d45.targets(),...d46.targets(),...a47.targets(),
      ...d48.targets(),...d49.targets(),...d50.targets(),...a51.targets(),...d52.targets(),...d53.targets(),...d54.targets(),...a55.targets(),
      ...d56.targets(),...d57.targets(),...d58.targets(),...a59.targets(),...d60.targets(),...d61.targets(),...d62.targets(),...a63.targets(),
    ]);
  }
}
