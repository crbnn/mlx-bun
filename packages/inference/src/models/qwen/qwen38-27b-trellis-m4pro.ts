import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { ModelConfig } from "../../artifacts/config";
import type { Weights } from "../../artifacts/weights";
import type { Cache } from "../../contracts/mlx/cache";
import type { ResidualBasis, TargetView } from "../../contracts/mlx/draft-target";
import type { MlxDeclaredGraph, MlxTokenPhases, MlxLayerCapture, MlxHeadForward, MlxRowTable, MlxPromptRange, MlxCacheLayer, MlxPhaseForward } from "../../contracts/mlx/graph";
import type { MediaEncoders, MlxPromptInput, Vision } from "../../contracts/mlx/media";
import type { MropeForwardState } from "../../contracts/mlx/positions";
import type { GraphCapabilities } from "../../contracts/portable/graph";
import type { AneProgramSpec } from "../../kernels/ane/linear";
import { Affine3MmaLinear as Affine3Mma, Affine3RowsLinear as Affine3Rows, Affine4MmaLinear as Affine4Mma, Affine4RowsLinear as Affine4Rows } from "../../layers/affine-verify-linear";
import { AneAffineSplit, AneTrellisMlpSplit as AneMlp, AneTrellisMlpSplitK3i as AneMlpK3i, aneAffineSplitProgram, aneMlpSplitProgram } from "../../layers/ane-prefill-split";
import { LoraState } from "../../layers/lora";
import { RMSNorm } from "../../layers/normalization";
import { QuantizedEmbedding } from "../../layers/quantized-embedding";
import { QuantizedLinear, QuantizedLinearPlain as Plain, QuantizedLinearRows as SeparateRows } from "../../layers/quantized-linear";
import { qwenAppendChunkSize } from "../../layers/qwen-append";
import { mropeInvFreq } from "../../layers/qwen-mrope";
import { FactoredDownK3iRow as DownK3iRow, FactoredDownK3iRows as DownK3iRows, FactoredDownRow as DownRow, FactoredDownRows as DownRows, MmaDown, MmaDownK3i } from "../../layers/trellis-down";
import { FactoredGateUpRow as GateRow, FactoredGateUpRows as GateRows, FactoredMixedGateUp as MixedGate, MmaGateUp as MmaGate } from "../../layers/trellis-gate-up";
import { TrellisFusedMLP, TrellisUnfusedMLP, TrellisReducePrompt as Reduce, TrellisScatterPrompt as Scatter, TrellisSplitKPrompt as SplitK, TrellisWidePrompt as Wide, TrellisTiledPrompt as Tiled, TrellisExpandReducePrompt as ExpandGate, TrellisExpandScatterPrompt as ExpandDown } from "../../layers/trellis-prompt";
import { GraphForwardScope, BoundedGraphForwardScope, SubmittedGraphForwardScope } from "../../runtime/graph-forward";
import { disposeResources } from "../../runtime/resources";
import { declareGraph } from "../capabilities";
import type { ReadPhase } from "../graph";
import { QwenDeltaLayer as Delta, QwenAttentionLayer as Attention } from "./blocks";
import { bindQwenMediaInput, qwenDraftTarget, qwenMediaEncoders, qwenResidualBasis } from "./media-input";
import { DeltaWeights, AttentionWeights, GateDownMLP, AneBoundMlp, AneWeightShares, canonicalQwenWeights } from "./written-layers";

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
  readonly cacheLayers:readonly MlxCacheLayer[]=[
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
  readonly appendKnown:MlxRowTable<MlxPhaseForward>;
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
  readonly #stockNarrow:Layers;
  readonly #stockFiveToEight:Layers;
  readonly #knownNarrow:Layers;
  readonly #knownFiveToEight:Layers;
  readonly #knownWide:Layers;
  readonly #knownTiled:Layers;
  readonly #knownExpanded:Layers;

  constructor(weights:Weights,readonly config:ModelConfig) {
    const w=canonicalQwenWeights(weights),c=config,t=config.text;
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
    // GPU chunks, media and known appends retain the stock Trellis arithmetic.
    this.#stockNarrow=[
      d0.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d1.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d2.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a3.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d4.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d5.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d6.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a7.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d8.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d9.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d10.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a11.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d12.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d13.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d14.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a15.prompt(w,c,Attention,TrellisUnfusedMLP,Reduce,Scatter,Plain),
      d16.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d17.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d18.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a19.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d20.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d21.prompt(w,c,Delta,TrellisUnfusedMLP,Reduce,Scatter,Plain,qk),d22.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a23.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d24.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d25.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d26.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a27.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d28.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d29.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d30.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a31.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d32.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d33.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d34.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a35.prompt(w,c,Attention,TrellisUnfusedMLP,Reduce,Scatter,Plain),
      d36.prompt(w,c,Delta,TrellisUnfusedMLP,Reduce,Scatter,Plain,qk),d37.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d38.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a39.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d40.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d41.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d42.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a43.prompt(w,c,Attention,TrellisUnfusedMLP,Reduce,Scatter,Plain),
      d44.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d45.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d46.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a47.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d48.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d49.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d50.prompt(w,c,Delta,TrellisUnfusedMLP,Reduce,Scatter,Plain,qk),a51.prompt(w,c,Attention,TrellisUnfusedMLP,Reduce,Scatter,Plain),
      d52.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d53.prompt(w,c,Delta,TrellisUnfusedMLP,Reduce,Scatter,Plain,qk),d54.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a55.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d56.prompt(w,c,Delta,TrellisUnfusedMLP,Reduce,Scatter,Plain,qk),d57.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d58.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),a59.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
      d60.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d61.fused(w,c,Delta,TrellisFusedMLP,Scatter,Plain,qk),d62.prompt(w,c,Delta,TrellisUnfusedMLP,Reduce,Scatter,Plain,qk),a63.fused(w,c,Attention,TrellisFusedMLP,Scatter,Plain),
    ];
    this.#stockFiveToEight=[
      d0.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d1.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d2.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a3.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d4.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d5.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d6.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a7.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d8.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d9.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d10.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a11.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d12.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d13.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d14.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a15.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d16.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d17.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d18.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a19.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d20.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d21.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d22.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a23.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d24.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d25.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d26.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a27.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d28.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d29.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d30.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a31.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d32.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d33.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d34.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a35.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d36.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d37.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d38.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a39.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d40.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d41.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d42.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a43.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d44.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d45.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d46.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a47.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d48.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d49.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d50.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a51.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d52.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d53.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d54.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a55.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d56.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d57.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d58.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a59.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
      d60.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d61.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),d62.prompt(w,c,Delta,TrellisUnfusedMLP,Wide,SplitK,Plain,qk),a63.prompt(w,c,Attention,TrellisUnfusedMLP,Wide,SplitK,Plain),
    ];
    this.#knownNarrow=[
      d0.withMlp(w,c,Delta,this.#stockNarrow[0].mlp,SeparateRows,qk),d1.withMlp(w,c,Delta,this.#stockNarrow[1].mlp,SeparateRows,qk),d2.withMlp(w,c,Delta,this.#stockNarrow[2].mlp,SeparateRows,qk),a3.withMlp(w,c,Attention,this.#stockNarrow[3].mlp,SeparateRows),
      d4.withMlp(w,c,Delta,this.#stockNarrow[4].mlp,SeparateRows,qk),d5.withMlp(w,c,Delta,this.#stockNarrow[5].mlp,SeparateRows,qk),d6.withMlp(w,c,Delta,this.#stockNarrow[6].mlp,SeparateRows,qk),a7.withMlp(w,c,Attention,this.#stockNarrow[7].mlp,SeparateRows),
      d8.withMlp(w,c,Delta,this.#stockNarrow[8].mlp,SeparateRows,qk),d9.withMlp(w,c,Delta,this.#stockNarrow[9].mlp,SeparateRows,qk),d10.withMlp(w,c,Delta,this.#stockNarrow[10].mlp,SeparateRows,qk),a11.withMlp(w,c,Attention,this.#stockNarrow[11].mlp,SeparateRows),
      d12.withMlp(w,c,Delta,this.#stockNarrow[12].mlp,SeparateRows,qk),d13.withMlp(w,c,Delta,this.#stockNarrow[13].mlp,SeparateRows,qk),d14.withMlp(w,c,Delta,this.#stockNarrow[14].mlp,SeparateRows,qk),a15.withMlp(w,c,Attention,this.#stockNarrow[15].mlp,SeparateRows),
      d16.withMlp(w,c,Delta,this.#stockNarrow[16].mlp,SeparateRows,qk),d17.withMlp(w,c,Delta,this.#stockNarrow[17].mlp,SeparateRows,qk),d18.withMlp(w,c,Delta,this.#stockNarrow[18].mlp,SeparateRows,qk),a19.withMlp(w,c,Attention,this.#stockNarrow[19].mlp,SeparateRows),
      d20.withMlp(w,c,Delta,this.#stockNarrow[20].mlp,SeparateRows,qk),d21.withMlp(w,c,Delta,this.#stockNarrow[21].mlp,SeparateRows,qk),d22.withMlp(w,c,Delta,this.#stockNarrow[22].mlp,SeparateRows,qk),a23.withMlp(w,c,Attention,this.#stockNarrow[23].mlp,SeparateRows),
      d24.withMlp(w,c,Delta,this.#stockNarrow[24].mlp,SeparateRows,qk),d25.withMlp(w,c,Delta,this.#stockNarrow[25].mlp,SeparateRows,qk),d26.withMlp(w,c,Delta,this.#stockNarrow[26].mlp,SeparateRows,qk),a27.withMlp(w,c,Attention,this.#stockNarrow[27].mlp,SeparateRows),
      d28.withMlp(w,c,Delta,this.#stockNarrow[28].mlp,SeparateRows,qk),d29.withMlp(w,c,Delta,this.#stockNarrow[29].mlp,SeparateRows,qk),d30.withMlp(w,c,Delta,this.#stockNarrow[30].mlp,SeparateRows,qk),a31.withMlp(w,c,Attention,this.#stockNarrow[31].mlp,SeparateRows),
      d32.withMlp(w,c,Delta,this.#stockNarrow[32].mlp,SeparateRows,qk),d33.withMlp(w,c,Delta,this.#stockNarrow[33].mlp,SeparateRows,qk),d34.withMlp(w,c,Delta,this.#stockNarrow[34].mlp,SeparateRows,qk),a35.withMlp(w,c,Attention,this.#stockNarrow[35].mlp,SeparateRows),
      d36.withMlp(w,c,Delta,this.#stockNarrow[36].mlp,SeparateRows,qk),d37.withMlp(w,c,Delta,this.#stockNarrow[37].mlp,SeparateRows,qk),d38.withMlp(w,c,Delta,this.#stockNarrow[38].mlp,SeparateRows,qk),a39.withMlp(w,c,Attention,this.#stockNarrow[39].mlp,SeparateRows),
      d40.withMlp(w,c,Delta,this.#stockNarrow[40].mlp,SeparateRows,qk),d41.withMlp(w,c,Delta,this.#stockNarrow[41].mlp,SeparateRows,qk),d42.withMlp(w,c,Delta,this.#stockNarrow[42].mlp,SeparateRows,qk),a43.withMlp(w,c,Attention,this.#stockNarrow[43].mlp,SeparateRows),
      d44.withMlp(w,c,Delta,this.#stockNarrow[44].mlp,SeparateRows,qk),d45.withMlp(w,c,Delta,this.#stockNarrow[45].mlp,SeparateRows,qk),d46.withMlp(w,c,Delta,this.#stockNarrow[46].mlp,SeparateRows,qk),a47.withMlp(w,c,Attention,this.#stockNarrow[47].mlp,SeparateRows),
      d48.withMlp(w,c,Delta,this.#stockNarrow[48].mlp,SeparateRows,qk),d49.withMlp(w,c,Delta,this.#stockNarrow[49].mlp,SeparateRows,qk),d50.withMlp(w,c,Delta,this.#stockNarrow[50].mlp,SeparateRows,qk),a51.withMlp(w,c,Attention,this.#stockNarrow[51].mlp,SeparateRows),
      d52.withMlp(w,c,Delta,this.#stockNarrow[52].mlp,SeparateRows,qk),d53.withMlp(w,c,Delta,this.#stockNarrow[53].mlp,SeparateRows,qk),d54.withMlp(w,c,Delta,this.#stockNarrow[54].mlp,SeparateRows,qk),a55.withMlp(w,c,Attention,this.#stockNarrow[55].mlp,SeparateRows),
      d56.withMlp(w,c,Delta,this.#stockNarrow[56].mlp,SeparateRows,qk),d57.withMlp(w,c,Delta,this.#stockNarrow[57].mlp,SeparateRows,qk),d58.withMlp(w,c,Delta,this.#stockNarrow[58].mlp,SeparateRows,qk),a59.withMlp(w,c,Attention,this.#stockNarrow[59].mlp,SeparateRows),
      d60.withMlp(w,c,Delta,this.#stockNarrow[60].mlp,SeparateRows,qk),d61.withMlp(w,c,Delta,this.#stockNarrow[61].mlp,SeparateRows,qk),d62.withMlp(w,c,Delta,this.#stockNarrow[62].mlp,SeparateRows,qk),a63.withMlp(w,c,Attention,this.#stockNarrow[63].mlp,SeparateRows),
    ];
    this.#knownFiveToEight=[
      d0.withMlp(w,c,Delta,this.#stockFiveToEight[0].mlp,SeparateRows,qk),d1.withMlp(w,c,Delta,this.#stockFiveToEight[1].mlp,SeparateRows,qk),d2.withMlp(w,c,Delta,this.#stockFiveToEight[2].mlp,SeparateRows,qk),a3.withMlp(w,c,Attention,this.#stockFiveToEight[3].mlp,SeparateRows),
      d4.withMlp(w,c,Delta,this.#stockFiveToEight[4].mlp,SeparateRows,qk),d5.withMlp(w,c,Delta,this.#stockFiveToEight[5].mlp,SeparateRows,qk),d6.withMlp(w,c,Delta,this.#stockFiveToEight[6].mlp,SeparateRows,qk),a7.withMlp(w,c,Attention,this.#stockFiveToEight[7].mlp,SeparateRows),
      d8.withMlp(w,c,Delta,this.#stockFiveToEight[8].mlp,SeparateRows,qk),d9.withMlp(w,c,Delta,this.#stockFiveToEight[9].mlp,SeparateRows,qk),d10.withMlp(w,c,Delta,this.#stockFiveToEight[10].mlp,SeparateRows,qk),a11.withMlp(w,c,Attention,this.#stockFiveToEight[11].mlp,SeparateRows),
      d12.withMlp(w,c,Delta,this.#stockFiveToEight[12].mlp,SeparateRows,qk),d13.withMlp(w,c,Delta,this.#stockFiveToEight[13].mlp,SeparateRows,qk),d14.withMlp(w,c,Delta,this.#stockFiveToEight[14].mlp,SeparateRows,qk),a15.withMlp(w,c,Attention,this.#stockFiveToEight[15].mlp,SeparateRows),
      d16.withMlp(w,c,Delta,this.#stockFiveToEight[16].mlp,SeparateRows,qk),d17.withMlp(w,c,Delta,this.#stockFiveToEight[17].mlp,SeparateRows,qk),d18.withMlp(w,c,Delta,this.#stockFiveToEight[18].mlp,SeparateRows,qk),a19.withMlp(w,c,Attention,this.#stockFiveToEight[19].mlp,SeparateRows),
      d20.withMlp(w,c,Delta,this.#stockFiveToEight[20].mlp,SeparateRows,qk),d21.withMlp(w,c,Delta,this.#stockFiveToEight[21].mlp,SeparateRows,qk),d22.withMlp(w,c,Delta,this.#stockFiveToEight[22].mlp,SeparateRows,qk),a23.withMlp(w,c,Attention,this.#stockFiveToEight[23].mlp,SeparateRows),
      d24.withMlp(w,c,Delta,this.#stockFiveToEight[24].mlp,SeparateRows,qk),d25.withMlp(w,c,Delta,this.#stockFiveToEight[25].mlp,SeparateRows,qk),d26.withMlp(w,c,Delta,this.#stockFiveToEight[26].mlp,SeparateRows,qk),a27.withMlp(w,c,Attention,this.#stockFiveToEight[27].mlp,SeparateRows),
      d28.withMlp(w,c,Delta,this.#stockFiveToEight[28].mlp,SeparateRows,qk),d29.withMlp(w,c,Delta,this.#stockFiveToEight[29].mlp,SeparateRows,qk),d30.withMlp(w,c,Delta,this.#stockFiveToEight[30].mlp,SeparateRows,qk),a31.withMlp(w,c,Attention,this.#stockFiveToEight[31].mlp,SeparateRows),
      d32.withMlp(w,c,Delta,this.#stockFiveToEight[32].mlp,SeparateRows,qk),d33.withMlp(w,c,Delta,this.#stockFiveToEight[33].mlp,SeparateRows,qk),d34.withMlp(w,c,Delta,this.#stockFiveToEight[34].mlp,SeparateRows,qk),a35.withMlp(w,c,Attention,this.#stockFiveToEight[35].mlp,SeparateRows),
      d36.withMlp(w,c,Delta,this.#stockFiveToEight[36].mlp,SeparateRows,qk),d37.withMlp(w,c,Delta,this.#stockFiveToEight[37].mlp,SeparateRows,qk),d38.withMlp(w,c,Delta,this.#stockFiveToEight[38].mlp,SeparateRows,qk),a39.withMlp(w,c,Attention,this.#stockFiveToEight[39].mlp,SeparateRows),
      d40.withMlp(w,c,Delta,this.#stockFiveToEight[40].mlp,SeparateRows,qk),d41.withMlp(w,c,Delta,this.#stockFiveToEight[41].mlp,SeparateRows,qk),d42.withMlp(w,c,Delta,this.#stockFiveToEight[42].mlp,SeparateRows,qk),a43.withMlp(w,c,Attention,this.#stockFiveToEight[43].mlp,SeparateRows),
      d44.withMlp(w,c,Delta,this.#stockFiveToEight[44].mlp,SeparateRows,qk),d45.withMlp(w,c,Delta,this.#stockFiveToEight[45].mlp,SeparateRows,qk),d46.withMlp(w,c,Delta,this.#stockFiveToEight[46].mlp,SeparateRows,qk),a47.withMlp(w,c,Attention,this.#stockFiveToEight[47].mlp,SeparateRows),
      d48.withMlp(w,c,Delta,this.#stockFiveToEight[48].mlp,SeparateRows,qk),d49.withMlp(w,c,Delta,this.#stockFiveToEight[49].mlp,SeparateRows,qk),d50.withMlp(w,c,Delta,this.#stockFiveToEight[50].mlp,SeparateRows,qk),a51.withMlp(w,c,Attention,this.#stockFiveToEight[51].mlp,SeparateRows),
      d52.withMlp(w,c,Delta,this.#stockFiveToEight[52].mlp,SeparateRows,qk),d53.withMlp(w,c,Delta,this.#stockFiveToEight[53].mlp,SeparateRows,qk),d54.withMlp(w,c,Delta,this.#stockFiveToEight[54].mlp,SeparateRows,qk),a55.withMlp(w,c,Attention,this.#stockFiveToEight[55].mlp,SeparateRows),
      d56.withMlp(w,c,Delta,this.#stockFiveToEight[56].mlp,SeparateRows,qk),d57.withMlp(w,c,Delta,this.#stockFiveToEight[57].mlp,SeparateRows,qk),d58.withMlp(w,c,Delta,this.#stockFiveToEight[58].mlp,SeparateRows,qk),a59.withMlp(w,c,Attention,this.#stockFiveToEight[59].mlp,SeparateRows),
      d60.withMlp(w,c,Delta,this.#stockFiveToEight[60].mlp,SeparateRows,qk),d61.withMlp(w,c,Delta,this.#stockFiveToEight[61].mlp,SeparateRows,qk),d62.withMlp(w,c,Delta,this.#stockFiveToEight[62].mlp,SeparateRows,qk),a63.withMlp(w,c,Attention,this.#stockFiveToEight[63].mlp,SeparateRows),
    ];
    this.#knownWide=[
      new Delta(w,c,d0.prefix,this.#knownNarrow[0].mixer,this.#wide[0].mlp),new Delta(w,c,d1.prefix,this.#knownNarrow[1].mixer,this.#wide[1].mlp),new Delta(w,c,d2.prefix,this.#knownNarrow[2].mixer,this.#wide[2].mlp),new Attention(w,c,a3.prefix,this.#knownNarrow[3].mixer,this.#wide[3].mlp),
      new Delta(w,c,d4.prefix,this.#knownNarrow[4].mixer,this.#wide[4].mlp),new Delta(w,c,d5.prefix,this.#knownNarrow[5].mixer,this.#wide[5].mlp),new Delta(w,c,d6.prefix,this.#knownNarrow[6].mixer,this.#wide[6].mlp),new Attention(w,c,a7.prefix,this.#knownNarrow[7].mixer,this.#wide[7].mlp),
      new Delta(w,c,d8.prefix,this.#knownNarrow[8].mixer,this.#wide[8].mlp),new Delta(w,c,d9.prefix,this.#knownNarrow[9].mixer,this.#wide[9].mlp),new Delta(w,c,d10.prefix,this.#knownNarrow[10].mixer,this.#wide[10].mlp),new Attention(w,c,a11.prefix,this.#knownNarrow[11].mixer,this.#wide[11].mlp),
      new Delta(w,c,d12.prefix,this.#knownNarrow[12].mixer,this.#wide[12].mlp),new Delta(w,c,d13.prefix,this.#knownNarrow[13].mixer,this.#wide[13].mlp),new Delta(w,c,d14.prefix,this.#knownNarrow[14].mixer,this.#wide[14].mlp),new Attention(w,c,a15.prefix,this.#knownNarrow[15].mixer,this.#wide[15].mlp),
      new Delta(w,c,d16.prefix,this.#knownNarrow[16].mixer,this.#wide[16].mlp),new Delta(w,c,d17.prefix,this.#knownNarrow[17].mixer,this.#wide[17].mlp),new Delta(w,c,d18.prefix,this.#knownNarrow[18].mixer,this.#wide[18].mlp),new Attention(w,c,a19.prefix,this.#knownNarrow[19].mixer,this.#wide[19].mlp),
      new Delta(w,c,d20.prefix,this.#knownNarrow[20].mixer,this.#wide[20].mlp),new Delta(w,c,d21.prefix,this.#knownNarrow[21].mixer,this.#wide[21].mlp),new Delta(w,c,d22.prefix,this.#knownNarrow[22].mixer,this.#wide[22].mlp),new Attention(w,c,a23.prefix,this.#knownNarrow[23].mixer,this.#wide[23].mlp),
      new Delta(w,c,d24.prefix,this.#knownNarrow[24].mixer,this.#wide[24].mlp),new Delta(w,c,d25.prefix,this.#knownNarrow[25].mixer,this.#wide[25].mlp),new Delta(w,c,d26.prefix,this.#knownNarrow[26].mixer,this.#wide[26].mlp),new Attention(w,c,a27.prefix,this.#knownNarrow[27].mixer,this.#wide[27].mlp),
      new Delta(w,c,d28.prefix,this.#knownNarrow[28].mixer,this.#wide[28].mlp),new Delta(w,c,d29.prefix,this.#knownNarrow[29].mixer,this.#wide[29].mlp),new Delta(w,c,d30.prefix,this.#knownNarrow[30].mixer,this.#wide[30].mlp),new Attention(w,c,a31.prefix,this.#knownNarrow[31].mixer,this.#wide[31].mlp),
      new Delta(w,c,d32.prefix,this.#knownNarrow[32].mixer,this.#wide[32].mlp),new Delta(w,c,d33.prefix,this.#knownNarrow[33].mixer,this.#wide[33].mlp),new Delta(w,c,d34.prefix,this.#knownNarrow[34].mixer,this.#wide[34].mlp),new Attention(w,c,a35.prefix,this.#knownNarrow[35].mixer,this.#wide[35].mlp),
      new Delta(w,c,d36.prefix,this.#knownNarrow[36].mixer,this.#wide[36].mlp),new Delta(w,c,d37.prefix,this.#knownNarrow[37].mixer,this.#wide[37].mlp),new Delta(w,c,d38.prefix,this.#knownNarrow[38].mixer,this.#wide[38].mlp),new Attention(w,c,a39.prefix,this.#knownNarrow[39].mixer,this.#wide[39].mlp),
      new Delta(w,c,d40.prefix,this.#knownNarrow[40].mixer,this.#wide[40].mlp),new Delta(w,c,d41.prefix,this.#knownNarrow[41].mixer,this.#wide[41].mlp),new Delta(w,c,d42.prefix,this.#knownNarrow[42].mixer,this.#wide[42].mlp),new Attention(w,c,a43.prefix,this.#knownNarrow[43].mixer,this.#wide[43].mlp),
      new Delta(w,c,d44.prefix,this.#knownNarrow[44].mixer,this.#wide[44].mlp),new Delta(w,c,d45.prefix,this.#knownNarrow[45].mixer,this.#wide[45].mlp),new Delta(w,c,d46.prefix,this.#knownNarrow[46].mixer,this.#wide[46].mlp),new Attention(w,c,a47.prefix,this.#knownNarrow[47].mixer,this.#wide[47].mlp),
      new Delta(w,c,d48.prefix,this.#knownNarrow[48].mixer,this.#wide[48].mlp),new Delta(w,c,d49.prefix,this.#knownNarrow[49].mixer,this.#wide[49].mlp),new Delta(w,c,d50.prefix,this.#knownNarrow[50].mixer,this.#wide[50].mlp),new Attention(w,c,a51.prefix,this.#knownNarrow[51].mixer,this.#wide[51].mlp),
      new Delta(w,c,d52.prefix,this.#knownNarrow[52].mixer,this.#wide[52].mlp),new Delta(w,c,d53.prefix,this.#knownNarrow[53].mixer,this.#wide[53].mlp),new Delta(w,c,d54.prefix,this.#knownNarrow[54].mixer,this.#wide[54].mlp),new Attention(w,c,a55.prefix,this.#knownNarrow[55].mixer,this.#wide[55].mlp),
      new Delta(w,c,d56.prefix,this.#knownNarrow[56].mixer,this.#wide[56].mlp),new Delta(w,c,d57.prefix,this.#knownNarrow[57].mixer,this.#wide[57].mlp),new Delta(w,c,d58.prefix,this.#knownNarrow[58].mixer,this.#wide[58].mlp),new Attention(w,c,a59.prefix,this.#knownNarrow[59].mixer,this.#wide[59].mlp),
      new Delta(w,c,d60.prefix,this.#knownNarrow[60].mixer,this.#wide[60].mlp),new Delta(w,c,d61.prefix,this.#knownNarrow[61].mixer,this.#wide[61].mlp),new Delta(w,c,d62.prefix,this.#knownNarrow[62].mixer,this.#wide[62].mlp),new Attention(w,c,a63.prefix,this.#knownNarrow[63].mixer,this.#wide[63].mlp),
    ];
    this.#knownTiled=[
      new Delta(w,c,d0.prefix,this.#knownNarrow[0].mixer,this.#tiled[0].mlp),new Delta(w,c,d1.prefix,this.#knownNarrow[1].mixer,this.#tiled[1].mlp),new Delta(w,c,d2.prefix,this.#knownNarrow[2].mixer,this.#tiled[2].mlp),new Attention(w,c,a3.prefix,this.#knownNarrow[3].mixer,this.#tiled[3].mlp),
      new Delta(w,c,d4.prefix,this.#knownNarrow[4].mixer,this.#tiled[4].mlp),new Delta(w,c,d5.prefix,this.#knownNarrow[5].mixer,this.#tiled[5].mlp),new Delta(w,c,d6.prefix,this.#knownNarrow[6].mixer,this.#tiled[6].mlp),new Attention(w,c,a7.prefix,this.#knownNarrow[7].mixer,this.#tiled[7].mlp),
      new Delta(w,c,d8.prefix,this.#knownNarrow[8].mixer,this.#tiled[8].mlp),new Delta(w,c,d9.prefix,this.#knownNarrow[9].mixer,this.#tiled[9].mlp),new Delta(w,c,d10.prefix,this.#knownNarrow[10].mixer,this.#tiled[10].mlp),new Attention(w,c,a11.prefix,this.#knownNarrow[11].mixer,this.#tiled[11].mlp),
      new Delta(w,c,d12.prefix,this.#knownNarrow[12].mixer,this.#tiled[12].mlp),new Delta(w,c,d13.prefix,this.#knownNarrow[13].mixer,this.#tiled[13].mlp),new Delta(w,c,d14.prefix,this.#knownNarrow[14].mixer,this.#tiled[14].mlp),new Attention(w,c,a15.prefix,this.#knownNarrow[15].mixer,this.#tiled[15].mlp),
      new Delta(w,c,d16.prefix,this.#knownNarrow[16].mixer,this.#tiled[16].mlp),new Delta(w,c,d17.prefix,this.#knownNarrow[17].mixer,this.#tiled[17].mlp),new Delta(w,c,d18.prefix,this.#knownNarrow[18].mixer,this.#tiled[18].mlp),new Attention(w,c,a19.prefix,this.#knownNarrow[19].mixer,this.#tiled[19].mlp),
      new Delta(w,c,d20.prefix,this.#knownNarrow[20].mixer,this.#tiled[20].mlp),new Delta(w,c,d21.prefix,this.#knownNarrow[21].mixer,this.#tiled[21].mlp),new Delta(w,c,d22.prefix,this.#knownNarrow[22].mixer,this.#tiled[22].mlp),new Attention(w,c,a23.prefix,this.#knownNarrow[23].mixer,this.#tiled[23].mlp),
      new Delta(w,c,d24.prefix,this.#knownNarrow[24].mixer,this.#tiled[24].mlp),new Delta(w,c,d25.prefix,this.#knownNarrow[25].mixer,this.#tiled[25].mlp),new Delta(w,c,d26.prefix,this.#knownNarrow[26].mixer,this.#tiled[26].mlp),new Attention(w,c,a27.prefix,this.#knownNarrow[27].mixer,this.#tiled[27].mlp),
      new Delta(w,c,d28.prefix,this.#knownNarrow[28].mixer,this.#tiled[28].mlp),new Delta(w,c,d29.prefix,this.#knownNarrow[29].mixer,this.#tiled[29].mlp),new Delta(w,c,d30.prefix,this.#knownNarrow[30].mixer,this.#tiled[30].mlp),new Attention(w,c,a31.prefix,this.#knownNarrow[31].mixer,this.#tiled[31].mlp),
      new Delta(w,c,d32.prefix,this.#knownNarrow[32].mixer,this.#tiled[32].mlp),new Delta(w,c,d33.prefix,this.#knownNarrow[33].mixer,this.#tiled[33].mlp),new Delta(w,c,d34.prefix,this.#knownNarrow[34].mixer,this.#tiled[34].mlp),new Attention(w,c,a35.prefix,this.#knownNarrow[35].mixer,this.#tiled[35].mlp),
      new Delta(w,c,d36.prefix,this.#knownNarrow[36].mixer,this.#tiled[36].mlp),new Delta(w,c,d37.prefix,this.#knownNarrow[37].mixer,this.#tiled[37].mlp),new Delta(w,c,d38.prefix,this.#knownNarrow[38].mixer,this.#tiled[38].mlp),new Attention(w,c,a39.prefix,this.#knownNarrow[39].mixer,this.#tiled[39].mlp),
      new Delta(w,c,d40.prefix,this.#knownNarrow[40].mixer,this.#tiled[40].mlp),new Delta(w,c,d41.prefix,this.#knownNarrow[41].mixer,this.#tiled[41].mlp),new Delta(w,c,d42.prefix,this.#knownNarrow[42].mixer,this.#tiled[42].mlp),new Attention(w,c,a43.prefix,this.#knownNarrow[43].mixer,this.#tiled[43].mlp),
      new Delta(w,c,d44.prefix,this.#knownNarrow[44].mixer,this.#tiled[44].mlp),new Delta(w,c,d45.prefix,this.#knownNarrow[45].mixer,this.#tiled[45].mlp),new Delta(w,c,d46.prefix,this.#knownNarrow[46].mixer,this.#tiled[46].mlp),new Attention(w,c,a47.prefix,this.#knownNarrow[47].mixer,this.#tiled[47].mlp),
      new Delta(w,c,d48.prefix,this.#knownNarrow[48].mixer,this.#tiled[48].mlp),new Delta(w,c,d49.prefix,this.#knownNarrow[49].mixer,this.#tiled[49].mlp),new Delta(w,c,d50.prefix,this.#knownNarrow[50].mixer,this.#tiled[50].mlp),new Attention(w,c,a51.prefix,this.#knownNarrow[51].mixer,this.#tiled[51].mlp),
      new Delta(w,c,d52.prefix,this.#knownNarrow[52].mixer,this.#tiled[52].mlp),new Delta(w,c,d53.prefix,this.#knownNarrow[53].mixer,this.#tiled[53].mlp),new Delta(w,c,d54.prefix,this.#knownNarrow[54].mixer,this.#tiled[54].mlp),new Attention(w,c,a55.prefix,this.#knownNarrow[55].mixer,this.#tiled[55].mlp),
      new Delta(w,c,d56.prefix,this.#knownNarrow[56].mixer,this.#tiled[56].mlp),new Delta(w,c,d57.prefix,this.#knownNarrow[57].mixer,this.#tiled[57].mlp),new Delta(w,c,d58.prefix,this.#knownNarrow[58].mixer,this.#tiled[58].mlp),new Attention(w,c,a59.prefix,this.#knownNarrow[59].mixer,this.#tiled[59].mlp),
      new Delta(w,c,d60.prefix,this.#knownNarrow[60].mixer,this.#tiled[60].mlp),new Delta(w,c,d61.prefix,this.#knownNarrow[61].mixer,this.#tiled[61].mlp),new Delta(w,c,d62.prefix,this.#knownNarrow[62].mixer,this.#tiled[62].mlp),new Attention(w,c,a63.prefix,this.#knownNarrow[63].mixer,this.#tiled[63].mlp),
    ];
    this.#knownExpanded=[
      new Delta(w,c,d0.prefix,this.#knownNarrow[0].mixer,this.#expanded[0].mlp),new Delta(w,c,d1.prefix,this.#knownNarrow[1].mixer,this.#expanded[1].mlp),new Delta(w,c,d2.prefix,this.#knownNarrow[2].mixer,this.#expanded[2].mlp),new Attention(w,c,a3.prefix,this.#knownNarrow[3].mixer,this.#expanded[3].mlp),
      new Delta(w,c,d4.prefix,this.#knownNarrow[4].mixer,this.#expanded[4].mlp),new Delta(w,c,d5.prefix,this.#knownNarrow[5].mixer,this.#expanded[5].mlp),new Delta(w,c,d6.prefix,this.#knownNarrow[6].mixer,this.#expanded[6].mlp),new Attention(w,c,a7.prefix,this.#knownNarrow[7].mixer,this.#expanded[7].mlp),
      new Delta(w,c,d8.prefix,this.#knownNarrow[8].mixer,this.#expanded[8].mlp),new Delta(w,c,d9.prefix,this.#knownNarrow[9].mixer,this.#expanded[9].mlp),new Delta(w,c,d10.prefix,this.#knownNarrow[10].mixer,this.#expanded[10].mlp),new Attention(w,c,a11.prefix,this.#knownNarrow[11].mixer,this.#expanded[11].mlp),
      new Delta(w,c,d12.prefix,this.#knownNarrow[12].mixer,this.#expanded[12].mlp),new Delta(w,c,d13.prefix,this.#knownNarrow[13].mixer,this.#expanded[13].mlp),new Delta(w,c,d14.prefix,this.#knownNarrow[14].mixer,this.#expanded[14].mlp),new Attention(w,c,a15.prefix,this.#knownNarrow[15].mixer,this.#expanded[15].mlp),
      new Delta(w,c,d16.prefix,this.#knownNarrow[16].mixer,this.#expanded[16].mlp),new Delta(w,c,d17.prefix,this.#knownNarrow[17].mixer,this.#expanded[17].mlp),new Delta(w,c,d18.prefix,this.#knownNarrow[18].mixer,this.#expanded[18].mlp),new Attention(w,c,a19.prefix,this.#knownNarrow[19].mixer,this.#expanded[19].mlp),
      new Delta(w,c,d20.prefix,this.#knownNarrow[20].mixer,this.#expanded[20].mlp),new Delta(w,c,d21.prefix,this.#knownNarrow[21].mixer,this.#expanded[21].mlp),new Delta(w,c,d22.prefix,this.#knownNarrow[22].mixer,this.#expanded[22].mlp),new Attention(w,c,a23.prefix,this.#knownNarrow[23].mixer,this.#expanded[23].mlp),
      new Delta(w,c,d24.prefix,this.#knownNarrow[24].mixer,this.#expanded[24].mlp),new Delta(w,c,d25.prefix,this.#knownNarrow[25].mixer,this.#expanded[25].mlp),new Delta(w,c,d26.prefix,this.#knownNarrow[26].mixer,this.#expanded[26].mlp),new Attention(w,c,a27.prefix,this.#knownNarrow[27].mixer,this.#expanded[27].mlp),
      new Delta(w,c,d28.prefix,this.#knownNarrow[28].mixer,this.#expanded[28].mlp),new Delta(w,c,d29.prefix,this.#knownNarrow[29].mixer,this.#expanded[29].mlp),new Delta(w,c,d30.prefix,this.#knownNarrow[30].mixer,this.#expanded[30].mlp),new Attention(w,c,a31.prefix,this.#knownNarrow[31].mixer,this.#expanded[31].mlp),
      new Delta(w,c,d32.prefix,this.#knownNarrow[32].mixer,this.#expanded[32].mlp),new Delta(w,c,d33.prefix,this.#knownNarrow[33].mixer,this.#expanded[33].mlp),new Delta(w,c,d34.prefix,this.#knownNarrow[34].mixer,this.#expanded[34].mlp),new Attention(w,c,a35.prefix,this.#knownNarrow[35].mixer,this.#expanded[35].mlp),
      new Delta(w,c,d36.prefix,this.#knownNarrow[36].mixer,this.#expanded[36].mlp),new Delta(w,c,d37.prefix,this.#knownNarrow[37].mixer,this.#expanded[37].mlp),new Delta(w,c,d38.prefix,this.#knownNarrow[38].mixer,this.#expanded[38].mlp),new Attention(w,c,a39.prefix,this.#knownNarrow[39].mixer,this.#expanded[39].mlp),
      new Delta(w,c,d40.prefix,this.#knownNarrow[40].mixer,this.#expanded[40].mlp),new Delta(w,c,d41.prefix,this.#knownNarrow[41].mixer,this.#expanded[41].mlp),new Delta(w,c,d42.prefix,this.#knownNarrow[42].mixer,this.#expanded[42].mlp),new Attention(w,c,a43.prefix,this.#knownNarrow[43].mixer,this.#expanded[43].mlp),
      new Delta(w,c,d44.prefix,this.#knownNarrow[44].mixer,this.#expanded[44].mlp),new Delta(w,c,d45.prefix,this.#knownNarrow[45].mixer,this.#expanded[45].mlp),new Delta(w,c,d46.prefix,this.#knownNarrow[46].mixer,this.#expanded[46].mlp),new Attention(w,c,a47.prefix,this.#knownNarrow[47].mixer,this.#expanded[47].mlp),
      new Delta(w,c,d48.prefix,this.#knownNarrow[48].mixer,this.#expanded[48].mlp),new Delta(w,c,d49.prefix,this.#knownNarrow[49].mixer,this.#expanded[49].mlp),new Delta(w,c,d50.prefix,this.#knownNarrow[50].mixer,this.#expanded[50].mlp),new Attention(w,c,a51.prefix,this.#knownNarrow[51].mixer,this.#expanded[51].mlp),
      new Delta(w,c,d52.prefix,this.#knownNarrow[52].mixer,this.#expanded[52].mlp),new Delta(w,c,d53.prefix,this.#knownNarrow[53].mixer,this.#expanded[53].mlp),new Delta(w,c,d54.prefix,this.#knownNarrow[54].mixer,this.#expanded[54].mlp),new Attention(w,c,a55.prefix,this.#knownNarrow[55].mixer,this.#expanded[55].mlp),
      new Delta(w,c,d56.prefix,this.#knownNarrow[56].mixer,this.#expanded[56].mlp),new Delta(w,c,d57.prefix,this.#knownNarrow[57].mixer,this.#expanded[57].mlp),new Delta(w,c,d58.prefix,this.#knownNarrow[58].mixer,this.#expanded[58].mlp),new Attention(w,c,a59.prefix,this.#knownNarrow[59].mixer,this.#expanded[59].mlp),
      new Delta(w,c,d60.prefix,this.#knownNarrow[60].mixer,this.#expanded[60].mlp),new Delta(w,c,d61.prefix,this.#knownNarrow[61].mixer,this.#expanded[61].mlp),new Delta(w,c,d62.prefix,this.#knownNarrow[62].mixer,this.#expanded[62].mlp),new Attention(w,c,a63.prefix,this.#knownNarrow[63].mixer,this.#expanded[63].mlp),
    ];
    const mediaNarrow:Layers=[
      this.#stockNarrow[0],this.#stockNarrow[1],this.#stockNarrow[2],a3.mediaLayer(w,c,Attention,this.#stockNarrow[3].mlp,Plain),
      this.#stockNarrow[4],this.#stockNarrow[5],this.#stockNarrow[6],a7.mediaLayer(w,c,Attention,this.#stockNarrow[7].mlp,Plain),
      this.#stockNarrow[8],this.#stockNarrow[9],this.#stockNarrow[10],a11.mediaLayer(w,c,Attention,this.#stockNarrow[11].mlp,Plain),
      this.#stockNarrow[12],this.#stockNarrow[13],this.#stockNarrow[14],a15.mediaLayer(w,c,Attention,this.#stockNarrow[15].mlp,Plain),
      this.#stockNarrow[16],this.#stockNarrow[17],this.#stockNarrow[18],a19.mediaLayer(w,c,Attention,this.#stockNarrow[19].mlp,Plain),
      this.#stockNarrow[20],this.#stockNarrow[21],this.#stockNarrow[22],a23.mediaLayer(w,c,Attention,this.#stockNarrow[23].mlp,Plain),
      this.#stockNarrow[24],this.#stockNarrow[25],this.#stockNarrow[26],a27.mediaLayer(w,c,Attention,this.#stockNarrow[27].mlp,Plain),
      this.#stockNarrow[28],this.#stockNarrow[29],this.#stockNarrow[30],a31.mediaLayer(w,c,Attention,this.#stockNarrow[31].mlp,Plain),
      this.#stockNarrow[32],this.#stockNarrow[33],this.#stockNarrow[34],a35.mediaLayer(w,c,Attention,this.#stockNarrow[35].mlp,Plain),
      this.#stockNarrow[36],this.#stockNarrow[37],this.#stockNarrow[38],a39.mediaLayer(w,c,Attention,this.#stockNarrow[39].mlp,Plain),
      this.#stockNarrow[40],this.#stockNarrow[41],this.#stockNarrow[42],a43.mediaLayer(w,c,Attention,this.#stockNarrow[43].mlp,Plain),
      this.#stockNarrow[44],this.#stockNarrow[45],this.#stockNarrow[46],a47.mediaLayer(w,c,Attention,this.#stockNarrow[47].mlp,Plain),
      this.#stockNarrow[48],this.#stockNarrow[49],this.#stockNarrow[50],a51.mediaLayer(w,c,Attention,this.#stockNarrow[51].mlp,Plain),
      this.#stockNarrow[52],this.#stockNarrow[53],this.#stockNarrow[54],a55.mediaLayer(w,c,Attention,this.#stockNarrow[55].mlp,Plain),
      this.#stockNarrow[56],this.#stockNarrow[57],this.#stockNarrow[58],a59.mediaLayer(w,c,Attention,this.#stockNarrow[59].mlp,Plain),
      this.#stockNarrow[60],this.#stockNarrow[61],this.#stockNarrow[62],a63.mediaLayer(w,c,Attention,this.#stockNarrow[63].mlp,Plain),
    ];
    const mediaFiveToEight:Layers=[
      this.#stockFiveToEight[0],this.#stockFiveToEight[1],this.#stockFiveToEight[2],a3.mediaLayer(w,c,Attention,this.#stockFiveToEight[3].mlp,Plain),
      this.#stockFiveToEight[4],this.#stockFiveToEight[5],this.#stockFiveToEight[6],a7.mediaLayer(w,c,Attention,this.#stockFiveToEight[7].mlp,Plain),
      this.#stockFiveToEight[8],this.#stockFiveToEight[9],this.#stockFiveToEight[10],a11.mediaLayer(w,c,Attention,this.#stockFiveToEight[11].mlp,Plain),
      this.#stockFiveToEight[12],this.#stockFiveToEight[13],this.#stockFiveToEight[14],a15.mediaLayer(w,c,Attention,this.#stockFiveToEight[15].mlp,Plain),
      this.#stockFiveToEight[16],this.#stockFiveToEight[17],this.#stockFiveToEight[18],a19.mediaLayer(w,c,Attention,this.#stockFiveToEight[19].mlp,Plain),
      this.#stockFiveToEight[20],this.#stockFiveToEight[21],this.#stockFiveToEight[22],a23.mediaLayer(w,c,Attention,this.#stockFiveToEight[23].mlp,Plain),
      this.#stockFiveToEight[24],this.#stockFiveToEight[25],this.#stockFiveToEight[26],a27.mediaLayer(w,c,Attention,this.#stockFiveToEight[27].mlp,Plain),
      this.#stockFiveToEight[28],this.#stockFiveToEight[29],this.#stockFiveToEight[30],a31.mediaLayer(w,c,Attention,this.#stockFiveToEight[31].mlp,Plain),
      this.#stockFiveToEight[32],this.#stockFiveToEight[33],this.#stockFiveToEight[34],a35.mediaLayer(w,c,Attention,this.#stockFiveToEight[35].mlp,Plain),
      this.#stockFiveToEight[36],this.#stockFiveToEight[37],this.#stockFiveToEight[38],a39.mediaLayer(w,c,Attention,this.#stockFiveToEight[39].mlp,Plain),
      this.#stockFiveToEight[40],this.#stockFiveToEight[41],this.#stockFiveToEight[42],a43.mediaLayer(w,c,Attention,this.#stockFiveToEight[43].mlp,Plain),
      this.#stockFiveToEight[44],this.#stockFiveToEight[45],this.#stockFiveToEight[46],a47.mediaLayer(w,c,Attention,this.#stockFiveToEight[47].mlp,Plain),
      this.#stockFiveToEight[48],this.#stockFiveToEight[49],this.#stockFiveToEight[50],a51.mediaLayer(w,c,Attention,this.#stockFiveToEight[51].mlp,Plain),
      this.#stockFiveToEight[52],this.#stockFiveToEight[53],this.#stockFiveToEight[54],a55.mediaLayer(w,c,Attention,this.#stockFiveToEight[55].mlp,Plain),
      this.#stockFiveToEight[56],this.#stockFiveToEight[57],this.#stockFiveToEight[58],a59.mediaLayer(w,c,Attention,this.#stockFiveToEight[59].mlp,Plain),
      this.#stockFiveToEight[60],this.#stockFiveToEight[61],this.#stockFiveToEight[62],a63.mediaLayer(w,c,Attention,this.#stockFiveToEight[63].mlp,Plain),
    ];
    const mediaWide:Layers=[
      this.#wide[0],this.#wide[1],this.#wide[2],a3.mediaLayer(w,c,Attention,this.#wide[3].mlp,Plain),
      this.#wide[4],this.#wide[5],this.#wide[6],a7.mediaLayer(w,c,Attention,this.#wide[7].mlp,Plain),
      this.#wide[8],this.#wide[9],this.#wide[10],a11.mediaLayer(w,c,Attention,this.#wide[11].mlp,Plain),
      this.#wide[12],this.#wide[13],this.#wide[14],a15.mediaLayer(w,c,Attention,this.#wide[15].mlp,Plain),
      this.#wide[16],this.#wide[17],this.#wide[18],a19.mediaLayer(w,c,Attention,this.#wide[19].mlp,Plain),
      this.#wide[20],this.#wide[21],this.#wide[22],a23.mediaLayer(w,c,Attention,this.#wide[23].mlp,Plain),
      this.#wide[24],this.#wide[25],this.#wide[26],a27.mediaLayer(w,c,Attention,this.#wide[27].mlp,Plain),
      this.#wide[28],this.#wide[29],this.#wide[30],a31.mediaLayer(w,c,Attention,this.#wide[31].mlp,Plain),
      this.#wide[32],this.#wide[33],this.#wide[34],a35.mediaLayer(w,c,Attention,this.#wide[35].mlp,Plain),
      this.#wide[36],this.#wide[37],this.#wide[38],a39.mediaLayer(w,c,Attention,this.#wide[39].mlp,Plain),
      this.#wide[40],this.#wide[41],this.#wide[42],a43.mediaLayer(w,c,Attention,this.#wide[43].mlp,Plain),
      this.#wide[44],this.#wide[45],this.#wide[46],a47.mediaLayer(w,c,Attention,this.#wide[47].mlp,Plain),
      this.#wide[48],this.#wide[49],this.#wide[50],a51.mediaLayer(w,c,Attention,this.#wide[51].mlp,Plain),
      this.#wide[52],this.#wide[53],this.#wide[54],a55.mediaLayer(w,c,Attention,this.#wide[55].mlp,Plain),
      this.#wide[56],this.#wide[57],this.#wide[58],a59.mediaLayer(w,c,Attention,this.#wide[59].mlp,Plain),
      this.#wide[60],this.#wide[61],this.#wide[62],a63.mediaLayer(w,c,Attention,this.#wide[63].mlp,Plain),
    ];
    const mediaTiled:Layers=[
      this.#tiled[0],this.#tiled[1],this.#tiled[2],a3.mediaLayer(w,c,Attention,this.#tiled[3].mlp,Plain),
      this.#tiled[4],this.#tiled[5],this.#tiled[6],a7.mediaLayer(w,c,Attention,this.#tiled[7].mlp,Plain),
      this.#tiled[8],this.#tiled[9],this.#tiled[10],a11.mediaLayer(w,c,Attention,this.#tiled[11].mlp,Plain),
      this.#tiled[12],this.#tiled[13],this.#tiled[14],a15.mediaLayer(w,c,Attention,this.#tiled[15].mlp,Plain),
      this.#tiled[16],this.#tiled[17],this.#tiled[18],a19.mediaLayer(w,c,Attention,this.#tiled[19].mlp,Plain),
      this.#tiled[20],this.#tiled[21],this.#tiled[22],a23.mediaLayer(w,c,Attention,this.#tiled[23].mlp,Plain),
      this.#tiled[24],this.#tiled[25],this.#tiled[26],a27.mediaLayer(w,c,Attention,this.#tiled[27].mlp,Plain),
      this.#tiled[28],this.#tiled[29],this.#tiled[30],a31.mediaLayer(w,c,Attention,this.#tiled[31].mlp,Plain),
      this.#tiled[32],this.#tiled[33],this.#tiled[34],a35.mediaLayer(w,c,Attention,this.#tiled[35].mlp,Plain),
      this.#tiled[36],this.#tiled[37],this.#tiled[38],a39.mediaLayer(w,c,Attention,this.#tiled[39].mlp,Plain),
      this.#tiled[40],this.#tiled[41],this.#tiled[42],a43.mediaLayer(w,c,Attention,this.#tiled[43].mlp,Plain),
      this.#tiled[44],this.#tiled[45],this.#tiled[46],a47.mediaLayer(w,c,Attention,this.#tiled[47].mlp,Plain),
      this.#tiled[48],this.#tiled[49],this.#tiled[50],a51.mediaLayer(w,c,Attention,this.#tiled[51].mlp,Plain),
      this.#tiled[52],this.#tiled[53],this.#tiled[54],a55.mediaLayer(w,c,Attention,this.#tiled[55].mlp,Plain),
      this.#tiled[56],this.#tiled[57],this.#tiled[58],a59.mediaLayer(w,c,Attention,this.#tiled[59].mlp,Plain),
      this.#tiled[60],this.#tiled[61],this.#tiled[62],a63.mediaLayer(w,c,Attention,this.#tiled[63].mlp,Plain),
    ];
    const mediaExpanded:Layers=[
      this.#expanded[0],this.#expanded[1],this.#expanded[2],a3.mediaLayer(w,c,Attention,this.#expanded[3].mlp,Plain),
      this.#expanded[4],this.#expanded[5],this.#expanded[6],a7.mediaLayer(w,c,Attention,this.#expanded[7].mlp,Plain),
      this.#expanded[8],this.#expanded[9],this.#expanded[10],a11.mediaLayer(w,c,Attention,this.#expanded[11].mlp,Plain),
      this.#expanded[12],this.#expanded[13],this.#expanded[14],a15.mediaLayer(w,c,Attention,this.#expanded[15].mlp,Plain),
      this.#expanded[16],this.#expanded[17],this.#expanded[18],a19.mediaLayer(w,c,Attention,this.#expanded[19].mlp,Plain),
      this.#expanded[20],this.#expanded[21],this.#expanded[22],a23.mediaLayer(w,c,Attention,this.#expanded[23].mlp,Plain),
      this.#expanded[24],this.#expanded[25],this.#expanded[26],a27.mediaLayer(w,c,Attention,this.#expanded[27].mlp,Plain),
      this.#expanded[28],this.#expanded[29],this.#expanded[30],a31.mediaLayer(w,c,Attention,this.#expanded[31].mlp,Plain),
      this.#expanded[32],this.#expanded[33],this.#expanded[34],a35.mediaLayer(w,c,Attention,this.#expanded[35].mlp,Plain),
      this.#expanded[36],this.#expanded[37],this.#expanded[38],a39.mediaLayer(w,c,Attention,this.#expanded[39].mlp,Plain),
      this.#expanded[40],this.#expanded[41],this.#expanded[42],a43.mediaLayer(w,c,Attention,this.#expanded[43].mlp,Plain),
      this.#expanded[44],this.#expanded[45],this.#expanded[46],a47.mediaLayer(w,c,Attention,this.#expanded[47].mlp,Plain),
      this.#expanded[48],this.#expanded[49],this.#expanded[50],a51.mediaLayer(w,c,Attention,this.#expanded[51].mlp,Plain),
      this.#expanded[52],this.#expanded[53],this.#expanded[54],a55.mediaLayer(w,c,Attention,this.#expanded[55].mlp,Plain),
      this.#expanded[56],this.#expanded[57],this.#expanded[58],a59.mediaLayer(w,c,Attention,this.#expanded[59].mlp,Plain),
      this.#expanded[60],this.#expanded[61],this.#expanded[62],a63.mediaLayer(w,c,Attention,this.#expanded[63].mlp,Plain),
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
