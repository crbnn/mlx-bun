/** CPU-only: LEAN=/path/to/lean bun proofs/check.ts [--self-test].
 * Lake resolves the pinned FloatLib/mathlib proof dependencies in proofs/.
 * No MLX import. */
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, dirname, join } from "node:path";

const root = new URL("../", import.meta.url);
const lean = process.env.LEAN ?? "lean";
const lake = process.env.LAKE ?? join(dirname(lean), "lake");
let dependencyPath = "";
// Reviewed manual translations, not a verified Metal/TypeScript frontend. Pins
// prevent source drift; changing a hash requires reviewing the affected proofs.
const sourcePins = {
  "layers/quantized-embedding.ts": "cdead9d17af0a47f7f178ee4bd6b2710364a485dd7001e4ca98c62dec991fe4c",
  "models/qwen/qwen38-27b-trellis-m1max.ts": "915f129881b5ce85f97b6bca5c98afe06dedc21be514a81137a75e058cac882b",
  "state/kv.ts": "526a42d976c7dd0a8e908dcf8b515b94d43269042dc3fc3023e1a5baf0d8a579",
  "state/attention-read.ts": "8081b76f6bd7676825511476c2275601c0c34cb5b68ef30637c7a1c8f9f1c50c",
  "layers/affine-verify-linear.ts": "ee519b24ad34a0322dee6f35d9a0ed9e3c37a367b78a6b09bd4d0353c81a3870",
  "layers/quantized-linear.ts": "bf1c24c17d04951c5ea296bcd2bf34f818ffd7aefc684793e7792a8e5c25e798",
  "kernels/quantization/affine-row.ts": "cb5bd05e24a13547f5672cbe661c0262bc18969eea62266b185fcc3e68a1d20b",
  "state/ssm.ts": "96b6f33dac6da5f75cfba8ab760b4dcdce92c73b88d4927b8afa1b413896a7d1",
  "kernels/delta/silu-rms-heads.ts": "52c59772bf95e2cd8f7f813a69a7dfac0c352dadc3c2a1e0e550c054dd301145",
  "kernels/residual-rms-norm.ts": "27353e109e7f91bc030fbd493785fcc4839eeccea2d585c22d610ba8f9e909d0",
  "kernels/gated-rms-norm.ts": "e402118d33622ef62a017e964c257e23456d03f96b0fdc855b48704ff0fb954f",
  "layers/delta-heads.ts": "3d44375dd8bf8be8ea6ebf9772a4c3f1321fa22f567e9e9c7e7da11301a87a6d",
  "kernels/delta/gated.ts": "3f0e11f777f57b6595d57d62ec3f8b8331cbfa7263183e1baa98892efe935aec",
  "models/qwen/blocks.ts": "2fa644511dbafc027cf4f191be69e932abda7d8c6a92ebe45f7d76332324645f",
  "layers/loading.ts": "8dba0d4a34db80b373fb99f36b8f72cfc4afc77f96241101b1c4352ac32ee550",
  "layers/normalization.ts": "3da742f8b9321aea3e05699361bb82a425b9c548f1a63340e2ca96fc7b8108c8",
  "kernels/trellis/codebook.ts": "0ba8624952767d077cdecc531fd9f49a159545b95ae4fc4ddd213ed13fa1e276",
  "kernels/trellis/down-codebook.ts": "5649124f12f6828241d88bb86f9e9f6bf965af2b3b5a298ddd8a2973277eec55",
  "kernels/trellis/gate-up-codebook.ts": "c0aec9b9f951c5fd94270a52f4bf4da43e7fb7d0ecda6ccd65e20571e8931f45",
  "kernels/trellis/launch.ts": "da0a19922ccc93935c5cab869de040d4b759271df08496c88c7371c386724052",
  "layers/trellis-codebook-gate-up.ts": "28d27e84cb669c18a4532723e8dbc66786e386a4390454121640a3e14e5526f5",
  "layers/trellis-codebook-down.ts": "48752c429a3ab558131f95c6d361b15b6795c3bb6ee161f8afdca71609437898",
  "models/qwen/written-layers.ts": "ebf7710ffdf202d3e25ec7721930e5d992df31fb3d7fa5bde9709aa7a608c2c9",
  "models/qwen/checkpoint.ts": "1ec846424a2d4e5a046bcb62b43a5e6b8e1f3d9c2b4b3f5f3d7180276362edd8",
  "kernels/sampling/normalized-argmax.ts": "1578cff1ef678cbff3e2e9a754c6d7ba369e455885127e8d15c8b61f22a87740",
};
// Exact upstream source for the reviewed MLX arithmetic and kernel translations.
// These pins do not establish correspondence with the bundled native binary.
const mlxRevision = "1f8e74e3f12f31365464a6867c6579f0e9b29d85";
const mlxSourcePins = {
  "mlx/backend/metal/kernels/unary_ops.h": "0a5492b65ae39ecb6d8b04e64ea5007e0a4ff60d8d9428559bfbfd03387ece2a",
  "mlx/backend/metal/kernels/bf16.h": "abd87446a310b77ac530ef52a324feae5cb285d03ec9613e3a88ebb71410fdcb",
  "mlx/backend/metal/scaled_dot_product_attention.cpp": "215ebac6495706cf01a2635c4dcec29802e130165ed8704f212d789679d2e673",
  "mlx/backend/metal/kernels/scaled_dot_product_attention.metal": "f6fefad1d91b01f05c12095e69f248e255f880d65b5ae9e9d8bc2714da56fb41",
  "mlx/backend/metal/kernels/sdpa_vector.h": "de098d50a67a865e2e64fb1700fc4819307861f009a65642679777a9a9cec7cd",
  "mlx/backend/metal/rope.cpp": "9ec2d2f298381af6445b1e2281dd90521988861ed170a602944ec0f1fca17ecc",
  "mlx/backend/metal/kernels/rope.metal": "2b35221ebea033ff43100b85eaaddec5c0e34e3cfd52e7c692573764b2b7d0ab",
  "mlx/backend/metal/conv.cpp": "ffec76d32b453b232245978ee067fac70a0205981d9ff38311e19db422dc0758",
  "mlx/backend/metal/kernels/conv.metal": "d0604f69e144ffb65dcfc99e6383880b6d2d61645fa73147f35230034a1cd874",
  "mlx/backend/metal/quantized.cpp": "44483286a359afb7f35d8a9c1658ffaec6ff6c01a7ab34ac7bb35cf6438804a6",
  "mlx/backend/metal/kernels/quantized.h": "2a007016da606afe569adb9adcc05e00f14558ad2e094bcb4f8974beb53c316f",
  "mlx/ops.cpp": "6eae219354d0f9be37a064cb327c863fc3a12dba6668fd61d428e8d09d6baaf2",
  "mlx/backend/metal/logsumexp.cpp": "e629a1fa0b8744c7ee7913ccdb37caf6d22ad76aa6945282aa6adc4afdd316e7",
  "mlx/backend/metal/kernels/logsumexp.h": "233f9a7072bed21f2b71f522c41aa5e86208e3bcfd18fafc87cf28a7d7f13154",
  "mlx/backend/metal/kernels/logsumexp.metal": "f9bec5e1e5a23d20bedf9ff8d29a8c03bbb5144bc5d751bbfe906d32ee894817",
};
const proofFiles = [
  "contracts/mlx/M1Max.lean",
  "contracts/mlx/Memory.lean",
  "contracts/mlx/Numerics.lean",
  "contracts/mlx/FlushToZero.lean",
  "contracts/mlx/Arithmetic.lean",
  "contracts/mlx/Float32Bounds.lean",
  "contracts/mlx/NativeDivisionBounds.lean",
  "contracts/mlx/Float32UnitInterval.lean",
  "contracts/mlx/NativeMultiplication.lean",
  "contracts/mlx/NativeFma.lean",
  "contracts/mlx/ArithmeticBounds.lean",
  "contracts/mlx/IntegerBounds.lean",
  "contracts/mlx/Division.lean",
  "contracts/mlx/BFloat16.lean",
  "contracts/mlx/DyadicBounds.lean",
  "contracts/mlx/DyadicCover.lean",
  "contracts/mlx/Negation.lean",
  "contracts/mlx/BFloat16Arithmetic.lean",
  "contracts/mlx/TensorWords.lean",
  "contracts/mlx/BFloat16Fma.lean",
  "contracts/mlx/BFloat16Bounds.lean",
  "contracts/mlx/BFloat16UnitInterval.lean",
  "contracts/mlx/BFloat16DyadicBounds.lean",
  "contracts/mlx/BFloat16ConversionBounds.lean",
  "contracts/mlx/DivisionBounds.lean",
  "contracts/mlx/DivisionDirectedBounds.lean",
  "contracts/mlx/ElementwiseAdd.lean",
  "contracts/mlx/Reduction.lean",
  "contracts/mlx/FlushedReduction.lean",
  "contracts/mlx/ArithmeticFold.lean",
  "contracts/mlx/ArithmeticFoldBounds.lean",
  "contracts/mlx/ArithmeticVaryingBounds.lean",
  "contracts/mlx/AdditionFoldBounds.lean",
  "contracts/mlx/ArithmeticDependencies.lean",
  "contracts/mlx/DotProduct.lean",
  "contracts/mlx/FlushedDotProduct.lean",
  "contracts/mlx/FlushedDotProductBounds.lean",
  "contracts/mlx/SquaredSum.lean",
  "contracts/mlx/SquaredSumBounds.lean",
  "contracts/mlx/SquaredSumDyadicBounds.lean",
  "contracts/mlx/NormalizationSchedule.lean",
  "contracts/mlx/NormalizationReduction.lean",
  "contracts/mlx/Normalization.lean",
  "contracts/mlx/NormalizationExecution.lean",
  "contracts/mlx/RmsNorm.lean",
  "contracts/mlx/NormalizationBounds.lean",
  "contracts/mlx/NormalizationDyadicBounds.lean",
  "contracts/mlx/NormalizationFloorBounds.lean",
  "contracts/mlx/RsqrtBounds.lean",
  "contracts/mlx/NormalizationUniform.lean",
  "kernels/trellis/Codebook.lean",
  "kernels/trellis/CodebookLoad.lean",
  "kernels/trellis/CodebookFlushToZero.lean",
  "kernels/trellis/CodebookAccuracy.lean",
  "kernels/trellis/CodebookRanges.lean",
  "kernels/trellis/CodebookInitialization.lean",
  "kernels/trellis/DownCodebook.lean",
  "kernels/trellis/GateUpCodebook.lean",
  "kernels/trellis/DownExecution.lean",
  "kernels/trellis/DownMemory.lean",
  "kernels/trellis/DownAccuracy.lean",
  "kernels/trellis/DownReduction.lean",
  "kernels/trellis/GateUpExecution.lean",
  "kernels/trellis/GateUpAccuracy.lean",
  "kernels/trellis/GateUpFlushToZero.lean",
  "kernels/trellis/DownFlushToZero.lean",
  "kernels/trellis/GateUpTail.lean",
  "kernels/trellis/SwiGLUStability.lean",
  "kernels/trellis/GateUpTailFlushToZero.lean",
  "kernels/trellis/GateUpSplitTail.lean",
  "kernels/StableSigmoid.lean",
  "kernels/StableSigmoidRanges.lean",
  "kernels/quantization/AffinePacking.lean",
  "kernels/quantization/AffineLift.lean",
  "kernels/quantization/AffineDequantize.lean",
  "kernels/quantization/AffineDequantizeBounds.lean",
  "kernels/quantization/AffineInput.lean",
  "kernels/quantization/AffineInputBounds.lean",
  "kernels/quantization/AffineSumBounds.lean",
  "kernels/quantization/AffineDot.lean",
  "kernels/quantization/AffineDotBounds.lean",
  "kernels/quantization/AffineTile.lean",
  "kernels/quantization/AffineTileBounds.lean",
  "kernels/quantization/AffineValueBounds.lean",
  "kernels/quantization/AffineScaledBounds.lean",
  "kernels/quantization/AffineShiftedDotBounds.lean",
  "kernels/quantization/AffineShiftedTileBounds.lean",
  "kernels/quantization/AffineLayout.lean",
  "kernels/quantization/AffineRow.lean",
  "kernels/quantization/AffineRowBounds.lean",
  "kernels/quantization/MlxQmv.lean",
  "kernels/delta/Gates.lean",
  "kernels/trellis/SigmoidRanges.lean",
  "kernels/trellis/GateUpSplitTailRanges.lean",
  "kernels/delta/Softplus.lean",
  "kernels/delta/SoftplusRanges.lean",
  "kernels/delta/Decay.lean",
  "kernels/delta/DecayRanges.lean",
  "kernels/delta/Gated.lean",
  "kernels/delta/GatedRanges.lean",
  "kernels/delta/Convolution.lean",
  "kernels/delta/ConvolutionBounds.lean",
  "kernels/delta/Layout.lean",
  "kernels/delta/Recurrence.lean",
  "kernels/delta/Inputs.lean",
  "kernels/delta/SiluRmsHeads.lean",
  "kernels/delta/SiluRmsHeadsRanges.lean",
  "kernels/attention/Rope.lean",
  "kernels/attention/RopeAngles.lean",
  "kernels/attention/OnlineSoftmax.lean",
  "kernels/attention/SdpaUpdate.lean",
  "kernels/attention/SdpaLoop.lean",
  "kernels/attention/SdpaNormalize.lean",
  "kernels/attention/SdpaScore.lean",
  "kernels/attention/SdpaMerge.lean",
  "kernels/attention/SdpaLayout.lean",
  "kernels/attention/SdpaVector.lean",
  "kernels/attention/OutputGate.lean",
  "kernels/ResidualRmsNorm.lean",
  "kernels/ResidualRmsNormRanges.lean",
  "kernels/GatedRmsNorm.lean",
  "kernels/GatedRmsNormRanges.lean",
  "layers/TrellisCodebookGateUp.lean",
  "layers/TrellisCodebookDown.lean",
  "layers/TrellisCodebookRanges.lean",
  "layers/Normalization.lean",
  "layers/DeltaHeads.lean",
  "layers/AffineVerifyLinear.lean",
  "layers/AffineVerifyLinearBounds.lean",
  "layers/AffineVerifyLinearShiftedBounds.lean",
  "layers/QuantizedEmbedding.lean",
  "state/Ssm.lean",
  "state/Kv.lean",
  "models/qwen/WrittenLayers.lean",
  "models/qwen/Blocks.lean",
  "models/qwen/NormalizationRanges.lean",
  "models/qwen/NormalizationUniformBounds.lean",
  "models/qwen/NormalizedAffineProjection.lean",
  "models/qwen/AttentionHeads.lean",
  "models/qwen/DeltaNet.lean",
  "models/qwen/TrellisMlp.lean",
  "models/qwen/TrellisMlpRanges.lean",
  "models/qwen/ResidualBlocks.lean",
  "models/qwen/ResidualMlpRanges.lean",
  "models/qwen/DeltaCache.lean",
  "models/qwen/DeltaInputRanges.lean",
  "models/qwen/DeltaHeadRanges.lean",
  "models/qwen/DeltaGateRanges.lean",
  "models/qwen/DeltaInitialRecurrenceRanges.lean",
  "models/qwen/DeltaOutputRanges.lean",
  "models/qwen/DeltaInitialRanges.lean",
  "models/qwen/ResidualDelta.lean",
  "models/qwen/ResidualDeltaInitialRanges.lean",
  "models/qwen/Attention.lean",
  "models/qwen/ResidualAttention.lean",
  "models/qwen/DecodeStage.lean",
  "models/qwen/DecodeChain.lean",
  "models/qwen/Qwen38TrellisM1Max.lean",
  "models/qwen/TokenInference.lean",
  "models/qwen/EmbeddingRanges.lean",
  "models/qwen/EmbeddingNormalization.lean",
  "models/qwen/InitialState.lean",
  "contracts/mlx/Float32OverflowMargin.lean",
  "contracts/mlx/FlushedIdentities.lean",
  "kernels/attention/OutputGateRanges.lean",
  "kernels/attention/RopeAnglesRanges.lean",
  "kernels/attention/RopeRanges.lean",
  "kernels/attention/SdpaExponentRanges.lean",
  "kernels/attention/SdpaScoreRanges.lean",
  "kernels/attention/SdpaInitialUpdateRanges.lean",
  "kernels/attention/SdpaInitialLoopRanges.lean",
  "kernels/attention/SdpaMergeRanges.lean",
  "kernels/attention/SdpaInitialRanges.lean",
  "models/qwen/AttentionInputRanges.lean",
  "models/qwen/AttentionInitialHeadsRanges.lean",
  "models/qwen/AttentionInitialRanges.lean",
  "models/qwen/ResidualAttentionInitialRanges.lean",
  "models/qwen/DecodeStageInitialRanges.lean",
  "models/qwen/DecodeChainInitialRanges.lean",
  "models/qwen/Qwen38TrellisM1MaxInitialRanges.lean",
  "models/qwen/TokenInitialRanges.lean",
  "models/qwen/CheckpointTransforms.lean",
  "models/qwen/CheckpointRanges.lean",
  "models/qwen/CheckpointNormalizationData.lean",
  "models/qwen/CheckpointNormalizationBinding.lean",
  "contracts/mlx/Argmax.lean",
  "kernels/sampling/ArgmaxPairs.lean",
  "kernels/sampling/NormalizedScores.lean",
  "kernels/sampling/NormalizedArgmaxLayout.lean",
  "kernels/sampling/NormalizedArgmaxExecution.lean",
  "contracts/mlx/Maximum.lean",
  "contracts/mlx/FlushOrder.lean",
  "contracts/mlx/NonnegativeArithmetic.lean",
  "kernels/sampling/LogSumExpLayout.lean",
  "kernels/sampling/LogSumExpExponentRanges.lean",
  "kernels/sampling/LogSumExpAccumulator.lean",
  "kernels/sampling/LogSumExpBlockRanges.lean",
  "kernels/sampling/LogSumExpLoopRanges.lean",
  "kernels/sampling/LogSumExpMergeRanges.lean",
  "kernels/sampling/LogSumExpRanges.lean",
  "kernels/sampling/GreedySampling.lean",
  "models/qwen/TokenSampledInitialRanges.lean",
  "contracts/mlx/RelativeRounding.lean",
  "contracts/mlx/RsqrtRelativeBounds.lean",
  "contracts/mlx/NormalizedCastBounds.lean",
  "contracts/mlx/SquaredSumRelativeBounds.lean",
  "kernels/delta/RecurrenceStability.lean",
  "kernels/delta/SiluRmsHeadsSumBounds.lean",
  "kernels/delta/SiluRmsHeadsEnergy.lean",
  "models/qwen/DeltaHeadEnergy.lean",
  "kernels/delta/RecurrenceNumericalStability.lean",
  "kernels/delta/GatedRecurrentRanges.lean",
  "models/qwen/DeltaRecurrentInputs.lean",
  "kernels/delta/RecurrenceGrowth.lean",
  "contracts/mlx/ContractiveMultiplication.lean",
  "kernels/delta/DecayStateBounds.lean",
  "contracts/mlx/FlushedReductionRelative.lean",
  "contracts/mlx/ArithmeticMaddRelative.lean",
  "kernels/delta/GatedDotRelative.lean",
  "kernels/delta/GatedPredictionRelative.lean",
  "contracts/mlx/SubtractionRelative.lean",
  "kernels/delta/GatedDeltaRelative.lean",
  "kernels/delta/GatedUpdateRelative.lean",
  "kernels/delta/RecurrenceSupportedRanges.lean",
  "kernels/delta/RecurrenceReadoutRanges.lean",
  "kernels/delta/RecurrenceSupportedError.lean",
  "models/qwen/DeltaSequenceRecurrenceRanges.lean",
  "models/qwen/DeltaSequenceRanges.lean",
  "models/qwen/AttentionSequenceHeadsRanges.lean",
  "contracts/mlx/TwoProductsRelative.lean",
  "kernels/attention/SdpaOrderedExponentRanges.lean",
  "kernels/attention/SdpaUpdateRelative.lean",
  "kernels/attention/SdpaUpdateBounds.lean",
  "kernels/attention/SdpaLoopBounds.lean",
  "kernels/attention/SdpaSequenceLoopRanges.lean",
  "contracts/mlx/WeightedReductionRelative.lean",
  "kernels/attention/SdpaMergeRelative.lean",
  "kernels/attention/SdpaSequenceRanges.lean",
  "models/qwen/AttentionSequenceRanges.lean",
  "kernels/delta/RecurrenceEnvelope.lean",
  "models/qwen/DeltaDecodeRecurrenceRanges.lean",
  "models/qwen/DeltaDecodeRanges.lean",
  "models/qwen/ResidualDeltaDecodeRanges.lean",
  "models/qwen/ResidualAttentionSequenceRanges.lean",
  "models/qwen/DecodeStageSequenceRanges.lean",
  "models/qwen/DecodeChainSequenceRanges.lean",
  "models/qwen/Qwen38TrellisM1MaxSequenceRanges.lean",
  "models/qwen/TokenSequenceRanges.lean",
  "models/qwen/TokenSampledSequenceRanges.lean",
  "models/qwen/TokenAutoregressive.lean",
];

const discoveredProofs = await Array.fromAsync(new Bun.Glob("**/*.lean").scan({
  cwd: new URL("./", import.meta.url).pathname, onlyFiles: true, dot: false,
}));
for (const file of discoveredProofs) {
  if (!proofFiles.includes(file)) throw new Error(`Proof module is missing from the checked build: ${file}`);
}

function checkSource(name: string, source: string, expected: string): void {
  if (new Bun.CryptoHasher("sha256").update(source).digest("hex") !== expected) {
    throw new Error(`${name} changed: review its translation and proofs before updating the source pin.`);
  }
}

// Audit every theorem in the current module, including private helpers and newly
// added theorems. collectAxioms walks transitive dependencies in the kernel env.
const audit = `
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let names := env.constants.foldStage2 (fun ns name info =>
    if info.isTheorem then ns.push name else ns) (#[] : Array Name)
  if names.isEmpty then throwError "No proof declarations found"
  for name in names do
    let axioms ← collectAxioms name
    for axiomName in axioms do
      unless #[\`propext, \`Classical.choice, \`Quot.sound].contains axiomName do
        throwError "Forbidden axiom {axiomName} in {name}"
  logInfo m!"Axiom audit passed: {names.size} theorems"
`;

async function run(args: string[], leanPath?: string) {
  const child = Bun.spawn([lean, ...args], {
    cwd: root.pathname, stdout: "pipe", stderr: "pipe",
    env: { ...process.env, ...(leanPath ? { LEAN_PATH: [leanPath, dependencyPath].filter(Boolean).join(delimiter) } : {}) },
  });
  const [stdout, stderr, code] = await Promise.all([
    new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited,
  ]);
  return { output: stdout + stderr, code };
}

const version = await run(["--version"]);
if (version.code !== 0 || !version.output.includes("version 4.35.0,")) {
  throw new Error(`Expected Lean 4.35.0; got ${version.output}`);
}
async function runLake(args: string[]) {
  const child = Bun.spawn([lake, ...args], {
    cwd: new URL("./", import.meta.url).pathname, stdout: "pipe", stderr: "pipe",
    env: { ...process.env, MATHLIB_NO_CACHE_ON_UPDATE: "1" },
  });
  const [stdout, stderr, code] = await Promise.all([
    new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited,
  ]);
  if (code !== 0) throw new Error(`Lake failed: ${stdout}${stderr}`);
  return stdout;
}
console.log("Building pinned numerical proof dependencies...");
process.stdout.write(await runLake(["build",
  "FloatLib.Floats.Formats.BinaryInterchange.Analysis.StandardModel",
  "FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.LeanModel.MulDiv",
  "FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.SqrtSemantics",
  "FloatLib.Floats.Formats.BinaryInterchange.IntervalSemantics.Order",
  "FloatLib.Floats.Formats.IEEE754.Native.Model",
  "FloatLib.Floats.Formats.IEEE754.Native.AddSub",
  "FloatLib.Floats.Formats.IEEE754.Native.Integer.Constructors",
  "FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Exact",
  "FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Proof",
  "FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.Constants",
  "FloatLib.Floats.Formats.BinaryInterchange.Operations.Classification.Semantics",
  "FloatLib.Floats.Formats.BinaryInterchange.Operations.Compare.Proof",
  "Mathlib.Analysis.SpecialFunctions.Exp",
  "Mathlib.Analysis.SpecialFunctions.ExpDeriv",
  "Mathlib.Analysis.SpecialFunctions.Log.Deriv",
  "Mathlib.Analysis.Calculus.MeanValue",
  "Mathlib.Analysis.Calculus.Deriv.Inv",
  "Mathlib.Analysis.Real.Sqrt",
  "Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds",
  "Mathlib.Data.List.FinRange",
  "Mathlib.Data.List.Nodup",
  "Mathlib.Algebra.Order.BigOperators.Ring.Finset",
  "Mathlib.Analysis.Complex.ExponentialBounds",
]));
dependencyPath = (await runLake(["env", "printenv", "LEAN_PATH"])).trim();
const sources = new Map<string, { source: string; expected: string }>();
for (const [name, expected] of Object.entries(sourcePins)) {
  const source = await Bun.file(new URL(`packages/inference/src/${name}`, root)).text();
  checkSource(name, source, expected);
  sources.set(name, { source, expected });
}
for (const [name, expected] of Object.entries(mlxSourcePins)) {
  const cached = new URL(`.lake/sources/mlx/${mlxRevision}/${name}`, import.meta.url);
  let source: string;
  if (await Bun.file(cached).exists()) {
    source = await Bun.file(cached).text();
  } else {
    const response = await fetch(`https://raw.githubusercontent.com/ml-explore/mlx/${mlxRevision}/${name}`);
    if (!response.ok) throw new Error(`Cannot fetch pinned MLX source ${name}: HTTP ${response.status}`);
    source = await response.text();
    checkSource(name, source, expected);
    await mkdir(dirname(cached.pathname), { recursive: true });
    await Bun.write(cached, source);
  }
  checkSource(name, source, expected);
  sources.set(`mlx@${mlxRevision}/${name}`, { source, expected });
}

const dir = await mkdtemp(join(tmpdir(), "mlx-bun-proof-"));
async function checkProof(name: string, text: string, build = false) {
  const path = join(dir, name);
  await mkdir(dirname(path), { recursive: true });
  await Bun.write(path, `import Lean\n${text}\n${audit}`);
  return run(["-R", dir, ...(build ? ["-o", path.replace(/\.lean$/, ".olean")] : []), path], dir);
}

try {
  for (const file of proofFiles) {
    console.log(`Checking ${file}...`);
    const result = await checkProof(file, await Bun.file(new URL(file, import.meta.url)).text(), true);
    process.stdout.write(result.output);
    if (result.code !== 0 || result.output.includes("sorryAx") || !result.output.includes("Axiom audit passed:")) {
      throw new Error(`Lean did not establish ${file} without forbidden assumptions.`);
    }
  }

  if (process.argv.includes("--self-test")) {
    for (const [name, { source, expected }] of sources) {
      let rejected = false;
      try { checkSource(name, source + "\n// source mutation\n", expected); } catch { rejected = true; }
      if (!rejected) throw new Error(`Source mutation was accepted: ${name}`);
    }
    // Small prefix retains the arithmetic contract and its dependencies, avoiding
    // repeated exhaustive floating-point checking for deliberately broken proofs.
    const source = await Bun.file(new URL("kernels/trellis/Codebook.lean", import.meta.url)).text();
    const proof = source.split("namespace PackedState")[0]! + "\nend Inference.Kernels.Trellis.Codebook\n";
    const wrongClaim = proof.replace("implementation s = specification s :=", "implementation s = specification s + 1 :=");
    const assumedClaim = proof.replace(
      /theorem correct[\s\S]*?(?=\ntheorem range)/,
      "axiom injected : ∀ s, implementation s = specification s\n" +
      "theorem correct (s : BitVec 32) : implementation s = specification s := injected s\n",
    );
    const nativeProof = proof.replace("import Std\n", "import Std.Tactic.BVDecide\n").replace(
      /theorem packedSum_bound[\s\S]*?(?=\ntheorem correct)/,
      "theorem packedSum_bound (x : BitVec 32) : packedSum x ≤ 1020 := by\n  unfold packedSum\n  bv_decide\n",
    );
    const uncheckedTheorem = proof + "\nnamespace Inference\naxiom newAssumption : False\ntheorem newClaim : False := newAssumption\nend Inference\n";
    const proofHole = proof + "\nnamespace Inference\ntheorem missingProof : False := by sorry\nend Inference\n";
    for (const [name, text, axiomCheck] of [
      ["FalseContract", wrongClaim, false],
      ["AssumedContract", assumedClaim, true],
      ["NativeComputation", nativeProof, true],
      ["NewUncheckedTheorem", uncheckedTheorem, true],
      ["ProofHole", proofHole, true],
    ] as const) {
      if (proof === text) throw new Error(`Negative proof test did not alter its target: ${name}`);
      const rejected = await checkProof(`${name}.lean`, text);
      if (rejected.code === 0) throw new Error(`Lean accepted ${name}.`);
      if (axiomCheck && !rejected.output.includes("Forbidden axiom")) {
        throw new Error(`${name} failed before the axiom audit:\n${rejected.output}`);
      }
    }
    console.log("Negative checks passed: source drift, false claims, custom axioms, native computation, unaudited additions, and proof holes rejected.");
  }
} finally {
  await rm(dir, { recursive: true, force: true });
}
console.log(`Checked ${proofFiles.length} proof modules: Trellis and affine numerical bounds, MLX fast-qmv source-model equivalence, packed windows, conditional M1 loads, scratch stores, flushed binary32/bfloat16 arithmetic and range bounds, SwiGLU and sigmoid, RMS normalization, Delta convolution/recurrence/cache, normalized-key stability and 262144-step Delta layer bounds, Q/K normalization and partial RoPE, full-context one-pass SDPA bounds and scratch transpose, plain KV append, quantized embedding, residual Delta/attention layer composition, initial cache state, conditional first-token greedy selection with returned-cache bounds, and looped logsumexp/normalized argmax source models.`);
console.log("Permitted axioms: propext, Classical.choice, Quot.sound. No native proof evaluation.");
console.log("Scope: the autoregressive source model covers embedding, all 64 layers, final normalization, logits, 1024-thread looped logsumexp, normalized greedy selection and returned-cache error bounds through 262144 steps, conditional on parameter and scalar/platform contracts. Each actual selected token and returned cache feeds the following call. Induction derives all intermediate range conditions, including the recurrent state and one-pass attention. Selection maximizes the rounded normalized score with lowest-index ties. Error bounds compare against a mathematical reference consuming the same emitted tokens; agreement with an independently generated exact-real token sequence requires per-step numerical margins. Attention bounds include nonzero-position RoPE, online-softmax numerator/denominator stability, group merge, division, gating, projection and KV append. This does not establish native dispatch to the modeled one-pass kernel. Normalization literals have checked range certificates; their host attachment checker is outside Lean. Full checkpoint/loader binding, a useful final accuracy bound, native memory/dispatch and hardware/compiler correspondence remain unproved.");
