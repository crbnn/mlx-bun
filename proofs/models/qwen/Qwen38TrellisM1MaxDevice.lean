import models.qwen.AutoregressiveImplementation
import models.qwen.TrellisMlpDevice
import models.qwen.Qwen38TrellisM1MaxMagnitude

namespace Inference.Models.Qwen.Qwen38TrellisM1MaxDevice
open Contracts.Mlx Kernels.Sampling TokenInference
open FloatLib FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 10000

abbrev Schedules (p : Pipeline) :=
  AutoregressiveImplementation.Schedules TrellisMlpDevice.implementation p RopeInputConstruction.angles

noncomputable def Schedules.canonical (p : Pipeline) : Schedules p :=
  AutoregressiveImplementation.Schedules.canonical TrellisMlpDevice.implementation p RopeInputConstruction.angles

-- A fixed proved MLP implementation is used at every stage. Its traces
-- include byte stores, producer completion, BF16 activation readback and
-- the two-launch down reduction. The other kernels retain their existing
-- source translations; no compiled-binary correspondence is asserted here.
noncomputable def run (p : Pipeline) (encoded : CheckpointDictionaryBinding.Encoding)
    (binding : CheckpointDictionaryBinding.Bindings p encoded)
    (primitive : Qwen38TrellisM1MaxSequenceRanges.Primitives p.graph)
    (sampling : LogSumExpRanges.Platform) (samplingPrimitive : LogSumExpRanges.Primitives sampling)
    (token : Fin 248320) (schedules : Schedules p) (steps : Nat) (supported : steps ≤ 262144) : TokenAutoregressive.State p steps :=
  AutoregressiveImplementation.run TrellisMlpDevice.implementation p (CheckpointDictionaryBinding.pipeline_parameters p encoded binding)
    primitive sampling samplingPrimitive token RopeInputConstruction.angles schedules steps supported

theorem run_eq (p : Pipeline) (encoded : CheckpointDictionaryBinding.Encoding)
    (binding : CheckpointDictionaryBinding.Bindings p encoded)
    (primitive : Qwen38TrellisM1MaxSequenceRanges.Primitives p.graph)
    (sampling : LogSumExpRanges.Platform) (samplingPrimitive : LogSumExpRanges.Primitives sampling)
    (token : Fin 248320) (schedules : Schedules p) (steps : Nat) (supported : steps ≤ 262144) :
    run p encoded binding primitive sampling samplingPrimitive token schedules steps supported=
      Qwen38TrellisM1MaxInference.run p encoded binding primitive sampling samplingPrimitive token steps supported :=
  AutoregressiveImplementation.run_eq TrellisMlpDevice.implementation p (CheckpointDictionaryBinding.pipeline_parameters p encoded binding)
    primitive sampling samplingPrimitive token RopeInputConstruction.angles schedules steps supported

theorem step_execution (p : Pipeline) (encoded : CheckpointDictionaryBinding.Encoding)
    (binding : CheckpointDictionaryBinding.Bindings p encoded)
    (primitive : Qwen38TrellisM1MaxSequenceRanges.Primitives p.graph)
    (sampling : LogSumExpRanges.Platform) (samplingPrimitive : LogSumExpRanges.Primitives sampling)
    (token : Fin 248320) (schedules : Schedules p) (time : Nat) (supported : time+1 ≤ 262144) :
    let before := run p encoded binding primitive sampling samplingPrimitive token schedules time (by omega)
    let after := run p encoded binding primitive sampling samplingPrimitive token schedules (time+1) supported
    let output := DecodeImplementation.pipeline TrellisMlpDevice.implementation p before.cache before.token
      (RopeInputConstruction.angles.value time) (schedules time before.cache before.token)
    after.token.val=GreedySampling.sample sampling output.1 ∧ after.cache=output.2 := by
  dsimp only
  rw [DecodeImplementation.pipeline_eq]
  simp only [run_eq]
  simpa only [TokenSampledSequenceRanges.run, RopeInputConstruction.angles] using
    Qwen38TrellisM1MaxInference.step_execution p encoded binding primitive sampling samplingPrimitive token time supported

theorem step_correct (p : Pipeline) (encoded : CheckpointDictionaryBinding.Encoding)
    (binding : CheckpointDictionaryBinding.Bindings p encoded)
    (primitive : Qwen38TrellisM1MaxSequenceRanges.Primitives p.graph)
    (sampling : LogSumExpRanges.Platform) (samplingPrimitive : LogSumExpRanges.Primitives sampling)
    (token : Fin 248320) (schedules : Schedules p) (time : Nat) (supported : time+1 ≤ 262144) :
    let parameters := CheckpointDictionaryBinding.pipeline_parameters p encoded binding
    let before := run p encoded binding primitive sampling samplingPrimitive token schedules time (by omega)
    let after := run p encoded binding primitive sampling samplingPrimitive token schedules (time+1) supported
    let logits := (DecodeImplementation.pipeline TrellisMlpDevice.implementation p before.cache before.token
      (RopeInputConstruction.angles.value time) (schedules time before.cache before.token)).1
    (∀ i, |Model.toReal (logits i)-(p.targetRun before.targetCache before.token (RopeInputConstruction.target time)).1 i| ≤
      (TokenSampledSequenceRanges.budget p parameters primitive before.cache before.cacheBound before.targetCache
        before.errorCache before.token (RopeInputConstruction.inputs time) (RopeInputConstruction.inputs_bound time (by omega))
          supported (RopeInputConstruction.target time) RopeInputConstruction.error).1 i) ∧
    LogSumExpRanges.Trace sampling (fun i => BFloat16.widen (logits i)) ∧
    (∀ i, (NormalizedScores.value (logits i) (LogSumExpRanges.value sampling logits)).isFinite=true) ∧
    (∀ i, GreedySampling.score sampling logits i ≤ GreedySampling.score sampling logits after.token) ∧
    (∀ i, GreedySampling.score sampling logits after.token=GreedySampling.score sampling logits i → after.token.val ≤ i.val) := by
  dsimp only
  rw [DecodeImplementation.pipeline_eq]
  simp only [run_eq]
  simpa only [TokenSampledSequenceRanges.logits, RopeInputConstruction.angles] using
    Qwen38TrellisM1MaxInference.step_correct p encoded binding primitive sampling samplingPrimitive token time supported

theorem output_bound (p : Pipeline) (encoded : CheckpointDictionaryBinding.Encoding)
    (binding : CheckpointDictionaryBinding.Bindings p encoded)
    (primitive : Qwen38TrellisM1MaxSequenceRanges.Primitives p.graph)
    (sampling : LogSumExpRanges.Platform) (samplingPrimitive : LogSumExpRanges.Primitives sampling)
    (token : Fin 248320) (schedules : Schedules p) (time : Nat) (supported : time+1 ≤ 262144) :
    let before := run p encoded binding primitive sampling samplingPrimitive token schedules time (by omega)
    let logits := (DecodeImplementation.pipeline TrellisMlpDevice.implementation p before.cache before.token
      (RopeInputConstruction.angles.value time) (schedules time before.cache before.token)).1
    ∀ i, Model.isFinite (logits i)=true ∧ |Model.toReal (logits i)| ≤ (2 : ℝ)^18 := by
  dsimp only
  rw [DecodeImplementation.pipeline_eq]
  simp only [run_eq]
  simpa only [TokenSampledSequenceRanges.logits, RopeInputConstruction.angles] using
    Qwen38TrellisM1MaxMagnitude.inference_output p encoded binding primitive sampling samplingPrimitive token time supported

theorem schedule_independent (p : Pipeline) (encoded : CheckpointDictionaryBinding.Encoding)
    (binding : CheckpointDictionaryBinding.Bindings p encoded)
    (primitive : Qwen38TrellisM1MaxSequenceRanges.Primitives p.graph)
    (sampling : LogSumExpRanges.Platform) (samplingPrimitive : LogSumExpRanges.Primitives sampling)
    (token : Fin 248320) (a b : Schedules p) (steps : Nat) (supported : steps ≤ 262144) :
    run p encoded binding primitive sampling samplingPrimitive token a steps supported=
      run p encoded binding primitive sampling samplingPrimitive token b steps supported := by
  rw [run_eq, run_eq]

end Inference.Models.Qwen.Qwen38TrellisM1MaxDevice
