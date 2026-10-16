import models.qwen.DeltaConvolution
import kernels.delta.GatedOutput

namespace Inference.Models.Qwen.DeltaDevice
open Contracts.Mlx Kernels.Delta
abbrev BF := BFloat16Arithmetic.Value
set_option maxRecDepth 10000

structure Storage (cache : DeltaNet.Cache) where
  history : State.ConvolutionStorage.Storage cache.history
  recurrent : GatedDevice.Storage cache.recurrent

def Storage.canonical (cache : DeltaNet.Cache) : Storage cache :=
  ⟨State.ConvolutionStorage.Storage.canonical cache.history, GatedDevice.Storage.canonical cache.recurrent⟩

structure Execution (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache) where
  convolution : DeltaConvolution.Execution layer input cache
  recurrence : GatedDevice.Execution layer.recurrence.reduce layer.recurrence.flags cache.recurrent
    (DeltaConvolution.inputValues layer input cache convolution).inputs
  inputBuffers : HEq recurrence.inputStorage (DeltaConvolution.inputStorage layer input cache convolution)
  normalization : Kernels.GatedRmsNormDevice.Execution layer.gated layer.headParams.width layer.headParams.eps
    recurrence.output (DeltaNet.zHead layer (fun _ => input) 0) layer.normWeight
  normalizationInput : normalization.storage.input=GatedOutput.buffer recurrence

def Execution.storage (execution : Execution layer input cache) : Storage cache :=
  ⟨execution.convolution.convolution.convolution.old, execution.recurrence.storage⟩

def Execution.fromStorage (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (storage : Storage cache) : Execution layer input cache :=
  let convolution := DeltaConvolution.Execution.fromStorage layer input cache storage.history
  let recurrence : GatedDevice.Execution layer.recurrence.reduce layer.recurrence.flags cache.recurrent
      (DeltaConvolution.inputValues layer input cache convolution).inputs :=
    { values := DeltaConvolution.inputValues layer input cache convolution
      binding := rfl
      inputStorage := DeltaConvolution.inputStorage layer input cache convolution
      storage := storage.recurrent
      stores := GatedBytes.Trace.canonical _ _ }
  ⟨convolution, recurrence, HEq.rfl, GatedOutput.normalization layer.gated layer.headParams.width layer.headParams.eps
    recurrence (DeltaNet.zHead layer (fun _ => input) 0) layer.normWeight, rfl⟩

def Execution.canonical (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache) : Execution layer input cache :=
  Execution.fromStorage layer input cache (Storage.canonical cache)

theorem output_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : execution.recurrence.output=DeltaNet.recurrentOutput layer cache (fun _ => input) 0 := by
  rw [execution.recurrence.output_eq, DeltaConvolution.inputs_eq]
  unfold GatedDevice.producedOutput
  rfl

theorem recurrent_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : execution.recurrence.state=(DeltaCache.after layer cache (fun _ => input) 1).recurrent := by
  rw [execution.recurrence.state_eq, DeltaConvolution.inputs_eq]
  unfold GatedDevice.producedState
  rfl

def gated (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : Fin 48 → Fin 128 → BF := execution.normalization.output

theorem gated_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : gated layer input cache execution=DeltaNet.gatedOutput layer cache (fun _ => input) 0 := by
  unfold gated
  rw [execution.normalization.output_eq, output_eq]
  rfl

def merged (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) (i : Fin 6144) : BF :=
  gated layer input cache execution (DeltaNet.headOf i) (DeltaNet.valueOf i)

theorem merged_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : merged layer input cache execution=DeltaNet.merged layer cache (fun _ => input) 0 := by
  unfold merged
  rw [gated_eq]
  rfl

def forward (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : Fin 5120 → BF := layer.out.forward (merged layer input cache execution)

theorem forward_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : forward layer input cache execution=layer.forward cache (fun _ => input) 0 := by
  unfold forward
  rw [merged_eq]
  rfl

-- Both cache fields now come from completed byte stores.
def cacheAfter (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : DeltaNet.Cache :=
  ⟨execution.convolution.convolution.retained, execution.recurrence.state⟩

theorem cache_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : cacheAfter layer input cache execution=DeltaCache.after layer cache (fun _ => input) 1 := by
  unfold cacheAfter
  rw [DeltaConvolution.history_eq, recurrent_eq]

def storageAfter (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : Storage (cacheAfter layer input cache execution) :=
  ⟨execution.convolution.convolution.after, execution.recurrence.after⟩

theorem normalization_input_bytes (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : execution.normalization.storage.input.bytes=execution.recurrence.stores.outputMemory :=
  congrArg ArrayStorage.BFBuffer.bytes execution.normalizationInput

def run (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : (Fin 5120 → BF) × DeltaNet.Cache :=
  (forward layer input cache execution, cacheAfter layer input cache execution)

theorem run_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) :
    run layer input cache execution=(layer.forward cache (fun _ => input) 0, DeltaCache.after layer cache (fun _ => input) 1) := by
  unfold run
  rw [forward_eq, cache_eq]

def implementation : DeltaExecution.Implementation where
  Trace := Execution
  run := run
  refines := run_eq
  canonical := Execution.canonical

end Inference.Models.Qwen.DeltaDevice
