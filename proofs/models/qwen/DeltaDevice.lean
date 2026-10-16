import models.qwen.DeltaExecution
import kernels.delta.GatedDevice

namespace Inference.Models.Qwen.DeltaDevice
open Contracts.Mlx Kernels.Delta
abbrev BF := BFloat16Arithmetic.Value

abbrev Execution (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache) :=
  GatedDevice.Execution layer.recurrence.reduce layer.recurrence.flags cache.recurrent
    (fun head row => DeltaNet.inputs layer cache (fun _ => input) head row 0)

@[irreducible] def inputValues (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache) : GatedInputs.Values :=
  let heads := State.Ssm.heads layer.recurrence layer.headParams (DeltaNet.convData layer cache (fun _ => input)) 0
  let gates := DeltaNet.gates layer (fun _ => input)
  { key := heads.k 0
    query := heads.q 0
    value := heads.v 0
    decay := fun h => Decay.value layer.recurrence.gates.decay (BFloat16.widen (gates.aLog h)) (gates.a 0 h) (gates.dtBias h)
    beta := fun h => Kernels.Trellis.GateUpSplitTail.sigmoidValue layer.recurrence.gates.expAbs (gates.b 0 h) }

theorem input_values_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache) :
    (inputValues layer input cache).inputs=(fun head row => DeltaNet.inputs layer cache (fun _ => input) head row 0) := by
  unfold inputValues GatedInputs.Values.inputs DeltaNet.inputs State.Ssm.inputs Kernels.Delta.Inputs.actual State.Ssm.projected
  rfl

def Execution.fromStorage (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (storage : GatedDevice.Storage cache.recurrent) : Execution layer input cache where
  values := inputValues layer input cache
  binding := input_values_eq layer input cache
  inputStorage := GatedInputs.Storage.canonical _
  storage := storage
  stores := GatedBytes.Trace.canonical _ _

def Execution.canonical (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache) : Execution layer input cache :=
  Execution.fromStorage layer input cache (GatedDevice.Storage.canonical cache.recurrent)

theorem output_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : execution.output=DeltaNet.recurrentOutput layer cache (fun _ => input) 0 := by
  rw [execution.output_eq]
  unfold GatedDevice.producedOutput
  rfl

theorem recurrent_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : execution.state=(DeltaCache.after layer cache (fun _ => input) 1).recurrent := by
  rw [execution.state_eq]
  unfold GatedDevice.producedState
  rfl

def gated (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) (head : Fin 48) : Fin 128 → BF :=
  Kernels.GatedRmsNorm.value layer.gated (execution.output head) (DeltaNet.zHead layer (fun _ => input) 0 head)
    layer.normWeight layer.headParams.width layer.headParams.eps

theorem gated_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : gated layer input cache execution=DeltaNet.gatedOutput layer cache (fun _ => input) 0 := by
  unfold gated
  rw [output_eq]
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

-- The convolution history retains its translated update. Recurrent state
-- and the row consumed by gated normalization use completed byte readback.
def cacheAfter (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : DeltaNet.Cache :=
  {DeltaCache.after layer cache (fun _ => input) 1 with recurrent := execution.state}

theorem cache_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : cacheAfter layer input cache execution=DeltaCache.after layer cache (fun _ => input) 1 := by
  unfold cacheAfter
  rw [recurrent_eq]

def storageAfter (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : GatedDevice.Storage (cacheAfter layer input cache execution).recurrent := execution.after

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
