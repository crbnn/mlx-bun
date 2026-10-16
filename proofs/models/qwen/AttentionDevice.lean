import models.qwen.AttentionExecution
import kernels.attention.SdpaDispatchedDevice
import state.KvDevice

namespace Inference.Models.Qwen.AttentionDevice
open Contracts.Mlx Contracts.Mlx.Numerics Kernels.Attention
open FloatLib FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
abbrev BF := BFloat16Arithmetic.Value

structure Execution (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) where
  storage : State.KvAllocation.Storage cache
  updated : State.KvDevice.Execution storage (layer.heads input angle).key (layer.heads input angle).value
  attention : SdpaDispatchedDevice.Execution layer.attention layer.mergeContracted layer.architectureSuffix
    (layer.heads input angle).query (State.KvBytes.cacheAfter updated.toKvBytes).keys (State.KvBytes.cacheAfter updated.toKvBytes).values

def Execution.fromStorage (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) (storage : State.KvAllocation.Storage cache) : Execution layer input angle cache where
  storage := storage
  updated := State.KvDevice.Execution.canonical storage (layer.heads input angle).key (layer.heads input angle).value
  attention := SdpaDispatchedDevice.Execution.canonical _ _ _ _ _ _

def Execution.canonical (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) : Execution layer input angle cache :=
  Execution.fromStorage layer input angle cache (State.KvAllocation.Storage.canonical cache)

def read (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) (execution : Execution layer input angle cache) : Fin 24 → Fin 256 → BF :=
  SdpaDispatchedDevice.forward execution.attention

theorem read_eq (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) (execution : Execution layer input angle cache) :
    read layer input angle cache execution=layer.read input angle cache := by
  funext head i
  unfold read
  rw [SdpaDispatchedDevice.forward_eq, State.KvBytes.cacheAfter_eq]
  rfl

def merged (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) (execution : Execution layer input angle cache) (i : Fin 6144) : BF :=
  OutputGate.forward layer.gate (read layer input angle cache execution (Attention.headIndex i) (Attention.coordinate i))
    ((layer.heads input angle).gate (Attention.headIndex i) (Attention.coordinate i))

theorem merged_eq (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) (execution : Execution layer input angle cache) :
    merged layer input angle cache execution=layer.merged input angle cache := by
  funext i
  unfold merged
  rw [read_eq]
  rfl

def forward (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) (execution : Execution layer input angle cache) : Fin 5120 → BF :=
  layer.output.forward (merged layer input angle cache execution)

theorem forward_eq (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) (execution : Execution layer input angle cache) :
    forward layer input angle cache execution=layer.forward input angle cache := by
  unfold forward
  rw [merged_eq]
  rfl

-- Attention consumes and returns the same appended bytes. KV traces include
-- zero fill, concurrent growth copies, donation/vector-copy preservation and
-- the strided slice update. Allocator, dispatch/visibility and compiler
-- correspondence must still implement these translated execution models.
def run (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) (execution : Execution layer input angle cache) : (Fin 5120 → BF) × State.Kv.Cache BF (n+1) :=
  (forward layer input angle cache execution, State.KvBytes.cacheAfter execution.updated.toKvBytes)

theorem run_eq (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) (execution : Execution layer input angle cache) :
    run layer input angle cache execution=(layer.forward input angle cache, layer.cacheAfter input angle cache) := by
  unfold run
  rw [forward_eq, State.KvBytes.cacheAfter_eq]
  rfl

def storageAfter (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) (execution : Execution layer input angle cache) :
    State.KvAllocation.Storage (layer.cacheAfter input angle cache) := execution.updated.after

theorem storage_keys (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) (execution : Execution layer input angle cache) :
    (storageAfter layer input angle cache execution).keys=execution.updated.toKvBytes.keys.memory := rfl

theorem storage_values (layer : Attention.Layer) (input : Fin 5120 → BF) (angle : RopeAngles.Inputs)
    (cache : State.Kv.Cache BF n) (execution : Execution layer input angle cache) :
    (storageAfter layer input angle cache execution).values=execution.updated.toKvBytes.values.memory := rfl

def implementation : AttentionExecution.Implementation where
  Trace := Execution
  run := run
  refines := run_eq
  canonical := Execution.canonical

theorem forward_error (layer : Attention.Layer) (input : Fin 5120 → BF) (target error : Fin 5120 → ℝ)
    (angle : RopeAngles.Inputs) (angleTarget angleError : RopeAngles.Target)
    (cache : State.Kv.Cache BF n) (cacheTarget cacheError : State.Kv.Cache ℝ n) (execution : Execution layer input angle cache)
    (hi : ∀ i, |Model.toReal (input i)-target i| ≤ error i) (hc : State.Kv.Error cache cacheTarget cacheError)
    (hcf : State.Kv.Finite cache) (hf : layer.Finite input angle cache)
    (ha : RopeAngles.InputError angle angleTarget angleError) (hb : 0 ≤ angleTarget.base) (i : Fin 5120) :
    |Model.toReal (forward layer input angle cache execution i)-layer.target target angleTarget cacheTarget i| ≤
      layer.budget input target error angle angleTarget angleError cache cacheTarget cacheError i := by
  rw [forward_eq]
  exact layer.forward_error input target error angle angleTarget angleError cache cacheTarget cacheError hi hc hcf hf ha hb i

end Inference.Models.Qwen.AttentionDevice
