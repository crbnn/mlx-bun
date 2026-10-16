import models.qwen.DeltaExecution
import kernels.delta.GatedInputs
import kernels.delta.ConvolutionDevice
import layers.DeltaHeadsDevice
import contracts.mlx.ArrayViews

namespace Inference.Models.Qwen.DeltaConvolution
open Contracts.Mlx Kernels.Delta
abbrev BF := BFloat16Arithmetic.Value

structure Execution (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache) where
  convolution : ConvolutionDevice.Execution layer.recurrence.convolutionContracted cache.history (layer.qkv.forward input) layer.convolutionWeight
  heads : Layers.DeltaHeadsDevice.Execution layer.recurrence.heads layer.headParams.width layer.headParams.eps
    convolution.output layer.headParams.qScale layer.headParams.kScale
  inputBuffer : heads.storage.input=ArrayViews.readback convolution.stores.buffer

def Execution.fromStorage (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (storage : State.ConvolutionStorage.Storage cache.history) : Execution layer input cache :=
  let convolution := ConvolutionDevice.Execution.fromStorage _ _ _ _ storage
  let inputs : Layers.DeltaHeadsDevice.Storage convolution.output layer.headParams.qScale layer.headParams.kScale :=
    ⟨ArrayViews.readback convolution.stores.buffer, ArrayStorage.BFBuffer.canonical _ (by decide), ArrayStorage.BFBuffer.canonical _ (by decide)⟩
  ⟨convolution, Layers.DeltaHeadsDevice.Execution.fromStorage _ _ _ _ _ _ inputs, rfl⟩

theorem output_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) :
    execution.convolution.output=State.Ssm.convolved layer.recurrence (DeltaNet.convData layer cache (fun _ => input)) 0 := by
  funext channel
  exact execution.convolution.window_eq channel

def heads (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : Layers.DeltaHeads.Output 1 BF := execution.heads.output

theorem heads_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) :
    heads layer input cache execution=State.Ssm.heads layer.recurrence layer.headParams (DeltaNet.convData layer cache (fun _ => input)) 0 := by
  unfold heads
  rw [execution.heads.output_eq, output_eq]
  rfl

@[irreducible] def inputValues (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : GatedInputs.Values :=
  let h := heads layer input cache execution
  let gates := DeltaNet.gates layer (fun _ => input)
  { key := h.k 0
    query := h.q 0
    value := h.v 0
    decay := fun head => Decay.value layer.recurrence.gates.decay (BFloat16.widen (gates.aLog head)) (gates.a 0 head) (gates.dtBias head)
    beta := fun head => Kernels.Trellis.GateUpSplitTail.sigmoidValue layer.recurrence.gates.expAbs (gates.b 0 head) }

theorem inputs_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) :
    (inputValues layer input cache execution).inputs=(fun head row => DeltaNet.inputs layer cache (fun _ => input) head row 0) := by
  unfold inputValues
  rw [heads_eq]
  unfold GatedInputs.Values.inputs DeltaNet.inputs State.Ssm.inputs Kernels.Delta.Inputs.actual State.Ssm.projected
  rfl

theorem history_eq (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : execution.convolution.retained=(DeltaCache.after layer cache (fun _ => input) 1).history := by
  rw [execution.convolution.history_eq]
  funext row channel
  unfold DeltaCache.after Convolution.tail Convolution.joined
  by_cases h : 1+row.val<3 <;> simp only [h, ↓reduceDIte]

def inputStorage (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : GatedInputs.Storage (inputValues layer input cache execution) := by
  unfold inputValues
  exact ⟨execution.heads.keyBuffer, execution.heads.queryBuffer, execution.heads.valueBuffer,
    ArrayStorage.FloatBuffer.canonical _ (by decide), ArrayStorage.BFBuffer.canonical _ (by decide)⟩

theorem head_input_bytes (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : execution.heads.storage.input.bytes=execution.convolution.stores.memory :=
  congrArg ArrayStorage.BFBuffer.bytes execution.inputBuffer

theorem key_bytes (layer : DeltaNet.Layer) (input : Fin 5120 → BF) (cache : DeltaNet.Cache)
    (execution : Execution layer input cache) : (inputStorage layer input cache execution).key.bytes=execution.heads.kStores.memory := by
  unfold inputStorage
  rfl

end Inference.Models.Qwen.DeltaConvolution
