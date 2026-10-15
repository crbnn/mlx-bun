import layers.TrellisCodebookGateUp
import layers.TrellisCodebookDown

namespace Inference.Models.Qwen.WrittenLayers
open Contracts.Mlx Contracts.Mlx.Numerics Kernels.Trellis
open Layers
open FloatLib FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

-- A static activation choice, corresponding to the kernel specialization.
-- The match below models specialization; it adds no runtime validation.
inductive Activation where
  | float (expNeg : Float32 → Float32)
  | split (expAbs : BFloat16Arithmetic.Value → BFloat16Arithmetic.Value)

def hidden (activation : Activation) (platform : TrellisCodebookGateUp.Platform)
    (gate up : TrellisCodebookGateUp.Projection) (input : Nat → BitVec 16) (row : Nat) : Float32 :=
  match activation with
  | .float expNeg => TrellisCodebookGateUp.floatOutput platform gate up input row expNeg
  | .split expAbs => TrellisCodebookGateUp.splitOutput platform gate up input row expAbs

-- This contract is a theorem about the mathematical function and numerical
-- error, not just shape compatibility. Constructors below discharge it from
-- kernel proofs. It can also be assembled row by row using the overflow
-- theorems, without changing the executed activation.
structure HiddenContract (activation : Activation) (platform : TrellisCodebookGateUp.Platform)
    (gate up : TrellisCodebookGateUp.Projection) (input : Nat → BitVec 16) where
  error : Nat → ℝ
  accuracy : ∀ row : Fin 17408,
    |real (hidden activation platform gate up input row.val)-TrellisCodebookGateUp.layerTarget gate up input row.val| ≤ error row.val

noncomputable def floatFiniteContract (p : TrellisCodebookGateUp.Platform)
    (gate up : TrellisCodebookGateUp.Projection) (input : Nat → BitVec 16)
    (expNeg : Float32 → Float32) (expError : Float32 → ℝ) (exponential : GateUpTail.ExpNegContract expNeg expError)
    (hg : ∀ row : Fin 17408, TrellisCodebookGateUp.Finite p gate input row.val)
    (hu : ∀ row : Fin 17408, TrellisCodebookGateUp.Finite p up input row.val)
    (hgb : ∀ row : Fin 17408, Model.isFinite (BFloat16.narrow (TrellisCodebookGateUp.raw p gate input row.val))=true)
    (hub : ∀ row : Fin 17408, Model.isFinite (BFloat16.narrow (TrellisCodebookGateUp.raw p up input row.val))=true)
    (hf : ∀ row : Fin 17408, GateUpTailFlushToZero.Finite expNeg
      (BFloat16.cast (TrellisCodebookGateUp.raw p gate input row.val)) (BFloat16.cast (TrellisCodebookGateUp.raw p up input row.val))) :
    HiddenContract (.float expNeg) p gate up input where
  error row := TrellisCodebookGateUp.layerBudget p gate up input row
    (real (BFloat16.cast (TrellisCodebookGateUp.raw p up input row)))
    (GateUpTailFlushToZero.tailBudget expNeg expError (BFloat16.cast (TrellisCodebookGateUp.raw p gate input row))
      (BFloat16.cast (TrellisCodebookGateUp.raw p up input row)))
  accuracy row := TrellisCodebookGateUp.float_finite_error p gate up input row.val expNeg expError exponential
    (hg row) (hu row) (hgb row) (hub row) (hf row)

noncomputable def splitFiniteContract (p : TrellisCodebookGateUp.Platform)
    (gate up : TrellisCodebookGateUp.Projection) (input : Nat → BitVec 16)
    (expAbs : BFloat16Arithmetic.Value → BFloat16Arithmetic.Value) (expError : BFloat16Arithmetic.Value → ℝ)
    (exponential : GateUpSplitTail.ExpAbsContract expAbs expError)
    (hg : ∀ row : Fin 17408, TrellisCodebookGateUp.Finite p gate input row.val)
    (hu : ∀ row : Fin 17408, TrellisCodebookGateUp.Finite p up input row.val)
    (hs : ∀ row : Fin 17408, GateUpSplitTail.SigmoidFinite expAbs (BFloat16.narrow (TrellisCodebookGateUp.raw p gate input row.val)))
    (hp : ∀ row : Fin 17408, GateUpSplitTail.ProductsFinite expAbs
      (BFloat16.narrow (TrellisCodebookGateUp.raw p gate input row.val)) (BFloat16.narrow (TrellisCodebookGateUp.raw p up input row.val))) :
    HiddenContract (.split expAbs) p gate up input where
  error row := TrellisCodebookGateUp.layerBudget p gate up input row
    (Model.toReal (BFloat16.narrow (TrellisCodebookGateUp.raw p up input row)))
    (GateUpSplitTail.tailBudget expAbs (BFloat16.narrow (TrellisCodebookGateUp.raw p gate input row))
      (BFloat16.narrow (TrellisCodebookGateUp.raw p up input row))
      (GateUpSplitTail.sigmoidBudget expAbs expError (BFloat16.narrow (TrellisCodebookGateUp.raw p gate input row))))
  accuracy row := TrellisCodebookGateUp.split_finite_error p gate up input row.val expAbs expError exponential
    (hg row) (hu row) (hs row) (hp row)

-- GateDownMLP.forward is gateUp.forward followed by down.forward. This theorem
-- composes their numerical contracts for the base single-row Trellis path.
-- Optional delta adapters and other dispatch paths are separate proof targets.
theorem gateDown_error (activation : Activation) (p : TrellisCodebookGateUp.Platform)
    (gate up : TrellisCodebookGateUp.Projection) (input : Nat → BitVec 16)
    (hmid : HiddenContract activation p gate up input)
    (load : M1Max.ThreadgroupLoad) (loads : M1Max.LoadContract load)
    (before : Nat → M1Max.WordMemory) (base : Nat) (ha : base%4=0) (hn : base+16384 ≤ 2^64)
    (words : DownExecution.PackedMatrix) (k : Nat) (hk : k=2 ∨ k=3 ∨ k=4)
    (scale : Nat → Float32) (col : Nat) (hc : col<5120)
    (reduce : DownExecution.SimdReduction) (hardware : GateUpFlushToZero.SimdContract reduce)
    (convert : Float32 → Float32) (conversion : BFloat16.RneContract convert)
    (hf : DownFlushToZero.Finite words k scale (hidden activation p gate up input) col hardware.tree)
    (hb : Model.isFinite (BFloat16.narrow (reduce (fun y =>
      DownFlushToZero.partialSum words k scale (hidden activation p gate up input) y.val col)))=true) :
    |real (DownFlushToZero.output load before base words k scale (hidden activation p gate up input) reduce convert col)-
      TrellisCodebookDown.target words k scale (TrellisCodebookGateUp.layerTarget gate up input) col| ≤
      DownFlushToZero.budget words k scale (hidden activation p gate up input) col hardware.tree +
        BFloat16.epsilon (real (reduce (fun y =>
          DownFlushToZero.partialSum words k scale (hidden activation p gate up input) y.val col))) +
        TrellisCodebookDown.propagatedBudget words k scale hmid.error col := by
  exact TrellisCodebookDown.output_error load loads before base ha hn words k hk scale
    (hidden activation p gate up input) (TrellisCodebookGateUp.layerTarget gate up input) hmid.error col hc
    reduce hardware convert conversion hf hb hmid.accuracy

end Inference.Models.Qwen.WrittenLayers
