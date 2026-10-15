import kernels.trellis.DownFlushToZero

namespace Inference.Layers.TrellisCodebookDown
open Contracts.Mlx Contracts.Mlx.Numerics Kernels.Trellis
open scoped BigOperators

-- The base single-row projection, before any optional delta adapter. Shape
-- reshaping, native ownership, and adapter execution are not asserted here.
noncomputable def weight (words : DownExecution.PackedMatrix) (k : Nat)
    (scale : Nat → Float32) (row col : Nat) : ℝ :=
  DownAccuracy.idealCoefficient words k row col * real (scale row)

noncomputable def target (words : DownExecution.PackedMatrix) (k : Nat)
    (scale : Nat → Float32) (input : Nat → ℝ) (col : Nat) : ℝ :=
  ∑ row : Fin 17408, weight words k scale row.val col * input row.val

noncomputable def propagatedBudget (words : DownExecution.PackedMatrix) (k : Nat)
    (scale : Nat → Float32) (inputError : Nat → ℝ) (col : Nat) : ℝ :=
  ∑ row : Fin 17408, |weight words k scale row.val col| * inputError row.val

theorem input_error (words : DownExecution.PackedMatrix) (k : Nat) (scale input : Nat → Float32)
    (ideal inputError : Nat → ℝ) (col : Nat)
    (h : ∀ row : Fin 17408, |real (input row.val)-ideal row.val| ≤ inputError row.val) :
    |DownReduction.dotProduct words k scale input col-target words k scale ideal col| ≤
      propagatedBudget words k scale inputError col := by
  have he : DownReduction.dotProduct words k scale input col-target words k scale ideal col =
      ∑ row : Fin 17408, weight words k scale row.val col * (real (input row.val)-ideal row.val) := by
    unfold DownReduction.dotProduct target weight
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro row _
    ring
  rw [he]
  calc
    _ ≤ ∑ row : Fin 17408, |weight words k scale row.val col * (real (input row.val)-ideal row.val)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ propagatedBudget words k scale inputError col := by
      apply Finset.sum_le_sum
      intro row _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left (h row) (abs_nonneg _)

theorem output_error (load : M1Max.ThreadgroupLoad) (loads : M1Max.LoadContract load)
    (before : Nat → M1Max.WordMemory) (base : Nat) (ha : base%4=0) (hn : base+16384 ≤ 2^64)
    (words : DownExecution.PackedMatrix) (k : Nat) (hk : k=2 ∨ k=3 ∨ k=4)
    (scale input : Nat → Float32) (ideal inputError : Nat → ℝ) (col : Nat) (hc : col<5120)
    (reduce : DownExecution.SimdReduction) (hardware : GateUpFlushToZero.SimdContract reduce)
    (convert : Float32 → Float32) (conversion : BFloat16.RneContract convert)
    (hf : DownFlushToZero.Finite words k scale input col hardware.tree)
    (hb : FloatLib.Floats.Formats.BinaryInterchange.Model.isFinite
      (BFloat16.narrow (reduce (fun y => DownFlushToZero.partialSum words k scale input y.val col)))=true)
    (hi : ∀ row : Fin 17408, |real (input row.val)-ideal row.val| ≤ inputError row.val) :
    |real (DownFlushToZero.output load before base words k scale input reduce convert col)-
      target words k scale ideal col| ≤
      DownFlushToZero.budget words k scale input col hardware.tree +
        BFloat16.epsilon (real (reduce (fun y => DownFlushToZero.partialSum words k scale input y.val col))) +
        propagatedBudget words k scale inputError col := by
  have localError := DownFlushToZero.output_bfloat_error load loads before base ha hn words k hk scale input
    col hc reduce hardware convert conversion hf hb
  have upstreamError := input_error words k scale input ideal inputError col hi
  calc
    _ = |(real (DownFlushToZero.output load before base words k scale input reduce convert col)-
          DownReduction.dotProduct words k scale input col) +
        (DownReduction.dotProduct words k scale input col-target words k scale ideal col)| := by congr 1; ring
    _ ≤ _ := (abs_add_le _ _).trans (add_le_add localError upstreamError)

end Inference.Layers.TrellisCodebookDown
