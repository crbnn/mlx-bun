import models.qwen.NormalizationRanges
import contracts.mlx.DyadicCover
import contracts.mlx.RsqrtBounds
import contracts.mlx.BFloat16DyadicBounds

namespace Inference.Models.Qwen.NormalizationUniformBounds
open Contracts.Mlx Contracts.Mlx.Numerics
open FloatLib FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
abbrev BF := BFloat16Arithmetic.Value

private theorem float_cap (exponent : Int) (lo : -149 ≤ exponent) (hi : exponent ≤ 100) :
    (DyadicBounds.ceiling 1 exponent).isFinite = true ∧ real (DyadicBounds.ceiling 1 exponent) = (2 : ℝ)^exponent := by
  simpa only [Nat.cast_one, one_mul] using DyadicBounds.ceiling_range 1 exponent (by decide) lo hi

private theorem bf_cap (exponent : Int) (lo : -133 ≤ exponent) (hi : exponent ≤ 119) :
    Model.isFinite (BFloat16DyadicBounds.ceiling 1 exponent) = true ∧
      Model.toReal (BFloat16DyadicBounds.ceiling 1 exponent) = (2 : ℝ)^exponent := by
  simpa only [Nat.cast_one, one_mul] using BFloat16DyadicBounds.ceiling_range 1 exponent (by decide) lo hi

-- These concrete capacities are intentionally conservative. They make the
-- normalization output bound uniform across input magnitudes up to 2^50.
def weightCap : BF := BFloat16DyadicBounds.ceiling 1 5
def outputCap : BF := BFloat16DyadicBounds.ceiling 1 16

theorem weightCap_real : Model.toReal weightCap = 32 := by
  rw [weightCap, (bf_cap 5 (by decide) (by decide)).2]
  norm_num

theorem outputCap_real : Model.toReal outputCap = 65536 := by
  rw [outputCap, (bf_cap 16 (by decide) (by decide)).2]
  norm_num

private def profile (exponent : Nat) (inverse : Int) (product : Nat)
    (bounded : exponent ≤ 50) (inverseLo : -149 ≤ inverse) (inverseHi : inverse ≤ 100)
    (productHi : product ≤ 11) (productEq : (exponent : Int)+inverse = (product : Int)) : NormalizationDyadicBounds.Profile where
  inputExponent := exponent
  inputLower := by omega
  inputUpper := by omega
  eps := Blocks.eps
  epsFinite := Blocks.eps_finite
  epsNormal := Blocks.eps_normal
  epsBound := by rw [Blocks.eps_value]; norm_num
  inverseCap := DyadicBounds.ceiling 1 inverse
  productCap := DyadicBounds.ceiling 1 product
  normalizedCap := BFloat16DyadicBounds.ceiling 1 product
  weightCap := weightCap
  outputCap := outputCap
  inverseFinite := (float_cap inverse inverseLo inverseHi).1
  productFinite := (float_cap product (by omega) (by omega)).1
  normalizedFinite := (bf_cap product (by omega) (by omega)).1
  outputFinite := (bf_cap 16 (by decide) (by decide)).1
  productCapacity := by
    rw [(float_cap inverse inverseLo inverseHi).2, (float_cap product (by omega) (by omega)).2,
      ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0), productEq]
  normalizedCapacity := by
    rw [(float_cap product (by omega) (by omega)).2, (bf_cap product (by omega) (by omega)).2]
  outputCapacity := by
    rw [weightCap, outputCap, (bf_cap 5 (by decide) (by decide)).2, (bf_cap 16 (by decide) (by decide)).2,
      (bf_cap product (by omega) (by omega)).2, ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    exact zpow_le_zpow_right₀ (by norm_num) (by omega)

theorem small_range (norm : Blocks.Norm)
    (division : DivisionBounds.Contract norm.platform.divide) (root : RsqrtBounds.Envelope norm.platform.rsqrt)
    (hw : ∀ i, Model.isFinite (norm.weight i) = true ∧ |Model.toReal (norm.weight i)| ≤ 32)
    (x : Fin 5120 → BF) (hx : ∀ i, Model.isFinite (x i) = true ∧ |Model.toReal (x i)| ≤ 1) :
    norm.Finite x ∧ ∀ i, |Model.toReal (norm.forward x i)| ≤ 65536 := by
  let p := profile 0 11 11 (by decide) (by decide) (by decide) (by decide) (by decide)
  have hr := RsqrtBounds.range norm.platform.rsqrt root Blocks.eps (DyadicBounds.ceiling 1 11)
    (FlushToZero.threshold_pos.trans Blocks.eps_normal)
    (by rw [(float_cap 11 (by decide) (by decide)).2]; positivity)
    (by rw [Blocks.eps_value, (float_cap 11 (by decide) (by decide)).2]; norm_num)
  have h := NormalizationRanges.ranges norm p rfl division hr
    (fun i => by simpa only [p, profile, weightCap_real] using hw i) x
    (fun i => by simpa only [p, profile, Int.natCast_zero, zpow_zero] using hx i)
  exact ⟨h.1, fun i => by simpa only [p, profile, outputCap_real] using h.2 i⟩

theorem bin_range (norm : Blocks.Norm)
    (division : DivisionDirectedBounds.Contract norm.platform.divide) (root : RsqrtBounds.Envelope norm.platform.rsqrt)
    (hw : ∀ i, Model.isFinite (norm.weight i) = true ∧ |Model.toReal (norm.weight i)| ≤ 32)
    (x : Fin 5120 → BF) (exponent : Nat) (positive : 0 < exponent) (bounded : exponent ≤ 50)
    (hx : ∀ i, Model.isFinite (x i) = true ∧ |Model.toReal (x i)| ≤ (2 : ℝ)^exponent)
    (i : Fin 5120) (lower : (2 : ℝ)^(exponent-1) < |Model.toReal (x i)|) :
    norm.Finite x ∧ ∀ j, |Model.toReal (norm.forward x j)| ≤ 65536 := by
  let e : Int := exponent
  let p := profile exponent (9-e) 9 bounded (by dsimp [e]; omega) (by dsimp [e]; omega) (by decide) (by dsimp [e]; omega)
  let squareFloor := DyadicBounds.ceiling 1 (2*e-2)
  let meanFloor := DyadicBounds.ceiling 1 (2*e-15)
  have hs := float_cap (2*e-2) (by dsimp [e]; omega) (by dsimp [e]; omega)
  have hm := float_cap (2*e-15) (by dsimp [e]; omega) (by dsimp [e]; omega)
  have hc := float_cap (9-e) (by dsimp [e]; omega) (by dsimp [e]; omega)
  have sn : FlushToZero.threshold < real squareFloor := by
    rw [hs.2]
    exact (by norm_num [FlushToZero.threshold] : FlushToZero.threshold < 1).trans_le
      (one_le_zpow₀ (by norm_num) (by dsimp [e]; omega))
  have mn : FlushToZero.threshold < real meanFloor := by
    rw [hm.2]
    exact (by norm_num [FlushToZero.threshold] : FlushToZero.threshold < (2 : ℝ)^(-13 : Int)).trans_le
      (zpow_le_zpow_right₀ (by norm_num) (by dsimp [e]; omega))
  have inputNormal : FlushToZero.threshold < |Model.toReal (x i)| := by
    have hone : 1 ≤ (2 : ℝ)^(exponent-1) := by
      simpa only [zpow_natCast] using one_le_zpow₀ (by norm_num : (1 : ℝ) ≤ 2) (show (0 : Int) ≤ ((exponent-1 : Nat) : Int) by omega)
    exact ((by norm_num [FlushToZero.threshold] : FlushToZero.threshold < 1).trans_le hone).trans lower
  have square : real squareFloor ≤ Model.toReal (x i) * Model.toReal (x i) := by
    rw [hs.2, show 2*e-2 = ((2*(exponent-1) : Nat) : Int) from by dsimp [e]; omega, zpow_natCast]
    have hsq := (sq_le_sq₀ (by positivity : 0 ≤ (2 : ℝ)^(exponent-1)) (abs_nonneg _)).mpr lower.le
    calc
      (2 : ℝ)^(2*(exponent-1)) = ((2 : ℝ)^(exponent-1))^2 := by rw [Nat.mul_comm 2 (exponent-1), pow_mul]
      _ ≤ |Model.toReal (x i)|^2 := hsq
      _ = _ := by rw [sq_abs, pow_two]
  have capacity : real meanFloor ≤ real squareFloor / 5120 := by
    rw [hm.2, hs.2]
    apply (le_div_iff₀ (by norm_num : (0 : ℝ) < 5120)).mpr
    rw [show 2*e-2 = (2*e-15)+13 from by ring, zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    have hp := zpow_nonneg (by norm_num : (0 : ℝ) ≤ 2) (2*e-15)
    norm_num
    nlinarith
  have rootRange := RsqrtBounds.range norm.platform.rsqrt root meanFloor (DyadicBounds.ceiling 1 (9-e))
    (FlushToZero.threshold_pos.trans mn) (by rw [hc.2]; positivity)
    (by rw [hm.2, hc.2, pow_two, ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0),
        ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0), show (2*e-15)+((9-e)+(9-e)) = 3 from by ring]; norm_num)
  have h := NormalizationRanges.floor_ranges norm p rfl division
    (fun j => by simpa only [p, profile, weightCap_real] using hw j) x
    (fun j => by simpa only [p, profile, zpow_natCast] using hx j) i squareFloor meanFloor hs.1 sn hm.1 mn
    inputNormal square capacity rootRange
  exact ⟨h.1, fun j => by simpa only [p, profile, outputCap_real] using h.2 j⟩

theorem ranges (norm : Blocks.Norm)
    (division : DivisionDirectedBounds.Contract norm.platform.divide) (root : RsqrtBounds.Envelope norm.platform.rsqrt)
    (hw : ∀ i, Model.isFinite (norm.weight i) = true ∧ |Model.toReal (norm.weight i)| ≤ 32)
    (x : Fin 5120 → BF) (hx : ∀ i, Model.isFinite (x i) = true ∧ |Model.toReal (x i)| ≤ (2 : ℝ)^(50 : Nat)) :
    norm.Finite x ∧ ∀ i, |Model.toReal (norm.forward x i)| ≤ 65536 := by
  rcases DyadicCover.cover (fun i => Model.toReal (x i)) 50 (fun i => (hx i).2) with small | ⟨exponent, positive, bounded, upper, i, lower⟩
  · exact small_range norm division.toContract root hw x (fun i => ⟨(hx i).1, small i⟩)
  · exact bin_range norm division root hw x exponent positive bounded (fun i => ⟨(hx i).1, upper i⟩) i lower

end Inference.Models.Qwen.NormalizationUniformBounds
