import contracts.mlx.Numerics
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Algebra.BigOperators.Fin

namespace Inference.Contracts.Mlx.Normalization
open Numerics
open scoped BigOperators

noncomputable def meanSquare {n : Nat} (x : Fin n → ℝ) : ℝ := (∑ i, x i ^ 2) / n
noncomputable def denominator {n : Nat} (x : Fin n → ℝ) (eps : ℝ) : ℝ := meanSquare x + eps
noncomputable def rmsNorm {n : Nat} (x weight : Fin n → ℝ) (eps : ℝ) (i : Fin n) : ℝ :=
  weight i * (x i / Real.sqrt (denominator x eps))

theorem meanSquare_nonnegative {n : Nat} (x : Fin n → ℝ) : 0 ≤ meanSquare x := by
  unfold meanSquare
  positivity

theorem denominator_positive {n : Nat} (x : Fin n → ℝ) (eps : ℝ) (he : 0<eps) :
    0 < denominator x eps := by
  have := meanSquare_nonnegative x
  unfold denominator
  linarith

-- This bound is independent of the input magnitude. It is useful for proving
-- subsequent projection ranges once rounding and weight errors are included.
theorem normalized_bound {n : Nat} (hn : 0<n) (x : Fin n → ℝ) (eps : ℝ) (he : 0<eps) (i : Fin n) :
    |x i / Real.sqrt (denominator x eps)| ≤ Real.sqrt n := by
  have hn' : (0 : ℝ)<n := by exact_mod_cast hn
  have hd := denominator_positive x eps he
  have hs : x i ^ 2 ≤ ∑ j : Fin n, x j ^ 2 := by
    exact Finset.single_le_sum (fun j _ => sq_nonneg (x j)) (Finset.mem_univ i)
  have hdEq : (n : ℝ)*denominator x eps = (∑ j : Fin n, x j ^ 2)+(n : ℝ)*eps := by
    unfold denominator meanSquare
    field_simp
  have boundSq : |x i| ^ 2 ≤ (Real.sqrt n * Real.sqrt (denominator x eps)) ^ 2 := by
    rw [sq_abs, mul_pow, Real.sq_sqrt hn'.le, Real.sq_sqrt hd.le, hdEq]
    nlinarith
  have hb : |x i| ≤ Real.sqrt n * Real.sqrt (denominator x eps) :=
    (sq_le_sq₀ (abs_nonneg _) (mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))).mp boundSq
  rw [abs_div, abs_of_pos (Real.sqrt_pos.mpr hd)]
  exact (div_le_iff₀ (Real.sqrt_pos.mpr hd)).mpr hb

theorem output_bound {n : Nat} (hn : 0<n) (x weight : Fin n → ℝ) (eps : ℝ) (he : 0<eps) (i : Fin n) :
    |rmsNorm x weight eps i| ≤ |weight i| * Real.sqrt n := by
  unfold rmsNorm
  rw [abs_mul]
  exact mul_le_mul_of_nonneg_left (normalized_bound hn x eps he i) (abs_nonneg _)

theorem output_bound_of_error {n : Nat} (hn : 0 < n) (x weight : Fin n → ℝ) (eps : ℝ)
    (he : 0 < eps) (i : Fin n) (actual error : ℝ)
    (ha : |actual - rmsNorm x weight eps i| ≤ error) :
    |actual| ≤ |weight i| * Real.sqrt n + error := by
  have h := (abs_add_le (actual - rmsNorm x weight eps i) (rmsNorm x weight eps i)).trans
    (add_le_add ha (output_bound hn x weight eps he i))
  simpa only [sub_add_cancel, add_comm error] using h

-- Loading Qwen normalization weights adds one in the tensor's dtype. A
-- weight error therefore contributes at most sqrt(width) times that error.
theorem weight_perturbation {n : Nat} (hn : 0 < n) (x weight target : Fin n → ℝ)
    (eps : ℝ) (he : 0 < eps) (i : Fin n) (weightError : ℝ)
    (hw : |weight i - target i| ≤ weightError) :
    |rmsNorm x weight eps i - rmsNorm x target eps i| ≤ weightError * Real.sqrt n := by
  unfold rmsNorm
  rw [← sub_mul, abs_mul]
  exact mul_le_mul hw (normalized_bound hn x eps he i) (abs_nonneg _) (le_trans (abs_nonneg _) hw)

theorem inverse_sqrt_difference (d t : ℝ) (hd : 0<d) (ht : 0<t) :
    |1/Real.sqrt d-1/Real.sqrt t| =
      |d-t|/(Real.sqrt d*Real.sqrt t*(Real.sqrt d+Real.sqrt t)) := by
  have sd := Real.sqrt_pos.mpr hd
  have st := Real.sqrt_pos.mpr ht
  have dd := Real.sq_sqrt hd.le
  have tt := Real.sq_sqrt ht.le
  have hf : d-t=(Real.sqrt d-Real.sqrt t)*(Real.sqrt d+Real.sqrt t) := by nlinarith
  have hi : 1/Real.sqrt d-1/Real.sqrt t = -(Real.sqrt d-Real.sqrt t)/(Real.sqrt d*Real.sqrt t) := by
    field_simp
    ring
  simp only [hi, hf, abs_div, abs_neg, abs_mul,
    abs_of_pos sd, abs_of_pos st, abs_of_pos (by linarith : 0<Real.sqrt d+Real.sqrt t)]
  field_simp

structure RsqrtContract (rsqrt : Float32 → Float32) (error : Float32 → ℝ) : Prop where
  accuracy : ∀ x, x.isFinite=true → 0<real x → (rsqrt x).isFinite=true →
    |real (rsqrt x)-1/Real.sqrt (real x)| ≤ error x

-- Only the primitive reciprocal-square-root error is a platform premise.
-- Perturbation of its argument is propagated here by a proved identity.
theorem rsqrt_error (rsqrt : Float32 → Float32) (error : Float32 → ℝ)
    (hardware : RsqrtContract rsqrt error) (x : Float32) (target inputError : ℝ)
    (hx : x.isFinite=true) (hp : 0<real x) (ht : 0<target) (ho : (rsqrt x).isFinite=true)
    (he : |real x-target| ≤ inputError) :
    |real (rsqrt x)-1/Real.sqrt target| ≤ error x +
      inputError/(Real.sqrt (real x)*Real.sqrt target*(Real.sqrt (real x)+Real.sqrt target)) := by
  have localError := hardware.accuracy x hx hp ho
  have propagated : |1/Real.sqrt (real x)-1/Real.sqrt target| ≤
      inputError/(Real.sqrt (real x)*Real.sqrt target*(Real.sqrt (real x)+Real.sqrt target)) := by
    rw [inverse_sqrt_difference _ _ hp ht]
    apply div_le_div_of_nonneg_right he
    positivity
  calc
    _ = |(real (rsqrt x)-1/Real.sqrt (real x))+(1/Real.sqrt (real x)-1/Real.sqrt target)| := by congr 1; ring
    _ ≤ _ := (abs_add_le _ _).trans (add_le_add localError propagated)

end Inference.Contracts.Mlx.Normalization
