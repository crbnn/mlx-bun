import contracts.mlx.NormalizationBounds
import contracts.mlx.SquaredSumDyadicBounds

namespace Inference.Contracts.Mlx.NormalizationDyadicBounds
open FloatLib FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange Numerics
open NormalizationExecution

structure Profile where
  inputExponent : Int
  inputLower : 0 ≤ inputExponent
  inputUpper : inputExponent+inputExponent ≤ 100
  eps : Float32
  epsFinite : eps.isFinite = true
  epsNormal : FlushToZero.threshold < real eps
  epsBound : |real eps| ≤ 1
  inverseCap : Float32
  productCap : Float32
  normalizedCap : BFloat16Arithmetic.Value
  weightCap : BFloat16Arithmetic.Value
  outputCap : BFloat16Arithmetic.Value
  inverseFinite : inverseCap.isFinite = true
  productFinite : productCap.isFinite = true
  normalizedFinite : Model.isFinite normalizedCap = true
  outputFinite : Model.isFinite outputCap = true
  productCapacity : (2 : ℝ)^inputExponent * real inverseCap ≤ real productCap
  normalizedCapacity : real productCap ≤ Model.toReal normalizedCap
  outputCapacity : Model.toReal weightCap * Model.toReal normalizedCap ≤ Model.toReal outputCap

theorem Profile.reduction (p : Profile) (platform : RmsNorm.Platform)
    (division : DivisionBounds.Contract platform.divide)
    (input : Fin n → Float32)
    (hx : ∀ i, (input i).isFinite = true ∧ |real (input i)| ≤ (2 : ℝ)^p.inputExponent)
    (positive : 0 < n) (size : n+1 < 16777216) :
    ReductionFinite platform.divide input (RmsNorm.expression platform n) (Float32.ofInt n) p.eps ∧
      real p.eps ≤ real (denominator platform.divide input (RmsNorm.expression platform n) (Float32.ofInt n) p.eps) := by
  let expr := RmsNorm.expression platform n
  have hlo : -149 ≤ p.inputExponent+p.inputExponent := by have := p.inputLower; omega
  have hs := SquaredSumDyadicBounds.covered_range input p.inputExponent hlo p.inputUpper hx expr
    (RmsNorm.coverage platform n) (by omega)
  have hw := IntegerBounds.natural n (by omega)
  have cs := DyadicBounds.ceiling_range n (p.inputExponent+p.inputExponent) (by omega) hlo p.inputUpper
  have cd := DyadicBounds.ceiling_range (n+1) (p.inputExponent+p.inputExponent) size hlo p.inputUpper
  have hd : Division.DivisorDomain (Float32.ofInt n) := by
    rw [Division.DivisorDomain, hw.2]
    constructor
    · exact_mod_cast positive
    · have hn : (n : ℝ) ≤ 16777216 := by exact_mod_cast (show n ≤ 16777216 by omega)
      exact hn.trans (by norm_num)
  have hsf := SquaredSum.finite_evaluate input expr hs.1
  have hmf := division.finite _ _ hsf hw.1 hd
  have hmb := division.bounded _ _ (DyadicBounds.ceiling n (p.inputExponent+p.inputExponent))
    hsf hw.1 hd cs.1 (by rw [cs.2, abs_of_nonneg hs.2.1]; exact hs.2.2)
  have hmn := platform.division.nonnegative _ _ hsf hw.1 hs.2.1 hd hmf
  have he : |real p.eps| ≤ (2 : ℝ)^(p.inputExponent+p.inputExponent) :=
    p.epsBound.trans (one_le_zpow₀ (by norm_num) (by have := p.inputLower; omega))
  have ha := Float32Bounds.add_range (mean platform.divide input expr (Float32.ofInt n)) p.eps
    (DyadicBounds.ceiling (n+1) (p.inputExponent+p.inputExponent)) hmf p.epsFinite cd.1
    (by rw [cd.2]; have hm := hmb; rw [cs.2] at hm; calc
        _ ≤ (n : ℝ) * (2 : ℝ)^(p.inputExponent+p.inputExponent) + (2 : ℝ)^(p.inputExponent+p.inputExponent) := add_le_add hm he
        _ = _ := by push_cast; ring)
  exact ⟨⟨hs.1, hw.1, hw.2, positive, hd.2, hmf, p.epsFinite, ha.1, p.epsNormal⟩,
    FlushToZero.add_ge_normal_right _ _ hmf p.epsFinite hmn p.epsNormal ha.1⟩

theorem Profile.finish (p : Profile) (platform : RmsNorm.Platform)
    (input : Fin n → Float32) (weight : Fin n → BFloat16Arithmetic.Value)
    (hx : ∀ i, (input i).isFinite = true ∧ |real (input i)| ≤ (2 : ℝ)^p.inputExponent)
    (hw : ∀ i, Model.isFinite (weight i) = true ∧ |Model.toReal (weight i)| ≤ Model.toReal p.weightCap)
    (reduction : ReductionFinite platform.divide input (RmsNorm.expression platform n) (Float32.ofInt n) p.eps)
    (inverse : (normalizer platform.divide input (RmsNorm.expression platform n) (Float32.ofInt n) p.eps platform.rsqrt).isFinite = true ∧
      |real (normalizer platform.divide input (RmsNorm.expression platform n) (Float32.ofInt n) p.eps platform.rsqrt)| ≤ real p.inverseCap) :
    Finite platform.divide input weight (RmsNorm.expression platform n) (Float32.ofInt n) p.eps platform.rsqrt ∧
      ∀ i, |real (output platform.divide input weight (RmsNorm.expression platform n) (Float32.ofInt n) p.eps platform.rsqrt i)| ≤
        Model.toReal p.outputCap := by
  let expr := RmsNorm.expression platform n
  have hm (i : Fin n) := NativeMultiplication.range (input i)
    (normalizer platform.divide input expr (Float32.ofInt n) p.eps platform.rsqrt) p.productCap
    (hx i).1 inverse.1 p.productFinite
    ((mul_le_mul (hx i).2 inverse.2 (abs_nonneg _) (zpow_nonneg (by norm_num) _)).trans p.productCapacity)
  have hb (i : Fin n) := BFloat16ConversionBounds.narrow_range
    (normalized platform.divide input expr (Float32.ofInt n) p.eps platform.rsqrt i) p.normalizedCap
    (FlushToZero.finite_flush _ (hm i).1) p.normalizedFinite ((hm i).2.trans p.normalizedCapacity)
  have ho (i : Fin n) := BFloat16Bounds.mul_range (weight i)
    (BFloat16.narrow (normalized platform.divide input expr (Float32.ofInt n) p.eps platform.rsqrt i)) p.outputCap
    (hw i).1 (hb i).1 p.outputFinite
    ((mul_le_mul (hw i).2 (hb i).2 (abs_nonneg _) ((abs_nonneg _).trans (hw i).2)).trans p.outputCapacity)
  refine ⟨{ toReductionFinite := reduction
            hNormalizer := inverse.1
            hNormalized := fun i => (hm i).1
            hNarrow := fun i => (hb i).1
            hWeight := fun i => (hw i).1
            hOutput := fun i => (ho i).1 }, ?_⟩
  intro i
  unfold output
  have hf : Model.isFinite (BFloat16Arithmetic.mul (weight i)
      (BFloat16.narrow (normalized platform.divide input expr (Float32.ofInt n) p.eps platform.rsqrt i))) = true :=
    BFloat16Arithmetic.finite_flush _ (ho i).1
  rw [BFloat16.widen_real _ hf]
  exact (ho i).2

theorem Profile.ranges (p : Profile) (platform : RmsNorm.Platform)
    (division : DivisionBounds.Contract platform.divide)
    (root : NormalizationBounds.RsqrtRange platform.rsqrt p.eps p.inverseCap)
    (input : Fin n → Float32) (weight : Fin n → BFloat16Arithmetic.Value)
    (hx : ∀ i, (input i).isFinite = true ∧ |real (input i)| ≤ (2 : ℝ)^p.inputExponent)
    (hw : ∀ i, Model.isFinite (weight i) = true ∧ |Model.toReal (weight i)| ≤ Model.toReal p.weightCap)
    (positive : 0 < n) (size : n+1 < 16777216) :
    Finite platform.divide input weight (RmsNorm.expression platform n) (Float32.ofInt n) p.eps platform.rsqrt ∧
      ∀ i, |real (output platform.divide input weight (RmsNorm.expression platform n) (Float32.ofInt n) p.eps platform.rsqrt i)| ≤
        Model.toReal p.outputCap := by
  have hr := p.reduction platform division input hx positive size
  have hd := denominator_finite platform.divide input (RmsNorm.expression platform n) (Float32.ofInt n) p.eps hr.1
  exact p.finish platform input weight hx hw hr.1 (root.range _ hd hr.2)

end Inference.Contracts.Mlx.NormalizationDyadicBounds
