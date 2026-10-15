import contracts.mlx.Normalization
import contracts.mlx.SquaredSum
import contracts.mlx.BFloat16
import contracts.mlx.BFloat16Arithmetic

namespace Inference.Contracts.Mlx.NormalizationExecution
open Numerics
open FloatLib FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open scoped BigOperators

-- Arithmetic of MLX v0.32.2 rms_norm.metal: float square reduction, divide
-- by width, add epsilon, precise rsqrt, float multiply, bf16 conversion, then
-- bf16 weight multiply. Source scheduling/coverage is an explicit obligation;
-- a generic reduction primitive is not assumed to compute the right sum.
def sum (input : Fin n → Float32) (expr : SquaredSum.Expr n) : Float32 := SquaredSum.evaluate input expr
def mean (input : Fin n → Float32) (expr : SquaredSum.Expr n) (width : Float32) : Float32 :=
  FlushToZero.div (sum input expr) width
def denominator (input : Fin n → Float32) (expr : SquaredSum.Expr n) (width eps : Float32) : Float32 :=
  FlushToZero.add (mean input expr width) eps
def normalizer (input : Fin n → Float32) (expr : SquaredSum.Expr n) (width eps : Float32)
    (rsqrt : Float32 → Float32) : Float32 := rsqrt (denominator input expr width eps)
def normalized (input : Fin n → Float32) (expr : SquaredSum.Expr n) (width eps : Float32)
    (rsqrt : Float32 → Float32) (i : Fin n) : Float32 := FlushToZero.mul (input i) (normalizer input expr width eps rsqrt)
def output (input : Fin n → Float32) (weight : Fin n → BFloat16Arithmetic.Value)
    (expr : SquaredSum.Expr n) (width eps : Float32) (rsqrt : Float32 → Float32) (i : Fin n) : Float32 :=
  BFloat16.widen (BFloat16Arithmetic.mul (weight i) (BFloat16.narrow (normalized input expr width eps rsqrt i)))

structure ReductionFinite (input : Fin n → Float32) (expr : SquaredSum.Expr n) (width eps : Float32) : Prop where
  hSum : SquaredSum.Finite input expr
  hWidth : width.isFinite=true
  hWidthValue : real width=(n : ℝ)
  hPositiveWidth : 0<n
  hMean : (FlushToZero.flush (sum input expr)/width).isFinite=true
  hEps : eps.isFinite=true
  hDenominator : (FlushToZero.flush (mean input expr width)+FlushToZero.flush eps).isFinite=true
  hNormalEps : FlushToZero.threshold < real eps

noncomputable def meanBudget (input : Fin n → Float32) (expr : SquaredSum.Expr n) (width : Float32) : ℝ :=
  FlushToZero.divBudget (sum input expr) width + SquaredSum.budget input expr / real width

noncomputable def denominatorBudget (input : Fin n → Float32) (expr : SquaredSum.Expr n)
    (width eps : Float32) (epsError : ℝ) : ℝ :=
  FlushToZero.addBudget (mean input expr width) eps + meanBudget input expr width + epsError

theorem mean_finite (input : Fin n → Float32) (expr : SquaredSum.Expr n) (width eps : Float32)
    (hf : ReductionFinite input expr width eps) : (mean input expr width).isFinite=true := by
  have hw : 1 ≤ real width := by rw [hf.hWidthValue]; exact_mod_cast hf.hPositiveWidth
  unfold mean FlushToZero.div
  rw [FlushToZero.flush_eq_of_one_le width hw]
  exact FlushToZero.finite_flush _ hf.hMean

theorem denominator_finite (input : Fin n → Float32) (expr : SquaredSum.Expr n) (width eps : Float32)
    (hf : ReductionFinite input expr width eps) : (denominator input expr width eps).isFinite=true :=
  FlushToZero.finite_flush _ hf.hDenominator

theorem denominator_positive (input : Fin n → Float32) (expr : SquaredSum.Expr n) (width eps : Float32)
    (hf : ReductionFinite input expr width eps) : 0 < real (denominator input expr width eps) := by
  have hw : 1 ≤ real width := by rw [hf.hWidthValue]; exact_mod_cast hf.hPositiveWidth
  have hm : 0 ≤ real (mean input expr width) :=
    FlushToZero.div_nonnegative _ _ hf.hWidth (SquaredSum.nonnegative input expr hf.hSum)
      (by linarith) (by rw [FlushToZero.flush_eq_of_one_le width hw]; exact hf.hMean)
  have hd := FlushToZero.add_ge_normal_right (mean input expr width) eps
    (mean_finite input expr width eps hf) hf.hEps hm hf.hNormalEps hf.hDenominator
  exact lt_of_lt_of_le (lt_trans FlushToZero.threshold_pos hf.hNormalEps) hd

theorem mean_error (input : Fin n → Float32) (expr : SquaredSum.Expr n)
    (coverage : (SquaredSum.indices expr).Perm (List.finRange n)) (width eps : Float32)
    (hf : ReductionFinite input expr width eps) :
    |real (mean input expr width)-Normalization.meanSquare (fun i => real (input i))| ≤ meanBudget input expr width := by
  have hw : 1 ≤ real width := by rw [hf.hWidthValue]; exact_mod_cast hf.hPositiveWidth
  have hs := SquaredSum.sum_error input expr coverage hf.hSum
  rw [← Fin.sum_univ_def] at hs
  have hd := FlushToZero.div_normal_error (sum input expr) width hf.hWidth hw hf.hMean
  have hi : |real (sum input expr)/real width-(∑ i, real (input i)^2)/real width| ≤
      SquaredSum.budget input expr/real width := by
    rw [← sub_div, abs_div, abs_of_nonneg (show 0 ≤ real width by linarith)]
    exact div_le_div_of_nonneg_right hs (by linarith)
  unfold Normalization.meanSquare meanBudget
  rw [← hf.hWidthValue]
  calc
    _ = |(real (mean input expr width)-real (sum input expr)/real width) +
        (real (sum input expr)/real width-(∑ i, real (input i)^2)/real width)| := by congr 1; ring
    _ ≤ _ := (abs_add_le _ _).trans (add_le_add hd hi)

theorem denominator_error (input : Fin n → Float32) (expr : SquaredSum.Expr n)
    (coverage : (SquaredSum.indices expr).Perm (List.finRange n)) (width eps : Float32)
    (idealEps epsError : ℝ) (he : |real eps-idealEps| ≤ epsError) (hf : ReductionFinite input expr width eps) :
    |real (denominator input expr width eps)-Normalization.denominator (fun i => real (input i)) idealEps| ≤
      denominatorBudget input expr width eps epsError := by
  have hm := mean_error input expr coverage width eps hf
  have ha := FlushToZero.add_error (mean input expr width) eps (mean_finite input expr width eps hf) hf.hEps hf.hDenominator
  change |real (denominator input expr width eps)-(real (mean input expr width)+real eps)| ≤
    FlushToZero.addBudget (mean input expr width) eps at ha
  unfold Normalization.denominator denominatorBudget
  calc
    _ = |(real (denominator input expr width eps)-(real (mean input expr width)+real eps)) +
        ((real (mean input expr width)-Normalization.meanSquare (fun i => real (input i)))+(real eps-idealEps))| := by congr 1; ring
    _ ≤ _ := by
      have h := (abs_add_le _ _).trans (add_le_add ha ((abs_add_le _ _).trans (add_le_add hm he)))
      linarith

noncomputable def normalizerBudget (input : Fin n → Float32) (expr : SquaredSum.Expr n)
    (width eps : Float32) (idealEps epsError : ℝ) (rsqrtError : Float32 → ℝ) : ℝ :=
  let d := real (denominator input expr width eps)
  let t := Normalization.denominator (fun i => real (input i)) idealEps
  rsqrtError (denominator input expr width eps)+denominatorBudget input expr width eps epsError /
    (Real.sqrt d*Real.sqrt t*(Real.sqrt d+Real.sqrt t))

theorem normalizer_error (input : Fin n → Float32) (expr : SquaredSum.Expr n)
    (coverage : (SquaredSum.indices expr).Perm (List.finRange n)) (width eps : Float32)
    (idealEps epsError : ℝ) (hp : 0 < idealEps) (he : |real eps-idealEps| ≤ epsError)
    (hf : ReductionFinite input expr width eps) (rsqrt : Float32 → Float32) (rsqrtError : Float32 → ℝ)
    (hardware : Normalization.RsqrtContract rsqrt rsqrtError)
    (ho : (normalizer input expr width eps rsqrt).isFinite=true) :
    |real (normalizer input expr width eps rsqrt)-
      1/Real.sqrt (Normalization.denominator (fun i => real (input i)) idealEps)| ≤
        normalizerBudget input expr width eps idealEps epsError rsqrtError := by
  exact Normalization.rsqrt_error rsqrt rsqrtError hardware _ _ _ (denominator_finite input expr width eps hf)
    (denominator_positive input expr width eps hf) (Normalization.denominator_positive _ _ hp) ho
    (denominator_error input expr coverage width eps idealEps epsError he hf)

noncomputable def normalizedBudget (input : Fin n → Float32) (expr : SquaredSum.Expr n)
    (width eps : Float32) (idealEps epsError : ℝ) (rsqrt : Float32 → Float32)
    (rsqrtError : Float32 → ℝ) (i : Fin n) : ℝ :=
  BFloat16.epsilon (real (normalized input expr width eps rsqrt i))+
    FlushToZero.mulBudget (input i) (normalizer input expr width eps rsqrt)+
    |real (input i)| *normalizerBudget input expr width eps idealEps epsError rsqrtError

theorem normalized_error (input : Fin n → Float32) (expr : SquaredSum.Expr n)
    (coverage : (SquaredSum.indices expr).Perm (List.finRange n)) (width eps : Float32)
    (idealEps epsError : ℝ) (hp : 0 < idealEps) (he : |real eps-idealEps| ≤ epsError)
    (hf : ReductionFinite input expr width eps) (rsqrt : Float32 → Float32) (rsqrtError : Float32 → ℝ)
    (hardware : Normalization.RsqrtContract rsqrt rsqrtError)
    (ho : (normalizer input expr width eps rsqrt).isFinite=true) (i : Fin n)
    (hm : (FlushToZero.flush (input i)*FlushToZero.flush (normalizer input expr width eps rsqrt)).isFinite=true)
    (hb : Model.isFinite (BFloat16.narrow (normalized input expr width eps rsqrt i))=true) :
    |Model.toReal (BFloat16.narrow (normalized input expr width eps rsqrt i))-
      real (input i)/Real.sqrt (Normalization.denominator (fun j => real (input j)) idealEps)| ≤
        normalizedBudget input expr width eps idealEps epsError rsqrt rsqrtError i := by
  have en := normalizer_error input expr coverage width eps idealEps epsError hp he hf rsqrt rsqrtError hardware ho
  have em := FlushToZero.mul_error (input i) (normalizer input expr width eps rsqrt) hm
  change |real (normalized input expr width eps rsqrt i)-real (input i)*real (normalizer input expr width eps rsqrt)| ≤
    FlushToZero.mulBudget (input i) (normalizer input expr width eps rsqrt) at em
  have eb := BFloat16.narrow_error (normalized input expr width eps rsqrt i) (FlushToZero.finite_flush _ hm) hb
  have ex := mul_le_mul_of_nonneg_left en (abs_nonneg (real (input i)))
  rw [← abs_mul] at ex
  unfold normalizedBudget
  calc
    _ = |(Model.toReal (BFloat16.narrow (normalized input expr width eps rsqrt i))-
          real (normalized input expr width eps rsqrt i)) +
        ((real (normalized input expr width eps rsqrt i)-real (input i)*real (normalizer input expr width eps rsqrt)) +
          real (input i)*(real (normalizer input expr width eps rsqrt)-
            1/Real.sqrt (Normalization.denominator (fun j => real (input j)) idealEps)))| := by congr 1; ring
    _ ≤ _ := by
      have h := (abs_add_le _ _).trans (add_le_add eb ((abs_add_le _ _).trans (add_le_add em ex)))
      linarith

theorem output_error (input : Fin n → Float32) (weight : Fin n → BFloat16Arithmetic.Value)
    (expr : SquaredSum.Expr n) (coverage : (SquaredSum.indices expr).Perm (List.finRange n)) (width eps : Float32)
    (idealEps epsError : ℝ) (hp : 0 < idealEps) (he : |real eps-idealEps| ≤ epsError)
    (hf : ReductionFinite input expr width eps) (rsqrt : Float32 → Float32) (rsqrtError : Float32 → ℝ)
    (hardware : Normalization.RsqrtContract rsqrt rsqrtError)
    (ho : (normalizer input expr width eps rsqrt).isFinite=true) (i : Fin n)
    (hm : (FlushToZero.flush (input i)*FlushToZero.flush (normalizer input expr width eps rsqrt)).isFinite=true)
    (hb : Model.isFinite (BFloat16.narrow (normalized input expr width eps rsqrt i))=true)
    (hw : Model.isFinite (weight i)=true)
    (hout : Model.isFinite (Model.mul (BFloat16Arithmetic.flush (weight i))
      (BFloat16Arithmetic.flush (BFloat16.narrow (normalized input expr width eps rsqrt i))))=true) :
    |real (output input weight expr width eps rsqrt i)-
      Normalization.rmsNorm (fun j => real (input j)) (fun j => Model.toReal (weight j)) idealEps i| ≤
      BFloat16Arithmetic.mulBudget (weight i) (BFloat16.narrow (normalized input expr width eps rsqrt i)) +
        |Model.toReal (weight i)| *normalizedBudget input expr width eps idealEps epsError rsqrt rsqrtError i := by
  have en := normalized_error input expr coverage width eps idealEps epsError hp he hf rsqrt rsqrtError hardware ho i hm hb
  have ew := BFloat16Arithmetic.mul_error (weight i) (BFloat16.narrow (normalized input expr width eps rsqrt i)) hw hb hout
  have ex := mul_le_mul_of_nonneg_left en (abs_nonneg (Model.toReal (weight i)))
  rw [← abs_mul] at ex
  unfold output Normalization.rmsNorm
  have hfout : Model.isFinite (BFloat16Arithmetic.mul (weight i) (BFloat16.narrow (normalized input expr width eps rsqrt i)))=true :=
    BFloat16Arithmetic.finite_flush _ hout
  rw [BFloat16.widen_real _ hfout]
  calc
    _ = |(Model.toReal (BFloat16Arithmetic.mul (weight i) (BFloat16.narrow (normalized input expr width eps rsqrt i)))-
          Model.toReal (weight i)*Model.toReal (BFloat16.narrow (normalized input expr width eps rsqrt i))) +
        Model.toReal (weight i)*(Model.toReal (BFloat16.narrow (normalized input expr width eps rsqrt i))-
          real (input i)/Real.sqrt (Normalization.denominator (fun j => real (input j)) idealEps))| := by congr 1; ring
    _ ≤ _ := (abs_add_le _ _).trans (add_le_add ew ex)

end Inference.Contracts.Mlx.NormalizationExecution
