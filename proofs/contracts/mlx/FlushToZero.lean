import contracts.mlx.Numerics

namespace Inference.Contracts.Mlx.FlushToZero
open Numerics
open FloatLib FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

-- Binary32 subnormals have an exponent of -149 and fewer than 24 significant
-- bits. Model input and output flushing explicitly; IEEE gradual underflow is
-- not an unconditional description of M1 Max arithmetic.
def flush (x : Float32) : Float32 :=
  match x.toModel.unpack with
  | .finite s m e _ =>
    if m<8388608 ∧ e = -149 then Float32.ofModel (Float32.Model.pack (.zero s)) else x
  | _ => x

noncomputable def threshold : ℝ := (2 : ℝ)^(-126 : ℤ)

theorem threshold_pos : 0 < threshold := by unfold threshold; positivity

theorem real_signed_zero (s : Float.Model.UnpackedFloat.Sign) :
    real (Float32.ofModel (Float32.Model.pack (.zero s))) = 0 := by
  rw [real_unpack]
  cases s <;> exact Model.unpackedToReal_zero _

theorem finite_flush (x : Float32) (hx : x.isFinite=true) : (flush x).isFinite=true := by
  unfold flush
  split
  · split
    · rename_i s _ _ _ _ _
      cases s <;> decide
    · exact hx
  · exact hx

theorem flush_abs_le (x : Float32) : |real (flush x)| ≤ |real x| := by
  unfold flush
  split
  · split
    · rw [real_signed_zero, abs_zero]
      exact abs_nonneg _
    · exact le_refl _
  · exact le_refl _

theorem flush_error (x : Float32) : |real (flush x)-real x| ≤ threshold := by
  unfold flush
  split
  · rename_i s m e hm hu
    split
    · rename_i h
      rw [real_signed_zero, zero_sub, abs_neg, real_unpack, hu, Model.unpackedToReal_finite]
      obtain ⟨hsmall, rfl⟩ := h
      have hm' : (m : ℝ) ≤ 8388608 := by exact_mod_cast (show m≤8388608 by omega)
      have hp : 0 < (2 : ℝ)^(-149 : ℤ) := by positivity
      have hbound := mul_le_mul_of_nonneg_right hm' hp.le
      cases s <;> norm_num [Model.modelSignBit, FloatLib.Floats.Formats.Flocq.bpow] at *
      all_goals
        change (m : ℝ) * (|(2 : ℝ)|^149)⁻¹ ≤ (2 : ℝ)^(-126 : ℤ)
        norm_num at hbound ⊢
        linarith
    · simpa using threshold_pos.le
  · simpa using threshold_pos.le

def mul (x y : Float32) : Float32 := flush (flush x * flush y)
def add (x y : Float32) : Float32 := flush (flush x + flush y)
def fma (x y z : Float32) : Float32 := flush (Float32.fma (flush x) (flush y) (flush z))

theorem product_input_error (x y : Float32) :
    |real (flush x)*real (flush y)-real x*real y| ≤ threshold*(|real x|+|real y|) := by
  have ex := flush_error x
  have ey := flush_error y
  have exy := mul_le_mul_of_nonneg_right ex (abs_nonneg (real y))
  have eyx := mul_le_mul_of_nonneg_left ey (abs_nonneg (real (flush x)))
  have hm := mul_le_mul_of_nonneg_right (flush_abs_le x) threshold_pos.le
  calc
    _ = |real (flush x)*(real (flush y)-real y)+(real (flush x)-real x)*real y| := by congr 1; ring
    _ ≤ _ := by
      have h := abs_add_le (real (flush x)*(real (flush y)-real y)) ((real (flush x)-real x)*real y)
      simp only [abs_mul] at h
      nlinarith

theorem mul_error (x y : Float32) (hf : (flush x*flush y).isFinite=true) :
    |real (mul x y)-real x*real y| ≤
      threshold + epsilon (real (flush x)*real (flush y)) + threshold*(|real x|+|real y|) := by
  have eo := flush_error (flush x*flush y)
  have er := Numerics.mul_error (flush x) (flush y) hf
  have ei := product_input_error x y
  unfold mul
  calc
    _ = |(real (flush (flush x*flush y))-real (flush x*flush y)) +
        ((real (flush x*flush y)-real (flush x)*real (flush y)) +
          (real (flush x)*real (flush y)-real x*real y))| := by congr 1; ring
    _ ≤ _ := by
      have h := (abs_add_le _ _).trans (add_le_add eo ((abs_add_le _ _).trans (add_le_add er ei)))
      linarith

theorem add_error (x y : Float32) (hx : x.isFinite=true) (hy : y.isFinite=true)
    (hf : (flush x+flush y).isFinite=true) :
    |real (add x y)-(real x+real y)| ≤
      3*threshold + epsilon (real (flush x)+real (flush y)) := by
  have eo := flush_error (flush x+flush y)
  have er := Numerics.add_error (flush x) (flush y) (finite_flush x hx) (finite_flush y hy) hf
  have ex := flush_error x
  have ey := flush_error y
  unfold add
  calc
    _ = |(real (flush (flush x+flush y))-real (flush x+flush y)) +
        ((real (flush x+flush y)-(real (flush x)+real (flush y))) +
          ((real (flush x)-real x)+(real (flush y)-real y)))| := by congr 1; ring
    _ ≤ _ := by
      have h := (abs_add_le _ _).trans (add_le_add eo ((abs_add_le _ _).trans (add_le_add er
        ((abs_add_le _ _).trans (add_le_add ex ey)))))
      linarith

theorem fma_error (x y z : Float32) (hx : x.isFinite=true) (hy : y.isFinite=true) (hz : z.isFinite=true)
    (hf : (Float32.fma (flush x) (flush y) (flush z)).isFinite=true) :
    |real (fma x y z)-(real x*real y+real z)| ≤
      2*threshold + epsilon (real (flush x)*real (flush y)+real (flush z)) + threshold*(|real x|+|real y|) := by
  have eo := flush_error (Float32.fma (flush x) (flush y) (flush z))
  have er := Numerics.fma_error (flush x) (flush y) (flush z) (finite_flush x hx) (finite_flush y hy) (finite_flush z hz) hf
  have ei := product_input_error x y
  have ez := flush_error z
  unfold fma
  calc
    _ = |(real (flush (Float32.fma (flush x) (flush y) (flush z)))-real (Float32.fma (flush x) (flush y) (flush z))) +
        ((real (Float32.fma (flush x) (flush y) (flush z))-(real (flush x)*real (flush y)+real (flush z))) +
          ((real (flush x)*real (flush y)-real x*real y)+(real (flush z)-real z)))| := by congr 1; ring
    _ ≤ _ := by
      have h := (abs_add_le _ _).trans (add_le_add eo ((abs_add_le _ _).trans (add_le_add er
        ((abs_add_le _ _).trans (add_le_add ei ez)))))
      linarith

-- A kernel-checked distinction between the two models, also observed by the
-- locked M1 Max probe. This finite witness cannot establish all hardware cases.
theorem underflow_witness :
    (Float32.ofBits 0x00800000 * Float32.ofBits 0x3f000000).toBits = 0x00400000 ∧
    (mul (Float32.ofBits 0x00800000) (Float32.ofBits 0x3f000000)).toBits = 0 := by decide +kernel

noncomputable def mulBudget (x y : Float32) : ℝ :=
  threshold + epsilon (real (flush x)*real (flush y)) + threshold*(|real x|+|real y|)

noncomputable def addBudget (x y : Float32) : ℝ :=
  3*threshold + epsilon (real (flush x)+real (flush y))

noncomputable def fmaBudget (x y z : Float32) : ℝ :=
  2*threshold + epsilon (real (flush x)*real (flush y)+real (flush z)) + threshold*(|real x|+|real y|)

theorem scaled_fma_error (code scale input acc : Float32)
    (hw : (flush code*flush scale).isFinite=true) (hx : input.isFinite=true) (ha : acc.isFinite=true)
    (ho : (Float32.fma (flush (mul code scale)) (flush input) (flush acc)).isFinite=true) :
    |real (fma (mul code scale) input acc)-(real code*real scale*real input+real acc)| ≤
      mulBudget code scale * |real input| + fmaBudget (mul code scale) input acc := by
  have em := mul_error code scale hw
  have ef := fma_error (mul code scale) input acc (finite_flush _ hw) hx ha ho
  have ep := mul_le_mul_of_nonneg_right em (abs_nonneg (real input))
  unfold mulBudget fmaBudget
  calc
    _ = |(real (fma (mul code scale) input acc)-(real (mul code scale)*real input+real acc)) +
        (real (mul code scale)-real code*real scale)*real input| := by congr 1; ring
    _ ≤ _ := by
      have e : |(real (mul code scale)-real code*real scale)*real input| ≤
          (threshold+epsilon (real (flush code)*real (flush scale))+threshold*(|real code|+|real scale|))* |real input| := by
        simpa only [abs_mul] using ep
      have h := (abs_add_le _ _).trans (add_le_add ef e)
      linarith

end Inference.Contracts.Mlx.FlushToZero
