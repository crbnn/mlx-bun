import contracts.mlx.Numerics
import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Proof

namespace Inference.Contracts.Mlx.BFloat16
open FloatLib FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange Numerics

-- Native Metal bfloat conversion is an instruction/compiler boundary. This
-- executable specification uses nearest-even with gradual underflow. Its
-- conformance on the target is a separate premise, not a numerical theorem.
def narrow (x : Float32) : Model FloatFormat.bfloat16 :=
  Model.cast FloatFormat.binary32 FloatFormat.bfloat16 (model x)

def cast (x : Float32) : Float32 :=
  Float32.ofModel (Model.toFloat32Model
    (Model.cast FloatFormat.bfloat16 FloatFormat.binary32 (narrow x)))

def widen (x : Model FloatFormat.bfloat16) : Float32 :=
  Float32.ofModel (Model.toFloat32Model (Model.cast FloatFormat.bfloat16 FloatFormat.binary32 x))

theorem bridge (x : Model FloatFormat.binary32) :
    real (Float32.ofModel (Model.toFloat32Model x)) = Model.toReal x := by
  rw [real_unpack]
  change Model.unpackedToReal (Model.toFloat32Model x).unpack = Model.toReal x
  rw [Model.unpack_toFloat32Model]
  exact (Model.toReal_eq_unpackedToReal_toModel (by decide) x).symm

theorem cast_real (x : Float32) (hx : x.isFinite=true)
    (hy : Model.isFinite (narrow x)=true) :
    real (cast x) = Model.roundAt FloatFormat.bfloat16 (real x) := by
  unfold cast
  rw [bridge, Model.cast_exact_of_compatibleWidening (narrow x)
    (by decide) (by decide) (by decide) (by decide) hy]
  exact Model.cast_eq_roundAt (by decide) (by decide) (model x)
    (by rw [finite_model]; exact hx) hy

noncomputable def epsilon (x : ℝ) : ℝ := Model.epsilonAt FloatFormat.bfloat16 x

theorem narrow_error (x : Float32) (hx : x.isFinite=true)
    (hy : Model.isFinite (narrow x)=true) :
    |Model.toReal (narrow x)-real x| ≤ epsilon (real x) := by
  have h := Model.cast_eq_roundAt (by decide) (by decide) (model x)
    (by rw [finite_model]; exact hx) hy
  change Model.toReal (narrow x) = Model.roundAt FloatFormat.bfloat16 (real x) at h
  rw [h]
  exact Model.abs_roundAt_sub_le _ _

theorem widen_real (x : Model FloatFormat.bfloat16) (hx : Model.isFinite x=true) :
    real (widen x) = Model.toReal x := by
  unfold widen
  rw [bridge]
  exact Model.cast_exact_of_compatibleWidening x (by decide) (by decide) (by decide) (by decide) hx

theorem cast_error (x : Float32) (hx : x.isFinite=true)
    (hy : Model.isFinite (narrow x)=true) :
    |real (cast x)-real x| ≤ epsilon (real x) := by
  rw [cast_real x hx hy]
  exact Model.abs_roundAt_sub_le _ _

structure RneContract (convert : Float32 → Float32) : Prop where
  execution : ∀ x, x.isFinite=true → Model.isFinite (narrow x)=true → convert x = cast x

theorem rne_error (convert : Float32 → Float32) (hardware : RneContract convert)
    (x : Float32) (hx : x.isFinite=true) (hy : Model.isFinite (narrow x)=true) :
    |real (convert x)-real x| ≤ epsilon (real x) := by
  rw [hardware.execution x hx hy]
  exact cast_error x hx hy

theorem modelContract : RneContract cast := ⟨fun _ _ _ => rfl⟩

end Inference.Contracts.Mlx.BFloat16
