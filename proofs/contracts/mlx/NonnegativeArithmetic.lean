import contracts.mlx.SquaredSumDyadicBounds
import contracts.mlx.FlushedIdentities

namespace Inference.Contracts.Mlx.NonnegativeArithmetic
open Numerics Arithmetic
open FloatLib FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

-- Only operations preserving nonnegativity are admitted. Finiteness is
-- established separately by an ArithmeticBounds certificate.
inductive Nonnegative (input : Nat → Float32) : Expr → Prop where
  | input (i : Nat) (h : 0 ≤ real (input i)) : Nonnegative input (.input i)
  | zero : Nonnegative input .zero
  | add {a b} : Nonnegative input a → Nonnegative input b → Nonnegative input (.add a b)
  | mul {a b} : Nonnegative input a → Nonnegative input b → Nonnegative input (.mul a b)
  | fma {a b c} : Nonnegative input a → Nonnegative input b → Nonnegative input c → Nonnegative input (.fma a b c)

theorem nonnegative (input : Nat → Float32) (expr : Expr)
    (h : Nonnegative input expr) (hf : Finite input expr) : 0 ≤ real (evaluate input expr) := by
  induction h with
  | input i hi => exact hi
  | zero => rw [evaluate, FlushedIdentities.zero_real]
  | add ha hb ia ib =>
    exact FlushToZero.add_nonnegative _ _ (finite_evaluate _ _ hf.1)
      (finite_evaluate _ _ hf.2.1) (ia hf.1) (ib hf.2.1) hf.2.2
  | mul ha hb ia ib =>
    exact FlushToZero.mul_nonnegative _ _
      (mul_nonneg (FlushToZero.flush_nonnegative _ (ia hf.1))
        (FlushToZero.flush_nonnegative _ (ib hf.2.1))) hf.2.2
  | fma ha hb hc ia ib ic =>
    apply FlushToZero.flush_nonnegative
    rw [fma_real _ _ _ (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.1))
      (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.2.1))
      (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.2.2.1)) hf.2.2.2]
    exact FlushToZero.round_nonnegative _ (add_nonneg
      (mul_nonneg (FlushToZero.flush_nonnegative _ (ia hf.1))
        (FlushToZero.flush_nonnegative _ (ib hf.2.1)))
      (FlushToZero.flush_nonnegative _ (ic hf.2.2.1)))

private theorem one_normal : FlushToZero.threshold < real (1 : Float32) := by
  rw [real_one]
  norm_num [FlushToZero.threshold]

private theorem flush_lower_one (x : Float32) (hx : 1 ≤ real x) : 1 ≤ real (FlushToZero.flush x) := by
  rw [FlushToZero.flush_eq_of_one_le x hx]
  exact hx

private theorem rounded_lower_one (x : ℝ) (hx : 1 ≤ x) : 1 ≤ Model.roundAt FloatFormat.binary32 x := by
  simpa only [real_one] using SquaredSumDyadicBounds.round_lower 1 (by decide) x (by rwa [real_one])

-- A maximum contributes exp(0)=1. This witness survives nonnegative sums
-- and multiplication by one, including a contracted multiply-add.
inductive LowerOne (input : Nat → Float32) : Expr → Prop where
  | input (i : Nat) (h : 1 ≤ real (input i)) : LowerOne input (.input i)
  | addLeft {a b} : LowerOne input a → Nonnegative input b → LowerOne input (.add a b)
  | addRight {a b} : Nonnegative input a → LowerOne input b → LowerOne input (.add a b)
  | mulRightOne {a b} : LowerOne input a → real (evaluate input b) = 1 → LowerOne input (.mul a b)
  | fmaProduct {a b c} : LowerOne input a → real (evaluate input b) = 1 → Nonnegative input c → LowerOne input (.fma a b c)
  | fmaAddend {a b c} : Nonnegative input a → Nonnegative input b → LowerOne input c → LowerOne input (.fma a b c)

theorem lower_one (input : Nat → Float32) (expr : Expr)
    (h : LowerOne input expr) (hf : Finite input expr) : 1 ≤ real (evaluate input expr) := by
  induction h with
  | input i hi => exact hi
  | addLeft ha hb ih =>
    apply flush_lower_one
    rw [add_real _ _ (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.1))
      (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.2.1)) hf.2.2]
    exact rounded_lower_one _ ((flush_lower_one _ (ih hf.1)).trans
      (le_add_of_nonneg_right (FlushToZero.flush_nonnegative _ (nonnegative _ _ hb hf.2.1))))
  | addRight ha hb ih =>
    apply flush_lower_one
    rw [add_real _ _ (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.1))
      (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.2.1)) hf.2.2]
    exact rounded_lower_one _ ((flush_lower_one _ (ih hf.2.1)).trans
      (le_add_of_nonneg_left (FlushToZero.flush_nonnegative _ (nonnegative _ _ ha hf.1))))
  | mulRightOne ha hb ih =>
    apply flush_lower_one
    rw [mul_real _ _ hf.2.2, FlushedIdentities.flush_one _ hb, mul_one]
    exact rounded_lower_one _ (flush_lower_one _ (ih hf.1))
  | fmaProduct ha hb hc ih =>
    apply flush_lower_one
    rw [fma_real _ _ _ (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.1))
      (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.2.1))
      (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.2.2.1)) hf.2.2.2,
      FlushedIdentities.flush_one _ hb, mul_one]
    exact rounded_lower_one _ ((flush_lower_one _ (ih hf.1)).trans
      (le_add_of_nonneg_right (FlushToZero.flush_nonnegative _ (nonnegative _ _ hc hf.2.2.1))))
  | fmaAddend ha hb hc ih =>
    apply flush_lower_one
    rw [fma_real _ _ _ (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.1))
      (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.2.1))
      (FlushToZero.finite_flush _ (finite_evaluate _ _ hf.2.2.1)) hf.2.2.2]
    exact rounded_lower_one _ ((flush_lower_one _ (ih hf.2.2.1)).trans
      (le_add_of_nonneg_left (mul_nonneg (FlushToZero.flush_nonnegative _ (nonnegative _ _ ha hf.1))
        (FlushToZero.flush_nonnegative _ (nonnegative _ _ hb hf.2.1)))))

end Inference.Contracts.Mlx.NonnegativeArithmetic
