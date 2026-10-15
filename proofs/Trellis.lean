import Std

/-
First arithmetic obligation for the Qwen3.8-27B Trellis M1 Max graph.
Run: bun proofs/check.ts (Lean 4.35.0 must be on PATH).

The implementation below is a manually reviewed translation of trellis_y in
packages/inference/src/kernels/trellis/codebook.ts. check.ts pins that source;
the pin prevents silent drift but does not prove the translation correct.

BitVec 32 models Metal uint arithmetic modulo 2^32. The signed cast and
subtraction are modeled by BitVec.toInt; the theorems establish that neither
the cast nor the signed subtraction overflows.

All proofs use ordinary proof terms checked by Lean's logical kernel. The
axiom checks below permit only propext, Classical.choice, and Quot.sound;
native computation assumptions, custom axioms, and proof holes are rejected.

This proves the integer helper, not floating-point decoding, GPU memory access,
a projection kernel, a layer, or the complete inference graph.
-/
namespace Trellis

def mix (s : BitVec 32) : BitVec 32 := s * 34038481 + 76625530

def packedSum (x : BitVec 32) : BitVec 32 :=
  let p := (x &&& 0x00FF00FF) + ((x >>> 8) &&& 0x00FF00FF)
  (p &&& 0xFFFF) + (p >>> 16)

def byteSum (x : BitVec 32) : BitVec 32 :=
  (x &&& 255#32) + ((x >>> 8) &&& 255#32) +
  ((x >>> 16) &&& 255#32) + ((x >>> 24) &&& 255#32)

def implementation (s : BitVec 32) : Int := (packedSum (mix s)).toInt - 510

-- The specification uses mathematical integers, not a wrapping byte sum.
def specification (s : BitVec 32) : Int :=
  let x := mix s
  (x &&& 255#32).toNat + ((x >>> 8) &&& 255#32).toNat +
    ((x >>> 16) &&& 255#32).toNat + ((x >>> 24) &&& 255#32).toNat - (510 : Int)

private theorem mask8 (n : Nat) : n &&& 255 = n % 256 := by
  exact Nat.and_two_pow_sub_one_eq_mod n 8

private theorem mask16 (n : Nat) : n &&& 65535 = n % 65536 := by
  exact Nat.and_two_pow_sub_one_eq_mod n 16

-- The alternating-byte mask is two independent 16-bit lanes.
private theorem maskPairs (n : Nat) :
    n &&& 16711935 = n % 256 + 65536 * (n / 65536 % 256) := by
  have hm := Nat.and_mod_two_pow (a := n) (b := 16711935) (n := 16)
  have hd := Nat.and_div_two_pow (a := n) (b := 16711935) (n := 16)
  simp only [show 2^16 = 65536 from rfl, show 16711935 % 65536 = 255 from rfl,
    show 16711935 / 65536 = 255 from rfl, mask8] at hm hd
  have he := Nat.mod_add_div (n &&& 16711935) 65536
  omega

-- Each lane holds at most 255 + 255, so neither lane carries into the other
-- and their packed sum cannot overflow uint32. Folding the lanes sums bytes.
theorem packedSum_toNat (x : BitVec 32) :
    (packedSum x).toNat = x.toNat % 256 + (x.toNat / 256) % 256 +
      (x.toNat / 65536) % 256 + (x.toNat / 16777216) % 256 := by
  have hp : x.toNat % 256 + 65536 * (x.toNat / 65536 % 256) +
      (x.toNat / 256 % 256 + 65536 * (x.toNat / 16777216 % 256)) < 4294967296 := by
    have h0 := Nat.mod_lt x.toNat (by decide : 0 < 256)
    have h1 := Nat.mod_lt (x.toNat / 256) (by decide : 0 < 256)
    have h2 := Nat.mod_lt (x.toNat / 65536) (by decide : 0 < 256)
    have h3 := Nat.mod_lt (x.toNat / 16777216) (by decide : 0 < 256)
    omega
  simp [packedSum, BitVec.toNat_add, BitVec.toNat_and, BitVec.toNat_ushiftRight,
    Nat.shiftRight_eq_div_pow, mask16, maskPairs, Nat.div_div_eq_div_mul]
  rw [Nat.mod_eq_of_lt hp]
  have hpack : x.toNat % 256 + 65536 * (x.toNat / 65536 % 256) +
      (x.toNat / 256 % 256 + 65536 * (x.toNat / 16777216 % 256)) =
      (x.toNat % 256 + x.toNat / 256 % 256) +
      65536 * (x.toNat / 65536 % 256 + x.toNat / 16777216 % 256) := by omega
  rw [hpack]
  have hlo : x.toNat % 256 + x.toNat / 256 % 256 < 65536 := by omega
  simp only [Nat.add_mul_mod_self_left, Nat.add_mul_div_left _ _ (by decide : 0 < 65536),
    Nat.mod_eq_of_lt hlo, Nat.div_eq_of_lt hlo, Nat.zero_add]
  omega

theorem byteSum_no_wrap (x : BitVec 32) :
    (byteSum x).toNat = (x &&& 255#32).toNat + ((x >>> 8) &&& 255#32).toNat +
      ((x >>> 16) &&& 255#32).toNat + ((x >>> 24) &&& 255#32).toNat := by
  have bound (v : BitVec 32) : (v &&& 255#32).toNat ≤ 255 := by
    simpa using (Nat.and_le_right (n := v.toNat) (m := 255))
  have h0 := bound x
  have h1 := bound (x >>> 8)
  have h2 := bound (x >>> 16)
  have h3 := bound (x >>> 24)
  simp only [byteSum, BitVec.toNat_add]
  omega

theorem packedSum_eq_byteSum (x : BitVec 32) : packedSum x = byteSum x := by
  apply BitVec.eq_of_toNat_eq
  rw [packedSum_toNat, byteSum_no_wrap]
  simp [BitVec.toNat_and, BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow, mask8]

theorem packedSum_bound (x : BitVec 32) : packedSum x ≤ 1020 := by
  simp only [BitVec.le_def, packedSum_toNat]
  change x.toNat % 256 + (x.toNat / 256) % 256 +
    (x.toNat / 65536) % 256 + (x.toNat / 16777216) % 256 ≤ 1020
  omega

theorem correct (s : BitVec 32) : implementation s = specification s := by
  have hb := packedSum_bound (mix s)
  have hn := byteSum_no_wrap (mix s)
  have he := packedSum_eq_byteSum (mix s)
  simp [BitVec.le_def] at hb
  simp only [implementation, specification, BitVec.toInt_eq_toNat_cond]
  rw [ite_eq_left (by omega)]
  rw [he, hn]
  omega

theorem range (s : BitVec 32) : -510 ≤ implementation s ∧ implementation s ≤ 510 := by
  have hb := packedSum_bound (mix s)
  simp [BitVec.le_def] at hb
  simp only [implementation, BitVec.toInt_eq_toNat_cond]
  rw [ite_eq_left (by omega)]
  omega

/-- info: 'Trellis.correct' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms correct
/-- info: 'Trellis.range' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms range
end Trellis
