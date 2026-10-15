import Std.Tactic.BVDecide

/-
First arithmetic obligation for the Qwen3.8-27B Trellis M1 Max graph.
Run: bun proofs/check.ts (Lean 4.35.0 must be on PATH).

The implementation below is a manually reviewed translation of trellis_y in
packages/inference/src/kernels/trellis/codebook.ts. check.ts pins that source;
the pin prevents silent drift but does not prove the translation correct.

BitVec 32 models Metal uint arithmetic modulo 2^32. The signed cast and
subtraction are modeled by BitVec.toInt; the theorems establish that neither
the cast nor the signed subtraction overflows.

bv_decide obtains a CaDiCaL certificate and checks it using Lean's verified
LRAT checker. Native evaluation of that checker trusts the Lean compiler in
addition to the logical kernel. No numerical axiom or sorry is introduced here.

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

theorem packedSum_eq_byteSum (x : BitVec 32) : packedSum x = byteSum x := by
  unfold packedSum byteSum
  bv_decide

theorem packedSum_bound (x : BitVec 32) : packedSum x ≤ 1020 := by
  unfold packedSum
  bv_decide

theorem byteSum_no_wrap (x : BitVec 32) :
    (byteSum x).toNat = (x &&& 255#32).toNat + ((x >>> 8) &&& 255#32).toNat +
      ((x >>> 16) &&& 255#32).toNat + ((x >>> 24) &&& 255#32).toNat := by
  have bound (v : BitVec 32) : (v &&& 255#32).toNat ≤ 255 := by
    have h : v &&& 255#32 ≤ 255#32 := by bv_decide
    simpa [BitVec.le_def] using h
  have h0 := bound x
  have h1 := bound (x >>> 8)
  have h2 := bound (x >>> 16)
  have h3 := bound (x >>> 24)
  simp only [byteSum, BitVec.toNat_add]
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

#print axioms correct
#print axioms range
end Trellis
