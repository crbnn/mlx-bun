import Std

namespace Inference.Kernels.Trellis.DownCodebook
/- Indexing obligations for down-codebook.ts with the pinned Qwen dimensions:
   R = 17408, C = 5120, SPLITS = 128, P = 16, U = 2, THREADS = 256.
   These theorems establish integer indexing and coverage, not float arithmetic,
   synchronization, or the hardware semantics of codebook loads and simd_sum. -/

def scratch (sg lane i : Nat) : Nat := sg*512 + ((lane/16)*256+255-lane%16*16)-i
def column (ty c : Nat) : Nat := (2*(ty/32)+c/512)*512+c%512
def splitRow (y sg r : Nat) : Nat := (y+32*(sg%4))*136+r

theorem scratch_bounds (sg lane i : Nat) (hs : sg<8) (hl : lane<32) (hi : i<16) :
    scratch sg lane i < 4096 := by unfold scratch; omega

-- Each of the eight SIMD groups, 32 lanes, and 16 entries writes a distinct
-- scratch location; the inverse also shows all 4096 locations are written.
theorem scratch_inverse (sg lane i : Nat) (hs : sg<8) (hl : lane<32) (hi : i<16) :
    scratch sg lane i / 512 = sg ∧
    (scratch sg lane i % 512)/256*16 + (255-scratch sg lane i%256)/16 = lane ∧
    (255-scratch sg lane i%256)%16 = i := by unfold scratch; omega

theorem scratch_coverage (j : Nat) (hj : j<4096) :
    scratch (j/512) ((j%512)/256*16+(255-j%256)/16) ((255-j%256)%16) = j := by
  unfold scratch
  omega

theorem partial_bounds (ty c : Nat) (ht : ty<160) (hc : c<1024) :
    ty%32 < 32 ∧ column ty c < 5120 ∧ ty%32*5120+column ty c < 32*5120 := by
  unfold column
  omega

theorem partial_inverse (ty c : Nat) (_ht : ty<160) (hc : c<1024) :
    32*(column ty c / 1024)+ty%32 = ty ∧ column ty c % 1024 = c := by
  unfold column
  omega

theorem partial_coverage (y col : Nat) (hy : y<32) (hc : col<5120) :
    (32*(col/1024)+y) < 160 ∧ col%1024 < 1024 ∧
    (32*(col/1024)+y)%32 = y ∧ column (32*(col/1024)+y) (col%1024) = col := by
  unfold column
  omega

theorem row_bounds (y sg r : Nat) (hy : y<32) (_hs : sg<8) (hr : r<136) :
    splitRow y sg r < 17408 := by unfold splitRow; omega

theorem row_coverage (r : Nat) (hr : r<17408) :
    r/136%32 < 32 ∧ r/136/32 < 4 ∧ r%136 < 136 ∧
    splitRow (r/136%32) (r/136/32) (r%136) = r := by
  unfold splitRow
  omega

theorem row_pair (pair u : Nat) (hp : pair<68) (hu : u<2) :
    2*pair+u < 136 ∧ (2*pair+u)/2=pair ∧ (2*pair+u)%2=u := by omega

def rowMajor (row block word k : Nat) := row*(20*(8*k))+block*(8*k)+word
def interleaved (row block word : Nat) := (block/2)*17408*48+(block%2)*24+row*48+word

theorem rowMajor_bounds (row block word k : Nat) (hr : row<17408) (hb : block<20)
    (hk : k=2 ∨ k=4) (hw : word<8*k) : rowMajor row block word k < 17408*20*(8*k) := by
  unfold rowMajor
  rcases hk with hk | hk <;> subst k <;> omega

theorem interleaved_bounds (row block word : Nat)
    (hr : row<17408) (hb : block<20) (hw : word<24) : interleaved row block word < 10*17408*48 := by
  unfold interleaved
  omega

theorem interleaved_inverse (row block word : Nat)
    (hr : row<17408) (_hb : block<20) (hw : word<24) :
    interleaved row block word % (17408*48) / 48 = row ∧
    interleaved row block word / (17408*48)*2 + interleaved row block word % 48 / 24 = block ∧
    interleaved row block word % 24 = word := by
  unfold interleaved
  omega

theorem fold_bounds (group sg lane i : Nat)
    (hg : group<160) (hs : sg<8) (hl : lane<32) (hi : i<4) :
    group*32+sg*4+i < 5120 ∧ lane*5120+group*32+sg*4+i < 32*5120 := by omega

#print axioms scratch_inverse
#print axioms scratch_coverage
#print axioms partial_inverse
#print axioms partial_coverage
#print axioms row_coverage
#print axioms interleaved_inverse
end Inference.Kernels.Trellis.DownCodebook
