import contracts.mlx.M1Max

namespace Inference.Contracts.Mlx.Memory
open M1Max

structure Store where
  word : Nat
  value : Float32

def write (memory : WordMemory) (store : Store) : WordMemory :=
  fun word => if word=store.word then store.value else memory word

def execute : List Store → WordMemory → WordMemory
  | [], memory => memory
  | store::rest, memory => execute rest (write memory store)

theorem frame (stores : List Store) (memory : WordMemory) (word : Nat)
    (h : ∀ store ∈ stores, store.word ≠ word) : execute stores memory word = memory word := by
  induction stores generalizing memory with
  | nil => rfl
  | cons store rest ih =>
    rw [execute, ih (write memory store) (fun s hs => h s (by simp [hs]))]
    unfold write
    exact ite_eq_right (Ne.symm (h store (by simp)))

-- Any ordering of stores which write the same value at an address establishes
-- that value there. No scheduling order or initial memory value is assumed.
theorem written_value (stores : List Store) (memory : WordMemory) (word : Nat) (value : Float32)
    (values : ∀ store ∈ stores, store.word=word → store.value=value)
    (covered : ∃ store ∈ stores, store.word=word) : execute stores memory word = value := by
  induction stores generalizing memory with
  | nil => simp at covered
  | cons store rest ih =>
    by_cases tail : ∃ s ∈ rest, s.word=word
    · exact ih (write memory store) (fun s hs => values s (by simp [hs])) tail
    · have frameRest : ∀ s ∈ rest, s.word ≠ word := by
        intro s hs he
        exact tail ⟨s, hs, he⟩
      have head : store.word=word := by
        obtain ⟨s, hs, he⟩ := covered
        rcases List.mem_cons.mp hs with h | h
        · subst s; exact he
        · exact False.elim (tail ⟨s, h, he⟩)
      rw [execute, frame rest (write memory store) word frameRest]
      simp only [write, head, ite_true]
      exact values store (by simp) head

theorem written_value_permuted (stores reordered : List Store) (order : reordered.Perm stores)
    (memory : WordMemory) (word : Nat) (value : Float32)
    (values : ∀ store ∈ stores, store.word=word → store.value=value)
    (covered : ∃ store ∈ stores, store.word=word) : execute reordered memory word = value := by
  apply written_value
  · intro s hs he
    exact values s (order.mem_iff.mp hs) he
  · obtain ⟨s, hs, he⟩ := covered
    exact ⟨s, order.mem_iff.mpr hs, he⟩

end Inference.Contracts.Mlx.Memory
