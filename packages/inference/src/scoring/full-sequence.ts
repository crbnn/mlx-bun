// Training forward pass: full-sequence logits with no KV reuse.
//
// B=1 (no padding): trainForward runs the graph's scoring forward (sequenceHidden) over a
// fresh cache exactly as before — bit-identical to the single-example trainer.
//
// B>1 (padded batch): every row in the batch is padded to a common length L,
// but the rows have different *true* lengths. A plain causal mask would let a
// real query attend to padded keys (and let padded queries pollute the cache
// view), so for batched forwards we build an explicit [B, 1, L, L] boolean
// mask that is the causal mask AND-ed with a per-row "key is within this row's
// valid length" mask, and route it through the model's attention via a cache
// wrapper whose makeMask returns it. The wrapper delegates every KV operation
// to a real cache (KVCache/RotatingKVCache are already shape-generic over B),
// so only the mask differs from the stock path. Window/offset handling is
// preserved: training always runs offset-0 full-sequence forwards, so the
// causal-with-window mask is built at offset 0, exactly matching what the
// underlying cache's makeMask would have produced for the unpadded case.

import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import { createCausalMask } from "../kernels/attention/masks";
import { type AttentionCache, type AttentionRead, type Cache, type Mask } from "../contracts/mlx/cache";
import { AttentionMasks, sdpaRead } from "../state/attention-read";
import { TrainingCache, TrainingSSMCache, trainingStandIn } from "../state/training-cache";
import type { RuntimeModel } from "../models/factory";
import { widthClass } from "../contracts/mlx/width";

export { TrainingCache, TrainingSSMCache };

/** Run a full-sequence forward for training.
 *  @param ids int32 array [B, L].
 *  @returns logits [B, L, V] (caller owns; dispose when done).
 *
 *  When `validLengths` is omitted (or every row is full-length L) the batch
 *  has no padding and the model's stock mask path is used unchanged — the
 *  B=1 result is bit-identical to the original trainer. When some row is
 *  shorter than L, a padding-aware batched mask is injected. */
export function trainForwardHidden(
  model: RuntimeModel,
  ids: MlxArray,
  validLengths?: number[],
): MlxArray {
  const [B, L] = ids.shape as [number, number];
  const width = widthClass(B * L);
  const needsPadMask =
    validLengths !== undefined && validLengths.some((v) => v < L);

  if (needsPadMask) {
    // B>1 padded path: dense cache wrapped with the per-row padding mask
    // (unchanged; not yet checkpointed — B=1 is the checkpointing target).
    const realCache = model.makeCache();
    const cache = wrapWithBatchedMask(realCache, B, L, validLengths!);
    try {
      return model.sequenceHidden(ids, cache, width);
    } finally {
      for (const c of realCache) c.dispose();
      for (const c of cache) if (c instanceof BatchedMaskCache) c.disposeOwnMask();
    }
  }

  // B=1 / no padding: stateless offset-0 caches (one per donor layer), so the
  // forward is pure and safe to recompute under gradient checkpointing. Each
  // keeps its layer's sliding window, and they share one mask memo.
  const probe = model.makeCache();
  const masks = new AttentionMasks();
  const cache: Cache[] = probe.map((c) => trainingStandIn(c, window => new TrainingCache(window, masks)));
  for (const c of probe) c.dispose();
  try {
    return model.sequenceHidden(ids, cache, width); // [B, L, hidden]
  } finally {
    for (const c of cache) c.dispose();
  }
}

/** Full-sequence forward returning logits [B, L, V]. Convenience over
 *  trainForwardHidden + logitsFromHidden. The SFT B=1 path deliberately skips
 *  this and applies the LM head only at response positions (responseOnlyCe)
 *  to avoid materializing the [B, L, vocab] logits — the dominant training
 *  memory term at long context with a 262k vocab. */
export function trainForward(
  model: RuntimeModel,
  ids: MlxArray,
  validLengths?: number[],
): MlxArray {
  const h = trainForwardHidden(model, ids, validLengths);
  const logits = model.logitsFromHidden(h);
  h.dispose();
  return logits;
}

/** Build the padding-aware additive-style boolean mask for one window type.
 *  Shape `[B, 1, L, L]`, true where query i may attend to key j:
 *    allow[b,0,i,j] = (j <= i)               (causal, offset 0)
 *                     AND (j < validLen[b])   (key within row b's true length)
 *                     [AND (i < j + window)   if a sliding window applies]
 *  Built from createCausalMask(L, 0, window) (the model's own [L, L] bool
 *  matrix) broadcast over B, masked by a per-row [B,1,1,L] key-validity mask.
 *  Caller owns the returned array. */
export function buildBatchedPadMask(
  B: number,
  L: number,
  validLengths: number[],
  windowSize: number | null,
): MlxArray {
  // [L, L] bool causal (+window) matrix at offset 0 — exactly the model's.
  const causal2d = createCausalMask(L, 0, windowSize); // [L, L] bool
  const causal4d = ops.reshape(causal2d, [1, 1, L, L]);
  causal2d.dispose();

  // Per-row key validity: keyValid[b, 0, 0, j] = j < validLengths[b].
  // Build as int32 {0,1} then compare to get bool, broadcasting over q (i).
  const keyValidData = new Int32Array(B * L);
  for (let b = 0; b < B; b++) {
    const vlen = validLengths[b]!;
    for (let j = 0; j < L; j++) keyValidData[b * L + j] = j < vlen ? 1 : 0;
  }
  const keyValidI = MlxArray.fromInt32(keyValidData, [B, 1, 1, L]);
  const zero = MlxArray.fromInt32(new Int32Array([0]), []);
  const keyValid = ops.less(zero, keyValidI); // 0 < x  → bool true where x==1
  zero.dispose();
  keyValidI.dispose();

  const allow = ops.logicalAnd(causal4d, keyValid); // broadcasts [1,1,L,L] & [B,1,1,L]
  causal4d.dispose();
  keyValid.dispose();
  return allow; // [B, 1, L, L] bool
}

/** Cache wrapper that forwards every KV op to a real cache but overrides
 *  makeMask to return a precomputed padding-aware batched mask. One mask is
 *  built per distinct window size (null = full attention, else the sliding
 *  window) and shared across all caches of that type; ownership of the mask
 *  arrays stays with the wrapper (disposeOwnMask), NOT the model — the model's
 *  forwardLayers only disposes the mask it pulled from makeMask, so makeMask
 *  hands back a fresh view each call and keeps the canonical copy itself.
 *
 *  Reads: the inner cache's `updateAndFetch` and the stock fused SDPA under a
 *  view of the canonical mask, the call a graph made after `makeMask` and
 *  `updateAndFetch` here before it read through the cache (Qwen 3.5's padded
 *  training forward). The mask is the same for every query count, so decode and
 *  window read alike. The inner cache is the graph's own plain cache. */
class BatchedMaskCache implements AttentionCache {
  /** Training-only adapter; never admitted, merged, or persisted. */
  signature(): string { return "train:batched-mask"; }
  constructor(
    private readonly inner: Cache,
    private readonly window: number | null,
    private readonly maskStore: Map<number | null, MlxArray>,
  ) {}

  get offset(): number {
    return this.inner.offset;
  }

  appendDecode(k: MlxArray, v: MlxArray): AttentionRead { return this.#read(k, v); }
  appendWindow(k: MlxArray, v: MlxArray): AttentionRead { return this.#read(k, v); }
  #read(k: MlxArray, v: MlxArray): AttentionRead {
    const mask = this.makeMask(k.shape[2]!, this.window);
    let keys: MlxArray, values: MlxArray;
    try { [keys, values] = this.inner.updateAndFetch(k, v); } catch (error) { mask.arr?.dispose(); throw error; }
    return sdpaRead(keys, values, { mask, release: () => mask.arr?.dispose() });
  }

  updateAndFetch(k: MlxArray, v: MlxArray): [MlxArray, MlxArray] {
    return this.inner.updateAndFetch(k, v);
  }

  makeMask(N: number, _windowSize: number | null): Mask {
    const canonical = this.maskStore.get(this.window)!;
    // Hand back a fresh full-range view so the model can dispose it without
    // touching our canonical copy. slice over all 4 axes = cheap view node.
    const [B, , L] = canonical.shape as [number, number, number, number];
    const view = canonical.slice([0, 0, 0, 0], [B, 1, L, L]);
    return { mode: "array", arr: view };
  }

  state(): MlxArray[] {
    return this.inner.state();
  }
  isTrimmable(): boolean {
    return this.inner.isTrimmable();
  }
  trim(n: number): void {
    this.inner.trim(n);
  }
  dispose(): void {
    this.inner.dispose();
  }
  /** Dispose the canonical mask copies (shared store; idempotent via clear). */
  disposeOwnMask(): void {
    for (const m of this.maskStore.values()) m.dispose();
    this.maskStore.clear();
  }
}

/** Wrap each real attention cache in a BatchedMaskCache over one
 *  padding-aware mask per distinct window (the full-attention mask for a
 *  cache without one). The state's stand-in for a recurrent layer is the
 *  stateless TrainingSSMCache (`trainingStandIn`). */
function wrapWithBatchedMask(
  realCache: Cache[],
  B: number,
  L: number,
  validLengths: number[],
): Cache[] {
  const store = new Map<number | null, MlxArray>();
  return realCache.map(c => trainingStandIn(c, window => {
    if (!store.has(window)) store.set(window, buildBatchedPadMask(B, L, validLengths, window));
    return new BatchedMaskCache(c, window, store);
  }));
}
