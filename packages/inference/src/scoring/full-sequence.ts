// Training forward pass: full-sequence logits with no KV reuse.
//
// Every forward runs the graph's scoring forward (sequenceHidden) over fresh
// stateless training caches (state/training-cache.ts), one per cache the graph
// composes: a TrainingCache with that layer's sliding window for attention, a
// TrainingSSMCache for recurrent state. They share one mask memo and attend
// with the training attention the run selected; the caches own their reads, so
// nothing here builds a mask or reads keys and values.
//
// B=1 (no padding): offset-0 causal (windowed) reads, safe to recompute under
// gradient checkpointing — bit-identical to the single-example trainer.
//
// B>1 (padded batch): every row in the batch is padded to a common length L,
// but the rows have different *true* lengths. A plain causal mask would let a
// padded query attend padded keys, so the caches take the rows' lengths and
// read under the padded-batch mask: causal (within the window) AND key within
// its row's true length, built at offset 0 like the unpadded read.

import type { MlxArray } from "@mlx-bun/mlx/array";
import { type Cache } from "../contracts/mlx/cache";
import { AttentionMasks } from "../state/attention-read";
import { selectedTrainingAttention, TrainingCache, TrainingSSMCache, trainingStandIn } from "../state/training-cache";
import type { MlxSequenceForward, MlxSequenceRange } from "../contracts/mlx/graph";
import type { RuntimeModel } from "../models/factory";

export { TrainingCache, TrainingSSMCache };
/** The padded-batch training mask (`paddedTrainingMask`): [B, 1, L, L] bool,
 *  causal (within `windowSize`) AND key within its row's valid length. Caller
 *  owns the result. */
export { paddedTrainingMask as buildBatchedPadMask } from "../state/training-cache";

/** Run a full-sequence forward for training.
 *  @param ids int32 array [B, L].
 *  @returns logits [B, L, V] (caller owns; dispose when done).
 *
 *  When `validLengths` is omitted (or every row is full-length L) the batch
 *  has no padding and the caches read causally — the B=1 result is
 *  bit-identical to the original trainer. When some row is shorter than L,
 *  the caches read under the padding-aware batched mask. */
export function trainForwardHidden(
  model: RuntimeModel,
  ids: MlxArray,
  validLengths?: number[],
): MlxArray {
  const [B, L] = ids.shape as [number, number];
  const width = (B * L);
  const rowLengths = validLengths !== undefined && validLengths.some((v) => v < L) ? validLengths : null;
  // Stateless offset-0 caches (one per cache the graph composes), so the
  // forward is pure and safe to recompute under gradient checkpointing. Each
  // keeps its layer's sliding window; recurrent layers take TrainingSSMCache,
  // whose causal recurrence never carries a right-padded position into a real
  // one, so it needs no mask.
  const probe = model.makeCache();
  const masks = new AttentionMasks(), attention = selectedTrainingAttention();
  const cache: Cache[] = probe.map((c) =>
    trainingStandIn(c, window => new TrainingCache(window, masks, attention, rowLengths)));
  for (const c of probe) c.dispose();
  try {
    return sequenceHiddenOperation(model, ids, cache, width); // [B, L, hidden]
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
  const logits = sequenceLogitsOperation(model,h);
  h.dispose();
  return logits;
}


/** Select the full-sequence operation from its exact token rows. */
type SequenceOperations = { readonly sequence?: readonly MlxSequenceRange[]; readonly sequenceHidden?: MlxSequenceForward };
export function sequenceHiddenOperation(model: SequenceOperations | object,
  ids: MlxArray, cache: Cache[], rows = ids.shape[0]! * ids.shape[1]!): MlxArray {
  const ports = model as SequenceOperations;
  if (ports.sequence) return ports.sequence.find(range => rows >= range.from && rows <= range.to)!.operation(ids, cache);
  if (!ports.sequenceHidden) throw new TypeError("the graph declares no full-sequence operation");
  return ports.sequenceHidden(ids, cache, rows);
}

/** Full-sequence heads may cover more rows than the serving scheduler's table. */
export function sequenceLogitsOperation(model: object, hidden: MlxArray): MlxArray {
  const ports=model as {readonly sequenceLogits?: readonly import("../contracts/mlx/graph").MlxHeadRange[]; logitsFromHidden?(hidden:MlxArray):MlxArray};
  const rows=hidden.shape.slice(0,-1).reduce((a,b)=>a*b,1);
  if (ports.sequenceLogits) {
    const range=ports.sequenceLogits.find(range=>rows>=range.from && rows<=range.to);
    if (!range) throw new Error(`graph has no sequence head for ${rows} rows`);
    return range.operation(hidden);
  }
  if (ports.logitsFromHidden) return ports.logitsFromHidden(hidden);
  throw new TypeError("graph declares no full-sequence head");
}
