import type { MlxArray } from "@mlx-bun/mlx/array";
import type { Cache, SharedKv } from "../../contracts/mlx/cache";
import type { HeadQuant, PrefixLayout, SegmentedPass, TrainableGraph } from "../../contracts/mlx/trainable";
import { setActivePrefixLayout } from "../../layers/prefix-layout";
import { AttentionMasks } from "../../state/attention-read";
import { prefixSharedCaches, selectedTrainingAttention, TrainingCache } from "../../state/training-cache";
import type { RMSNorm } from "../../layers/normalization";
import type { QuantizedEmbedding } from "../../layers/quantized-embedding";
import type { QuantizedLinear } from "../../layers/quantized-linear";

/** What the training declaration reads from a MiniCPM5 model (structural, so this
 *  module does not depend on the model file). */
export interface MiniCpmTraining {
  readonly embed: QuantizedEmbedding;
  readonly layers: readonly unknown[];
  readonly finalNorm: RMSNorm;
  readonly lmHead: QuantizedLinear;
  /** The training forward over layers `[aIdx, bIdx)`; `h` stays the caller's. */
  runLayerRange(h: MlxArray, aIdx: number, bIdx: number, cache: Cache[]): MlxArray;
}

/** One stateless training cache per layer, full attention, sharing one mask
 *  memo and attending with the run's training attention: prefix-shared caches
 *  under a layout, plain ones otherwise. */
function trainingCaches(model: MiniCpmTraining, prefix?: PrefixLayout): Cache[] {
  const attention = selectedTrainingAttention(), windows = model.layers.map(() => null);
  if (prefix) return prefixSharedCaches(prefix, windows, attention);
  const masks = new AttentionMasks();
  return windows.map(window => new TrainingCache(window, masks, attention));
}

/** One segmented pass over a MiniCPM5 sequence: plain full attention with no KV
 *  sharing, so each layer reads its own stateless cache (prefix-shared caches
 *  read under the block-sparse mask when a layout is given). */
class MiniCpmSegmentedPass implements SegmentedPass {
  readonly input: MlxArray;
  private readonly caches: Cache[];

  constructor(private readonly model: MiniCpmTraining, ids: MlxArray, private readonly prefix?: PrefixLayout) {
    this.caches = trainingCaches(model, prefix);
    if (prefix) setActivePrefixLayout(prefix);
    try {
      this.input = model.embed.encode(ids); // [1, T, hidden]
    } catch (error) {
      this.releaseCaches();
      throw error;
    }
  }

  private releaseCaches(): void {
    if (this.prefix) setActivePrefixLayout(null);
    for (const c of this.caches) c.dispose();
  }

  runRange(h: MlxArray, lo: number, hi: number, _donorKvIn: Map<number, SharedKv>) {
    return { h: this.model.runLayerRange(h, lo, hi, this.caches), donorKvOut: new Map<number, SharedKv>() };
  }

  dispose(): void { this.releaseCaches(); }
}

/** MiniCPM5's training declaration. */
export function miniCpmTrainable(model: MiniCpmTraining): TrainableGraph {
  return {
    lmHead(): HeadQuant {
      const h = model.lmHead;
      return { w: h.w, scales: h.scales, biases: h.biases, spec: h.spec, softcap: null };
    },
    flashAttention: { supported: true },
    segmented: {
      layerCount: model.layers.length,
      sharedKv: { reusedDonors: new Set(), donorOf: model.layers.map(() => null) },
      finalNorm: (h) => model.finalNorm.forward(h),
      begin: (ids, prefix) => new MiniCpmSegmentedPass(model, ids, prefix),
    },
    prefixShared: {
      /** The training forward (`runLayerRange`) over the whole concat, whose
       *  prefix-shared caches read under the block-sparse mask and whose
       *  queries and keys take the block-wise RoPE of the active layout. */
      prefixSharedHidden(ids: MlxArray, layout: PrefixLayout): MlxArray {
        const caches = trainingCaches(model, layout);
        setActivePrefixLayout(layout);
        try {
          const h0 = model.embed.encode(ids);
          let h: MlxArray;
          try { h = model.runLayerRange(h0, 0, model.layers.length, caches); } finally { h0.dispose(); }
          try { return model.finalNorm.forward(h); } finally { h.dispose(); } // [1, T, hidden], post-finalNorm
        } finally {
          setActivePrefixLayout(null);
          for (const c of caches) c.dispose();
        }
      },
    },
  };
}
