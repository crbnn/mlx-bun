// NOT A PATTERN TO FOLLOW, deleted with the D2 oracle. The deprecated entry
// points the generic Gemma 4 graph carried until E2 deleted them (Gemma4Model
// at d71467c4: forwardHidden, its forwardLayers hook, forwardEmbeddings, the
// hiddenTap capture and forwardHiddenMixed), frozen here so the pre-D2
// generated classes, which override that hook and inherit the rest, still run
// exactly as they did. They derive the phase from the input's shape, which no
// graph may do any more.
import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { Cache } from "../../../src/contracts/mlx/cache";
import type { DonorRead, Gemma4Phase } from "../../../src/models/gemma4/blocks";
import { mapPackedTokens } from "../../../src/input/token-groups";
import { Gemma4Model } from "../../../src/models/gemma4/model";
import type { TokenGroup } from "../frozen-token-groups";

const shapePhase = (input: MlxArray): "decode" | "window" => input.shape[1] === 1 ? "decode" : "window";

export class DeprecatedGemma4Base extends Gemma4Model {
  /** The layer tap the speculative binding set around a deprecated forward. */
  hiddenTap: { layers: Set<number>; pos?: number; captured: Map<number, MlxArray> } | null = null;

  /** ids [B, L] to the final-norm hidden [B, L, hidden]. */
  forwardHidden(ids: MlxArray, cache: Cache[]): MlxArray {
    return this.forwardLayers(this.embedTokens(ids), cache, null, ids);
  }

  /** Pre-merged (unscaled) input embeddings to hidden states, as the deprecated entry took them. */
  forwardEmbeddings(embeds: MlxArray, cache: Cache[], bidir: MlxArray | null, ids: MlxArray | null = null,
    multimodal: MlxArray | null = null): MlxArray {
    const h = ops.mulScalar(embeds, this.embedScale);
    if (this.perLayerWidth === 0) return this.forwardLayers(h, cache, bidir, null);
    if (!ids) throw new Error("per-layer-input multimodal models require token ids");
    const zeroing = multimodal ?? bidir;
    let plIds = ids;
    if (zeroing) {
      const cond = ops.reshape(zeroing, ids.shape);
      const zero = ops.zeros(ids.shape, ids.dtype);
      plIds = ops.where(cond, zero, ids);
      cond.dispose();
      zero.dispose();
    }
    const out = this.forwardLayers(h, cache, bidir, plIds);
    if (plIds !== ids) plIds.dispose();
    return out;
  }

  /** The hook the generated classes override: the phase from `h0`'s shape, a window over an image mask the
   *  bidirectional read, the hidden tap capturing each layer and the final hidden. */
  protected forwardLayers(h0: MlxArray, cache: Cache[], bidir: MlxArray | null, ids: MlxArray | null): MlxArray {
    const read = shapePhase(h0);
    const out = this.forwardDecoder(h0, cache, bidir && read === "window" ? { bidirectional: bidir } : read, ids,
      this.hiddenTap ? (layer, h) => this.captureLayer(layer, h) : null);
    this.captureLayer(this.layers.length, out);
    return out;
  }

  /** Store layer `i`'s residual stream if the hidden tap requests it, copied as bf16. */
  protected captureLayer(i: number, h: MlxArray): void {
    const tap = this.hiddenTap;
    if (!tap || !tap.layers.has(i)) return;
    const H = h.shape[2]!;
    const src = tap.pos !== undefined ? h.slice([0, tap.pos, 0], [1, tap.pos + 1, H]) : h;
    const copy = src.astype(this.embed.scales.dtype);
    if (src !== h) src.dispose();
    tap.captured.set(i, copy);
  }

  /** Mixed token execution as the generic graph ran it: attention per group, feed-forward packed. */
  forwardHiddenMixed(work: readonly TokenGroup[]): MlxArray[] {
    if (work.length === 1 && !work[0]!.captureLayer) return [this.forwardHidden(work[0]!.ids, work[0]!.cache)];
    const groups: Array<{ ids: MlxArray; cache: Cache[]; h: MlxArray; phase: Gemma4Phase;
      perLayer: MlxArray | null; donors: (DonorRead | null)[] }> = [];
    const results: MlxArray[] = [];
    const packedMlp = this.composition.packedMixedMlp && !work.some(group => group.preserveTokenGeometry);
    try {
      for (const { ids, cache } of work) {
        const group = { ids, cache, h: this.embedTokens(ids), phase: shapePhase(ids) as Gemma4Phase,
          perLayer: null as MlxArray | null, donors: Array<DonorRead | null>(this.numDonors).fill(null) };
        groups.push(group);
        if (this.perLayerWidth > 0) group.perLayer = this.computePerLayerInputs(ids, group.h);
      }
      for (let i = 0; i < this.layers.length; i++) {
        const layer = this.layers[i]!, ci = this.cacheIndex[i]!;
        const mids: MlxArray[] = [], inputs: MlxArray[] = [];
        try {
          for (const group of groups) {
            const { h, donor } = layer.forwardAttn(group.h, ci === -1 ? null : group.cache[ci]!, group.phase,
              ci === -1 ? group.donors[this.previousKvs[i]!]! : null);
            mids.push(h);
            if (ci !== -1) group.donors[ci] = donor;
            if (group.perLayer) {
              const [B, L] = group.ids.shape;
              using part = group.perLayer.slice([0, 0, i, 0], [B!, L!, i + 1, this.perLayerWidth]);
              inputs.push(ops.reshape(part, [B!, L!, this.perLayerWidth]));
            }
          }
          let packedInput: MlxArray | null = null;
          if (packedMlp && inputs.length) {
            const flat = inputs.map(input => ops.reshape(input, [1, -1, this.perLayerWidth]));
            try { packedInput = ops.concatAxis(flat, 1); }
            finally { for (const input of flat) input.dispose(); }
          }
          using pli = packedInput;
          const outputs: MlxArray[] = [];
          try {
            if (packedMlp) outputs.push(...mapPackedTokens(mids, packed => layer.forwardMlp(packed, pli)));
            else for (const [row, mid] of mids.entries()) outputs.push(layer.forwardMlp(mid, inputs[row] ?? null));
          } catch (error) { for (const output of outputs) output.dispose(); throw error; }
          for (const [index, group] of groups.entries()) { group.h.dispose(); group.h = outputs[index]!; }
          for (const [index, group] of groups.entries()) work[index]!.captureLayer?.(i, group.h);
        } finally {
          for (const mid of mids) mid.dispose();
          for (const input of inputs) input.dispose();
        }
      }
      for (const [index, group] of groups.entries()) {
        const hidden = this.finalNorm.forward(group.h); results.push(hidden);
        work[index]!.captureLayer?.(this.layers.length, hidden);
      }
      return results;
    } catch (error) { for (const result of results) result.dispose(); throw error; }
    finally {
      for (const group of groups) {
        group.h.dispose(); group.perLayer?.dispose();
        for (const donor of group.donors) donor?.dispose();
      }
    }
  }
}

