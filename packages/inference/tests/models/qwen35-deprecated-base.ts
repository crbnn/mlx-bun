// NOT A PATTERN TO FOLLOW, deleted with the pre-D1 M4 Pro oracle. The deprecated
// entry points the generic Qwen 3.5 graph carried until E2 deleted them
// (Qwen35Model at 1590c494: forwardHidden, forwardHiddenAtPositions,
// forwardEmbeddingsAtPositions, the hiddenTap capture inside the layer
// loop, forwardHiddenMixed and deprecatedPhase), frozen here so the pre-D1 M4 Pro
// graph, which overrides some of them and inherits the rest, still runs exactly
// as it did. They derive the phase and the width from the input's shape, which no
// graph may do any more.
import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { Cache } from "../../src/contracts/mlx/cache";
import { widthClass, type MlxLayerCapture, type MlxWidthClass } from "../../src/contracts/mlx/graph";
import type { ReadPhase } from "../../src/models/graph";
import { Qwen35Model } from "../../src/models/qwen/qwen3_5";
import { mapTokenGroups, type TokenGroup } from "./frozen-token-groups";

/** Positions per sequence above which the deprecated entry points evaluated each layer. */
const MATVEC_MAX_M = 4;

/** The phase of a deprecated forward from its input's shape: a committed append of one row is a committed
 *  span; otherwise one position per row is a decode and more is a window. */
export function deprecatedPhase(input: MlxArray, committedAppend = false): ReadPhase {
  const [rows, tokens] = input.shape as [number, number];
  if (committedAppend && rows === 1) return "committed";
  return tokens === 1 ? "decode" : "window";
}

export class DeprecatedQwen35Base extends Qwen35Model {
  /** The layer tap the speculative binding set around a deprecated forward. */
  hiddenTap: { layers: Set<number>; pos?: number; captured: Map<number, MlxArray> } | null = null;

  /** ids [B, L] to the final-norm hidden, the phase and width read from the shape. */
  forwardHidden(ids: MlxArray, cache: Cache[]): MlxArray {
    const [rows, tokens] = ids.shape as [number, number];
    return this.forwardLayers(this.embed.encode(ids), cache, deprecatedPhase(ids), undefined, null,
      widthClass(rows * tokens), tokens > MATVEC_MAX_M);
  }

  /** Token ids at request positions, the phase and width read from the shape. */
  forwardHiddenAtPositions(ids: MlxArray, cache: Cache[], positions: MlxArray): MlxArray {
    const [rows, tokens] = ids.shape as [number, number];
    return this.forwardLayers(this.embed.encode(ids), cache, deprecatedPhase(ids), positions, null,
      widthClass(rows * tokens), tokens > MATVEC_MAX_M);
  }

  /** Embeddings at request positions, the phase and width read from the shape. */
  forwardEmbeddingsAtPositions(embeds: MlxArray, cache: Cache[], positions: MlxArray): MlxArray {
    const [rows, tokens] = embeds.shape as [number, number];
    return this.forwardLayers(ops.contiguous(embeds), cache, deprecatedPhase(embeds), positions, null,
      widthClass(rows * tokens), tokens > MATVEC_MAX_M);
  }

  /** The layer loop with the hidden tap capturing each layer's output before `capture` sees it. */
  protected override forwardLayers(h0: MlxArray, cache: Cache[], phase: ReadPhase, positions: MlxArray | undefined,
    capture: MlxLayerCapture | null, width: MlxWidthClass, bounded: boolean): MlxArray {
    const tapped: MlxLayerCapture | null = this.hiddenTap
      ? (layer, h) => { this.captureLayer(layer, h); capture?.(layer, h); } : capture;
    return super.forwardLayers(h0, cache, phase, positions, tapped, width, bounded);
  }

  /** Store layer `i`'s residual stream if the tap requests it (a contiguous copy). */
  protected captureLayer(i: number, h: MlxArray): void {
    const tap = this.hiddenTap;
    if (!tap || !tap.layers.has(i)) return;
    const H = h.shape[2]!;
    const src = tap.pos !== undefined ? h.slice([0, tap.pos, 0], [1, tap.pos + 1, H]) : h;
    const copy = ops.contiguous(src);
    if (src !== h) src.dispose();
    tap.captured.set(i, copy);
  }

  /** Attention and DeltaNet retain disjoint row state; only tokenwise MLP work packs. */
  forwardHiddenMixed(work: readonly TokenGroup[]): MlxArray[] {
    if (work.length === 1 && !work[0]!.captureLayer) return [this.forwardHidden(work[0]!.ids, work[0]!.cache)];
    const groups: Array<{ h: MlxArray; phase: ReadPhase }> = [];
    const results: MlxArray[] = [];
    const bounded = work.some(group => group.ids.shape[1]! > MATVEC_MAX_M);
    try {
      for (const { ids } of work) groups.push({ h: this.embed.encode(ids), phase: deprecatedPhase(ids) });
      for (const [i, layer] of this.layers.entries()) {
        const mids: MlxArray[] = [];
        try {
          for (const [row, group] of groups.entries())
            mids.push(layer.forwardAttn(group.h, work[row]!.cache[i]!, group.phase));
          const outputs = mapTokenGroups(work, mids, hidden => layer.forwardMlp(hidden, "window",
            widthClass(hidden.size / hidden.shape.at(-1)!)), this.composition.packedMixedMlp);
          for (const [row, group] of groups.entries()) { group.h.dispose(); group.h = outputs[row]!; }
          if (bounded) {
            const caches = work.map(group => group.cache[i]!);
            const state = caches.map(cache => cache.state());
            try { ops.evalAll([...groups.map(group => group.h), ...state.flat()]); }
            finally { for (const [row, cache] of caches.entries())
              if (cache.stateNeedsDispose) for (const value of state[row]!) value.dispose(); }
          }
          for (const [row, group] of groups.entries()) work[row]!.captureLayer?.(i, group.h);
        } finally { for (const mid of mids) mid.dispose(); }
      }
      for (const [row, group] of groups.entries()) {
        const hidden = this.finalNorm.forward(group.h); results.push(hidden);
        work[row]!.captureLayer?.(this.layers.length, hidden);
      }
      return results;
    } catch (error) { for (const result of results) result.dispose(); throw error; }
    finally { for (const group of groups) group.h.dispose(); }
  }
}
