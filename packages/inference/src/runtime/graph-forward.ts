import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";

/** Owns intermediates of a written forward. The graph calls its layers itself;
 * this scope only releases the previous output and records the new one. */
export class GraphForwardScope {
  #hidden: MlxArray | null;
  constructor(hidden: MlxArray, readonly capture: ((layer: number, hidden: MlxArray) => void) | null) { this.#hidden = hidden; }
  get hidden(): MlxArray { return this.#hidden!; }
  accept(layer: number, next: MlxArray): void {
    this.#hidden!.dispose();
    this.#hidden = next;
    this.capture?.(layer, next);
  }
  bound(layer: number, next: MlxArray, cache: { state(): MlxArray[]; readonly stateNeedsDispose?: boolean }): void {
    this.#hidden!.dispose();
    this.#hidden = next;
    const state = cache.state();
    try { ops.evalAll([next, ...state]); }
    finally { if (cache.stateNeedsDispose) for (const array of state) array.dispose(); }
    this.capture?.(layer, next);
  }
  submit(): void { ops.asyncEvalAll([this.#hidden!]); }
  finish(layers: number, output: MlxArray): MlxArray {
    this.#hidden!.dispose();
    this.#hidden = null;
    try { this.capture?.(layers, output); return output; }
    catch (error) { output.dispose(); throw error; }
  }
  [Symbol.dispose](): void { this.#hidden?.dispose(); this.#hidden = null; }
}
