// The scheduler's side of the graph contract: it names the phase it is in and
// the rows it is loading, and this resolves the operation the graph built for
// them. Bound once per graph; nothing here inspects a tensor.
import {
  widthClass, type MlxCapturingForward, type MlxHeadForward, type MlxLayerCapture, type MlxPhaseForward, type MlxTokenPhases,
} from "../contracts/mlx/graph";
import type { Composition } from "../contracts/portable/composition";
import type { PrefillPhase } from "../contracts/portable/prefill";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxForwardStep } from "../contracts/mlx/forward-work";

/** What a dispatch reads from its graph: the named phases and the composition
 *  record they were built for (absent for a graph that composes none). */
export interface MlxPhasedGraph extends MlxTokenPhases {
  readonly composition?: Composition;
}

/** One prompt step the scheduler runs: the phase its planner named and its
 *  rows, `sequences` sequences of `tokens` positions each. */
export interface MlxPromptStep {
  readonly phase: PrefillPhase;
  readonly sequences: number;
  readonly tokens: number;
}

/** A verify window: `sequences` sequences of `positions` (depth + 1) each. */
export interface MlxVerifyWindow {
  readonly sequences: number;
  readonly positions: number;
}

/** A prompt step as the mixed iteration's hook takes it: one position per
 *  sequence is a decode group, any other piece a window. */
export function promptWorkStep(step: MlxPromptStep, capture: MlxLayerCapture | null,
  own: () => Promise<MlxArray>): MlxForwardStep {
  const rows = step.sequences * step.tokens;
  return { phase: step.phase === "decode" ? "decode" : "window", rows, width: widthClass(rows), capture, own };
}

/** A verify window as the mixed iteration's hook takes it. */
export function verifyWorkStep(window: MlxVerifyWindow, capture: MlxLayerCapture | null,
  own: () => Promise<MlxArray>): MlxForwardStep {
  const rows = window.sequences * window.positions;
  return { phase: "verify", rows, width: widthClass(rows), capture, own };
}

/** The phase operations of one graph, indexed by what the scheduler decides.
 *
 *  The Neural Engine is a unit the scheduler owns: read once, here, whether the
 *  composition built a Neural Engine chunk operation and for how many tokens.
 *  A whole chunk of one sequence of exactly that size runs on it; every other
 *  chunk runs on the GPU. One scheduler drives one dispatch at a time, so the
 *  unit serves one chunk at a time. */
export class MlxPhaseDispatch {
  readonly #phases: MlxTokenPhases | null;
  readonly #ane: { readonly forward: MlxCapturingForward; readonly tokens: number } | null;

  /** `graph`: the phases `tokenPhasesOf` resolved, or null for a graph whose
   *  method runs none (denoising), which then refuses every token phase. */
  constructor(graph: MlxPhasedGraph | null) {
    this.#phases = graph;
    const ane = graph?.prefillChunk.ane;
    this.#ane = ane && graph.composition ? Object.freeze({ forward: ane, tokens: graph.composition.prefillChunkTokens }) : null;
  }

  get #graph(): MlxTokenPhases {
    if (!this.#phases) throw new Error("this graph's method runs no token phases");
    return this.#phases;
  }

  /** Whether the composition built a Neural Engine chunk operation. */
  get neuralEngine(): boolean { return this.#ane !== null; }

  /** One prompt step of `sequences` sequences of `tokens` positions each, in
   *  the `phase` its planner named. */
  prompt(phase: PrefillPhase, sequences: number, tokens: number): MlxCapturingForward {
    if (phase === "decode") return this.#graph.decode[widthClass(sequences)];
    if (phase === "tail") return this.#graph.prefillTail[widthClass(sequences * tokens)];
    return this.#ane && sequences === 1 && tokens === this.#ane.tokens ? this.#ane.forward : this.#graph.prefillChunk.gpu;
  }

  /** One new token for each of `sequences` sequences. */
  decode(sequences: number): MlxCapturingForward { return this.#graph.decode[widthClass(sequences)]; }

  /** A verify window of `positions` (depth + 1) per sequence. */
  verify(sequences: number, positions: number): MlxCapturingForward {
    return this.#graph.verify[widthClass(sequences * positions)];
  }

  /** `tokens` known tokens appended to one sequence. */
  known(tokens: number): MlxPhaseForward { return this.#graph.appendKnown[widthClass(tokens)]; }

  /** The head over a hidden of `rows` rows. */
  logits(rows: number): MlxHeadForward { return this.#graph.logits[widthClass(rows)]; }
}
