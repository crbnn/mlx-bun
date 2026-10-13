// The scheduler's side of the graph contract: it names the phase it is in and
// the rows it is loading, and this resolves the operation the graph built for
// them. Bound once per graph; nothing here inspects a tensor.
import {
  widthClass, type MlxCapturingForward, type MlxHeadForward, type MlxLayerCapture, type MlxNeuralEngineChunk, type MlxPhaseForward,
  type MlxTokenPhases,
} from "../contracts/mlx/graph";
import type { PrefillPhase } from "../contracts/portable/prefill";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxForwardStep } from "../contracts/mlx/forward-work";

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
 *  The Neural Engine is a unit the scheduler owns: read once, here, which
 *  Neural Engine operations the composition built and the row range each takes.
 *  A prompt piece of one sequence, a whole chunk or a tail, whose positions one
 *  range holds runs on that operation; every other piece runs on the GPU. One
 *  scheduler drives one dispatch at a time, so the unit serves one piece at a
 *  time. */
export class MlxPhaseDispatch {
  readonly #phases: MlxTokenPhases | null;
  readonly #ane: readonly MlxNeuralEngineChunk[];

  /** `graph`: the phases `tokenPhasesOf` resolved, or null for a graph whose
   *  method runs none (denoising), which then refuses every token phase. */
  constructor(graph: MlxTokenPhases | null) {
    this.#phases = graph;
    this.#ane = graph?.prefillChunk.ane ?? [];
  }

  get #graph(): MlxTokenPhases {
    if (!this.#phases) throw new Error("this graph's method runs no token phases");
    return this.#phases;
  }

  /** Whether the composition built a Neural Engine operation. */
  get neuralEngine(): boolean { return this.#ane.length > 0; }

  /** One prompt step of `sequences` sequences of `tokens` positions each, in
   *  the `phase` its planner named. */
  prompt(phase: PrefillPhase, sequences: number, tokens: number): MlxCapturingForward {
    if (phase === "decode") return this.#graph.decode[widthClass(sequences)];
    const unit = sequences === 1 ? this.#ane.find(ane => tokens >= ane.minRows && tokens <= ane.rows) : undefined;
    if (unit) return unit.forward;
    return phase === "tail" ? this.#graph.prefillTail[widthClass(sequences * tokens)] : this.#graph.prefillChunk.gpu;
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
