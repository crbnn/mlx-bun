// The scheduler's side of the graph contract: it names the phase it is in and
// the rows it is loading, and this resolves the operation the graph built for
// them. Bound once per graph; nothing here inspects a tensor.
import { type MlxCapturingForward, type MlxHeadForward, type MlxLayerCapture, type MlxPromptRange, type MlxPhaseForward, type MlxTokenPhases, type MlxKnownOperations } from "../contracts/mlx/graph";
import type { PrefillPhase } from "../contracts/portable/prefill";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxForwardStep } from "../contracts/mlx/forward-work";

/** Index the declared whole-span operation without changing its geometry. */
export function knownOperationFor(operations: MlxKnownOperations, rows: number): MlxPhaseForward {
  const operation = Array.isArray(operations)
    ? operations.find(range => rows >= range.from && rows <= range.to)?.operation
    : (operations as Readonly<Record<number, MlxPhaseForward>>)[rows];
  if (!operation) throw new Error(`graph has no known-span operation for ${rows} rows`);
  return operation;
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
  return { phase: step.phase === "decode" ? "decode" : "window", rows, capture, own };
}

/** A verify window as the mixed iteration's hook takes it. */
export function verifyWorkStep(window: MlxVerifyWindow, capture: MlxLayerCapture | null,
  own: () => Promise<MlxArray>): MlxForwardStep {
  const rows = window.sequences * window.positions;
  return { phase: "verify", rows, capture, own };
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
  readonly #prompt: readonly MlxPromptRange[];

  /** `graph`: the phases `tokenPhasesOf` resolved, or null for a graph whose
   *  method runs none (denoising), which then refuses every token phase. */
  constructor(graph: MlxTokenPhases | null) {
    this.#phases = graph;
    this.#prompt = graph?.prompt ?? [];
  }

  get #graph(): MlxTokenPhases {
    if (!this.#phases) throw new Error("this graph's method runs no token phases");
    return this.#phases;
  }

  /** Whether the composition built a Neural Engine operation. */
  get neuralEngine(): boolean { return this.#prompt.some(range => range.unit === "ane"); }

  /** One prompt step of `sequences` sequences of `tokens` positions each, in
   *  the `phase` its planner named. */
  prompt(phase: PrefillPhase, sequences: number, tokens: number): MlxCapturingForward {
    if (phase === "decode") return this.#graph.decode[sequences]!;
    const positions = sequences > 1 ? sequences * tokens : tokens;
    const ranges = sequences > 1
      ? phase === "chunk" ? this.#graph.multiSequencePromptChunk ?? this.#graph.multiSequencePrompt : this.#graph.multiSequencePrompt
      : phase === "chunk" ? this.#graph.promptChunk ?? this.#prompt : this.#prompt;
    const range = ranges.find(range => positions >= range.from && positions <= range.to);
    if (!range) throw new Error(`graph has no prompt operation for ${tokens} positions`);
    return range.operation;
  }

  /** One new token for each of `sequences` sequences. */
  decode(sequences: number): MlxCapturingForward { return this.#graph.decode[sequences]!; }

  /** A verify window of `positions` (depth + 1) per sequence. */
  verify(sequences: number, positions: number): MlxCapturingForward {
    return this.#graph.verify[sequences * positions]!;
  }

  /** `tokens` known tokens appended to one sequence. */
  known(tokens: number): MlxPhaseForward { return knownOperationFor(this.#graph.appendKnown, tokens); }

  /** The head over a hidden of `rows` rows. */
  logits(rows: number): MlxHeadForward { return this.#graph.logits[rows]!; }
}
