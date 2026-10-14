import type { MlxArray } from "@mlx-bun/mlx/array";
import type { AssistantDonors } from "./attention";
import type { DraftProjection } from "./draft-projection";
import type { MlxRowTable } from "./row-table";

/** Read-only target state and embedding, independent of draft scheduling. */
export interface AssistantRowsTarget {
  readonly hiddenSize: number;
  embed(ids: MlxArray): MlxArray;
  readDonors(): AssistantDonors & {
    readonly positions: readonly number[];
    dispose(): void;
  };
}

/** What a recurrent multi-token-prediction head reads from its target: the
 *  final hidden layer's output (tapped before the final norm), the token
 *  embedding, and the output head. */
export interface RecurrentMtpTarget {
  readonly hiddenSize: number;
  readonly layerCount: number;
  embed(ids: MlxArray): MlxArray;
  /** The output head per row count or range, indexed by the rows the drafter projects. */
  readonly logits: MlxRowTable<(hidden: MlxArray) => MlxArray>;
  /** The untied quantized vocabulary projection, when the drafter may project
   *  onto a row subset of it (frequency-ranked draft vocabulary). Borrowed. */
  readonly vocabularyHead?: {
    readonly w: MlxArray; readonly scales: MlxArray; readonly biases: MlxArray | null;
    readonly spec: { bits: number; groupSize: number; mode: string };
    readonly vocabSize: number;
  };
}

/** Multi-layer hidden-state taps and the embedding/head projection a drafter
 *  conditions on. */
export interface HiddenLayerTapsTarget {
  readonly layerCount: number;
  readonly projection: DraftProjection;
}

/** A residual stream in another basis than the trained model's: the TurboQuant
 *  R1 fold (packages/quantize/src/rotate.ts) stores h·R1, R1 = diag(s)·M, with s
 *  the seed's lane-0 sign vector and M MLX's hadamard_transform/√n, and moves
 *  the final-norm gain into the head. */
export interface ResidualBasis {
  /** Seed of R1's sign vector. */
  readonly r1Seed: number;
  /** The final-norm gain the fold moved into the head (effective, 1+w), one per hidden channel. */
  readonly finalGain: Float32Array;
}

/** A projection kernel the target's composition measured on its device, built over a 4-bit, group-64 affine
 *  projection a drafter quantized itself (its packed weight, scales and biases, borrowed for the kernel's life):
 *  the forward over [.., inputs] activations, caller-owned output. */
export type DraftProjectionKernel = (w: MlxArray, scales: MlxArray, biases: MlxArray) => (x: MlxArray) => MlxArray;
export interface DraftProjectionRange { readonly from: number; readonly to: number; readonly operation: DraftProjectionKernel }
export type DraftProjectionKernels = MlxRowTable<DraftProjectionKernel> | readonly DraftProjectionRange[];

/** Graph-declared ports over the target's live state, named for what they
 * provide. Sources request only the ports they consume and never inspect a
 * concrete model or cache; a source refuses a target that lacks one with
 * `targetLacks(port)` before any draft-side allocation. */
export interface TargetView {
  /** Opaque identity for providers borrowing weights from one exact target. */
  readonly identity: object;
  readonly assistantRows?: AssistantRowsTarget;
  readonly hiddenLayerTaps?: HiddenLayerTapsTarget;
  readonly recurrentMtp?: RecurrentMtpTarget;
  /** Declared by a graph whose residual stream carries a rotation fold; absent,
   *  the stream is in the trained model's basis. The same object on every view. */
  readonly residualBasis?: ResidualBasis;
  /** Declared by a graph whose composition measured the kernels a drafter's own 4-bit projections run on its
   *  device: one per row count or range of the rows they project. Absent, the drafter keeps its own rule. The same
   *  table on every view. */
  readonly draftProjections?: DraftProjectionKernels;
}

export type TargetPort = "assistantRows" | "hiddenLayerTaps" | "recurrentMtp";

/** The refusal for a target graph that does not provide a requested port. */
export function targetLacks(port: TargetPort): Error {
  return new Error(`target graph does not provide ${port}`);
}

