// The draft contract: a provider owns the draft machinery for the server's
// lifetime and opens row groups that the speculative group (the one served
// lane, `execution/speculative-group.ts`) drafts, verifies and commits through.
// A single request is a group of one row. Every drafter shares that one
// verify/accept executor (designed in `02d723a:docs/design/speculative-decoding.md`
// §1); the providers differ only in what fills the draft:
//   - TwoModelProvider — mlx-lm parity (L1 oracle: mlx_lm.server --draft-model),
//     a full second model; reads no target port.
//   - AssistantProvider — the optiq KV-borrowing Gemma drafter
//     (models/gemma4/assistant.ts; L2 oracle: optiq spec_generate): reads the
//     target's donor K/V and the true hidden at each row's anchor.
//   - DflashProvider, Dflash2Provider, DeepspecProvider — block drafters over
//     the target's tapped hidden layers, projected once into context rows.
//   - QwenMtpProvider, Glm52NativeMtpProvider — multi-token-prediction heads.
//   - NgramProvider — model-free prompt lookup.
// KV-borrowing providers read target state through the `target` view their
// rows open with (see [[dspark-seam-kv-borrowing]]).

import type { MlxArray } from "@mlx-bun/mlx/array";
import type { PreparedStateChange } from "../../contracts/portable/resources";
import type { MlxDraftRows } from "./round";

export { targetLacks, type AssistantRowsTarget, type HiddenLayerTapsTarget, type RecurrentMtpTarget, type TargetPort, type TargetView } from "../../contracts/mlx/draft-target";
import type { TargetView } from "../../contracts/mlx/draft-target";

/** How a draft group's rows choose their tokens: each row's draw, and the
 *  selection the group resolved when it bound its requests. A drafter that
 *  selects over its target's head binds one selection operation from
 *  `selection` when its rows open; its draft loop calls that operation. */
export interface DraftRowSampling {
  /** Borrow [B,V] log-probabilities; return owned [B] IDs on device: each row's
   *  own draw (argmax under the greedy draft policy). */
  sample(logprobs: MlxArray, steps: readonly number[]): MlxArray;
  readonly selection: DraftSelection;
}

/** The selection every row of one draft group shares, resolved once per request
 *  from its sampling and the vocabulary its drafter declares.
 *  - `sampled`: each row's `sample` over the full head's log-probabilities.
 *  - `greedy`: every row drafts by argmax (the composition's `greedyDraft`), so a
 *    drafter may select with the fused normalized argmax over the full head's
 *    logits instead of calling `sample`; the selected IDs are identical.
 *  - `listed`: the frequency-ranked draft vocabulary its drafter declared
 *    (ascending token ids) and each row's coupled draw over logits [B, N]
 *    restricted to them, returning full-vocabulary ids [B]. Bound only for
 *    requests whose draft sampler reproduces on the subset. */
export type DraftSelection =
  | { readonly kind: "sampled" }
  | { readonly kind: "greedy" }
  | { readonly kind: "listed"; readonly ids: readonly number[];
      sample(logits: MlxArray, head: import("./draft-vocab").DraftVocabularyHead, steps: readonly number[]): MlxArray };

/** Request-owned constraints can propose known continuations without changing
 * their committed state. The target verifier still decides every output. */
export interface DraftRowConstraints {
  propose(row: number, maxTokens: number): Promise<number[]>;
}

/** Method-owned state membership, independent of scheduling and cache tiers.
 * Open/append borrow checkpoints; capture returns owned immutable state.
 * Membership changes and capture occur only between committed rounds.
 * The caller retains the provider's residency lease for the group's lifetime. */
export interface DraftRowGroup extends MlxDraftRows {
  /** Consume externally chosen target inputs at a committed round boundary.
   * Context and tokens are borrowed rectangular windows; lengths retain each
   * row's consumed prefix, excluding right padding and the pending output. */
  consume?(tokens: MlxArray, context: MlxArray, lengths: readonly number[]): void | Promise<void>;
  readonly namespace: string;
  readonly tapLayers: readonly number[];
  readonly rowCount: number;
  append(checkpoints: readonly DraftRowCheckpoint[]): void;
  prepareAppend(checkpoints: readonly DraftRowCheckpoint[]): PreparedStateChange;
  filterRows(keep: readonly number[]): void;
  capture(row: number): DraftRowCheckpoint;
  dispose(): void;
}

/** Draft preparation consumes each target chunk at the same batch geometry.
 * Inputs are borrowed and contain only new tokens/context, with equal chunk
 * lengths across rows. A null checkpoint appends a cold request. The provider
 * owns alignment and retained hidden state; scheduling and persistence do not.
 * Membership and capture occur between prefill calls. Capture returns owned
 * state after at least one target token has been consumed for that row. */
export interface DraftPrefillGroup {
  readonly namespace: string;
  /** Target prefill shape required by this provider's oracle. Most mlx-lm
   *  providers leave the final prompt token pending (`tail-split`); the MTP
   *  heads start from a full-prompt target forward. */
  readonly prefillMode: "tail-split" | "full";
  readonly tapLayers: readonly number[];
  readonly rowCount: number;
  append(checkpoints: readonly (DraftRowCheckpoint | null)[]): void;
  prepareAppend(checkpoints: readonly (DraftRowCheckpoint | null)[]): PreparedStateChange;
  prefill(tokens: MlxArray, context?: MlxArray): void | Promise<void>;
  /** Resolve borrowed backing before its external restore lease is released. */
  materialize(): void;
  filterRows(keep: readonly number[]): void;
  capture(row: number): DraftRowCheckpoint;
  dispose(): void;
}

export interface GroupedDraftProvider {
  readonly supportsExternalTokens?: boolean;
  /** The frequency-ranked draft vocabulary (ascending token ids) this provider's
   *  rows can select over for this target, or null. The binding resolves it once
   *  for the bound target; a request whose draft sampler reproduces on it drafts
   *  over it (`DraftSelection` `listed`). Absent: none. */
  draftVocabulary?(target: TargetView): readonly number[] | null;
  /** Resolve persistence identity without allocating a draft row. */
  checkpointNamespace?(): string;
  /** Draft state remains valid when target forwards run under a mounted
   * adapter context. Unqualified learned draft graphs leave this absent. */
  readonly supportsTargetAdapters?: boolean;
  /** Target hidden layers every row of this provider taps for this target, in
   * capture order. A provider whose rows tap must implement it; rows opened for
   * the target tap exactly this list. The binding resolves it once and admits the
   * provider only when the target forward can capture these layers. Absent: no taps. */
  targetTapLayers?(target: TargetView): readonly number[];
  openPrefill(options: { target: TargetView;
    checkpoints: readonly (DraftRowCheckpoint | null)[] }): DraftPrefillGroup;
  open(options: { target: TargetView; sampling: DraftRowSampling; constraints?: DraftRowConstraints;
    checkpoints: readonly DraftRowCheckpoint[] }): DraftRowGroup;
}

/** Server-lifetime owner of the draft machinery (the loaded draft model). Its
 * grouped rows are how every request drafts, one request or several. */
export interface DraftProvider {
  /** Human-readable id (registry id / path tail) for logs + cache namespacing. */
  readonly id: string;
  readonly weightsBytes: number;
  readonly grouped: GroupedDraftProvider;
  dispose(): void;
}

import { DraftRowCheckpoint } from "../../contracts/mlx/draft-checkpoint";
export { type DraftRowCheckpoint } from "../../contracts/mlx/draft-checkpoint";
export { type DraftProjection } from "../../contracts/mlx/draft-projection";

import { DraftProjection } from "../../contracts/mlx/draft-projection";
