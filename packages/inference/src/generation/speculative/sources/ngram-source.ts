// NgramProvider — MODEL-FREE prompt-lookup speculative drafting behind the
// grouped draft rows (packages/inference/src/generation/speculative/source.ts).
// No weights, no caches, no tokenizer:
// drafts are copied from the request's own token stream. When the trailing
// k-gram of (prompt + emitted-so-far) has occurred earlier in that stream,
// the tokens that followed the earlier occurrence are proposed as the draft;
// the shared speculative group makes the result LOSSLESS by construction
// (drafts are only proposals — the target's own samples decide every emitted
// token, at any temperature; only the acceptance rate moves). Best case:
// agentic/RAG/code-edit traffic that re-emits spans already present in context.
//
// Prior art (ported, not invented): "Prompt Lookup Decoding" (Apoorv Saxena,
// github.com/apoorvumang/prompt-lookup-decoding) and vLLM's `ngram` proposer
// (vllm/v1 spec_decode ngram_proposer) — both match the LONGEST k-gram first
// (k = max..min) and take the FIRST occurrence scanning left-to-right; we
// follow that exactly. Defaults mirror Saxena (max_ngram_size=3,
// num_pred_tokens=10 → our --num-draft-tokens default for this kind).
//
// Token history: each row keeps the committed stream it has seen. Prefill
// appends the target's prompt chunks (to len-1 under the tail-split shape, the
// last prompt token being the first round's pending token); draft appends the
// row's pending token, then looks up; commit appends the accepted drafts.
// Invariant at matching time: history == prompt + every emitted token, ending
// at the pending token. A round with no match drafts nothing, which the group
// verifies as one plain target step, so ngram speculation is structurally never
// worse than plain decode by more than the host lookup and verification overhead.

import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { CheckpointAttachment } from "../../../contracts/mlx/checkpoint";
import type { DraftProvider, DraftRowGroup, DraftPrefillGroup, GroupedDraftProvider, DraftRowConstraints } from "../source";
import type { DraftRowCheckpoint } from "../../../contracts/mlx/draft-checkpoint";
import { applyStateChanges } from "../../../runtime/resources";

// Histories are append-only between prefill/restore replacements. Weak keys
// release derived indexes with retired rows; checkpoints still contain tokens
// only. Ordered positions preserve the original first-occurrence policy.
const ngramPositions = new WeakMap<readonly number[], { length: number; positions: Map<number, number[]> }>();
function proposeNgram(h: readonly number[], max: number, min: number, n: number): number[] {
  let index = ngramPositions.get(h);
  if (!index || h.length < index.length) {
    index = { length: 0, positions: new Map() };
    ngramPositions.set(h, index);
  }
  for (let i = index.length; i < h.length; i++) {
    const token = h[i]!;
    const positions = index.positions.get(token);
    if (positions) positions.push(i); else index.positions.set(token, [i]);
  }
  index.length = h.length;
  const L = h.length;
  for (let k = Math.min(max, L - 1); k >= min; k--) {
    const tailAt = L - k;
    const positions = index.positions.get(h[tailAt]!) ?? [];
    search: for (const i of positions) {
      if (i + k >= L) break;
      for (let j = 0; j < k; j++) if (h[i + j] !== h[tailAt + j]) continue search;
      return h.slice(i + k, i + k + n);
    }
  }
  return [];
}

const namespace = (max: number, min: number) => `ngram-history-v1:${max}:${min}`;
const captureHistory = (history: readonly number[], max: number, min: number): CheckpointAttachment => ({
  schema: "ngram-history-v1", metadata: { max, min }, tensors: [ops.fromInt32([...history], [history.length])],
});
function restoreHistory(state: DraftRowCheckpoint, max: number, min: number): number[] {
  const { attachment, processedTokens } = state;
  if (attachment.schema !== "ngram-history-v1" || attachment.metadata.max !== max || attachment.metadata.min !== min ||
      attachment.tensors.length !== 1 || attachment.tensors[0]!.shape.length !== 1 || attachment.tensors[0]!.shape[0] !== processedTokens)
    throw new Error("ngram checkpoint does not match its token coverage or lookup policy");
  return attachment.tensors[0]!.toIntTokens();
}

/** Each row owns only its committed token history. Verification and sampling
 * remain in the shared method; target storage and scheduling are not needed. */
class NgramRows implements DraftRowGroup, DraftPrefillGroup {
  readonly prefillMode = "tail-split" as const;
  readonly tapLayers: readonly number[] = [];
  readonly namespace: string;
  #histories: number[][] = [];
  #drafts: number[][] = [];
  constructor(readonly max: number, readonly min: number, checkpoints: readonly (DraftRowCheckpoint | null)[],
    readonly constraints?: DraftRowConstraints) {
    this.namespace = namespace(max, min);
    this.append(checkpoints);
  }
  get rowCount() { return this.#histories.length; }
  prepareAppend(checkpoints: readonly (DraftRowCheckpoint | null)[]) {
    let next = [...this.#histories, ...checkpoints.map(state => state ? restoreHistory(state, this.max, this.min) : [])];
    return { commit: () => { const old = this.#histories; this.#histories = next; next = old; }, dispose: () => { next = []; } };
  }
  append(checkpoints: readonly (DraftRowCheckpoint | null)[]) { applyStateChanges([() => this.prepareAppend(checkpoints)]); }
  filterRows(keep: readonly number[]) { this.#histories = keep.map(row => this.#histories[row]!); this.#drafts = []; }
  prefill(tokens: MlxArray): void {
    using packed = ops.contiguous(tokens);
    const ids = packed.toIntTokens(), length = tokens.shape[1]!;
    for (let row = 0; row < this.rowCount; row++) this.#histories[row]!.push(...ids.slice(row * length, (row + 1) * length));
  }
  materialize(): void {}
  draft(pending: readonly number[], depth: number): number[][] | Promise<number[][]> {
    if (this.constraints) {
      for (const [row, history] of this.#histories.entries()) history.push(pending[row]!);
      return Promise.all(this.#histories.map((_, row) => this.constraints!.propose(row, depth)))
        .then(proposals => this.#drafts = proposals);
    }
    this.#drafts = this.#histories.map((history, row) => {
      history.push(pending[row]!);
      return proposeNgram(history, this.max, this.min, depth);
    });
    return this.#drafts;
  }
  commit(accepted: readonly number[]) {
    for (let row = 0; row < this.rowCount; row++) this.#histories[row]!.push(...this.#drafts[row]!.slice(0, accepted[row]!));
    this.#drafts = [];
  }
  capture(row: number): DraftRowCheckpoint {
    const history = this.#histories[row]!;
    return { processedTokens: history.length, attachment: captureHistory(history, this.max, this.min) };
  }
  dispose() { this.#histories = []; this.#drafts = []; }
}

export interface NgramOptions {
  /** Longest suffix k-gram tried first (Saxena max_ngram_size). */
  max?: number;
  /** Shortest k-gram tried before giving up (vLLM prompt_lookup_min). */
  min?: number;
}

export class NgramProvider implements DraftProvider {
  readonly id = "ngram";
  readonly weightsBytes = 0;
  readonly max: number;
  readonly min: number;
  readonly grouped: GroupedDraftProvider = {
    checkpointNamespace: () => namespace(this.max, this.min),
    supportsTargetAdapters: true,
    open: options => new NgramRows(this.max, this.min, options.checkpoints),
    openPrefill: options => new NgramRows(this.max, this.min, options.checkpoints),
  };

  constructor(opts: NgramOptions = {}) {
    this.max = Math.max(1, opts.max ?? 3);
    this.min = Math.max(1, Math.min(opts.min ?? 1, this.max));
  }

  dispose(): void {}
}

/** Constraint candidates share committed-history ownership with prompt lookup.
 * They never perform ngram lookup. */
export function constraintDraftProvider(): { id: string; grouped: GroupedDraftProvider } {
  return {
    id: "grammar",
    grouped: {
      checkpointNamespace: () => namespace(3, 1),
      supportsTargetAdapters: true,
      openPrefill: options => new NgramRows(3, 1, options.checkpoints),
      open: options => new NgramRows(3, 1, options.checkpoints,
        options.constraints ?? { propose: async () => [] }),
    },
  };
}
