// Native GLM-5.2 MTP draft provider.
//
// The target and MTP row share embeddings, the output head, and the resident
// dense target weights. Only the MTP layer's own dense tensors and int8 routed
// experts add model weight residency. The shared speculative group owns
// verification, grammar composition, target-cache rollback, sampling, and
// telemetry. Prompt lookup remains the model-free alternative DraftProvider;
// selecting one provider per request prevents two independent draft histories
// from advancing the target or grammar state twice.

import type { NativeMtpHead } from "../../../contracts/mlx/drafter";
import { nativeMtpHeadOf } from "../../../models/drafter-loaders";
import type { DraftProvider, GroupedDraftProvider, TargetView, DraftRowSampling, DraftRowGroup, DraftPrefillGroup } from "../source";
import { DRAFT_CHECKPOINT_SCHEMA, type DraftRowCheckpoint } from "../../../contracts/mlx/draft-checkpoint";
import { Glm52MtpRows } from "../bindings/glm52-mtp-rows";
import { type Glm52MtpRowState } from "../../../state/speculative/glm52-mtp-state";
import { captureGlm52MtpState, restoreGlm52MtpState } from "../../../state/speculative/glm52-mtp-state";
import { applyStateChanges, disposeResources } from "../../../runtime/resources";

export class Glm52NativeMtpProvider implements DraftProvider {
  readonly grouped: GroupedDraftProvider = {
    checkpointNamespace: () => DRAFT_CHECKPOINT_SCHEMA.glm52Mtp,
    open: options => this.#openRows(options.target, options.sampling, options.checkpoints),
    openPrefill: options => this.#openRows(options.target, null, options.checkpoints),
  };
  readonly id = "glm52-native-mtp";
  readonly weightsBytes = 0;
  readonly #head: NativeMtpHead;

  /** Binds the native MTP head `model` declares (`GraphCapabilities.nativeDraft`); it owns every shared and MTP weight. */
  constructor(readonly model: object) {
    this.#head = nativeMtpHeadOf(model);
  }

  #openRows(target: TargetView, sampling: DraftRowSampling | null,
    checkpoints: readonly (DraftRowCheckpoint | null)[]): DraftRowGroup & DraftPrefillGroup {
    if (target.identity !== this.model) throw new Error("native MTP provider was opened for a different target");
    const rows = new Glm52MtpRows(this.#head, sampling);
    const prepareAppend = (checkpoints: readonly (DraftRowCheckpoint | null)[]) => {
      const states: (Glm52MtpRowState | null)[] = [];
      try {
        for (const checkpoint of checkpoints) states.push(checkpoint ? restoreGlm52MtpState(checkpoint, () => rows.makeCache()) : null);
        return rows.prepareAppend(states);
      } finally { disposeResources(states.flatMap(state => state ? [state.cache, state.hidden] : [])); }
    };
    const append = (states: readonly (DraftRowCheckpoint | null)[]) => applyStateChanges([() => prepareAppend(states)]);
    try { append(checkpoints); } catch (error) { rows.dispose(); throw error; }
    return {
      namespace: DRAFT_CHECKPOINT_SCHEMA.glm52Mtp, prefillMode: "full", tapLayers: [],
      get rowCount() { return rows.rowCount; },
      append, prepareAppend, prefill: (tokens, context) => rows.prefill(tokens, context!),
      materialize: rows.materialize.bind(rows), filterRows: rows.filterRows.bind(rows),
      draft: rows.draft.bind(rows), commit: rows.commit.bind(rows),
      capture(row) {
        const state = rows.extractRow(row);
        try { return captureGlm52MtpState(state); }
        finally { state.cache.dispose(); state.hidden.dispose(); }
      },
      dispose: rows.dispose.bind(rows),
    };
  }

  dispose(): void {
    // The target model owns every shared/MTP weight and the expert tier.
  }
}
