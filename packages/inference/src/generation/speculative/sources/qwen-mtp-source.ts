// Native Qwen3.8 MTP draft provider — the Qwen-TRAINED multi-step prediction
// head, split from the raw release's last shard and published as
// mlx-community/Qwen3.8-27B-MTP-* (model_type "qwen3_5_mtp", block_size 3).
//
// Reference implementation (locate + port + gate, don't design):
// mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py. Mechanism, per
// step: h = fc(concat(rms_emb(embed(token)), rms_hidden(hidden))) → ONE
// full-attention decoder layer (own KVCache, positions continue the target
// sequence) → norm → the TARGET's lm_head → next draft token; recursive for
// the block ("trained with multiple steps"). The drafter is NOT standalone:
// it binds the target's embed_tokens + lm_head (mtp_use_dedicated_embeddings
// false) and consumes the target's PRE-final-norm last-layer hidden (mlx-vlm
// captures it with skip_final_norm=True) — carried here by the provider's tap
// layer, NOT the target's post-norm output.
//
// Row convention (predict-2-ahead): drafter KV row at position p is built
// from (embed(token at position p+1), hidden at position p). Invariant
// across rounds: after commit, the drafter offset equals the next pending
// token's position minus zero — i.e. a draft always opens by building the
// pending token's row from the TARGET's TRUE hidden at the emitted position
// (held over from the verify tap). Computationally identical to mlx-vlm's
// commit-time "seed" append, just performed at the top of the next round.
//
// Companion projections may be dense or quantized according to their metadata.
// Published drafter norms are already sanitized to runtime layout; no +1.0 shift.

import { loadModelConfig, type ModelConfig } from "../../../artifacts/config";
import { Weights } from "../../../artifacts/weights";
import type { RecurrentMtpModule } from "../../../contracts/mlx/drafter";
import { DRAFT_CHECKPOINT_SCHEMA } from "../../../contracts/mlx/draft-checkpoint";
import { loadQwenMtpModule } from "../../../models/drafter-loaders";
import type { DraftProvider, DraftRowGroup, DraftPrefillGroup, DraftRowSampling, GroupedDraftProvider, RecurrentMtpTarget, TargetView } from "../source";
import { targetLacks } from "../source";
import type { DraftRowCheckpoint } from "../../../contracts/mlx/draft-checkpoint";
import { QwenMtpRows } from "../bindings/qwen-mtp-rows";
import { configuredDraftVocabulary } from "../draft-vocab";
import { type MtpRowState } from "../../../state/speculative/qwen-mtp-state";
import { captureQwenMtpState, restoreQwenMtpState } from "../../../state/speculative/qwen-mtp-state";
import { applyStateChanges, cleanupFailure, disposeResources } from "../../../runtime/resources";
import { artifactIdentity } from "../../../artifacts/identity";
import { configFingerprint } from "../../../artifacts/fingerprint";
import type { PreparedStateChange } from "../../../contracts/portable/resources";

/** MTP drafts from the target's last hidden layer, before its final norm. */
const mtpTapLayers = (target: RecurrentMtpTarget): readonly number[] => Object.freeze([target.layerCount - 1]);

export class QwenMtpProvider implements DraftProvider {
  readonly grouped: GroupedDraftProvider = {
    supportsExternalTokens: true,
    targetTapLayers: view => mtpTapLayers(this.#target(view)),
    // A list shipped beside the companion (or named by MLX_BUN_SPEC_DRAFT_VOCAB),
    // for a target whose quantized head the rows can project a row subset of.
    draftVocabulary: view => {
      const head = this.#target(view).vocabularyHead;
      return head ? configuredDraftVocabulary(head.vocabSize, this.#dir) : null;
    },
    checkpointNamespace: () => this.#checkpointNamespace,
    open: options => this.#openRows(options.target, options.sampling, options.checkpoints),
    openPrefill: options => this.#openRows(options.target, null, options.checkpoints),
  };
  readonly id: string;
  readonly weightsBytes: number;
  readonly #module: RecurrentMtpModule;
  readonly #config: ModelConfig;
  readonly #resources: DisposableStack;
  readonly #checkpointNamespace: string;
  /** The companion's directory, where its optional draft vocabulary ships. */
  readonly #dir: string;

  private constructor(id: string, dir: string, config: ModelConfig, weightsBytes: number, module: RecurrentMtpModule, resources: DisposableStack, checkpointNamespace: string) {
    this.id = id;
    this.#dir = dir;
    this.#config = config;
    this.#module = module;
    this.weightsBytes = weightsBytes;
    this.#resources = resources;
    this.#checkpointNamespace = checkpointNamespace;
  }

  static async load(dir: string): Promise<QwenMtpProvider> {
    const config = await loadModelConfig(dir);
    const weights = await Weights.open(dir);
    using resources = new DisposableStack();
    resources.defer(() => weights.dispose());
    const weightsBytes = [...weights.shards.files.values()]
      .reduce((a, f) => a + f.mmap.size, 0);
    const module = await loadQwenMtpModule(weights, config, resources);
    // Stable across restarts and distinct for differently folded/quantized
    // companions. Hash once at provider load, outside inference execution.
    const identity = await artifactIdentity(configFingerprint(config),
      [...weights.shards.files].map(([name, shard]) => ({ name, path: shard.path })));
    return new QwenMtpProvider(
      dir.split("/").filter(Boolean).at(-1) ?? "qwen-mtp", dir,
      config, weightsBytes, module, resources.move(), `${DRAFT_CHECKPOINT_SCHEMA.qwenMtp}:${identity}`,
    );
  }

  #target(view: TargetView): RecurrentMtpTarget {
    if (this.#resources.disposed)
      throw new Error("qwen MTP provider is disposed");
    const target = view.recurrentMtp;
    if (!target) throw targetLacks("recurrentMtp");
    if (target.hiddenSize !== this.#config.text.hiddenSize) {
      throw new Error(
        `qwen MTP drafter hidden ${this.#config.text.hiddenSize} != target ` +
        `${target.hiddenSize} — split from a different checkpoint?`,
      );
    }
    return target;
  }

  #openRows(view: TargetView, sampling: DraftRowSampling | null,
    checkpoints: readonly (DraftRowCheckpoint | null)[]): DraftRowGroup & DraftPrefillGroup {
    const target = this.#target(view);
    const rows = new QwenMtpRows(target, this.#module, sampling, []);
    const prepareAppend = (checkpoints: readonly (DraftRowCheckpoint | null)[]) => {
      const states: Array<MtpRowState | null> = [];
      let change: PreparedStateChange | undefined;
      try {
        for (const checkpoint of checkpoints) states.push(checkpoint ? restoreQwenMtpState(checkpoint) : null);
        change = rows.prepareAppend(states);
        disposeResources(states.splice(0).flatMap(state => state ? [state.cache, state.hidden] : []));
        return change;
      } catch (error) {
        return cleanupFailure(error, () => disposeResources([
          ...states.flatMap(state => state ? [state.cache, state.hidden] : []), ...(change ? [change] : []),
        ]));
      }
    };
    const append = (checkpoints: readonly (DraftRowCheckpoint | null)[]) => applyStateChanges([() => prepareAppend(checkpoints)]);
    try { append(checkpoints); }
    catch (error) { return cleanupFailure(error, () => rows.dispose()); }
    return {
      namespace: this.#checkpointNamespace, prefillMode: "full", tapLayers: mtpTapLayers(target),
      get rowCount() { return rows.rowCount; },
      append, prepareAppend, prefill: (tokens, context) => rows.prefill(tokens, context!), materialize: rows.materialize.bind(rows),
      filterRows: rows.filterRows.bind(rows),
      draft: rows.draft.bind(rows), draftDevice: rows.draftDevice.bind(rows), preview: rows.preview.bind(rows), commit: rows.commit.bind(rows),
      consume: rows.consume.bind(rows),
      capture(row) {
        const state = rows.extractRow(row);
        try { return captureQwenMtpState(state); }
        finally { disposeResources([state.cache, state.hidden]); }
      },
      dispose: rows.dispose.bind(rows),
    };
  }

  dispose(): void {
    // Release cached transpose views before their native weight maps. MLX
    // retains buffers needed by outstanding GPU commands until completion.
    this.#resources.dispose();
  }
}
