// Two-model speculative drafting — mlx-lm parity (`mlx_lm.server
// --draft-model`). A full second model drafts autoregressively in the grouped
// draft rows (bindings/two-model-rows.ts); the shared speculative group
// verifies with the target. L1 oracle:
// mlx_lm.server with the same target/draft pair, greedy, token-for-token
// (spec-vs-spec — both batch the verify lm-head, so neither is bit-exact to
// stock decode at knife-edges).
//
// Faithfulness notes (read from the oracle venv's generate.py):
//  - drafts are sampled with the REQUEST sampler (generate.py:593-601) —
//    greedy drafting under a temperature>0 request is not parity;
//  - draft-cache rewind is max(n - kAccept - 1, 0) (generate.py:589-591),
//    with the all-accept re-feed of the last draft kept pending in the rows
//    and joined with the next target token (generate.py:645-648);
//  - logits processors do NOT run on draft steps (they're target-side).

import { loadModelConfig } from "../../../artifacts/config";
import { Weights } from "../../../artifacts/weights";
import { createModel, type RuntimeModel } from "../../../models/factory";
import type { DraftProvider, GroupedDraftProvider } from "../source";
import { standaloneDraftGroups } from "../bindings/two-model-rows";
import { bindMlxAutoregressiveGraph } from "../../bindings/autoregressive";
import { artifactIdentity } from "../../../artifacts/identity";
import { configFingerprint } from "../../../artifacts/fingerprint";
import { cleanupFailure, disposeResources } from "../../../runtime/resources";

export class TwoModelProvider implements DraftProvider {
  readonly id: string;
  readonly weightsBytes: number;
  readonly grouped: GroupedDraftProvider;

  private constructor(
    readonly model: RuntimeModel,
    private readonly weights: Weights,
    id: string,
    namespace: string,
  ) {
    this.id = id;
    this.grouped = standaloneDraftGroups(bindMlxAutoregressiveGraph(model), namespace);
    this.weightsBytes = [...weights.shards.files.values()].reduce(
      (a, f) => a + f.mmap.size,
      0,
    );
  }

  /** Load the draft model from a snapshot dir. Vocab-size mismatch with the
   *  target is a WARNING (mlx-lm parity, server.py:363-368); the tokenizer-
   *  family hard check lives at the server (it owns both tokenizers). */
  static async load(modelDir: string, targetVocabSize?: number): Promise<TwoModelProvider> {
    const config = await loadModelConfig(modelDir);
    const weights = await Weights.open(modelDir);
    let model: RuntimeModel | null = null;
    try {
      model = createModel(weights, config);
      if (
        targetVocabSize !== undefined &&
        config.text.vocabSize !== targetVocabSize
      )
        console.warn(
          `draft model vocab size ${config.text.vocabSize} != target ${targetVocabSize} — ` +
            `speculation will accept nothing useful (mlx-lm warns identically)`,
        );
      const identity = await artifactIdentity(configFingerprint(config),
        [...weights.shards.files].map(([name, shard]) => ({ name, path: shard.path })));
      return new TwoModelProvider(model, weights, modelDir.split("/").filter(Boolean).at(-1)!,
        `standalone-draft-v1:${identity}`);
    } catch (error) { return cleanupFailure(error, () => disposeResources([...(model ? [model] : []), weights])); }
  }

  /** The draft graph releases what it built (constants, Neural Engine programs),
   *  then its weights go. */
  dispose(): void {
    disposeResources([this.model, this.weights]);
  }
}
