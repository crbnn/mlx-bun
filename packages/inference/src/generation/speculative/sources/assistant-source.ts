import { artifactIdentity } from "../../../artifacts/identity";
import { assistantGroups } from "../bindings/assistant-rows";
// AssistantProvider — the optiq KV-borrowing Gemma "-assistant" drafter
// (packages/inference/src/models/gemma4/assistant.ts) behind the grouped draft
// rows. L2 oracle: optiq spec_generate. Its rows plug into the shared
// speculative group, so they compose with grammar, logits processors,
// admission accounting, and stats like any other provider's.
//
// KV-borrowing (why the rows read a TargetView): the drafter has NO cache of
// its own — each draft step reads the target's donor K/V (last sliding + last
// full layer, in chronological order) and, for the block's first step, the
// target's hidden at each row's anchor position, which the rows keep from the
// target's prefill and verify windows. See [[dspark-seam-kv-borrowing]].
//
// Drafting is GREEDY (the drafter's own argmax head), independent of the
// request sampler — drafts are only proposals; the target's verify (which does
// honor the sampler) decides every emitted token, so correctness holds at any
// temperature and only the acceptance RATE moves.

import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import type { AssistantDrafterModel } from "../../../contracts/mlx/drafter";
import { DRAFT_CHECKPOINT_SCHEMA } from "../../../contracts/mlx/draft-checkpoint";
import { loadAssistantDrafter } from "../../../models/drafter-loaders";
import type { DraftProvider } from "../source";

/** Sum of the drafter's on-disk safetensors — admission accounting only. */
function safetensorsBytes(dir: string): number {
  let total = 0;
  for (const f of readdirSync(dir))
    if (f.endsWith(".safetensors")) total += statSync(join(dir, f)).size;
  return total;
}

export class AssistantProvider implements DraftProvider {
  readonly id: string;
  readonly weightsBytes: number;
  readonly grouped: import("../source").GroupedDraftProvider;

  private constructor(
    private readonly drafter: AssistantDrafterModel,
    id: string,
    weightsBytes: number, namespace: string,
  ) {
    this.id = id;
    this.weightsBytes = weightsBytes;
    this.grouped = assistantGroups(drafter,namespace);
  }

  static async load(modelDir: string): Promise<AssistantProvider> {
    const drafter = await loadAssistantDrafter(modelDir);
    const id = modelDir.split("/").filter(Boolean).at(-1)!;
    try {
      const identity = await artifactIdentity(await Bun.file(`${modelDir}/config.json`).text(),
        readdirSync(modelDir).filter(file => file.endsWith(".safetensors"))
          .map(name => ({ name, path: join(modelDir, name) })));
      return new AssistantProvider(drafter, id, safetensorsBytes(modelDir), `${DRAFT_CHECKPOINT_SCHEMA.assistant}:${identity}`);
    } catch (error) { drafter.dispose(); throw error; }
  }

  dispose(): void {
    this.drafter.dispose();
  }
}
