// DeepspecProvider — DeepSeek's released DSpark drafter checkpoints
// (Gemma4DSparkModel, e.g. deepseek-ai/dspark_gemma4_12b_block7) behind the
// grouped draft rows. The reference implementation is DeepSpec
// (github.com/deepseek-ai/DeepSpec, MIT); the module port with transcription
// notes lives in packages/inference/src/models/speculative/deepspec.ts. Oracle: DeepSpec's eval
// at temperature 0 (RNG-free — verify degenerates to argmax token-match),
// `02d723a:scripts/oracle/oracle-dspark-deepspec.py` → dspark-deepspec-compare.ts (deleted 2026-08-23; git history).
//
// Row mapping (why this fits the shared speculative group): the reference
// recycles target hidden states from each verify pass into the drafter's
// growing context — exactly the tapped context the rows consume:
//   prefill                  → project the prompt's tapped hiddens into the
//                              per-layer context K/V rows (their DynamicCache
//                              of PROJECTED context rows, positions 0..Lp-1);
//   draft                    → one block forward per row against its rows
//                              (anchor = the row's pending token at its
//                              context length); confidence truncation may
//                              return 0..depth tokens (ℓ=0 = plain step);
//   commit                   → project + append ONLY the accepted rows
//                              [anchor + accepted drafts] (their crop()
//                              semantics: rejected rows never enter the
//                              context).
// Context rows are cached post-norm-post-rope — bit-equivalent to their
// in-round concat (per-row RMSNorm/RoPE; argued in projectContextKVRows's doc).

import { artifactIdentity } from "../../../artifacts/identity";
import { deepspecGroups } from "../bindings/deepspec-rows";
import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import type { DeepspecDrafterModel } from "../../../contracts/mlx/drafter";
import { loadDeepspecDrafter } from "../../../models/drafter-loaders";
import type { DraftProvider } from "../source";

function safetensorsBytes(dir: string): number {
  let total = 0;
  for (const f of readdirSync(dir))
    if (f.endsWith(".safetensors")) total += statSync(join(dir, f)).size;
  return total;
}

export class DeepspecProvider implements DraftProvider {
  readonly id: string;
  readonly weightsBytes: number;
  readonly grouped: import("../source").GroupedDraftProvider;
  /** Trained block width (config block_size, e.g. 7) — the server pins
   *  numDraftTokens to this. */
  readonly gamma: number;

  private constructor(private readonly drafter: DeepspecDrafterModel, id: string, weightsBytes: number, namespace: string) {
    this.id = id;
    this.weightsBytes = weightsBytes;
    this.gamma = drafter.gamma;
    this.grouped = deepspecGroups(drafter, namespace);
  }

  static async load(modelDir: string): Promise<DeepspecProvider> {
    const drafter = await loadDeepspecDrafter(modelDir);
    const id = modelDir.split("/").filter(Boolean).at(-1)!;
    try {
      const identity = await artifactIdentity(await Bun.file(`${modelDir}/config.json`).text(),
        readdirSync(modelDir).filter(file => file.endsWith(".safetensors"))
          .map(name => ({ name, path: join(modelDir, name) })));
      return new DeepspecProvider(drafter, id, safetensorsBytes(modelDir), `deepspec-context-v1:${identity}`);
    } catch (error) { drafter.dispose(); throw error; }
  }

  dispose(): void {
    this.drafter.dispose(); // owned MlxArrays (transpose views + mmap'd weights)
  }
}
