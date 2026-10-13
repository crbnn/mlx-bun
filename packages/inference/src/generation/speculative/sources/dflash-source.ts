// DflashProvider — DSpark (faithful DFlash + Markov + confidence,
// packages/inference/src/models/speculative/dflash.ts) behind the grouped draft
// rows. L3 (KL/quality-gated). Its rows plug into the shared speculative group.
//
// H_ctx (why the rows tap): DSpark drafts by attending to a GROWING
// multi-layer context — the target's tapped hiddens over the accepted stream
// (paper Eq 2-3). The provider declares its tap layers, so the group taps the
// target's prefill (→ seeds each row's context) and every verify forward (→
// grown at commit by the accepted window; rejected tips are never appended,
// mirroring the target-cache trim). See [[dspark-seam-kv-borrowing]].
//
// v1 acceptance: the group verifies with TOKEN-MATCH acceptance
// (mlx-lm/optiq style — lossless at any temperature), NOT the paper's
// distribution-level rejection-sampling verify. Greedy is identical; temp>0 is
// still lossless but lower-acceptance. Drafting here is greedy (proposals; the
// target verify decides). Confidence-scheduled pruning (Alg 1) may return fewer
// than the requested tokens: active when the checkpoint carries STS thresholds
// (cfg.sts, §3.2.1) or via the MLX_BUN_DSPARK_MINCONF override; uncalibrated
// checkpoints draft fixed-length.

import { artifactIdentity } from "../../../artifacts/identity";
import { projectedDraftGroups } from "../bindings/projected-draft-rows";
import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import type { DsparkDrafterModel } from "../../../contracts/mlx/drafter";
import { loadDsparkDrafter } from "../../../models/drafter-loaders";
import type { DraftProvider } from "../source";
import { targetLacks } from "../source";
import { runtimeValue } from "../../../runtime/config";

function safetensorsBytes(dir: string): number {
  let total = 0;
  for (const f of readdirSync(dir))
    if (f.endsWith(".safetensors")) total += statSync(join(dir, f)).size;
  return total;
}

export class DflashProvider implements DraftProvider {
  readonly id: string;
  readonly weightsBytes: number;
  readonly grouped: import("../source").GroupedDraftProvider;
  /** The trained block width — the server pins numDraftTokens to this so the
   *  group never asks for more positions than the block was trained for. */
  readonly gamma: number;

  private constructor(private readonly drafter: DsparkDrafterModel, id: string, weightsBytes: number, namespace: string) {
    this.id = id;
    this.weightsBytes = weightsBytes;
    this.gamma = drafter.cfg.gamma;
    this.grouped = projectedDraftGroups(namespace, drafter.cfg.tapLayers, target => {
      if (!target.hiddenLayerTaps) throw targetLacks("hiddenLayerTaps");
      const projection = target.hiddenLayerTaps.projection;
      const minConf = runtimeValue("MLX_BUN_DSPARK_MINCONF");
      return {
        namespace, schema: "dflash-context-v1",
        layers: drafter.cfg.nLayers,
        project: hidden => drafter.projectContextRows(hidden),
        draft(context, pending, _positions, depth) {
          return drafter.forwardRows(projection, null, pending, depth, {
            collectLogits: false, collectConfidence: false,
            thresholds: drafter.cfg.sts?.thresholds,
            ...(minConf ? { minConf: Number(minConf) } : {}),
          }, context).tokens;
        },
      };
    });
  }

  static async load(modelDir: string): Promise<DflashProvider> {
    const drafter = await loadDsparkDrafter(modelDir); // variant dispatch (dspark|legacy dflash)
    const id = modelDir.split("/").filter(Boolean).at(-1)!;
    try {
      const identity = await artifactIdentity(await Bun.file(`${modelDir}/dspark.json`).text(),
        readdirSync(modelDir).filter(file => file.endsWith(".safetensors"))
          .map(name => ({ name, path: join(modelDir, name) })));
      return new DflashProvider(drafter, id, safetensorsBytes(modelDir), `dflash-context-v1:${identity}`);
    } catch (error) { drafter.dispose(); throw error; }
  }

  dispose(): void {
    // Provider-owned weights are released when the mounted draft is closed.
    this.drafter.dispose();
  }
}
