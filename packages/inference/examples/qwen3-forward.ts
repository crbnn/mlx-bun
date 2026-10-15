import { bindGraphCaches } from "@mlx-bun/inference/models";
import { phaseRowLimit } from "@mlx-bun/inference/models/graph";
import { MlxPhaseDispatch } from "@mlx-bun/inference/generation";
import { loadModelConfig, Weights } from "@mlx-bun/inference/artifacts";
import { resolveComposition } from "@mlx-bun/inference";

import { Qwen3Model } from "@mlx-bun/inference/models/qwen3";
import { argmaxLastPosition, lastPositionLogits } from "@mlx-bun/inference/scoring";
import * as ops from "@mlx-bun/mlx/ops";

/** Explicit forward passes through the graph's named phases: the caller
 * supplies tokens, names the phase of each call and owns the live cache. A
 * prompt shorter than the prefill chunk is one prompt tail. */
export async function forwardTokens(directory: string, promptTokens: number[]) {
  const config = await loadModelConfig(directory);
  const weights = await Weights.open(directory);
  try {
    const graph = bindGraphCaches(new Qwen3Model(weights, config, phaseRowLimit(resolveComposition(config))), resolveComposition(config));
    const state = graph.makeCache();
    try {
      using prompt = ops.fromInt32(promptTokens, [1, promptTokens.length]);
      using promptHidden = await new MlxPhaseDispatch(graph).prompt("tail", 1, (promptTokens.length))(prompt, state, null);
      using prefill = graph.logits[(promptTokens.length)]!(promptHidden);
      const token = argmaxLastPosition(prefill);
      using next = ops.fromInt32([token], [1, 1]);
      using nextHidden = await graph.decode[1]!(next, state, null);
      using continuation = graph.logits[1]!(nextHidden);
      return { token, nextToken: argmaxLastPosition(continuation),
        logits: lastPositionLogits(continuation), offset: state[0]!.offset };
    } finally {
      for (const cache of state) cache.dispose();
    }
  } finally {
    weights.dispose();
  }
}

if (import.meta.main) {
  const directory = process.argv[2], ids = process.argv[3];
  if (!directory || !ids) throw new Error('Usage: bun qwen3-forward.ts <checkpoint-directory> <token-ids-as-JSON>');
  console.log(await forwardTokens(directory, JSON.parse(ids)));
}
