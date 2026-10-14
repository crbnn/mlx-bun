import type { ModelConfig } from "../../artifacts/config";
import { TURBOQUANT_FOLD_FILE, readTurboQuantFold } from "../../artifacts/turboquant-fold";
import type { QuantizedEmbedding } from "../../layers/quantized-embedding";
import type { QuantizedLinear } from "../../layers/quantized-linear";
import { mropePositionIds } from "../../layers/qwen-mrope";
import { type MropeRequestState } from "../../contracts/mlx/positions";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { Cache } from "../../contracts/mlx/cache";
import type { MediaEncoders, MlxDecodeState, MlxPromptInput, VisionEncoder, MlxPositionedEmbeddingsRange, MlxPositionedRange, MlxPositionedForward } from "../../contracts/mlx/media";
import type { MlxHeadForward, MlxRowTable } from "../../contracts/mlx/graph";
import type { ResidualBasis, TargetView } from "../../contracts/mlx/draft-target";
import { existsSync } from "node:fs";
import { join } from "node:path";
import * as ops from "@mlx-bun/mlx/ops";
import { Dtype } from "@mlx-bun/mlx/ffi";
import { bindEmbeddingsInput } from "../media-input";

/** What the Qwen graph's media and draft ports read from it. */
export interface QwenMediaGraph {
  readonly config: ModelConfig;
  readonly embed: QuantizedEmbedding;
  readonly lmHead: QuantizedLinear | null;
  /** The full-attention layer whose cache supplies logical positions. */
  readonly faIdx: number;
  /** The vision prefill phase: a prompt's prepared embeddings (borrowed) at request positions. */
  prefillEmbeddingsAtPositions?(embeddings: MlxArray, caches: Cache[], positions: MlxArray, width: number): MlxArray;
  /** A prompt's token ids after a reused prefix, at request positions. */
  prefillAtPositions?(ids: MlxArray, caches: Cache[], positions: MlxArray, width: number): MlxArray;
  /** One new token per row at request positions. */
  decodeAtPositions?(ids: MlxArray, caches: Cache[], positions: MlxArray, width: number): MlxArray;
  readonly mediaPrefill?: readonly MlxPositionedEmbeddingsRange[];
  readonly mediaPrompt?: readonly MlxPositionedRange[];
  readonly mediaDecode?: MlxRowTable<MlxPositionedForward>;
  readonly logits: MlxRowTable<MlxHeadForward>;
}

class QwenMediaDecodeState implements MlxDecodeState {
  readonly key = "qwen-interleaved-mrope";
  constructor(readonly model: QwenMediaGraph, readonly delta: number) {}

  decode(ids: MlxArray, caches: Cache[], rows: readonly MlxDecodeState[], width: number): MlxArray {
    const B = rows.length, L = ids.shape[1]!;
    const cache = caches[this.model.faIdx]!;
    // The cache supplies logical positions, including left padding, on device.
    // No offset readback or request-global position mutation is needed.
    const offsets = (cache as Cache & { ropeOffsetArr?: MlxArray }).ropeOffsetArr;
    using deltas = ops.fromInt32(rows.map(row => (row as QwenMediaDecodeState).delta), [B]);
    using scalar = offsets ? null : ops.fromInt32([cache.offset], []);
    using starts = ops.add(offsets ?? scalar!, deltas);
    using startRows = ops.reshape(starts, [B, 1]);
    using indices = ops.arange(0, L, 1, Dtype.int32);
    using positions = ops.add(startRows, indices);
    using axis = ops.reshape(positions, [1, B, L]);
    using grid = ops.concatAxis([axis, axis, axis], 0);
    return this.model.mediaDecode ? this.model.mediaDecode[width]!(ids, caches, grid)
      : this.model.decodeAtPositions!(ids, caches, grid, width);
  }
}

/** The prompt owns its grid; only its continuation delta survives into decode. */
export function bindQwenMediaInput(model: QwenMediaGraph, embeddings: MlxArray,
  state: MropeRequestState): MlxPromptInput {
  const input = bindEmbeddingsInput((ids, caches, start, phase, width) => {
    using positions = mropePositionIds(state, start, ids.shape[1]!);
    if (start === 0) return model.mediaPrefill
      ? model.mediaPrefill.find(range => width >= range.from && width <= range.to)!.operation(embeddings, caches, positions)
      : model.prefillEmbeddingsAtPositions!(embeddings, caches, positions, width);
    if (phase === "decode") return model.mediaDecode ? model.mediaDecode[width]!(ids, caches, positions)
      : model.decodeAtPositions!(ids, caches, positions, width);
    return model.mediaPrompt
      ? model.mediaPrompt.find(range => width >= range.from && width <= range.to)!.operation(ids, caches, positions)
      : model.prefillAtPositions!(ids, caches, positions, width);
  });
  return { ...input, decodeState: new QwenMediaDecodeState(model, state.delta) };
}

/** Tower weights arrive either as the OptiQ-convention sidecar or in the main
 * shards (mlx-vlm convention); the tower loader handles both. The encoder is a
 * Qwen3VLVisionTower riding the shared lazy slot; the Qwen media route is its
 * only consumer and reads it through the structural contract it defines. */
export async function qwenMediaEncoders(model: QwenMediaGraph, modelDir: string): Promise<MediaEncoders> {
  const hasSidecar = existsSync(`${modelDir}/optiq/optiq_vision.safetensors`);
  if (!hasSidecar && model.config.raw.vision_config === undefined) return { vision: null, audio: null };
  const { Qwen3VLVisionTower } = await import("../vision/qwen3vl");
  return { vision: () => Qwen3VLVisionTower.load(modelDir) as unknown as VisionEncoder, audio: null };
}

/** The residual basis a rotated Qwen artifact declares in its fold record
 * (`turboquant_fold.json`), for its graph's constructor to read once. A record
 * whose stream carries no R1 rotation declares none. A missing record, or one
 * without `final_gain`, declares none either: DFlash 2 then drafts in its
 * trained basis (output exact, acceptance collapsed), and one line names the
 * file to update. */
export function qwenResidualBasis(graph: string, modelDir: string): ResidualBasis | undefined {
  const fold = readTurboQuantFold(modelDir);
  if (fold && !fold.r1) return undefined;
  if (fold?.finalGain) return Object.freeze({ r1Seed: fold.seed, finalGain: fold.finalGain });
  console.error(`[graph] ${graph}: ${join(modelDir, TURBOQUANT_FOLD_FILE)} ${fold ? "records no final_gain" : "is missing"}; ` +
    "DFlash 2 drafts unrotated (output exact, acceptance collapses) until it records the final-norm gain the fold moved into lm_head");
  return undefined;
}

/** Ports draft sources read from this graph: its embedding, head, layer taps,
 * the MTP geometry a checkpoint-native draft head needs and, from a graph whose
 * residual stream carries a rotation fold, that basis (the same object on every view). */
export function qwenDraftTarget(model: QwenMediaGraph, residualBasis?: ResidualBasis): TargetView {
  return Object.freeze({ identity: model, ...(residualBasis ? { residualBasis } : {}), hiddenLayerTaps: Object.freeze({
    layerCount: model.config.text.numHiddenLayers,
    projection: Object.freeze({ embed: model.embed, logits: model.logits }),
  }), recurrentMtp: Object.freeze({
    hiddenSize: model.config.text.hiddenSize, layerCount: model.config.text.numHiddenLayers,
    embed: model.embed.encode.bind(model.embed), logits: model.logits,
    vocabularyHead: model.lmHead ? Object.freeze({ w: model.lmHead.w, scales: model.lmHead.scales,
      biases: model.lmHead.biases, spec: model.lmHead.spec, vocabSize: model.config.text.vocabSize }) : undefined,
  }) });
}
