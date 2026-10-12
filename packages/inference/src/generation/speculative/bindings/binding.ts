import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { Cache } from "../../../contracts/mlx/cache";
import type { MlxLayerCapture } from "../../../contracts/mlx/graph";
import type { MlxTokenGraph } from "../../../models/graph";
import { declaredGraph, tokenPhasesOf } from "../../../models/capabilities";
import type { Sampler } from "../../../sampling/index";
import type { DraftProvider, DraftSource } from "../source";
import { runtimeConfig, type RuntimeConfig } from "../../../runtime/config";
import type { MlxModelMemory } from "../../bindings/autoregressive";
import type { SpeculativeTransaction } from "../../../contracts/portable/rollback";
import { bindCacheRollback } from "../../../state/rollback";
import type { GraphDescriptor } from "../../../contracts/portable/graph";
import { bindLegacyDraftTarget } from "./draft-target";
import type { MlxForwardWork } from "../../../contracts/mlx/forward-work";
import type { PrefillPolicy } from "../../../contracts/portable/prefill";
import { COMPOSITION_DEFAULTS, type KvStorageScheme } from "../../../contracts/portable/composition";
import { resolveMlxPrefillPolicy } from "../../bindings/prefill-policy";
import { MlxPhaseDispatch, promptWorkStep, verifyWorkStep, type MlxPromptStep, type MlxVerifyWindow } from "../../phase-dispatch";

/** A target forward's post-final-norm hidden and, when the drafter taps layers,
 *  the tapped layers concatenated on the feature axis [B, L, m*H] (null
 *  otherwise). Both belong to the caller. */
export interface MlxTargetForward {
  readonly hidden: MlxArray;
  readonly ctxML: MlxArray | null;
}

/** Bound target operations and draft construction for one speculative run.
 * A replacement graph supplies this entire port, including any hidden taps.
 * The verifier never needs a MlxTokenGraph or an artifact-family check. Each
 * forward names its phase; `work`, when given, is the mixed iteration's hook
 * and runs the same rows inside a shared forward. */
export interface MlxSpeculativeBinding {
  readonly runtime?: RuntimeConfig;
  readonly prefillPolicy?: PrefillPolicy;
  readonly memory?: MlxModelMemory;
  readonly descriptor: GraphDescriptor;
  readonly eosTokenIds: readonly number[];
  readonly prefillTailSplit: boolean;
  /** The composition's KV scheme, which `makeCache` composes; absent: bf16. */
  readonly kv?: KvStorageScheme;
  makeCache(): Cache[];
  openDraft(sampler: Sampler, caches: Cache[]): DraftSource;
  bindRollback(caches: Cache[]): SpeculativeTransaction;
  /** One prompt step in the phase its planner named, capturing `tapLayers`. */
  prefill(ids: MlxArray, caches: Cache[], step: MlxPromptStep, tapLayers?: readonly number[],
    work?: MlxForwardWork): Promise<MlxTargetForward>;
  /** A verify window, capturing `tapLayers`. */
  verify(ids: MlxArray, caches: Cache[], window: MlxVerifyWindow, tapLayers?: readonly number[],
    work?: MlxForwardWork): Promise<MlxTargetForward>;
  /** One new token per sequence, outside a speculative round. */
  decode(ids: MlxArray, caches: Cache[], sequences: number): Promise<MlxArray>;
  /** Whether the forwards can capture these hidden layers (a provider's declared taps). */
  supportsTapLayers(layers: readonly number[]): boolean;
  /** The head over a hidden of `rows` rows (sequences times positions). */
  projectLogits(hidden: MlxArray, rows: number): MlxArray;
}

export type MlxSpeculativeTargetBinding = Omit<MlxSpeculativeBinding, "openDraft">;

/** The target graph's phase operations for speculation, bound once. */
export function bindSpeculativeTargetModel(model: MlxTokenGraph): MlxSpeculativeTargetBinding {
  const runtime = runtimeConfig();
  const dispatch = new MlxPhaseDispatch(tokenPhasesOf(model));
  /** The decoder's layer count, the post-final-norm capture's index; read where a tap needs it. */
  const layerCount = () => model.config.text.numHiddenLayers;
  return {
    runtime,
    prefillPolicy: resolveMlxPrefillPolicy(model, runtime),
    memory: model,
    descriptor: Object.freeze({ id: `legacy-spec:${model.config.modelType}`, backend: "mlx",
      graphAbi: "mlx-hidden-bsh-v1", stateAbi: "legacy-cache-array-v1", artifact: "legacy-resident-model" }),
    eosTokenIds: model.config.eosTokenIds,
    prefillTailSplit: (model.composition ?? COMPOSITION_DEFAULTS).prefillTailSplit,
    ...(model.composition ? { kv: model.composition.kv } : {}),
    makeCache: model.makeCache.bind(model),
    bindRollback: bindCacheRollback,
    prefill: (ids, caches, step, tapLayers, work) => withCapture(tapLayers, layerCount, capture => {
      const own = async () => dispatch.prompt(step.phase, step.sequences, step.tokens)(ids, caches, capture);
      return work ? work(ids, caches, promptWorkStep(step, capture, own)) : own();
    }),
    verify: (ids, caches, window, tapLayers, work) => withCapture(tapLayers, layerCount, capture => {
      const own = async () => dispatch.verify(window.sequences, window.positions)(ids, caches, capture);
      return work ? work(ids, caches, verifyWorkStep(window, capture, own)) : own();
    }),
    decode: async (ids, caches, sequences) => dispatch.decode(sequences)(ids, caches, null),
    // The capture sees layers 0..nLayers-1 and then the post-final-norm output as index nLayers.
    supportsTapLayers: layers => !layers.length || (declaredGraph(model).graphCapabilities.hiddenLayerTaps && layers.every(layer =>
      Number.isInteger(layer) && layer >= 0 && layer <= layerCount())),
    projectLogits: (hidden, rows) => dispatch.logits(rows)(hidden),
  };
}

/** Serial compatibility binds draft construction separately from target work. */
export function bindLegacySpeculativeModel(model: MlxTokenGraph, provider: DraftProvider): MlxSpeculativeBinding {
  return { ...bindSpeculativeTargetModel(model),
    openDraft: (sampler, caches) => provider.open({ sampler, target: bindLegacyDraftTarget(model, caches) }) };
}

export function assertMlxSpeculativeBinding(binding: MlxSpeculativeBinding): void {
  const descriptor = binding.descriptor;
  if (descriptor.backend !== "mlx" || descriptor.graphAbi !== "mlx-hidden-bsh-v1" ||
      descriptor.stateAbi !== "legacy-cache-array-v1")
    throw new Error(`speculative binding ${descriptor.id} has an incompatible backend, graph, or state ABI`);
}

/** Run `forward` with a capture of `tapLayers` (none without taps) and return
 *  its hidden and the tapped layers [B, L, m*H], concatenated in `tapLayers`
 *  order on the feature axis. The capture copies each tapped layer as the
 *  graph hands it over (index `layerCount` is the post-final-norm output), so
 *  every graph and the mixed iteration capture the same way. */
async function withCapture(tapLayers: readonly number[] | undefined, layerCount: () => number,
  forward: (capture: MlxLayerCapture | null) => MlxArray | Promise<MlxArray>): Promise<MlxTargetForward> {
  if (!tapLayers?.length) return { hidden: await forward(null), ctxML: null };
  const cap = new Map<number, MlxArray>();
  const layers = new Set(tapLayers);
  let hidden: MlxArray | null = null;
  try {
    hidden = await forward((layer, h) => { if (layers.has(layer)) cap.set(layer, ops.contiguous(h)); });
    const final = layerCount();
    if (layers.has(final) && !cap.has(final)) cap.set(final, ops.contiguous(hidden));
    const perLayer = tapLayers.map(layer => {
      const captured = cap.get(layer);
      if (!captured) throw new Error(`spec tap: layer ${layer} not captured`);
      return captured;
    });
    const ctxML = ops.concatAxis(perLayer, 2);
    for (const [, a] of cap) a.dispose();
    cap.clear();
    const out = { hidden, ctxML };
    hidden = null; // ownership returned to the caller
    return out;
  } finally {
    // On any throw, free the partial captures and the orphaned hidden.
    hidden?.dispose();
    for (const [, a] of cap) a.dispose();
  }
}

export { bindLegacyDraftTarget } from "./draft-target";
