// Model construction consumes one declared profile and one composition. Exact
// artifact profiles outrank family profiles; the profile names the family's
// implementation, and the family's graph table picks the graph for the
// composition. Request-level methods are resolved elsewhere and are never
// changed by this module.

import { loadModelConfig, type ModelConfig } from "../artifacts/config";
import { totalmem } from "node:os";
import { Weights } from "../artifacts/weights";
import { Gemma4Model } from "./gemma4/model";
import { configFingerprint } from "../artifacts/fingerprint";
import { qwen35GraphFingerprint } from "./qwen/checkpoint";
import {
  COMPOSITION as QWEN38_TRELLIS_M4PRO_COMPOSITION, QWEN38_TRELLIS_M4PRO_GRAPH, Qwen38TrellisM4Pro,
  anePrograms as qwen38TrellisM4ProAnePrograms,
} from "./qwen/qwen38-27b-trellis-m4pro";
import {
  COMPOSITION as QWEN38_TRELLIS_M1MAX_COMPOSITION, QWEN38_TRELLIS_M1MAX_GRAPH, Qwen38TrellisM1Max,
} from "./qwen/qwen38-27b-trellis-m1max";
import { GENERATED_GRAPHS, type GeneratedGemma4Graph } from "./gemma4/generated/index";
import { MiniCPM5Model } from "./minicpm5/model";
import { Qwen35Model } from "./qwen/qwen3_5";
import { Qwen3Model } from "./qwen/qwen3";
import { Qwen3MoeModel } from "./qwen/qwen3-moe";
import { DiffusionGemmaModel } from "./diffusion-gemma/model";
import { Glm52Model } from "./glm52/model";
import {
  GLM52_G5_DEFAULT_CONTEXT_TOKENS,
  GLM52_G5_DEFAULT_MAX_GENERATION_TOKENS,
  GLM52_G5_DEFAULT_PROCESS_LIMIT_BYTES,
  planGlm52MemoryForArtifact,
  type Glm52MemoryPlan,
} from "../artifacts/glm52/memory";
import { UniversalDenseModel } from "./universal/dense";
import { WhisperModel } from "./whisper/model";
import { genericArgsFor } from "./universal/archs";
import { familyForGraph } from "./families";
import { assertComposedCaches, resolveComposition } from "./composition";
import type { Composition } from "../contracts/portable/composition";
import {
  GraphTable,
  ModelImplementationRegistry,
  type GraphEntry,
  type ModelImplementation,
  type ModelImplementationProvider,
} from "./implementation";
import {
  assertResolvedModelProfile,
  resolveModelProfile,
  type ResolvedModelProfile,
  type ResolveModelProfileOptions,
  type ModelGraph,
} from "./profile";

export type RuntimeModel =
  | Gemma4Model | GeneratedGemma4Graph | MiniCPM5Model | Qwen35Model | Qwen38TrellisM4Pro | Qwen38TrellisM1Max | Qwen3Model | Qwen3MoeModel
  | DiffusionGemmaModel | Glm52Model | UniversalDenseModel;

/** A family's generic graph: the table entry that constrains nothing. */
function generic(id: string, create: GraphEntry<Weights, RuntimeModel>["create"]): GraphEntry<Weights, RuntimeModel> {
  return { id, serves: {}, create };
}

/** Every resident family's graphs, keyed by the graph its family declares. Each
 * table lists the compositions its specialized graphs were built and qualified
 * for, plus the family's generic graph; see {@link GraphTable} for how one is
 * selected. Adding a graph means adding an entry here (or to the list a table
 * spreads in). */
export const MLX_GRAPH_TABLES: Readonly<Record<ModelGraph, GraphTable<Weights, RuntimeModel>>> = Object.freeze({
  "diffusion-gemma": new GraphTable([generic("diffusion-gemma", (weights, config, composition) => new DiffusionGemmaModel(weights, config, composition))]),
  "minicpm5": new GraphTable([generic("minicpm5", (weights, config, composition) => new MiniCPM5Model(weights, config, composition))]),
  // Our published Trellis quant has its generated graphs (scripts/gen-qwen38.ts), keyed by its graph
  // fingerprint (architecture plus the complete quantization table) and the GPU each was built for.
  "qwen3.5": new GraphTable([
    { id: QWEN38_TRELLIS_M4PRO_GRAPH, serves: QWEN38_TRELLIS_M4PRO_COMPOSITION,
      create: (weights, config, composition) => new Qwen38TrellisM4Pro(weights, config, composition),
      anePrograms: composition => qwen38TrellisM4ProAnePrograms(composition) },
    // The M1 Max composition builds no Neural Engine operation (its device's table has none): no programs to install.
    { id: QWEN38_TRELLIS_M1MAX_GRAPH, serves: QWEN38_TRELLIS_M1MAX_COMPOSITION,
      create: (weights, config, composition) => new Qwen38TrellisM1Max(weights, config, composition) },
    generic("qwen3.5", (weights, config, composition) => new Qwen35Model(weights, config, composition)),
  ], qwen35GraphFingerprint),
  "qwen3-moe": new GraphTable([generic("qwen3-moe", (weights, config, composition) => new Qwen3MoeModel(weights, config, composition))]),
  "qwen3": new GraphTable([generic("qwen3", (weights, config, composition) => new Qwen3Model(weights, config, composition))]),
  "gemma4": new GraphTable([...GENERATED_GRAPHS, generic("gemma4", (weights, config, composition) => new Gemma4Model(weights, config, composition))],
    configFingerprint),
  "universal-dense": new GraphTable([generic("universal-dense", (weights, config, composition) =>
    new UniversalDenseModel(weights, config, composition, genericArgsFor(config)!))]),
});

/** A resident family's native construction: its graph table selects the graph for
 * the composition it receives. Its loader and loop are the family's own declaration. */
function residentImplementation(graph: ModelGraph, table: GraphTable<Weights, RuntimeModel>): ModelImplementation<Weights, RuntimeModel> {
  const family = familyForGraph(graph);
  if (!family) throw new Error(`graph table ${graph} names a graph no model family declares`);
  return { id: graph, graph, loader: family.loader, loop: family.loop,
    create: (weights, config, _profile, composition) => table.select(config, composition).create(weights, config, composition) };
}

/** Engine-owned registrations, one per resident graph of `MODEL_FAMILIES`. Exact
 * quant profiles can name additional implementations in a composed registry,
 * without changing sessions or files. */
export const MLX_MODEL_IMPLEMENTATIONS = new ModelImplementationRegistry<Weights, RuntimeModel>(
  Object.entries(MLX_GRAPH_TABLES).map(([graph, table]) => residentImplementation(graph, table)));

export interface ModelOpenOptions<Model = RuntimeModel> extends Glm52RuntimeOpenOptions {
  readonly profiles?: ResolveModelProfileOptions;
  readonly implementations?: ModelImplementationProvider<Weights, Model>;
  /** The composition the graph is built for; absent resolves the default record
   * (`resolveComposition(config)`). The direct Colibri runtime plans its own
   * resources from the options above and takes none. */
  readonly composition?: Composition;
}

export interface Glm52RuntimeOpenOptions {
  /** Whole-process ceiling. Defaults to the smaller of the validated 25 GiB
   * preset and physical RAM. */
  memoryBudgetBytes?: number;
  /** Context reserved by the exact GLM resource equation. */
  contextTokens?: number;
  /** Generated-token allowance within contextTokens. */
  maxGenerationTokens?: number;
  /** Maximum ordinary continuous-batch rows. */
  batchSize?: number;
  /** Native checkpoint MTP row; on by default. */
  enableMtp?: boolean;
  mtpDraftTokens?: number;
  /** Test/diagnostic override; normal callers use physical RAM. */
  machineBytes?: number;
  /** Native expert-I/O dylib override. Packaged/dev defaults are automatic. */
  libraryPath?: string;
}

/** Open the direct Colibri artifact through the validated bounded expert
 * runtime. The header-only plan runs before any resident tensor or expert slab
 * is opened, so an impossible configuration fails without committing memory. */
export async function openGlm52RuntimeModel(
  modelDir: string,
  options: Glm52RuntimeOpenOptions = {},
): Promise<{ model: Glm52Model; plan: Glm52MemoryPlan }> {
  const machineBytes = options.machineBytes ?? totalmem();
  const processLimitBytes = options.memoryBudgetBytes ??
    Math.min(GLM52_G5_DEFAULT_PROCESS_LIMIT_BYTES, machineBytes);
  const plan = await planGlm52MemoryForArtifact(modelDir, {
    machineBytes,
    processLimitBytes,
    contextTokens: options.contextTokens ?? GLM52_G5_DEFAULT_CONTEXT_TOKENS,
    maxGenerationTokens:
      options.maxGenerationTokens ?? GLM52_G5_DEFAULT_MAX_GENERATION_TOKENS,
    batchSize: options.batchSize ?? 1,
    enableMtp: options.enableMtp !== false,
    mtpDraftTokens: options.mtpDraftTokens,
  });
  const model = await Glm52Model.openStreamed(modelDir, {
    budgetBytes: plan.processLimitBytes,
    reserveBytes: plan.runtimeReserveBytes,
    workingSlots: plan.mainWorkingSlots,
    maxSlotsPerLayer: 1,
    workers: 2,
    libraryPath: options.libraryPath,
    decodeKernel: "metal",
    enableMtp: plan.enableMtp,
    mtpDraftTokens: plan.mtpDraftTokens,
  });
  if (model.expertRuntime?.plan.plannedBytes !== plan.plannedProcessBytes) {
    const actual = model.expertRuntime?.plan.plannedBytes ?? 0;
    model.dispose();
    throw new Error(
      `GLM runtime plan ${actual} != preflight resource equation ` +
      `${plan.plannedProcessBytes}`,
    );
  }
  return { model, plan };
}

/** Open a Whisper checkpoint (encoder-decoder speech model) through the same
 * profile resolution as text models. Whisper never enters the chat loop, so
 * it is not a RuntimeModel: the transcription engine
 * (packages/inference/src/transcription/whisper/)
 * owns it. */
export async function openWhisperModel(
  modelDir: string, options: { readonly profiles?: ResolveModelProfileOptions } = {},
): Promise<{ model: WhisperModel; profile: ResolvedModelProfile; config: ModelConfig }> {
  const config = await loadModelConfig(modelDir);
  const profile = resolveModelProfile(config, options.profiles);
  if (profile.profile.execution.graph !== "whisper")
    throw new Error(`${modelDir} is not a Whisper checkpoint (profile ${profile.profile.id})`);
  const weights = await Weights.open(modelDir);
  try {
    return { model: new WhisperModel(weights, config), profile, config };
  } catch (error) {
    weights.dispose();
    throw error;
  }
}

/**
 * Artifact-aware construction. GLM-5.2's Colibri snapshot has no ordinary
 * model.safetensors.index.json, so it must bypass Weights.open and use its
 * dedicated header catalog/tensor source.
 */
export function openModel<Model>(
  modelDir: string,
  options: ModelOpenOptions<Model> & { readonly implementations: ModelImplementationProvider<Weights, Model> },
): Promise<Model>;
export function openModel(modelDir: string, options?: ModelOpenOptions): Promise<RuntimeModel>;
export async function openModel<Model>(
  modelDir: string,
  options: ModelOpenOptions<Model> = {},
): Promise<Model | RuntimeModel> {
  const config = await loadModelConfig(modelDir);
  const profile = resolveModelProfile(config, options.profiles);
  if (profile.profile.execution.loader === "colibri") {
    if (options.implementations || profile.profile.execution.implementation !== undefined)
      throw new Error("custom Colibri implementations require a streamed loader binding; refusing to fall back");
    return (await openGlm52RuntimeModel(modelDir, options)).model;
  }
  if (profile.profile.execution.loop === "encoder-decoder")
    throw new Error(
      `${modelDir} is a speech model (${profile.profile.id}); open it with openWhisperModel ` +
      "or serve it as a transcription model — it has no chat/completion loop",
    );
  // Resolve before opening weights; missing or incompatible code allocates nothing.
  const implementation = (options.implementations ?? MLX_MODEL_IMPLEMENTATIONS).select(config, profile);
  const composition = options.composition ?? resolveComposition(config);
  const weights = await Weights.open(modelDir);
  try { return composed(implementation.create(weights, config, profile, composition), composition); }
  catch (error) { weights.dispose(); throw error; }
}

/** Construct the binding named by a validated profile for `composition`, which
 * the implementation receives (absent: the default record,
 * `resolveComposition(config)`). A supplied registry can return any
 * backend-owned interface; the compatibility default returns the legacy model
 * union. The caller retains ownership of the supplied weights. */
export function createModel<Model>(
  weights: Weights, config: ModelConfig, composition: Composition | undefined,
  resolved: ResolvedModelProfile | undefined, implementations: ModelImplementationProvider<Weights, Model>,
): Model;
export function createModel(
  weights: Weights, config: ModelConfig, composition?: Composition, resolved?: ResolvedModelProfile,
  implementations?: ModelImplementationProvider<Weights, RuntimeModel>,
): RuntimeModel;
export function createModel<Model>(
  weights: Weights,
  config: ModelConfig,
  composition: Composition = resolveComposition(config),
  resolved: ResolvedModelProfile = resolveModelProfile(config),
  implementations: ModelImplementationProvider<Weights, Model> | ModelImplementationProvider<Weights, RuntimeModel> = MLX_MODEL_IMPLEMENTATIONS,
): Model | RuntimeModel {
  assertResolvedModelProfile(config, resolved);
  if (resolved.profile.execution.loader === "colibri")
    throw new Error(
      `${resolved.profile.id} uses the direct Colibri container; construct it with ` +
      "openModel(modelDir) or Glm52Model.open(modelDir)",
    );
  return composed(implementations.select(config, resolved).create(weights, config, resolved, composition), composition);
}

/** `model`, once its caches can serve `composition` (`assertComposedCaches`);
 * a refused model is released. */
function composed<Model>(model: Model, composition: Composition): Model {
  try { assertComposedCaches(model as object, composition); }
  catch (error) { (model as { dispose?(): void }).dispose?.(); throw error; }
  return model;
}
