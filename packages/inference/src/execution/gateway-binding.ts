import { pagedDecodeCompiles } from "../state/paged/cache";
import { speculativePrefixNamespace, captureSpeculativeOptions } from "../generation/speculative/cache-identity";
import { createOrdinaryContinuationRequest } from "./continuation-request";
import type { ExecutionContext } from "../contracts/portable/scheduling";
import type { ModelConfig } from "../artifacts/config";
import type { MlxTokenGraph } from "../models/graph";
import { declaredGraph } from "../models/capabilities";
import { BF16_KV, COMPOSITION_DEFAULTS, type KvStorageScheme } from "../contracts/portable/composition";
import { runtimeConfig, type RuntimeConfig } from "../runtime/config";
import type { CacheServiceGraph } from "../state/cache-service";
import { disposeResources } from "../runtime/resources";
import { MlxBatchExecutionGroup } from "./batch-group";
import { type MlxBatchExecutionGroupOptions, type MlxGroupMethodRequest } from "./batch-types";
import type { DraftProvider } from "../generation/speculative/source";
import { constraintDraftProvider } from "../generation/speculative/sources/ngram-source";
import { bindSpeculativeGroupRequests } from "./speculative-group";
import { bindGrammarGroupRequests } from "./grammar-group";
import { bindFillGroupRequests } from "./fill-group";
import { bindDenoisingGroupRequests } from "./denoising-group";
import { bindLegacyDenoisingModel } from "../generation/bindings/denoising";
import type { GenerateOptions } from "../generation/index";
import type { ExecutionRequirements, ResolvedExecution } from "../contracts/portable/execution";
import type { GraphCapabilities } from "../contracts/portable/graph";
import { resolveExecution } from "./plan";
import type { MlxPromptInput, Vision } from "../contracts/mlx/media";

export interface MlxBatchGroup extends Pick<MlxBatchExecutionGroup,
  "activeRows" | "pendingRows" | "submit" | "kick" | "close"> {
  runPreparation?<T>(work: () => Promise<T>, signal?: AbortSignal): Promise<T>;
}

/** Planning consumes the graph's declared capabilities (`declaredGraph`), the
 * request, and dynamic state; it never inspects a model's class, type, or name,
 * nor how the composition's caches store keys and values: every cache lego
 * serves every method. Request state, prefix reuse and persistence are the
 * cache service's (`state/cache-service`), handed to `createBatchGroup`. */
export interface MlxGatewayBinding {
  mediaInput?(input: Vision): MlxPromptInput;
  /** A replayable ordinary request when the plan qualified one: its sampler
   * and the key the cache service checkpoints it under every `interval` tokens
   * (the service's `checkpointEveryTokens`). */
  continuationRequest?(execution: ResolvedExecution | undefined, options: GenerateOptions, prompt: number[],
    onToken: Parameters<typeof createOrdinaryContinuationRequest>[0]["onToken"],
    interval: number | undefined): ReturnType<typeof createOrdinaryContinuationRequest> | undefined;
  readonly config: ModelConfig;
  readonly runtime: RuntimeConfig;
  /** What the composition's cache service builds request state from: the
   * bound graph's caches, config and composition. */
  readonly graph: CacheServiceGraph;
  /** What the bound graph declares, as resolved when it was bound. */
  readonly capabilities: GraphCapabilities;
  /** `scheduling.checkpoints`: the cache service keeps generation checkpoints. */
  plan(request: ExecutionRequirements, options: GenerateOptions, scheduling: { checkpoints: boolean }): ResolvedExecution;
  bindAdapterContext?(adapters: string[], key: string): ExecutionContext;
  /** Whether a request's prefix may be reused and its state kept (the batch
   * request's `reusesPrefix`): media token IDs alone do not identify prepared
   * embeddings, so an unidentified media prefix takes fresh state. */
  reusesPrefix(execution: ResolvedExecution | undefined): boolean;
  /** The prefix namespace of a request whose own namespace is `base` (its
   * adapters, a prepared media prefix): its method's over it. The cache service
   * adds the composition's paging. Null: the method keeps no prefixes. */
  prefixNamespace?(execution: ResolvedExecution | undefined, options: GenerateOptions, base: string): string | null;
  methodRequest?(execution: ResolvedExecution | undefined, options: GenerateOptions): MlxGroupMethodRequest | undefined;
  createBatchGroup(options: MlxBatchExecutionGroupOptions): MlxBatchGroup;
}

/** `adaptiveDepth`: each round drafts up to `numDraftTokens`, chosen from the
 *  decayed acceptance rate and the graph's measured `verifyRoundCosts` (the
 *  graph must declare them); otherwise every round drafts `numDraftTokens`.
 *  The graph's composition names its KV scheme and scheduler switches (a
 *  graph without one runs the defaults): every request's, so the binding
 *  reads them once, here. */
export function bindMlxGateway(model: MlxTokenGraph,
  draft?: { provider: DraftProvider; numDraftTokens: number; adaptiveDepth?: boolean }): MlxGatewayBinding {
  const kv: KvStorageScheme = model.composition?.kv ?? BF16_KV;
  // The composition facts this binding plans with: the graph's record, else the defaults.
  const facts = model.composition ?? COMPOSITION_DEFAULTS;
  const runtime = runtimeConfig();
  // The declarations are resolved once, when the graph is bound; planning reads
  // only these and the request.
  const graph = declaredGraph(model);
  const caps = graph.graphCapabilities;
  const verification = caps.speculation;
  // Denoising rows interleave through their own grouped method. Token-level
  // methods (speculation, grammar proposals, fill) never bind to this graph.
  const denoising = caps.method === "denoising" ? bindDenoisingGroupRequests(bindLegacyDenoisingModel(model)) : undefined;
  // Grouped speculation binds any provider whose taps the graph's forward
  // captures; one that cannot bind is refused by placement rather than served
  // ordinarily.
  const speculative = !denoising && draft?.provider.grouped
    ? bindSpeculativeGroupRequests(model, draft.provider, draft.numDraftTokens, draft.adaptiveDepth ?? false) : undefined;
  // Grammar jump: verified proposals where the graph's verifier qualifies them,
  // committed spans otherwise.
  const grammarProvider = caps.method === "autoregressive" && verification.grammarProposals && facts.grammarJump
    ? constraintDraftProvider() : undefined;
  const grammarProposals = grammarProvider ? bindSpeculativeGroupRequests(model, grammarProvider,
    Math.max(1, Math.trunc(runtime.number("MLX_BUN_GRAMMAR_DRAFT_TOKENS", 3)))) : undefined;
  const grammarSpans = caps.method === "autoregressive" && !verification.grammarProposals
    ? bindGrammarGroupRequests(model) : undefined;
  const adapterState = model.loraState;
  const fillRequests = !denoising ? bindFillGroupRequests(model) : undefined;
  const mediaInput = caps.media && caps.media.input !== "pixels"
    ? (input: Vision): MlxPromptInput => graph.bindMediaInput!(input) : undefined;
  return {
    mediaInput,
    config: model.config, runtime, graph: model, capabilities: caps,
    continuationRequest(execution, options, prompt, onToken, interval) {
      if (!execution?.checkpoint || execution.mechanism !== "continuous") return undefined;
      if (interval === undefined) throw new Error("qualified continuation requires bound persistence services");
      // The cache service keys checkpoints in the request state's namespace, as its prefixes are.
      return createOrdinaryContinuationRequest({ options, prompt, onToken, execution, interval });
    },
    // Media token IDs alone do not identify the prepared embeddings, so an
    // unidentified media prefix takes fresh state and reuses none.
    reusesPrefix: execution => !(execution?.method === "autoregressive" && !execution.promptCache),
    // The method's namespace: the request's base, or the speculative group's over it.
    prefixNamespace: (execution, options, base) => {
      if (execution?.method !== "speculative") return base;
      const namespace = (execution.grammarJump ? grammarProvider : draft?.provider)?.grouped?.checkpointNamespace?.();
      return namespace === undefined ? null : speculativePrefixNamespace(namespace, base, captureSpeculativeOptions(options));
    },
    methodRequest: (execution, options) => execution?.method === "denoising" ? denoising?.(options)
      : execution?.method === "speculative"
      ? (execution.grammarJump ? grammarProposals : speculative)?.(
        options.fill && !execution.fill ? { ...options, fill: undefined } : options)
      : execution?.grammarJump ? grammarSpans?.(options)
      : execution?.fill ? fillRequests?.(options) : undefined,
    ...(adapterState ? { bindAdapterContext(adapters: string[], key: string): ExecutionContext {
      // Denoising applies each row's adapters around that row's own units, so
      // its rows share one neutral group context whatever their adapters.
      if (denoising) return { key: "", enter: () => () => {} };
      const selected = [...adapters];
      return { key, enter() {
        const previous = adapterState.active;
        adapterState.active = selected;
        return () => { adapterState.active = previous; };
      } };
    } } : {}),
    plan(request, options, scheduling) {
      const sharedMethod = request.hasDraft ? speculative : grammarProposals;
      const provider = request.hasDraft ? draft?.provider : grammarProvider;
      // A graph whose verifier does not qualify adapters, or a provider that
      // cannot serve the target's adapters, serves adapter requests ordinarily
      // even when a draft is configured.
      const sharedSpeculativeAdapters = verification.adapters && !!sharedMethod &&
        provider?.grouped?.supportsTargetAdapters === true;
      // Such a drafted request opens no draft provider and keeps the ordinary checkpoint rules.
      const ignoredDraft = request.hasDraft && request.hasAdapters && !sharedSpeculativeAdapters;
      return resolveExecution(request, {
        checkpoints: scheduling.checkpoints,
        sharedCheckpoints: scheduling.checkpoints &&
          (!request.hasDraft || ignoredDraft) && !request.hasVision && !request.hasGrammar &&
          !request.wantsLogprobs && !options.fill,
        mediaBatch: !!mediaInput,
        mediaPrefixCache: runtime.flag("MLX_BUN_MEDIA_PREFIX_CACHE", true),
        groupedMethods: denoising ? ["denoising"] : sharedMethod ? ["autoregressive", "speculative"] : ["autoregressive"],
        sharedGrammarProposals: !!grammarProposals,
        // Committed spans take ordinary grammar rows; prepared media keeps masking.
        sharedGrammarJump: !!grammarSpans && !request.hasVision,
        sharedFill: !!fillRequests && !!options.fill,
        sharedSpeculativeEcho: verification.externalTokens && !!options.fill?.plan.echo &&
          provider?.grouped?.supportsExternalTokens === true,
        speculativeLogprobs: !!sharedMethod && verification.logprobs,
        sharedSpeculativeAdapters,
        method: caps.method,
        compiledDecode: caps.compiledDecode,
        // The batch group coordinates per-row grammar for every token method;
        // masks come from the shared sampler, so no graph qualifies or declines
        // them. Denoising samples canvases, which an AR token mask cannot apply to.
        grammarBatch: !denoising,
      }, { fill: !!options.fill, compiledDecode: (!model.composition?.pagedKv || pagedDecodeCompiles(model.composition.pagedKv, model.composition.kv)), grammarJump: facts.grammarJump });
    },
    createBatchGroup: (options) => new MlxBatchExecutionGroup(model, options),
  };
}
