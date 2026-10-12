// The composition's cache service: one per loaded model, constructed with its
// budget. It owns the state every request runs on: fresh caches (paged when the
// composition pages), prefix reuse from RAM and saved state, SSD offload,
// eviction, the KV budget requests are admitted under, and the generation
// checkpoints of replayable requests. The scheduler asks for a request's state
// (`acquire`: the caches plus the prompt tokens they already cover, 0 for a
// miss), offers back state it produced (`publish`, `checkpoint`) and ends the
// request (`release`). It never learns what is resident, what was evicted, or
// where (RAM, SSD); what the service holds is in its own `report`.

import { join } from "node:path";
import type { Cache } from "../contracts/mlx/cache";
import type { CheckpointAttachment } from "../contracts/mlx/checkpoint";
import { BF16_KV, type KvStorageScheme } from "../contracts/portable/composition";
import type { ObjectCache } from "../contracts/portable/object-cache";
import type { ModelConfig } from "../artifacts/config";
import { runtimeConfig, withRuntimeConfig, type RuntimeConfig } from "../runtime/config";
import { disposeResources } from "../runtime/resources";
import { ContinuationPersistence, type ContinuationAttempt, type ContinuationState,
  type ContinuationStore } from "./continuation-persistence";
import type { DurabilityFlushResult, DurabilitySnapshotStats } from "./durability";
import { batchRowKvBytes } from "./kv-budget";
import { kvSchemeOf, type KvScheme } from "./kv-scheme";
import { rowAdmission } from "./layout";
import { cloneKvCaches, composeAtLayer, legacyCacheCodecs } from "./persistence";
import type { CacheCodecProvider } from "./persistence-types";
import { PromptCache, type ColdTier } from "./prefix-cache";
import { maybePageKv, pagedPrefixNamespace, type PagedKv } from "./request-policy";
import type { RetentionPolicy } from "./retention-policy";
import { SsdCacheStore, type SsdIndexEntry, type SsdStoreOptions } from "./ssd-cache";
import { TieredPromptCache } from "./tiered-prefix-cache";

export { type ContinuationState } from "./continuation-persistence";

/** The loaded graph the service builds request state for: its fresh caches,
 * the config its KV is projected from, and the composition's KV scheme and
 * paging. */
export interface CacheServiceGraph {
  makeCache(): Cache[];
  readonly config: ModelConfig;
  readonly composition?: { readonly kv: KvStorageScheme; readonly pagedKv: PagedKv | null } | null;
}

/** What a replayable generation gives the service to checkpoint it. */
export interface ContinuationRequest {
  /** The checkpoint key of this generation in the state namespace `namespace`
   * of a store whose identity is `identity`. */
  key(namespace: string, identity: unknown): string;
  /** The sampler seed the generation starts from, and whether the client chose it. */
  readonly seed: number;
  readonly seedWasExplicit: boolean;
}

/** One request as the cache sees it: fields of the request the scheduler admits.
 * The request object identifies it from `admit` to `release`. */
export interface StateRequest {
  readonly promptIds: number[];
  /** Tokens the request may generate: with the prompt, its KV projection and page capacity. */
  readonly maxTokens: number;
  readonly cacheSessionId?: string;
  /** False for a request whose prefix has no identity (prepared media without
   * one): it takes fresh state, reuses none, and its state is not kept. */
  readonly reusesPrefix?: boolean;
  /** A replayable generation: the service checkpoints it and resumes it. */
  readonly continuation?: ContinuationRequest;
}

/** A request's state, owned by the caller. */
export interface AcquiredState {
  /** Restored, a reused prefix (composed as the graph's caches), or fresh. */
  readonly caches: Cache[];
  /** Prompt tokens `caches` already cover; 0 is a miss. */
  readonly reusedTokens: number;
  /** Method state stored with a reused prefix. */
  readonly attachments?: CheckpointAttachment[];
  /** Releases the backing of reused state; run it after disposing `caches`. */
  readonly retain?: () => void;
  /** A resumed generation: `caches` cover its prompt and generated tokens. */
  readonly continuation?: ContinuationState;
}

/** State a request offers back for reuse, covering the tokens it is published with. */
export interface PublishedState {
  readonly caches: Cache[];
  readonly attachments?: CheckpointAttachment[];
  readonly retain?: () => void;
}

/**
 * The scheduler's port to the composition's cache service. `acquire` and
 * `release` bracket a request; between them the scheduler offers state it
 * produced (`publish` at prefix boundaries and when a row finishes,
 * `checkpoint` for a replayable generation) and marks execution boundaries.
 * Admission under the KV budget runs before `acquire`. Nothing here says what
 * is resident or where.
 */
export interface RequestStateCache {
  /** The snapshot form of the composition's caches: how the scheduler copies
   * live serial state it publishes. */
  readonly codecs: CacheCodecProvider;
  /** Admit `request` beside the requests already holding state: true admits it
   * (it holds its projected KV bytes until `release`), false means it waits.
   * Throws for a request that can never be admitted: one past the caches' own
   * position limit, or one over the KV budget alone. */
  admit(request: StateRequest): boolean;
  /** Whether `admit` would admit `request` now; holds nothing. */
  fits(request: StateRequest): boolean;
  /** `request`'s state: its resumed generation, else the longest reusable prefix
   * of its prompt in `namespace` (its adapters, prepared media, method), else
   * fresh state. Ownership passes to the caller. */
  acquire(request: StateRequest, namespace: string): AcquiredState;
  /** Whether state `request` publishes may be kept: the scheduler skips
   * copying state the cache would drop. */
  keeps(request: StateRequest): boolean;
  /** Offer state covering `tokens` for reuse in `namespace`; the cache keeps or
   * drops it by its own policy. Ownership transfers on return; on a throw it
   * stays with the caller. */
  publish(request: StateRequest, namespace: string, tokens: number[], state: PublishedState): void;
  /** Save a resumable point of `request`'s generation (owned caches; the cache
   * disposes them whatever happens). */
  checkpoint(request: StateRequest, state: Omit<ContinuationState, "seed">): void;
  /** `request` is done: its KV bytes return to the budget; a completed
   * generation's checkpoints are removed, an interrupted one keeps them. */
  release(request: StateRequest, completed: boolean): void;
  /** An execution boundary passed: the cache may give optional state back to
   * the allocator. Never changes a running request. */
  boundary(): void;
}

/** What the service holds, as it reports itself (`/stats`, telemetry, residency). */
export interface KvCacheReport {
  /** Reusable prefixes held in RAM. */
  readonly prefixes: {
    readonly entries: number; readonly bytes: number; readonly maxBytes: number;
    readonly hits: number; readonly misses: number; readonly sessionHits: number; readonly sessionMisses: number;
    readonly prefixScans: number; readonly objectHits: number; readonly objectMisses: number;
    readonly objectRestores: number; readonly demotions: number;
  };
  /** Saved state on disk; null without it. */
  readonly saved: {
    readonly entries: number; readonly bytes: number; readonly maxBytes: number;
    readonly restores: number; readonly spills: number; readonly restoreMsLast: number;
    readonly longestDurablePrefixTokens: number;
  } | null;
  /** Writes not yet durable. */
  readonly pending: DurabilitySnapshotStats;
  /** Projected KV bytes of the admitted requests, and the budget they are admitted under. */
  readonly kv: { readonly projectedBytes: number; readonly budgetBytes: number | null };
}

/** The parts a service composes. `createKvCacheService` builds them from the
 * serve flags; CPU tests and replacement runtimes may supply their own. */
export interface KvCacheParts {
  /** Reusable prefixes, tiered over saved state when there is any. Null: the
   * service reuses nothing and keeps nothing. */
  readonly prefixes?: PromptCache | null;
  /** Saved state the prefixes tier over; reported, never reached by callers. */
  readonly saved?: SsdCacheStore | null;
  /** Generation checkpoints, written every `everyTokens` generated tokens. */
  readonly continuations?: { readonly store: ContinuationStore; readonly persistence: ContinuationPersistence;
    readonly everyTokens: number } | null;
}

export interface KvCacheSettings {
  /** Aggregate projected KV bytes the admitted requests may hold (`--kv-budget`); absent: unlimited. */
  readonly kvBudgetBytes?: number;
  /** The snapshot form of the composition's caches (default the built-in codecs). */
  readonly codecs?: CacheCodecProvider;
  /** Artifact, implementation, state ABI and codec identity captured at load: part of every checkpoint key. */
  readonly identity?: unknown;
  /** The scope paged state is built in. */
  readonly runtime?: RuntimeConfig;
  /** Demote RAM entries idle this long to saved state (milliseconds), on `schedule`'s timer. */
  readonly idleDemotion?: { readonly afterMs: number; schedule(run: () => void, intervalMs: number): () => void };
  /** Runs once after `close` cleared the state (the saved-state budget's release). */
  readonly closed?: () => void;
}

interface Lease {
  /** Admitted under the KV budget: the request holds its projected bytes until `release`. */
  admitted: boolean;
  /** Its projected KV bytes, once something asked. */
  bytes?: number;
  continuation?: { key: string; namespace: string; attempt: ContinuationAttempt; seed: number };
}

const NO_PENDING: DurabilitySnapshotStats = Object.freeze({ pendingSnapshots: 0, pendingSpills: 0,
  pendingSpillBytes: 0, droppedSpills: 0, failedSpills: 0 });

/** The composition's cache service (see the file header). */
export class KvCacheService implements RequestStateCache {
  readonly codecs: CacheCodecProvider;
  readonly #graph: CacheServiceGraph;
  readonly #kv: KvStorageScheme;
  readonly #paged: PagedKv | null;
  readonly #direct: boolean;
  readonly #runtime: RuntimeConfig;
  readonly #prefixes: PromptCache | null;
  readonly #saved: SsdCacheStore | null;
  readonly #continuations: NonNullable<KvCacheParts["continuations"]> | null;
  readonly #identity: unknown;
  readonly #kvBudget: number | undefined;
  readonly #scheme: KvScheme;
  #rowAdmission: ReturnType<typeof rowAdmission> | undefined;
  readonly #leases = new Map<StateRequest, Lease>();
  readonly #closed: (() => void) | undefined;
  #stopIdle: (() => void) | undefined;
  #closing: Promise<DurabilityFlushResult> | undefined;

  constructor(graph: CacheServiceGraph, parts: KvCacheParts = {}, settings: KvCacheSettings = {}) {
    this.#graph = graph;
    this.#kv = graph.composition?.kv ?? BF16_KV;
    this.#paged = graph.composition?.pagedKv ?? null;
    this.#runtime = settings.runtime ?? runtimeConfig();
    this.#direct = this.#paged ? this.#runtime.flag("MLX_BUN_PAGED_ATTN", false) : false;
    this.#prefixes = parts.prefixes ?? null;
    this.#saved = parts.saved ?? null;
    this.#continuations = parts.continuations ?? null;
    this.codecs = settings.codecs ?? legacyCacheCodecs;
    this.#identity = settings.identity;
    this.#kvBudget = settings.kvBudgetBytes;
    this.#scheme = kvSchemeOf(this.#kv);
    this.#closed = settings.closed;
    const idle = settings.idleDemotion, prefixes = this.#prefixes;
    if (idle && prefixes && idle.afterMs > 0)
      this.#stopIdle = idle.schedule(() => prefixes.demoteIdle(idle.afterMs), Math.max(30_000, idle.afterMs / 4));
  }

  /** Generation checkpoints every this many tokens; undefined: the service keeps none. */
  get checkpointEveryTokens(): number | undefined { return this.#continuations?.everyTokens; }

  /** The state namespace of a request whose own namespace is `namespace`: the composition's paging over it. */
  #namespace(namespace: string): string {
    return this.#paged ? pagedPrefixNamespace(this.#kv, this.#paged, namespace, this.#direct) : namespace;
  }

  /** Storage that projects its own rows (a layout such as GLM-5.2's compressed
   * state) bills a request by its projection; the rest by the KV scheme. Read
   * from the graph's fresh caches at the first admission. */
  get #admission(): ReturnType<typeof rowAdmission> {
    if (this.#rowAdmission === undefined) {
      const proto = this.#graph.makeCache(); // fresh caches hold no buffers
      try { this.#rowAdmission = rowAdmission(proto); } finally { disposeResources(proto); }
    }
    return this.#rowAdmission;
  }

  #rowBytes(request: StateRequest): number {
    const admission = this.#admission;
    return admission
      ? admission.bytes(request.promptIds.length + request.maxTokens)
      : batchRowKvBytes(this.#graph.config, request.promptIds.length, request.maxTokens, this.#scheme);
  }

  /** Projected KV of every admitted request (its full prompt and completion). */
  get #projectedBytes(): number {
    let total = 0;
    for (const [request, lease] of this.#leases) if (lease.admitted) total += lease.bytes ??= this.#rowBytes(request);
    return total;
  }

  admit(request: StateRequest): boolean {
    const requested = request.promptIds.length + request.maxTokens;
    const limit = this.#admission?.maxTokens ?? Infinity;
    if (requested > limit)
      throw new RangeError(`context limit: prompt ${request.promptIds.length} + max_tokens ${request.maxTokens} exceeds ${limit}`);
    let need: number | undefined;
    if (this.#kvBudget !== undefined) {
      need = this.#rowBytes(request);
      const alone = ![...this.#leases.values()].some(lease => lease.admitted);
      if (need > this.#kvBudget && alone)
        throw new Error(`kv budget: request needs ~${(need / 1e9).toFixed(2)} GB KV ` +
          `(prompt ${request.promptIds.length} + max_tokens ${request.maxTokens}), ` +
          `over --kv-budget ${(this.#kvBudget / 1e9).toFixed(2)} GB — lower max_tokens or raise the budget`);
      if (this.#projectedBytes + need > this.#kvBudget) return false;
    }
    const lease = this.#lease(request);
    lease.admitted = true; lease.bytes = need;
    return true;
  }

  fits(request: StateRequest): boolean {
    return this.#kvBudget === undefined || this.#projectedBytes + this.#rowBytes(request) <= this.#kvBudget;
  }

  #lease(request: StateRequest): Lease {
    let lease = this.#leases.get(request);
    if (!lease) this.#leases.set(request, lease = { admitted: false });
    return lease;
  }

  keeps(request: StateRequest): boolean {
    return this.#prefixes !== null && request.reusesPrefix !== false;
  }

  acquire(request: StateRequest, namespace: string): AcquiredState {
    const lease = this.#lease(request);
    const stateNamespace = this.#namespace(namespace);
    if (request.continuation) {
      const resumed = this.#resume(request, request.continuation, lease, stateNamespace);
      if (resumed) return { caches: resumed.caches, reusedTokens: request.promptIds.length, continuation: resumed };
    }
    if (this.keeps(request)) {
      const hit = this.#prefixes!.take(request.promptIds, stateNamespace, request.cacheSessionId);
      if (hit) return { caches: hit.caches, reusedTokens: hit.tokens.length,
        ...(hit.attachments ? { attachments: hit.attachments } : {}), ...(hit.retain ? { retain: hit.retain } : {}) };
    }
    return { caches: this.#fresh(request), reusedTokens: 0 };
  }

  /** Fresh state of the composition: the graph's caches, paged when it pages. */
  #fresh(request: StateRequest): Cache[] {
    const paged = this.#paged;
    if (!paged) return this.#graph.makeCache();
    return withRuntimeConfig(this.#runtime, () => {
      const caches = this.#graph.makeCache();
      try { maybePageKv(caches, this.#kv, paged, request.promptIds.length + request.maxTokens); return caches; }
      catch (error) { disposeResources(caches); throw error; }
    });
  }

  /** The generation checkpoint `request` resumes from, or null. Opens the
   * request's checkpoint writes either way. */
  #resume(request: StateRequest, continuation: ContinuationRequest, lease: Lease, namespace: string): ContinuationState | null {
    const continuations = this.#continuations;
    if (!continuations) throw new Error("qualified continuation requires bound persistence services");
    const key = continuation.key(namespace, this.#identity);
    const run = lease.continuation = { key, namespace, attempt: continuations.persistence.begin(key), seed: continuation.seed };
    if (!continuations.persistence.canRestore(key)) return null;
    const prompt = request.promptIds;
    const entry = continuations.store.findGenerationCheckpoint(prompt, key, namespace);
    // A restored entry takes the caches the graph composes.
    const restored = entry ? continuations.store.restore(entry, this.#graph) : null;
    if (!restored) return null;
    const metadata = restored.header.generationCheckpoint;
    if (!metadata || metadata.originalPromptTokens !== prompt.length ||
        metadata.generatedTokens !== restored.tokens.length - prompt.length) {
      disposeResources(restored.caches);
      throw new Error("generation checkpoint has inconsistent continuation metadata");
    }
    run.seed = metadata.seed;
    return { caches: restored.caches, cacheTokens: restored.tokens,
      generatedTokens: metadata.generatedTokens, pendingToken: metadata.pendingToken, seed: run.seed };
  }

  publish(request: StateRequest, namespace: string, tokens: number[], state: PublishedState): void {
    if (!this.keeps(request)) throw new Error("this cache keeps no state for the request");
    this.#prefixes!.put(tokens, state.caches, this.#namespace(namespace), state.retain, state.attachments, request.cacheSessionId);
  }

  checkpoint(request: StateRequest, state: Omit<ContinuationState, "seed">): void {
    const run = this.#leases.get(request)?.continuation;
    if (!run || !this.#continuations) {
      disposeResources(state.caches);
      throw new Error("owned continuation requires persistence coordinator");
    }
    this.#continuations.persistence.enqueue(run.attempt, state, { key: run.key, cacheNs: run.namespace,
      originalPromptTokens: request.promptIds.length, generatedTokens: state.generatedTokens,
      pendingToken: state.pendingToken, seed: run.seed, seedWasExplicit: request.continuation?.seedWasExplicit === true });
  }

  release(request: StateRequest, completed: boolean): void {
    const lease = this.#leases.get(request);
    if (!lease) return;
    this.#leases.delete(request);
    const run = lease.continuation, persistence = this.#continuations?.persistence;
    if (run && persistence) {
      if (completed) persistence.complete(run.attempt);
      else persistence.release(run.attempt);
    }
  }

  boundary(): void { this.#prefixes?.reclaim(); }

  /** Start reading `promptIds`' saved prefix in `namespace` before the request
   * is admitted. The returned release ends the request's interest. */
  async prefetch(promptIds: number[], namespace: string, sessionId?: string): Promise<() => void> {
    return await this.#prefixes?.prefetch(promptIds, this.#namespace(namespace), sessionId) ?? (() => {});
  }

  /** The longest prefix of `promptIds` held in RAM under `namespace`, which is
   * looked up as given (not under the composition's paging). */
  peekPrefixLen(promptIds: number[], namespace: string): number {
    return this.#prefixes?.peekPrefixLen(promptIds, namespace) ?? 0;
  }

  /** Exact tensor objects (encoder outputs) under the same budget and storage; absent without a prefix store. */
  get objects(): ObjectCache<CheckpointAttachment[]> | undefined { return this.#prefixes?.objects; }

  /** Drop a session's affinity without deleting reusable state. */
  closeSession(sessionId: string): void { this.#prefixes?.closeSession(sessionId); }

  /** Observes every prefix the cache stores, with its tokens. */
  set onPut(observe: ((tokens: number[], namespace: string) => void) | null) {
    if (this.#prefixes) this.#prefixes.onPut = observe;
  }

  /** The token prefixes saved state holds (header metadata only). */
  savedPrefixes(): Iterable<readonly number[]> { return this.#saved?.tokenPrefixes() ?? []; }

  report(): KvCacheReport {
    const prefixes = this.#prefixes, saved = this.#saved;
    return {
      prefixes: {
        entries: prefixes?.size ?? 0, bytes: prefixes?.totalBytes ?? 0, maxBytes: prefixes?.maxBytes ?? 0,
        hits: prefixes?.hits ?? 0, misses: prefixes?.misses ?? 0,
        sessionHits: prefixes?.sessionHits ?? 0, sessionMisses: prefixes?.sessionMisses ?? 0,
        prefixScans: prefixes?.prefixScans ?? 0, objectHits: prefixes?.objectHits ?? 0,
        objectMisses: prefixes?.objectMisses ?? 0, objectRestores: prefixes?.objectRestores ?? 0,
        demotions: prefixes?.demotions ?? 0,
      },
      saved: saved ? { entries: saved.entries, bytes: saved.totalBytes, maxBytes: saved.maxBytes,
        restores: saved.stats.restores, spills: saved.stats.spills, restoreMsLast: saved.stats.restoreMsLast,
        longestDurablePrefixTokens: saved.longestDurablePrefixTokens } : null,
      pending: this.stats(),
      kv: { projectedBytes: this.#projectedBytes, budgetBytes: this.#kvBudget ?? null },
    };
  }

  /** Pending persistence: prefix snapshots and spills, and checkpoint writes. */
  stats(): DurabilitySnapshotStats {
    const tiered = this.#prefixes instanceof TieredPromptCache ? this.#prefixes : null;
    const current = tiered?.durability.stats ?? NO_PENDING;
    const pending = this.#continuations?.persistence.stats;
    return { ...current, pendingSpills: current.pendingSpills + (pending?.pendingCount ?? 0),
      pendingSpillBytes: current.pendingSpillBytes + (pending?.pendingBytes ?? 0),
      droppedSpills: current.droppedSpills + (pending?.dropped ?? 0), failedSpills: current.failedSpills + (pending?.failed ?? 0) };
  }

  /** Make every kept prefix and checkpoint durable, as far as storage allows. */
  async flush(): Promise<DurabilityFlushResult> {
    const started = performance.now();
    const tiered = this.#prefixes instanceof TieredPromptCache ? this.#prefixes : null;
    const errors: unknown[] = [];
    let checkpointResult: Awaited<ReturnType<ContinuationPersistence["flush"]>> | undefined;
    let result: DurabilityFlushResult | undefined;
    try { checkpointResult = await this.#continuations?.persistence.flush(); } catch (error) { errors.push(error); }
    try { result = await tiered?.durability.flush(); } catch (error) { errors.push(error); }
    // Also drain after a failed coordinator flush before releasing RAM views.
    try { await tiered?.spillQueue.drain(); } catch (error) { errors.push(error); }
    if (errors.length === 1) throw errors[0];
    if (errors.length) throw new AggregateError(errors, "cache flush failed");
    const current = this.stats();
    return { ...current, durable: (result?.durable ?? current.pendingSpills === 0) &&
      (checkpointResult?.durable ?? true) && current.pendingSpills === 0,
      flushedSnapshots: result?.flushedSnapshots ?? 0, missingSnapshots: result?.missingSnapshots ?? 0,
      elapsedMs: performance.now() - started };
  }

  /** Stop idle demotion ahead of draining requests; the state stays. */
  stopIdleDemotion(): void { const stop = this.#stopIdle; this.#stopIdle = undefined; stop?.(); }

  /** After every request drained: flush, then clear the state even when
   * persistence fails (`durable: false` reports it). Idempotent. */
  close(): Promise<DurabilityFlushResult> {
    return this.#closing ??= (async () => {
      let result: DurabilityFlushResult | undefined; const errors: unknown[] = [];
      try { this.stopIdleDemotion(); } catch (error) { errors.push(error); }
      try { result = await this.flush(); } catch (error) { errors.push(error); }
      try { this.#prefixes?.clear(); } catch (error) { errors.push(error); }
      try { this.#closed?.(); } catch (error) { errors.push(error); }
      if (errors.length === 1) throw errors[0];
      if (errors.length) throw new AggregateError(errors, "cache cleanup failed");
      return result!;
    })();
  }
}

/** The saved-state root's one byte budget (`--ssd-cache-max`), shared by every
 * store under it: a live store counts against it and enforces the room it is lent. */
export interface SavedStateBudget {
  readonly maxBytes: number;
  attach(member: { readonly dir: string; bytes(): number }): { limit(): number; detach(): void };
  /** Evict what idle directories hold until the root fits. */
  reclaim(): unknown;
}

/** Saved state under a directory the caller owns. */
export interface SavedStateOptions {
  /** The saved-state root; this store's directory is `<dir>/<configFingerprint>/`. */
  readonly dir: string;
  /** Written into every file and enforced on scan and load. */
  readonly identity: Pick<SsdStoreOptions, "configFingerprint" | "modelId" | "tokenizerHash">;
  readonly budget: SavedStateBudget;
  /** Verify every tensor hash on load (`--ssd-cache-verify`). */
  readonly verify?: boolean;
  readonly storage?: SsdStoreOptions["storage"];
  /** Write each kept prefix to saved state as it is kept (default true). */
  readonly writeBehind?: boolean;
  /** Read a saved prefix ahead of admission (default true). */
  readonly prefetch?: boolean;
  /** Demote RAM entries idle this long (milliseconds) to saved state; 0 never. */
  readonly demoteIdleMs?: number;
  readonly schedule?: (run: () => void, intervalMs: number) => () => void;
  /** Generation checkpoints every this many generated tokens; absent: none. */
  readonly checkpointEveryTokens?: number;
  /** Bytes of checkpoint snapshots the writer may queue. */
  readonly checkpointQueueBytes?: number;
}

export interface KvCacheServiceOptions extends Pick<KvCacheSettings, "kvBudgetBytes" | "codecs" | "identity" | "runtime"> {
  /** RAM bytes of reusable prefixes (`--prompt-cache`), 0 keeping none; absent: no prefix store. */
  readonly promptCacheBytes?: number;
  readonly retention?: RetentionPolicy;
  /** Whether live memory is over what optional state may use, given the bytes
   * the prefix store holds or has queued for writing. */
  readonly pressure?: (heldBytes: number) => boolean;
  readonly saved?: SavedStateOptions;
}

const defaultSchedule = (run: () => void, intervalMs: number) => {
  const timer = setInterval(run, intervalMs); timer.unref(); return () => clearInterval(timer);
};

/** The service the serve flags compose for `graph`: RAM prefixes under
 * `promptCacheBytes`, tiered over saved state when `saved` names a directory,
 * with generation checkpoints when it names an interval. The caller closes it
 * after every request drained and before disposing the graph. */
export function createKvCacheService(graph: CacheServiceGraph, options: KvCacheServiceOptions = {}): KvCacheService {
  const codecs = options.codecs ?? legacyCacheCodecs;
  // A reused prefix comes back as the graph's composed caches, which convert as
  // the caches that stored it would have.
  const clone = (caches: Cache[]): Cache[] => {
    const cloned = cloneKvCaches(caches, codecs);
    const prototypes = graph.makeCache();
    try { return cloned.map((cache, layer) => composeAtLayer(prototypes[layer]!, cache)); }
    finally { disposeResources(prototypes); }
  };
  const savedOptions = options.saved;
  let saved: SsdCacheStore | null = null;
  let member: ReturnType<SavedStateBudget["attach"]> | undefined;
  let prefixes: PromptCache | null = null;
  try {
    if (savedOptions) {
      const budget = savedOptions.budget;
      const created = saved = new SsdCacheStore({ codecs, dir: savedOptions.dir, maxBytes: budget.maxBytes,
        limit: () => member?.limit() ?? budget.maxBytes, ...savedOptions.identity,
        verify: savedOptions.verify, storage: savedOptions.storage });
      member = budget.attach({ dir: join(savedOptions.dir, savedOptions.identity.configFingerprint), bytes: () => created.totalBytes });
      saved.scan();
      // A lowered budget, or another process's leftovers, takes effect at once; this store's directory is live and untouched.
      budget.reclaim();
    }
    const store = saved;
    if (options.promptCacheBytes !== undefined) {
      const cold: ColdTier | null = store ? {
        find: (tokens, ns) => { const hit = store.find(tokens, ns); return hit ? { prefixLen: hit.prefixLen, handle: hit.entry } : null; },
        findExact: (tokens, ns) => { const hit = store.findExact(tokens, ns); return hit ? { prefixLen: hit.prefixLen, handle: hit.entry } : null; },
        restoreObjectAsync: async handle => {
          const loaded = await store.restoreAsync(handle as SsdIndexEntry, { makeCache: () => [] });
          return loaded ? { ...loaded, retain() {} } : null;
        },
        restore: handle => {
          const loaded = store.restore(handle as SsdIndexEntry, graph);
          return loaded ? { ...loaded, retain() {} } : null;
        },
        ...(savedOptions?.prefetch !== false ? {
          restoreAsync: async (handle: unknown) => {
            const loaded = await store.restoreAsync(handle as SsdIndexEntry, graph);
            return loaded ? { ...loaded, retain() {} } : null;
          },
        } : {}),
        store: (tokens, caches, ns, attachments) => store.hasDurablePrefix(tokens, ns) || store.store(tokens, caches, ns, attachments),
      } : null;
      const tiered = store && cold ? new TieredPromptCache(options.promptCacheBytes, store, cold, clone,
        savedOptions?.writeBehind !== false) : null;
      const created = prefixes = tiered ?? new PromptCache(options.promptCacheBytes, null, null, clone);
      if (options.retention) created.retention = options.retention;
      const pressure = options.pressure;
      if (pressure) created.pressure = { overBudget: () =>
        pressure(Math.max(created.totalBytes, tiered?.spillQueue.pendingBytes ?? 0)) };
    }
    const every = savedOptions?.checkpointEveryTokens;
    const continuations = store && every ? { store, everyTokens: every,
      persistence: new ContinuationPersistence(store, { maxBytes: savedOptions!.checkpointQueueBytes ?? 2 * 1024 ** 3 }) } : null;
    const budget = savedOptions?.budget;
    return new KvCacheService(graph, { prefixes, saved, continuations }, {
      kvBudgetBytes: options.kvBudgetBytes, codecs, identity: options.identity, runtime: options.runtime,
      ...(store && savedOptions ? { idleDemotion: { afterMs: savedOptions.demoteIdleMs ?? 300_000,
        schedule: savedOptions.schedule ?? defaultSchedule } } : {}),
      // The directory is idle from here: its files count against the root's budget and may be evicted.
      ...(member ? { closed: () => { member!.detach(); budget!.reclaim(); } } : {}),
    });
  } catch (error) {
    const errors: unknown[] = [error];
    try { prefixes?.clear(); } catch (cleanup) { errors.push(cleanup); }
    try { member?.detach(); } catch (cleanup) { errors.push(cleanup); }
    if (errors.length > 1) throw new AggregateError(errors, "cache construction and cleanup failed");
    throw error;
  }
}
