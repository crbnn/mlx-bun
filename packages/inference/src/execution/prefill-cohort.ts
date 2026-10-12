import type { Cache } from "../contracts/mlx/cache";
import type { MlxArray } from "@mlx-bun/mlx/array";
import { nextPrefillStep, prefillPhase } from "../contracts/portable/prefill";
import { disposeResources, cleanupFailure } from "../runtime/resources";
import type { Row } from "./batch-types";
import type { KvMaintenance } from "../contracts/mlx/cache";
import type { ContinuationState, RequestStateCache } from "../state/cache-service";
import { MlxPrefillRows, type MlxPrefillForward, type MlxPrefillState } from "./prefill-rows";
import { widthClass } from "../contracts/mlx/graph";
import { promptWorkStep } from "../generation/phase-dispatch";

export interface PrefillState extends MlxPrefillState {
  snapAt: number | null;
  continuation?: ContinuationState;
  closePrefill?: () => void;
}

interface PrefillHost {
  chunkSize: number;
  tailSplit: boolean;
  /** Each row's state comes from the cache service, and boundary snapshots return to it. */
  cache: RequestStateCache;
  maintain?: KvMaintenance;
  /** One cohort forward of `ids` [sequences, tokens] in the phase its planner named. */
  forward(ids: MlxArray, caches: Cache[], step: MlxPrefillForward): Promise<MlxArray>;
  /** Target projection preserves the forward batch geometry before sampling. */
  project(hidden: MlxArray, caches: Cache[], completed: readonly PrefillState[], step: MlxPrefillForward): MlxArray;
  /** Borrows [1,1,V] logits; takes ownership of the completed request caches. */
  complete(state: PrefillState, logits: MlxArray): Promise<void>;
  resume?(state: PrefillState): Promise<void>;
  reject(row: Row, error: unknown): void;
}

/** Ordinary sampling/cache policy over the shared target preparation driver. */
export class MlxPrefillCohort extends MlxPrefillRows<PrefillState> {
  override get canAdmit(): boolean {
    return super.canAdmit && this.supportsMixedWork;
  }
  get supportsMixedWork(): boolean { return !this.rows.some(row => row.req.promptInput); }
  constructor(readonly host: PrefillHost) {
    super({
      codecs: host.cache.codecs, maintain: host.maintain,
      open(row) {
        let owned: { caches: Cache[]; retain?: () => void } | undefined;
        let closeCache: (() => void) | undefined;
        let closePrefill: (() => void) | undefined;
        try {
          closeCache = row.req.trace?.begin("cache.lookup_restore", { mechanism: "continuous" });
          // The row's resumed generation, a reused prefix composed again as the
          // graph's caches, or fresh state: each converts inside its own appends.
          const acquired = owned = host.cache.acquire(row.req, row.cacheNamespace ?? "");
          if (acquired.continuation)
            return { row, solo: acquired.caches, pos: row.req.promptIds.length,
              snapAt: null, continuation: acquired.continuation };
          closeCache?.(); closeCache = undefined;
          row.cachedTokens = acquired.reusedTokens;
          const length = row.req.promptIds.length;
          const boundary = Math.min(row.req.snapshotAt ?? length, length - 1);
          const snapAt = host.cache.keeps(row.req) && boundary >= 256 && boundary > acquired.reusedTokens ? boundary : null;
          closePrefill = row.req.trace?.begin("prefill.total", {
            mechanism: "continuous", promptTokens: row.promptTokens, cachedTokens: row.cachedTokens,
          });
          return { row, solo: acquired.caches, pos: acquired.reusedTokens, retain: acquired.retain, snapAt, closePrefill };
        } catch (error) {
          return cleanupFailure(error, () => disposeResources([...(owned?.caches ?? []),
            { dispose: () => owned?.retain?.() }, { dispose: () => closePrefill?.() }]));
        } finally { closeCache?.(); }
      },
      ready: state => !!state.continuation,
      plan(state) {
        if (state.row.req.promptInput) return { kind: "final", start: state.pos,
          end: state.row.promptTokens, snapshot: false, batchYield: false, atomic: true,
          phase: prefillPhase(state.pos, state.row.promptTokens, false) };
        return nextPrefillStep({ length: state.row.promptTokens, position: state.pos,
        chunkSize: state.row.req.prefillChunkSize ?? host.chunkSize,
        tailSplit: host.tailSplit, snapshotAt: state.snapAt }); },
      forward: (ids, caches, states, step, work) => states[0]?.row.req.promptInput
        ? Promise.resolve(states[0].row.req.promptInput.prefill(ids, caches, states[0].pos,
          step.phase === "decode" ? "decode" : "tail", widthClass(step.tokens)))
        : work ? work(ids, caches, promptWorkStep(step, null, () => host.forward(ids, caches, step))) : host.forward(ids, caches, step),
      // A media prompt input returns its prompt's last position: one row per sequence to project.
      project: (hidden, caches, completed, step) => host.project(hidden, caches, completed,
        completed.some(state => state.row.req.promptInput) ? { ...step, tokens: 1 } : step),
      complete: (state, logits) => state.continuation ? host.resume!(state) : host.complete(state, logits!), reject: host.reject.bind(host),
      checkpoint(state, capture) {
        let snapshot: Cache[] | undefined;
        try {
          snapshot = capture();
          host.cache.publish(state.row.req, state.row.cacheNamespace ?? "", state.row.req.promptIds.slice(0, state.pos), { caches: snapshot });
          snapshot = undefined;
        } catch (error) {
          if (snapshot) disposeResources(snapshot);
          console.warn(`batch-lane boundary snapshot skipped: ${(error as Error).message}`);
        }
        state.snapAt = null;
      },
      close(state) { const close = state.closePrefill; state.closePrefill = undefined; close?.(); },
    });
  }
}
