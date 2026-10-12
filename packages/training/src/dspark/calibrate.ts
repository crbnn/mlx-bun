// STS calibration (§3.2.1) of a trained drafter: run the production speculative
// group over a prompt set, one request at a time (a batch of one), with
// confidence pruning DISABLED, so every draft position's confidence AND its real
// accept/reject outcome are visible (pruning would hide the positions it drops),
// fit per-position thresholds (sts.ts) and write them into the checkpoint's
// dspark.json (`config.sts`). Every other field of the checkpoint is preserved
// verbatim.

import type { RuntimeModel } from "@mlx-bun/inference/models";
import type { ChatTemplate, ChatMessage, LoadedTokenizer } from "@mlx-bun/inference/input";
import { bindMlxGateway } from "@mlx-bun/inference/execution";
import { DflashSource, projectedDraftGroups, targetLacks, type DraftProvider, type DraftRowGroup, type DraftSource,
  type GroupedDraftProvider, type TargetView } from "@mlx-bun/inference/generation/speculative";
import { loadDsparkDrafter, type DflashDrafter, type StsCalibration } from "@mlx-bun/inference/generation/speculative/loader";
import { fitStsThresholds, type ConfSample } from "./sts";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/** The drafter operations calibration drafts with. */
type CalibrationDrafter = Pick<DflashDrafter, "cfg" | "forwardRows" | "projectContextRows" | "forwardInfer" | "dispose">;

/** Rows that report, at each commit, every drafted position's confidence and
 *  whether the target accepted it. `confidence` is the last draft's, per row. */
function observedRows(rows: DraftRowGroup, confidence: () => readonly (readonly number[])[],
  observe: (samples: ConfSample[]) => void): DraftRowGroup {
  let drafted: number[] = [];
  return {
    get namespace() { return rows.namespace; }, get tapLayers() { return rows.tapLayers; }, get rowCount() { return rows.rowCount; },
    append: checkpoints => rows.append(checkpoints), prepareAppend: checkpoints => rows.prepareAppend(checkpoints),
    filterRows: keep => rows.filterRows(keep), capture: row => rows.capture(row), dispose: () => rows.dispose(),
    async draft(pending, depth, steps) {
      const proposals = await rows.draft(pending, depth, steps);
      drafted = proposals.map(tokens => tokens.length);
      return proposals;
    },
    commit(accepted, context) {
      const conf = confidence();
      // `drafted` positions this round verified; the first `accepted` matched.
      for (const [row, count] of drafted.entries())
        observe(Array.from({ length: count }, (_, pos) => ({ pos, conf: conf[row]![pos]!, accepted: pos < accepted[row]! })));
      drafted = [];
      return rows.commit(accepted, context);
    },
  };
}

/** A DSpark draft provider whose rows draft the FULL unpruned block and report
 *  each round's (confidence, accepted) outcomes once the target has verified it. */
export class CalibrationProvider implements DraftProvider {
  readonly id: string;
  readonly weightsBytes = 0;
  readonly grouped: GroupedDraftProvider;
  constructor(private readonly drafter: CalibrationDrafter, private readonly observe: (samples: ConfSample[]) => void, id: string) {
    this.id = id;
    const namespace = `dspark-calibration:${id}`;
    let confidence: readonly (readonly number[])[] = [];
    const groups = projectedDraftGroups(namespace, drafter.cfg.tapLayers, target => {
      if (!target.hiddenLayerTaps) throw targetLacks("hiddenLayerTaps");
      const projection = target.hiddenLayerTaps.projection;
      return {
        namespace, schema: "dflash-context-v1", layers: drafter.cfg.nLayers,
        project: hidden => drafter.projectContextRows(hidden),
        draft(context, pending, _positions, depth) {
          // No thresholds and no minimum confidence: every position is drafted
          // and scored, whatever the checkpoint carries.
          const block = drafter.forwardRows(projection, null, pending, depth,
            { collectLogits: false, collectConfidence: true }, context);
          confidence = block.conf;
          return block.tokens;
        },
      };
    });
    this.grouped = { ...groups, open: options => observedRows(groups.open(options), () => confidence, observe) };
  }

  open({ target }: { target: TargetView }): DraftSource {
    let confidence: number[] = [];
    const drafter = this.drafter;
    const unpruned: Pick<DflashDrafter, "cfg" | "forwardInfer"> = {
      cfg: { ...drafter.cfg, sts: undefined },
      forwardInfer(model, hCtx, anchor, gamma, opts) {
        const block = drafter.forwardInfer(model, hCtx, anchor, gamma, { ...opts, thresholds: [], minConf: undefined });
        confidence = block.conf;
        return block;
      },
    };
    const inner = new DflashSource(unpruned, target);
    const observe = this.observe;
    return {
      weightsBytes: 0,
      tapLayers: inner.tapLayers,
      prefill: (ids, context) => inner.prefill(ids, context),
      draft: (...args) => inner.draft(...args),
      commit(drafted, accepted, context) {
        // `drafted` = positions this round verified; the first `accepted` matched.
        observe(Array.from({ length: drafted }, (_, pos) => ({ pos, conf: confidence[pos]!, accepted: pos < accepted })));
        return inner.commit(drafted, accepted, context);
      },
      dispose: () => inner.dispose(),
    };
  }

  dispose(): void { this.drafter.dispose(); }
}

export interface CalibrateOptions {
  /** Prompts to run (chat messages or a plain user message). At most `count` are used. */
  prompts: readonly (string | ChatMessage[])[];
  count?: number;
  maxTokens?: number;
  /** Acceptance precision the kept positions must reach. Default 0.5. */
  target?: number;
  /** Positions with fewer observations are never pruned. Default 50. */
  minSamples?: number;
  onPrompt?: (event: { index: number; promptTokens: number; generated: number; acceptance: number }) => void;
  signal?: AbortSignal;
}

export interface CalibrationFit { sts: StsCalibration; positionSamples: number[]; gamma: number }

/** Collect (confidence, accepted) samples and fit thresholds for the drafter in
 *  `drafterDir`. Does not write; see `writeSts`. */
export async function calibrateDrafter(
  model: RuntimeModel, tokenizer: LoadedTokenizer, template: ChatTemplate, drafterDir: string, opts: CalibrateOptions,
): Promise<CalibrationFit> {
  const drafter = loadDsparkDrafter(drafterDir);
  const gamma = drafter.cfg.gamma, samples: ConfSample[] = [];
  const provider = new CalibrationProvider(drafter, round => samples.push(...round), drafterDir);
  try {
    // The served lane: placement, the speculative method and its group, one row.
    const binding = bindMlxGateway(model, { provider, numDraftTokens: gamma });
    const group = binding.createBatchGroup({ maxBatch: 1 });
    try {
      const count = Math.min(opts.count ?? 32, opts.prompts.length);
      for (let index = 0; index < count; index++) {
        opts.signal?.throwIfAborted();
        const row = opts.prompts[index]!;
        const messages: ChatMessage[] = typeof row === "string" ? [{ role: "user", content: row }] : row;
        let ids = tokenizer.encode(template.render(messages, { addGenerationPrompt: true }), true);
        if (ids.length >= 2 && ids[0] === ids[1] && ids[0] === tokenizer.bosTokenId) ids = ids.slice(1);
        const maxTokens = opts.maxTokens ?? 128, eosTokenIds = [...model.config.eosTokenIds];
        const options = { maxTokens, temperature: 0, eosTokenIds };
        const plan = binding.plan({ hasVision: false, hasAdapters: false, hasRepetitionPenalty: false, userSeed: false,
          hasLogitsExtras: false, hasGrammar: false, wantsLogprobs: false, hasDraft: true }, options,
          { continuous: true, quantizedBatch: false, checkpoints: false });
        if (plan.method !== "speculative" || plan.mechanism !== "continuous")
          throw new Error(`calibration needs speculative placement for this target and drafter (${plan.reasons.join(", ") || plan.method})`);
        const stats = await group.submit({ method: binding.methodRequest!(plan, options)!, promptIds: ids, maxTokens, eosTokenIds,
          onToken() {}, ...(opts.signal ? { signal: opts.signal } : {}) });
        const spec = stats.spec!;
        opts.onPrompt?.({ index, promptTokens: ids.length, generated: stats.generatedTokens, acceptance: spec.drafted ? spec.accepted / spec.drafted : 0 });
      }
    } finally { await group.close(); }
  } finally { provider.dispose(); }
  const sts = fitStsThresholds(samples, gamma, opts.target ?? 0.5, opts.minSamples ?? 50);
  const positionSamples = new Array<number>(gamma).fill(0);
  for (const sample of samples) if (sample.pos >= 0 && sample.pos < gamma) positionSamples[sample.pos]!++;
  return { sts, positionSamples, gamma };
}

/** The calibration a checkpoint carries, if any (read without loading weights). */
export function readSts(dir: string): StsCalibration | undefined {
  return (JSON.parse(readFileSync(join(dir, "dspark.json"), "utf8")) as { config: { sts?: StsCalibration } }).config.sts;
}

/** Set `config.sts` in `<dir>/dspark.json`, preserving every other field. */
export function writeSts(dir: string, sts: StsCalibration): string {
  const path = join(dir, "dspark.json");
  if (!existsSync(path)) throw new Error(`${dir} has no dspark.json — copy the checkpoint directory first`);
  const meta = JSON.parse(readFileSync(path, "utf8")) as { config: Record<string, unknown> };
  meta.config = { ...meta.config, sts };
  writeFileSync(path, JSON.stringify(meta, null, 2));
  return path;
}
