// Streaming generation — port of mlx-lm's generate_step:
// - prefill in chunks; cache state evaluated per chunk (bounded transient
//   memory), logits never computed for non-final prefill positions
// - decode pipelining via mx.async_eval: step n+1's graph is built and
//   dispatched before step n's token is read back, so the GPU never idles
//   on the JS round-trip
// - opt-in early token-zero yield reduces latency before that pipeline starts
// - sampling stays on-device; only the chosen token id crosses to JS

import { gpuStream } from "@mlx-bun/mlx/array";
import {
maxRecommendedWorkingSetSize,
setWiredLimit,
synchronize
} from "@mlx-bun/mlx/ffi";
import { GenerateStats,GeneratedToken } from "./types";
import {
type MlxModelMemory
} from "./bindings/autoregressive";
import { runtimeValue } from "../runtime/config";

// Scoped wired limit, raised only for models above WIRE_THRESHOLD of the
// recommended working set. mlx-lm's wired_limit context wires
// unconditionally per generation; we deviate with a measured justification:
// - 26B-A4B (16.4 GB = 92% of the 17.8 GiB working set) NEEDS wiring —
//   8.6 tok/s without, 32.3 with (Metal evicts weight buffers per token).
// - 12B/e4b (≤47%) hit reference parity WITHOUT wiring, and wiring in a
//   multi-model process (the test suite) pins memory the OTHER resident
//   models need — async GPU exec OOM, which is uncatchable (the mlx
//   completion-handler throw terminates the process).
// Scope semantics match the reference: set → generate → synchronize →
// restore; nothing stays pinned between generations. Re-entrant: only
// the outermost wiring scope touches the limit.
// macOS 26.6 reports a 26.8e9 B (24.96 GiB, 0.78 × RAM) recommended set on
// a 32 GB M1 Max (2026-09-08; the earlier "24 GB machine" attribution was
// wrong — 24.96 GiB exceeds a 24 GB box's RAM). At 0.75 × that the old
// fraction stopped wiring the 13-16 GiB Qwen/GLM models even though they
// page heavily without an explicit wired limit.
// 0.4, not 0.5 (Josh, 2026-10-13): 0.5 is 13.4e9 B on that M1 Max, which
// left the 11.96e9 B Qwen3.8-27B Trellis unwired. Wired there, bit-identical,
// paired A/B: GPU prefill 1k/8k 136/142 -> 143/145 tok/s; with the Neural
// Engine split 171/173 -> 172/177, its first GPU step after the split
// 0.15-1.4 s -> 61-69 ms; one-row decode unchanged (60.0 ms). 0.4 is 10.7e9 B
// there: the 12.7-13.0e9 B Qwen3.8 3-bit artifacts are wired too, the 8-9 GiB
// models stay unwired. Where the set is 17.8 GiB (the 26B-A4B line above) it
// is 7.1 GiB, which wires those as well.
const WIRE_THRESHOLD = 0.4;
let wiredScopeDepth = 0;
let wiredOldLimit = 0;

type WiredModelMemory = MlxModelMemory;

/** Bytes whose MLX graph must stay resident while a model executes. */
export function wiredWorkingSetBytes(model: WiredModelMemory): number {
  const planned = model.expertRuntime?.plan.plannedBytes;
  return typeof planned === "number" && Number.isFinite(planned) && planned > 0
    ? Math.max(model.weightsBytes, planned)
    : model.weightsBytes;
}

export function modelNeedsWiredLimit(
  model: WiredModelMemory,
  recommendedBytes = maxRecommendedWorkingSetSize(),
  force = runtimeValue("MLX_BUN_FORCE_WIRE") === "1",
): boolean {
  return force ||
    wiredWorkingSetBytes(model) > WIRE_THRESHOLD * recommendedBytes;
}

function enterWiredScope(): void {
  if (wiredScopeDepth++ === 0)
    wiredOldLimit = setWiredLimit(maxRecommendedWorkingSetSize());
}
function exitWiredScope(): void {
  if (--wiredScopeDepth === 0) {
    synchronize(gpuStream);
    setWiredLimit(wiredOldLimit);
  }
}

/** Hold the process wired limit while one model owns GPU execution. */
export function acquireModelWiredLimit(model: WiredModelMemory): () => void {
  if (!modelNeedsWiredLimit(model)) return () => {};
  enterWiredScope();
  let released = false;
  return () => {
    if (released) return;
    released = true;
    exitWiredScope();
  };
}

/** Hold the model's active-adapter list for exactly this generation. */
export async function* adapterScoped(
  model: { loraState: { active: string[] } },
  adapters: string[],
  inner: AsyncGenerator<GeneratedToken, GenerateStats>,
): AsyncGenerator<GeneratedToken, GenerateStats> {
  const previous = model.loraState.active;
  model.loraState.active = adapters;
  try {
    return yield* inner;
  } finally {
    model.loraState.active = previous;
  }
}

/** Wrap the generator so the wired limit is held exactly while it runs
 *  (incl. early break/return/throw — finally fires on .return()). */
export async function* wiredScoped(
  inner: AsyncGenerator<GeneratedToken, GenerateStats>,
): AsyncGenerator<GeneratedToken, GenerateStats> {
  enterWiredScope();
  try {
    return yield* inner;
  } finally {
    exitWiredScope();
  }
}

/** Publish the streamed model's route ledger on every generator exit path. */
export async function* usageScoped(
  model: WiredModelMemory,
  inner: AsyncGenerator<GeneratedToken, GenerateStats>,
): AsyncGenerator<GeneratedToken, GenerateStats> {
  try {
    return yield* inner;
  } finally {
    if (model.expertRuntime?.finishUsage)
      await model.expertRuntime.finishUsage();
    else
      model.expertRuntime?.flushUsage?.();
  }
}

/** Apply generate()'s scoped wiring policy to non-generator execution paths. */
export async function withModelWiredLimit<T>(
  model: WiredModelMemory,
  run: () => Promise<T>,
): Promise<T> {
  const release = acquireModelWiredLimit(model);
  try {
    return await run();
  } finally {
    release();
  }
}

/** Apply generate()'s usage safe-point to direct/non-generator paths. */
export async function withModelUsageFlush<T>(
  model: WiredModelMemory,
  run: () => Promise<T>,
): Promise<T> {
  try {
    return await run();
  } finally {
    if (model.expertRuntime?.finishUsage)
      await model.expertRuntime.finishUsage();
    else
      model.expertRuntime?.flushUsage?.();
  }
}
