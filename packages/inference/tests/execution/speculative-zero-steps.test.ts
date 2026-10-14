// Rounds without drafts in an adaptive speculative group pipeline as plain decode does: each step's window is fed by
// the previous step's samples on the device and submitted before they are read. A stand-in graph whose head names
// the input's successor makes every sample known; costs under which drafting never pays keep the gate at zero.
import { expect, test } from "bun:test";
import { bindSpeculativeGroupRequests } from "../../src/execution/speculative-group";
import type { MlxGroupMethodHost, MlxGroupPreparation, Row } from "../../src/execution/batch-types";
import type { Cache } from "../../src/contracts/mlx/cache";
import { AttentionMasks } from "../../src/state/attention-read";
import { composedFullAttentionCache } from "../../src/state/bf16-first-kv";
import { unfusedAffineKernels } from "../../src/state/affine-attention";
import { denseTurboQuantKernels } from "../../src/state/turboquant-attention";
import { kvOf } from "../models/test-composition";
import type { RuntimeModel } from "../../src/models/factory";
import { captureKvAttention } from "../../src/state/kv-attention-view";
import { NgramProvider } from "../../src/generation/speculative/sources/ngram-source";
import { projectedDraftGroups } from "../../src/generation/speculative/bindings/projected-draft-rows";
import type { DraftProvider } from "../../src/generation/speculative/source";
import { PromptCache } from "../../src/state/prefix-cache";
import { KvCacheService } from "../../src/state/cache-service";
import { runtimeConfig } from "../../src/runtime/config";
import * as ops from "@mlx-bun/mlx/ops";
import { Dtype } from "@mlx-bun/mlx/ffi";
import type { MlxArray } from "@mlx-bun/mlx/array";
import { declareGraph } from "../../src/models/capabilities";
import { stubPhases } from "../models/phase-forward";

const VOCABULARY = 256;

/** A block drafter (projected context rows, as DFlash 2 drafts) that proposes `pending + 1 + offset`, ...: right at
 *  offset 0 for the successor head, wrong otherwise. Its rows draft without committing, so they can preview. */
function successorDrafter(offset: { value: number }) {
  return { id: "successor", grouped: projectedDraftGroups("successor-v1", [], () => ({
    namespace: "successor-v1", schema: "successor-v1", layers: 1,
    project(hidden: MlxArray) {
      using rows = ops.reshape(hidden, [hidden.shape[0]!, 1, hidden.shape[1]!, 1]);
      using values = rows.astype(Dtype.float32);
      return [{ k: ops.copyOf(values), v: ops.copyOf(values) }];
    },
    draft: () => { throw new Error("successor drafts on the device"); },
    draftDevice(_context: unknown, pending: readonly number[], _positions: unknown, depth: number) {
      return ops.fromInt32(pending.flatMap(token => Array.from({ length: depth }, (_, i) => token + 1 + i + offset.value)),
        [pending.length, depth]);
    },
  })), dispose() {} };
}

function fixture(provider: Pick<DraftProvider, "id" | "grouped"> & { dispose(): void } = new NgramProvider(),
  costs: [number, number][] = [[1, 1], [2, 50], [3, 60]]) {
  /** Each forward's input ids, read after the step ran (reading them earlier would end the pipelining). */
  const inputs: MlxArray[] = [];
  const puts: { tokens: number[]; offset: number }[] = [];
  const published: Map<Row, number[]> = new Map(), finished: Map<Row, string> = new Map();
  const failures: { row: Row; error: unknown }[] = [], joined: Row[] = [];
  const scheme = kvOf({});
  const model = { config: { modelType: "fixture", eosTokenIds: [] }, graphCapabilities: declareGraph(),
    // By default drafting never pays: a two-position window costs fifty one-position steps.
    verifyRoundCosts: new Map(costs),
    makeCache: () => [composedFullAttentionCache(scheme, 0, new AttentionMasks(), denseTurboQuantKernels)],
    ...stubPhases((ids: MlxArray, caches: Cache[]) => {
      inputs.push(ops.copyOf(ids));
      using kv = ops.reshape(ids, [ids.shape[0]!, 1, ids.shape[1]!, 1]);
      using values = kv.astype(Dtype.float32);
      for (const cache of caches) captureKvAttention(cache, values, values, unfusedAffineKernels(4, 64, values.dtype)).dispose();
      return ops.reshape(ids, [ids.shape[0]!, ids.shape[1]!, 1]);
    }, (hidden: MlxArray) => {
      // The head names each position's successor: logits one-hot at id + 1.
      using ids = hidden.astype(Dtype.int32);
      using one = ops.fromInt32([1], [1]);
      using next = ops.add(ids, one);
      using columns = ops.arange(0, VOCABULARY, 1, Dtype.int32);
      using hot = ops.equal(next, columns);
      return hot.astype(Dtype.float32);
    }),
  } as unknown as RuntimeModel;
  const cache = new PromptCache(1024 ** 2);
  const prefixes: Pick<PromptCache, "take" | "put" | "reclaim"> = { take: cache.take.bind(cache), reclaim() {},
    put(tokens, caches, namespace, retain, attachments) {
      puts.push({ tokens: [...tokens], offset: caches[0]!.offset });
      cache.put(tokens, caches, namespace, retain, attachments);
    } };
  const host: MlxGroupMethodHost = { rows: joined, runtime: runtimeConfig(), prefillChunkSize: 8, prefillTailSplit: true,
    kvMaintenance: undefined, cache: new KvCacheService(model, { prefixes: prefixes as PromptCache }),
    join(row) { joined.push(row); },
    filterRows(keep) {
      const kept = keep.map(index => joined[index]!);
      method.filterRows(keep, false); joined.splice(0, joined.length, ...kept);
    },
    async publish(row, token) { published.get(row)!.push(token); return row.req.onToken(token); },
    finish(row, reason) { finished.set(row, reason); },
  };
  const binding = bindSpeculativeGroupRequests(model, provider, 2, true)!({ temperature: 0, maxTokens: 6 });
  const method = binding.open(host);
  const row = (tokens: number[], options: { maxTokens?: number; eos?: number[]; signal?: AbortSignal;
    onToken?: (token: number) => boolean | void } = {}): Row => {
    const request: Row = { req: { method: binding, promptIds: tokens, maxTokens: options.maxTokens ?? 6,
      eosTokenIds: options.eos ?? [], signal: options.signal, onToken: options.onToken ?? (() => {}) },
      resolve() {}, reject(error) { failures.push({ row: request, error }); },
      current: 0, generated: 0, sampled: 0, promptTokens: tokens.length, cachedTokens: 0,
      admittedAt: 0, firstTokenAt: 0, fed: [], fedTainted: false, merged: false };
    published.set(request, []);
    return request;
  };
  const read = () => inputs.map(ids => ids.toIntTokens());
  return { method, host, row, read, puts, published, finished, failures, joined,
    dispose() { method.dispose(); cache.clear(); provider.dispose(); for (const ids of inputs) ids.dispose(); } };
}

async function drain(preparation: MlxGroupPreparation) {
  for (let step = 0; step < 30; step++) if (await preparation.advance()) return;
  throw new Error("preparation did not drain");
}

/** Prefill `rows` together, then advance until every row has left the group. */
async function run(f: ReturnType<typeof fixture>, rows: Row[], between?: (advance: number) => void) {
  const preparation = f.method.prepare(rows[0]!);
  try {
    for (const row of rows.slice(1)) preparation.admit!(row);
    await drain(preparation);
  } finally { preparation.dispose(); }
  for (let advance = 0; f.joined.length; advance++) {
    if (advance > 40) throw new Error("rows did not finish");
    between?.(advance);
    await f.method.advance();
  }
}

test("a row's steps read the previous samples on the device; its budget ends it with its processed state stored", async () => {
  const f = fixture(), a = f.row([1, 2, 3, 4]);
  try {
    await run(f, [a]);
    expect(f.published.get(a)).toEqual([5, 6, 7, 8, 9, 10]);
    expect(f.finished.get(a)).toBe("length");
    // The prompt's tail split leaves 4 pending; each step is fed the one before's sample.
    expect(f.read()).toEqual([[1, 2, 3], [4], [5], [6], [7], [8], [9]]);
    // The last sample (10) is delivered without another step; the stored state covers what was processed.
    expect(f.puts.at(-1)).toEqual({ tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9], offset: 9 });
    expect(a.spec!.rounds).toBe(7);
    expect(f.failures).toEqual([]);
  } finally { f.dispose(); }
});

test("an EOS sample ends its row and drops the step built from it, as an unpipelined round would", async () => {
  const f = fixture(), a = f.row([1, 2, 3, 4], { eos: [8] });
  try {
    await run(f, [a]);
    expect(f.published.get(a)).toEqual([5, 6, 7]);
    expect(f.finished.get(a)).toBe("stop");
    // The step fed 8 ran before 8 was read; its write is rolled back.
    expect(f.read().at(-1)).toEqual([8]);
    expect(f.puts.at(-1)).toEqual({ tokens: [1, 2, 3, 4, 5, 6, 7], offset: 7 });
  } finally { f.dispose(); }
});

test("a consumer that stops keeps the state of the tokens it took", async () => {
  const f = fixture(), a = f.row([1, 2, 3, 4], { onToken: token => token !== 6 });
  try {
    await run(f, [a]);
    expect(f.published.get(a)).toEqual([5, 6]);
    expect(f.finished.get(a)).toBe("stop");
    expect(f.puts.at(-1)).toEqual({ tokens: [1, 2, 3, 4, 5], offset: 5 });
  } finally { f.dispose(); }
});

test("rows that end at different steps drop only their own step; a cancelled row stores nothing", async () => {
  const f = fixture(), abort = new AbortController();
  const a = f.row([1, 2, 3, 4], { eos: [7] }), b = f.row([20, 21, 22], { maxTokens: 5 });
  const c = f.row([40, 41], { signal: abort.signal });
  try {
    await run(f, [a, b, c], advance => { if (advance === 2) abort.abort(new Error("cancelled")); });
    expect(f.published.get(a)).toEqual([5, 6]);
    expect(f.published.get(b)).toEqual([23, 24, 25, 26, 27]);
    expect(f.finished.get(a)).toBe("stop");
    expect(f.finished.get(b)).toBe("length");
    expect(f.failures.map(failure => failure.row)).toEqual([c]);
    expect(f.puts.filter(put => put.tokens[0] === 1).at(-1)).toEqual({ tokens: [1, 2, 3, 4, 5, 6], offset: 6 });
    expect(f.puts.filter(put => put.tokens[0] === 20).at(-1)).toEqual({ tokens: [20, 21, 22, 23, 24, 25, 26], offset: 7 });
    expect(f.puts.some(put => put.tokens[0] === 40 && put.tokens.length > 1)).toBe(false);
  } finally { f.dispose(); }
});

test("a row that joins while a step is in flight waits for its delivery; both continue", async () => {
  const f = fixture(), a = f.row([1, 2, 3, 4], { maxTokens: 6 }), b = f.row([30, 31, 32], { maxTokens: 4 });
  try {
    const first = f.method.prepare(a);
    try { await drain(first); } finally { first.dispose(); }
    await f.method.advance();
    await f.method.advance();
    expect(f.published.get(a)).toEqual([5]);
    const second = f.method.prepare(b);
    try { await drain(second); } finally { second.dispose(); }
    expect(f.joined).toEqual([a, b]);
    for (let advance = 0; f.joined.length; advance++) {
      if (advance > 20) throw new Error("rows did not finish");
      await f.method.advance();
    }
    expect(f.published.get(a)).toEqual([5, 6, 7, 8, 9, 10]);
    expect(f.published.get(b)).toEqual([33, 34, 35, 36]);
    expect(f.puts.filter(put => put.tokens[0] === 1).at(-1)).toEqual({ tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9], offset: 9 });
    expect(f.puts.filter(put => put.tokens[0] === 30).at(-1)).toEqual({ tokens: [30, 31, 32, 33, 34, 35], offset: 6 });
    expect(f.failures).toEqual([]);
  } finally { f.dispose(); }
});

test("a drafter that previews probes without verify windows, and drafts again once its previews match", async () => {
  const offset = { value: 5 };
  // Drafting pays when the drafts are right: three positions cost 1.5 steps.
  const f = fixture(successorDrafter(offset), [[1, 1], [2, 1.4], [3, 1.5]]);
  const a = f.row([1, 2, 3, 4], { maxTokens: 160 });
  try {
    const first = f.method.prepare(a);
    try { await drain(first); } finally { first.dispose(); }
    // Wrong drafts: the optimistic first rounds fail and the gate stops drafting.
    for (let advance = 0; advance < 30; advance++) await f.method.advance();
    const settled = f.read().length;
    expect(f.read().some(ids => ids.length > 1)).toBe(true);
    for (let advance = 0; advance < 20; advance++) await f.method.advance();
    // Its probes compare previews with the samples: every forward since is one decode position.
    expect(f.read().slice(settled).every(ids => ids.length === 1)).toBe(true);
    // Right drafts: a probe's preview matches and verify windows return.
    offset.value = 0;
    while (f.joined.length) await f.method.advance();
    expect(f.read().slice(settled).some(ids => ids.length > 1)).toBe(true);
    // Every token is the successor head's, whatever the drafts were.
    expect(f.published.get(a)).toEqual(Array.from({ length: 160 }, (_, i) => 5 + i));
    expect(f.failures).toEqual([]);
  } finally { f.dispose(); }
});
