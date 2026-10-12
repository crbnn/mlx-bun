import type { AttentionRead, Cache, CommittedAttentionCache, KvAttentionView, KvDonorRows, Mask, SpeculativeRowLayout } from "../../contracts/mlx/cache";
import { buildBatchedDecodeMask, mergeKVRows } from "../batched-mask";
import * as ops from "@mlx-bun/mlx/ops";
import { runtimeFlag } from "../../runtime/config";
import { MlxArray } from "@mlx-bun/mlx/array";
import { disposeResources } from "../../runtime/resources";
import { PagedKVCache, poolBlocksFor, type PagedQuantization } from "./cache";

/** Block storage owns row membership; attention consumes the same gathered
 * K/V representation as the existing paged cache. No scheduler policy here.
 *
 * Reads: fixed at construction by the pages' representation, as the graphs
 * read these rows before the cache owned the read. Gathered bf16 pages (not
 * `direct`, no `quantization`): each row appends to its own blocks, the rows'
 * gathered keys and values join left-padded along the batch, and one fused
 * SDPA attends under the padding mask of the append (`buildBatchedDecodeMask`),
 * for decode and windows alike. Reading each row alone over its own length
 * reduces in another order than this read over the padded length and differs
 * in the last bits (up to 4e-3 on decode at head dim 512). Direct or affine
 * pages: each row reads with its own `PagedKVCache` read for the phase and the
 * outputs join along the batch. A committed span is the window read, as on
 * the serial pages.
 *
 * The same layout serves continuous decoding, prefill cohorts and
 * verification: every row owns its blocks and its offset, so the speculative
 * transaction keeps each row's accepted prefix by trimming that row's blocks
 * back to it, which returns the rejected tail's blocks to the row's pool in the
 * order they were taken. The donor rows are each row's gathered positions,
 * joined left-padded as the joined read joins them. */
export class PagedKvRows implements CommittedAttentionCache, SpeculativeRowLayout {
  #rows: PagedKVCache[] = [];
  #rope?: MlxArray;
  #ropeOffset = -1;
  #beforeRound?: number[];
  readonly #decode: (keys: MlxArray, values: MlxArray) => AttentionRead;
  readonly #window: (keys: MlxArray, values: MlxArray) => AttentionRead;

  constructor(readonly capacityTokens: number, readonly blockSize: number, readonly direct = runtimeFlag("MLX_BUN_PAGED_ATTN", false), readonly quantization?: PagedQuantization) {
    const alone = direct || quantization !== undefined;
    this.#decode = alone ? (keys, values) => this.#rowReads(keys, values, (row, k, v) => row.appendDecode(k, v))
      : (keys, values) => this.#joinedRead(keys, values);
    this.#window = alone ? (keys, values) => this.#rowReads(keys, values, (row, k, v) => row.appendWindow(k, v))
      : (keys, values) => this.#joinedRead(keys, values);
  }
  signature(): string { return "kv:paged-rows"; }
  get batchSize(): number { return this.#rows.length; }
  get offset(): number { return Math.max(0, ...this.#rows.map(row => row.offset)); }
  get rowOffsets(): number[] { return this.#rows.map(row => row.offset); }
  get leftPad(): number[] { const width = this.offset; return this.#rows.map(row => width - row.offset); }
  get ropeOffsetArr(): MlxArray | undefined {
    if (this.#rows.length < 2) return undefined;
    if (!this.#rope || this.#ropeOffset !== this.offset) {
      this.releaseRopeArr();
      this.#rope = MlxArray.fromInt32(Int32Array.from(this.rowOffsets), [this.batchSize]);
      this.#ropeOffset = this.offset;
    }
    return this.#rope;
  }
  releaseRopeArr(): void { this.#rope?.dispose(); this.#rope = undefined; this.#ropeOffset = -1; }
  makeEmptyBatch(): PagedKvRows { return new PagedKvRows(this.capacityTokens, this.blockSize, this.direct, this.quantization); }

  mergeRows(rows: readonly Cache[]): void {
    const next: PagedKVCache[] = [];
    try {
      for (const source of rows) {
        if (source instanceof PagedKvRows) {
          for (const row of source.#rows) next.push(row.clone());
        } else if (source instanceof PagedKVCache) next.push(source.clone());
        else throw new Error(`Paged rows cannot merge ${source.signature()}`);
      }
    } catch (error) { disposeResources(next); throw error; }
    const previous = this.#rows; this.#rows = next;
    this.releaseRopeArr(); disposeResources(previous);
  }
  extractRow(row: number): PagedKVCache { return this.#rows[row]!.clone(); }
  filterRows(keep: readonly number[]): void {
    const previous = this.#rows;
    this.#rows = keep.map(row => previous[row]!);
    const retained = new Set(keep);
    this.releaseRopeArr();
    disposeResources(previous.filter((_, row) => !retained.has(row)));
  }

  appendDecode(keys: MlxArray, values: MlxArray): AttentionRead {
    return this.#decode(keys, values);
  }

  appendWindow(keys: MlxArray, values: MlxArray): AttentionRead {
    return this.#window(keys, values);
  }

  /** The window read, as on the serial pages. */
  appendCommitted(keys: MlxArray, values: MlxArray): AttentionRead {
    return this.#window(keys, values);
  }

  specRoundBegin(): void { this.#beforeRound = this.rowOffsets; }
  specRoundCommit(): void { this.#beforeRound = undefined; }
  specRoundRollback(keep: number | readonly number[]): void {
    const before = this.#beforeRound!;
    for (const [row, storage] of this.#rows.entries())
      storage.trim(storage.offset - before[row]! - (typeof keep === "number" ? keep : keep[row]!));
    this.#beforeRound = undefined;
    this.releaseRopeArr();
  }

  /** Each row's gathered positions, joined left-padded to the longest row. */
  captureDonorRows(): KvDonorRows {
    const gathered: MlxArray[] = [];
    try {
      const rows = this.#rows.map(row => {
        const { keys, values } = row.captureDonorRows();
        gathered.push(keys, values);
        return { keys, values };
      });
      const { keys, values, leftPad, width } = mergeKVRows(rows);
      return { keys, values, offsets: this.rowOffsets, starts: leftPad, ends: leftPad.map(() => width) };
    } finally { disposeResources(gathered); }
  }

  /** The graphs' read of gathered bf16 rows before the cache owned it: the
   * padding mask of an `N`-position append built before it, each row's
   * gathered keys and values joined left-padded along the batch, one fused
   * SDPA. */
  #joinedRead(keys: MlxArray, values: MlxArray): AttentionRead {
    const N = keys.shape[2]!;
    const mask = buildBatchedDecodeMask(this.batchSize, N, this.offset + N, this.leftPad, null);
    let joined: [MlxArray, MlxArray];
    try { joined = this.#joinRows(keys, values); } catch (error) { mask.dispose(); throw error; }
    const [k, v] = joined;
    return {
      attend: (q, scale) => ops.sdpa(q, k, v, scale, "array", mask),
      dispose() { k.dispose(); v.dispose(); mask.dispose(); },
    };
  }

  /** Append each row's slice to its blocks, then join the rows' gathered keys
   * and values left-padded to the longest row; the caller owns both. */
  #joinRows(keys: MlxArray, values: MlxArray): [MlxArray, MlxArray] {
    const fetched: Array<{ keys: MlxArray; values: MlxArray }> = [];
    try {
      for (const [row, storage] of this.#rows.entries()) {
        using k = keys.slice([row, 0, 0, 0], [row + 1, ...keys.shape.slice(1)]);
        using v = values.slice([row, 0, 0, 0], [row + 1, ...values.shape.slice(1)]);
        const [nextK, nextV] = storage.appendAndGather(k, v);
        fetched.push({ keys: nextK, values: nextV });
      }
      const result = mergeKVRows(fetched);
      return [result.keys, result.values];
    } finally { disposeResources(fetched.flatMap(row => [row.keys, row.values])); }
  }

  #rowReads(keys: MlxArray, values: MlxArray,
    read: (row: PagedKVCache, k: MlxArray, v: MlxArray) => AttentionRead): AttentionRead {
    const reads: AttentionRead[] = [];
    try {
      for (const [row, storage] of this.#rows.entries()) {
        using k = keys.slice([row, 0, 0, 0], [row + 1, ...keys.shape.slice(1)]);
        using v = values.slice([row, 0, 0, 0], [row + 1, ...values.shape.slice(1)]);
        reads.push(read(storage, k, v));
      }
    } catch (error) { disposeResources(reads); throw error; }
    return {
      attend(q, scale) {
        const outputs: MlxArray[] = [];
        try {
          for (const [row, rowRead] of reads.entries()) {
            using query = q.slice([row, 0, 0, 0], [row + 1, ...q.shape.slice(1)]);
            outputs.push(rowRead.attend(query, scale));
          }
          return ops.concatAxis(outputs, 0);
        } finally { disposeResources(outputs); }
      },
      dispose() { disposeResources(reads); },
    };
  }

  get attentionState(): this | undefined { return this.direct || this.quantization ? this : undefined; }
  appendAndFetch(keys: MlxArray, values: MlxArray): KvAttentionView {
    const views: KvAttentionView[] = [];
    try {
      for (const [row, storage] of this.#rows.entries()) {
        using k = keys.slice([row, 0, 0, 0], [row + 1, ...keys.shape.slice(1)]);
        using v = values.slice([row, 0, 0, 0], [row + 1, ...values.shape.slice(1)]);
        views.push(storage.appendAndFetch(k, v));
      }
    } catch (error) { disposeResources(views); throw error; }
    return {
      attend(q, scale, mask) {
        const outputs: MlxArray[] = [];
        try {
          for (const [row, view] of views.entries()) {
            using query = q.slice([row, 0, 0, 0], [row + 1, ...q.shape.slice(1)]);
            outputs.push(view.attend(query, scale, mask));
          }
          return ops.concatAxis(outputs, 0);
        } finally { disposeResources(outputs); }
      },
      dispose() { disposeResources(views); },
    };
  }

  updateAndFetch(keys: MlxArray, values: MlxArray): [MlxArray, MlxArray] {
    if (this.#rows.length === 1) return this.#rows[0]!.updateAndFetch(keys, values);
    return this.#joinRows(keys, values);
  }
  makeMask(tokens: number, window: number | null): Mask {
    if (this.attentionState && window === null) return { mode: tokens === 1 ? "" : "causal", arr: null };
    if (this.#rows.length === 1) return this.#rows[0]!.makeMask(tokens, window);
    return { mode: "array", arr: buildBatchedDecodeMask(this.batchSize, tokens,
      this.offset + tokens, this.leftPad, window) };
  }
  state(): MlxArray[] { return this.#rows.flatMap(row => row.state()); }
  isTrimmable(): boolean { return true; }
  trim(tokens: number): void { for (const row of this.#rows) row.trim(tokens); this.releaseRopeArr(); }
  projectedBytes(tokens: number): number {
    const pool = this.#rows.find(row => row.pool)?.pool;
    if (!pool) return 0;
    const bytesPerBlock = pool.arrays().reduce((bytes, array) => bytes + array.nbytes, 0) / pool.numBlocks;
    return poolBlocksFor(tokens, this.blockSize) * bytesPerBlock;
  }
  dispose(): void { const rows = this.#rows; this.#rows = []; this.#beforeRound = undefined; this.releaseRopeArr(); disposeResources(rows); }
}
