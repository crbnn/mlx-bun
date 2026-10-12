import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import type { BatchableCache, Cache, CompiledDecodeCache, DecodeSlot, DecodeStepInputs, DecodeStepPlan, Mask, DenseKvReads } from "../contracts/mlx/cache";
import { KvTensorRows } from "./kv-tensor-rows";
import type { KvStorage } from "./kv-storage";
import { leaseCacheStates } from "./leases";
import { minimumReusableOffset } from "./views";
import { withResource, disposeResources } from "../runtime/resources";
import { cloneKvCaches } from "./persistence";

/** Encoded layouts retain physical row alignment across precision changes. */
export interface TransitionedKvLayout extends BatchableCache, KvStorage {
  alignRows?(leftPad: readonly number[]): void;
}
export interface SpeculativeTransitionedKvLayout extends TransitionedKvLayout {
  specRoundBegin(): void;
  specRoundCommit(): void;
  specRoundRollback(keep: number | readonly number[]): void;
}
/** One precision transition over a row layout, built with the layout's scheme:
 * the same conversion the single-sequence legos make (`state/bf16-first-kv`). */
export interface KvRowTransition<Layout extends TransitionedKvLayout> {
  readonly signature: string;
  readonly conversionOffset?: number;
  /** Converts, in place, every row whose stored length has reached the
   * conversion offset. Converted rows, and rows below the offset, stay. */
  settle(rows: Cache[]): void;
  /** Pure: whether `row`, reading dense now, still reads dense after `settle`
   * next runs on it. */
  keepsDenseReads(row: Cache): boolean;
  converted(row: Cache): boolean;
  makeLayout(): Layout;
  packRows?(layout: Layout, rows: readonly Cache[]): void;
  prepareRows?(rows: Cache[]): Cache[];
  extractRow?(row: Cache): Cache;
  rollbackRow?(row: Cache, before: number, keep: number, preserveColumns: boolean): void;
  /** How the lego these rows came from composes storage for serial use
   * (`ComposedStorage.over`): a row this layout hands back to serial use is
   * that lego again. Absent for rows that came as storage. */
  compose?(storage: Cache): Cache;
  /** The lego of this transition over one row's storage (taking ownership):
   * what a compiled decode step drives this layout's lone row through. */
  lone(storage: Cache): Cache & CompiledDecodeCache;
  /** `row` handed to serial use as it is (ownership moves), or null when the
   * row is a wrapper whose storage must be extracted. Absent: every row is
   * extracted. */
  adopt?(row: Cache): Cache | null;
}


export interface TransitioningKvPositions {
  readonly rowOffsets: readonly number[];
  readonly leftPad: readonly number[];
  readonly offset: number;
  readonly batchSize: number | null;
  readonly ropeOffsetArr: MlxArray | undefined;
  sync(rows: readonly Cache[], preserve: boolean): void;
  filterRows(keep: readonly number[]): void;
  makeMask(tokens: number, window: number | null): Mask;
  dispose(): void;
}
export class FullKvPositions extends KvTensorRows implements TransitioningKvPositions {
  sync(rows: readonly Cache[], preserve: boolean): void {
    this.mergeRows(preserve
      ? [{ planes: [], rowOffsets: rows.map(row => row.offset), leftPad: [...this.leftPad] }]
      : rows.map(row => ({ planes: [], rowOffsets: [row.offset], leftPad: [0] })));
  }
}

/** Owns membership, committed positions and precision transitions. Codec and
 * attention adapters own tensor access; scheduling and persistence use the
 * existing batch and checkpoint contracts. Row conversion never examines an
 * uncommitted speculative suffix.
 *
 * Rows convert before an append outside prefill, and during prefill only at
 * each row's committed chunk boundary (`prefillMaintenance.commitPrefill`), so
 * a sibling that splits a row's planned chunk across forwards does not move
 * that row's precision boundary. This is why the transition runs here rather
 * than at the end of each append, as the single-sequence legos run it.
 *
 * Compiled decode: a layout holding one row (the lone row a batch adopted)
 * drives it through a compiled decode step as the lego of its transition
 * (`KvRowTransition.lone`): the first step that prepares this layout moves the
 * row's storage into that lego, which converts at the same position the rows
 * convert (the append that reaches the start is read before the conversion), and
 * every compiled step after reaches the row there. Its offsets, state and
 * signature read through the lego; any other operation (an ordinary append, a
 * merge, an extraction, a speculative round) first moves the storage back into
 * the rows. A one-row layout reads each decode position with the same bits as
 * its lego (`tests/state/compiled-decode-protocol.test.ts`). */
export abstract class TransitioningKvRows<Layout extends TransitionedKvLayout> implements BatchableCache, CompiledDecodeCache {
  readonly #positions: TransitioningKvPositions;
  #rows: Cache[] = [];
  #packed?: Layout;
  /** The lone row as its lego, while a compiled decode step drives it. */
  #lone: (Cache & CompiledDecodeCache) | null = null;
  #prefilling = false;

  readonly stateNeedsDispose = true;
  constructor(readonly transition: KvRowTransition<Layout>, row?: Cache, positions: TransitioningKvPositions = new FullKvPositions()) {
    this.#positions = positions;
    if (row) { this.rows = transition.prepareRows?.([row]) ?? [row]; this.syncPositions(); }
  }
  // Every access to the rows, the packed layout or the positions finds the row
  // home first (see the class).
  protected get rows(): Cache[] { this.#home(); return this.#rows; }
  protected set rows(rows: Cache[]) { this.#home(); this.#rows = rows; }
  protected get packed(): Layout | undefined { this.#home(); return this.#packed; }
  protected set packed(packed: Layout | undefined) { this.#home(); this.#packed = packed; }
  protected get positions(): TransitioningKvPositions { this.#home(); return this.#positions; }
  /** The lone row's storage, moved back from its lego into the rows. */
  #home(): void {
    const lone = this.#lone;
    if (!lone) return;
    this.#lone = null;
    const storage = lone.composed ? lone.composed.release() : lone;
    if (storage !== lone) lone.dispose();
    this.#rows = this.transition.prepareRows?.([storage]) ?? [storage];
    this.#positions.sync(this.#rows, false);
  }
  /** The lone row as its lego, moving its storage there on the first call. */
  #loneLego(): Cache & CompiledDecodeCache {
    if (this.#lone) return this.#lone;
    if (this.batchSize !== 1) throw new Error(`compiled decode drives a layout of one row; this one holds ${this.batchSize}`);
    const packed = this.#packed, row = this.#rows[0]!;
    const adopted = packed ? null : this.transition.adopt?.(row) ?? null;
    const storage = adopted ?? this.extractStorageRow(0);
    let lego: Cache & CompiledDecodeCache;
    try { lego = this.transition.lone(storage); }
    catch (error) { if (!adopted) storage.dispose(); throw error; }
    this.#rows = []; this.#packed = undefined;
    if (packed) packed.dispose();
    else if (!adopted) row.dispose();
    this.#lone = lego;
    return lego;
  }
  /** The fetch of the coming compiled step: the lego's while it holds the row,
   * else the rows' (`decodePhaseOfRows`), without moving the row. */
  decodePhase(): DecodeStepPlan["fetch"] { return this.#lone ? this.#lone.decodePhase() : this.decodePhaseOfRows(); }
  /** The fetch a lone row's lego would choose, from this layout's positions. */
  protected decodePhaseOfRows(): DecodeStepPlan["fetch"] { return "concat"; }
  prepareDecodeStep(): DecodeStepPlan { return this.#loneLego().prepareDecodeStep(); }
  decodeSlot(plan: DecodeStepPlan): DecodeSlot { return this.#loneLego().decodeSlot(plan); }
  decodeInputs(plan: DecodeStepPlan, step: DecodeStepInputs): MlxArray[] { return this.#loneLego().decodeInputs(plan, step); }
  commitDecodeStep(slot: DecodeSlot, outputs: MlxArray[]): MlxArray[] { return this.#loneLego().commitDecodeStep(slot, outputs); }
  signature(): string { return this.transition.signature; }
  /** Plain reads per row: the row reads dense now, by its own capability or
   * the packed layout's, and still reads dense after the transition its next
   * append runs (deferred during prefill; a packed layout is past it). */
  readonly denseKvReads: DenseKvReads = { appendable: row => {
    if (this.packed) return this.packed.denseKvReads?.appendable(row) ?? false;
    const cache = this.rows[row];
    return !!cache?.denseKvReads?.appendable(0) && (this.#prefilling || this.transition.keepsDenseReads(cache));
  } };
  get prefillMaintenance() { return this; }
  beginPrefill(): void { this.#prefilling = true; }
  commitPrefill(indices: readonly number[]): void {
    if (this.packed) return;
    for (const index of indices) {
      const row = [this.rows[index]!];
      try { this.transition.settle(row); }
      finally { this.rows[index] = row[0]!; }
    }
  }
  endPrefill(): void { this.#prefilling = false; }
  get minimumReusableOffset(): number {
    if (this.#lone) return this.#lone.minimumReusableOffset ?? 0;
    return this.packed?.minimumReusableOffset ?? minimumReusableOffset(this.rows);
  }
  get rowOffsets(): readonly number[] { return this.#lone ? [this.#lone.offset] : this.packed?.rowOffsets ?? this.positions.rowOffsets; }
  get leftPad(): readonly number[] { return this.#lone ? [0] : this.packed?.leftPad ?? this.positions.leftPad; }
  get offset(): number { return this.#lone ? this.#lone.offset : this.packed?.offset ?? this.positions.offset; }
  maxAppendTokens(): number {
    if (this.packed) return this.packed.maxAppendTokens?.() ?? Infinity;
    const end = this.transition.conversionOffset;
    if (end === undefined) return Infinity;
    return Math.min(Infinity, ...this.rowOffsets.map((offset, row) =>
      offset < end && !this.transition.converted(this.rows[row]!) ? end - offset : Infinity));
  }
  get batchSize(): number | null { return this.#lone ? 1 : this.packed?.batchSize ?? this.positions.batchSize; }
  get ropeOffsetArr(): MlxArray | undefined {
    if (this.#lone) return this.#lone.ropeOffsetArr;
    return this.packed ? this.packed.ropeOffsetArr : this.positions.ropeOffsetArr;
  }
  bytesPerToken(): number { return this.packed?.bytesPerToken?.() ?? Math.max(0, ...this.rows.map(row => row.bytesPerToken?.() ?? 0)); }
  projectedBytes(tokens: number): number { return this.bytesPerToken() * tokens; }
  makeMask(tokens: number, window: number | null): Mask { return (this.packed ?? this.positions).makeMask(tokens, window); }
  isTrimmable(): boolean { return this.packed?.isTrimmable() ?? this.rows.every(row => row.isTrimmable()); }
  trim(count: number): void {
    if (this.packed) this.packed.trim(count);
    else { for (const row of this.rows) row.trim(count); this.syncPositions(true); }
  }
  protected syncPositions(preserve = false): void { this.positions.sync(this.rows, preserve); }
  /** Reject incompatible storage before appending any row: rows already
   * converted refuse with no change; otherwise the transition runs (deferred
   * during prefill), then every row must still be plain. */
  protected advancePlain(): void {
    const plain = () => !this.packed && this.rows.every(row => !this.transition.converted(row));
    if (!plain()) throw new Error("mixed precision rows use their attention state");
    this.advance();
    if (!plain()) throw new Error("mixed precision rows use their attention state");
  }
  protected advance(): void {
    if (this.packed) return;
    if (!this.#prefilling) this.transition.settle(this.rows);
    if (this.rows.length && this.rows.every(row => this.transition.converted(row))) {
      const packed = this.transition.makeLayout();
      try {
        if (this.transition.packRows) this.transition.packRows(packed, this.rows);
        else packed.mergeRows(this.rows);
        packed.alignRows?.(this.positions.leftPad);
      }
      catch (error) { packed.dispose(); throw error; }
      const previous = this.rows; this.rows = []; this.packed = packed;
      this.positions.dispose(); disposeResources(previous);
    }
  }
  abstract updateAndFetch(k: MlxArray, v: MlxArray): [MlxArray, MlxArray];
  abstract makeEmptyBatch(): TransitioningKvRows<Layout>;
  mergeRows(rows: readonly Cache[]): void {
    const owned: Cache[] = [];
    try {
      for (const row of rows) {
        if (row instanceof TransitioningKvRows) {
          for (let index = 0; index < row.batchSize!; index++) owned.push(row.extractStorageRow(index));
        } else owned.push(...cloneKvCaches([row.composed ? row.composed.storage() : row]));
      }
    } catch (error) { disposeResources(owned); throw error; }
    let prepared: Cache[];
    try { prepared = this.transition.prepareRows?.(owned) ?? owned; }
    catch (error) { disposeResources(owned); throw error; }
    this.dispose(); this.rows = prepared; this.syncPositions();
  }
  filterRows(keep: readonly number[]): void {
    if (this.packed) { this.packed.filterRows(keep); return; }
    const previous = this.rows; this.rows = keep.map(index => previous[index]!);
    disposeResources(previous.filter((_, index) => !keep.includes(index))); this.positions.filterRows(keep);
  }
  /** A row's storage, owned by the caller: what another row layout merges. */
  extractStorageRow(row: number): Cache {
    return this.packed?.extractRow(row) ?? (this.transition.extractRow?.(this.rows[row]!) ?? cloneKvCaches([this.rows[row]!])[0]!);
  }
  /** A row handed back to serial use: its storage, composed again by the lego
   * the row came from (`KvRowTransition.compose`). */
  extractRow(row: number): Cache {
    const storage = this.extractStorageRow(row);
    return this.transition.compose ? this.transition.compose(storage) : storage;
  }
  state(): MlxArray[] {
    return withResource(leaseCacheStates(this.#lone ? [this.#lone] : this.packed ? [this.packed] : this.rows), arrays => {
      const views: MlxArray[] = [];
      try { for (const array of arrays) views.push(ops.contiguous(array)); return views; }
      catch (error) { disposeResources(views); throw error; }
    });
  }
  dispose(): void {
    this.#prefilling = false;
    const lone = this.#lone; this.#lone = null; lone?.dispose();
    const previous = this.#rows; this.#rows = [];
    this.#packed?.dispose(); this.#packed = undefined; this.#positions.dispose(); disposeResources(previous);
  }
}

/** Rollback is a method capability, separate from ordinary cache membership. */
export abstract class SpeculativeTransitioningKvRows<Layout extends SpeculativeTransitionedKvLayout> extends TransitioningKvRows<Layout> {
  #beforeRound?: number[];
  specRoundBegin(): void {
    this.advance(); this.#beforeRound = [...this.rowOffsets]; this.packed?.specRoundBegin();
  }
  specRoundCommit(): void {
    this.packed?.specRoundCommit(); this.#beforeRound = undefined; this.advance();
  }
  specRoundRollback(keep: number | readonly number[]): void {
    if (this.packed) this.packed.specRoundRollback(keep);
    else {
      for (const [index, row] of this.rows.entries()) {
        const count = typeof keep === "number" ? keep : keep[index]!;
        const before = this.#beforeRound![index]!;
        if (this.transition.rollbackRow) this.transition.rollbackRow(row, before, count, this.rows.length > 1);
        else row.trim(row.offset - before - count);
      }
      this.syncPositions(true);
    }
    this.#beforeRound = undefined; this.advance();
  }
  override dispose(): void { this.#beforeRound = undefined; super.dispose(); }
}
