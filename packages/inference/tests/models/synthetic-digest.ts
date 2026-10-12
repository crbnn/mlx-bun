// Recorded digests of synthetic-weight cases: the gate for numerical facts no
// artifact holds (GLM-5.2, DiffusionGemma, Gemma 2, Qwen 3 and Qwen 3 MoE). A
// case hashes every tensor it is handed (dtype, shape and bytes, in order) under
// the label it names; its suite's fixture (tests/fixtures/synthetic-digests/
// <suite>.json) holds the hashes recorded on each GPU architecture, so a change
// to any bit of any listed tensor fails the case at the first label that moved.
// MLX chooses kernels by device class, so the hashes are per architecture: one
// with no record skips its cases, which is not a pass. After a change whose
// numbers are meant to move, record the suite again with
// `bun packages/inference/scripts/synthetic-digests.ts record <suite>` and say
// why in the commit.
import { afterAll, expect, test } from "bun:test";
import { createHash, type Hash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { MlxArray } from "@mlx-bun/mlx/array";
import { MLX_VERSION, deviceArchitecture } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";
import type { BatchableCache, Cache } from "../../src/contracts/mlx/cache";

/** Set by the record script: write this architecture's hashes instead of checking them. */
export const RECORD_ENV = "MLX_BUN_RECORD_SYNTHETIC_DIGESTS";
export const FIXTURES = join(import.meta.dir, "../fixtures/synthetic-digests");

/** One label: how many tensors it hashed and the first 16 hex digits of their SHA-256. */
type Entry = [label: string, tensors: number, hash: string];
interface Fixture {
  readonly suite: string;
  readonly recorded: Record<string, { readonly mlx: string; readonly cases: Record<string, Entry[]> }>;
}

/** What a case hands its digest. Every tensor stays the caller's. */
export interface DigestCase {
  /** Hash `arrays` under `label`. */
  tensors(label: string, ...arrays: MlxArray[]): void;
  /** Hash every layer's signature, offset, row offsets and stored tensors under `label`. */
  caches(label: string, caches: readonly Cache[]): void;
  /** Hash a JSON value (a count, a selection mode) under `label`. */
  value(label: string, value: unknown): void;
}

class Recorder implements DigestCase {
  readonly entries: Entry[] = [];
  #label: string | null = null;
  #hash: Hash | null = null;
  #count = 0;

  #open(label: string): Hash {
    if (this.#label !== label) {
      this.close();
      this.#label = label;
      this.#hash = createHash("sha256");
      this.#count = 0;
    }
    return this.#hash!;
  }
  close(): void {
    if (this.#label !== null) this.entries.push([this.#label, this.#count, this.#hash!.digest("hex").slice(0, 16)]);
    this.#label = null; this.#hash = null;
  }
  tensors(label: string, ...arrays: MlxArray[]): void {
    const hash = this.#open(label);
    for (const array of arrays) {
      using flat = ops.contiguous(array);
      const bytes = flat.rawBytesView();
      hash.update(`${flat.dtypeName}[${flat.shape.join(",")}]`);
      hash.update(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
      this.#count++;
    }
  }
  caches(label: string, caches: readonly Cache[]): void {
    for (const [layer, cache] of caches.entries()) {
      this.value(label, [layer, cache.signature(), cache.offset, (cache as Partial<BatchableCache>).rowOffsets ?? null]);
      const state = cache.state();
      try { this.tensors(label, ...state); }
      finally { if (cache.stateNeedsDispose) for (const array of state) array.dispose(); }
    }
  }
  value(label: string, value: unknown): void {
    this.#open(label).update(JSON.stringify(value));
  }
}

/** The suite's digest test: `digest.test(name, run)` registers a test whose run hands
 * its tensors to the digest; the run's labels and hashes must equal the recorded ones. */
export function syntheticDigests(suite: string) {
  const path = join(FIXTURES, `${suite}.json`);
  const architecture = deviceArchitecture();
  const recording = process.env[RECORD_ENV] === "1";
  const fixture: Fixture = existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : { suite, recorded: {} };
  const recorded = fixture.recorded[architecture];
  const fresh: Record<string, Entry[]> = {};
  if (recording) afterAll(() => {
    const cases = Object.fromEntries(Object.entries(fresh).sort(([a], [b]) => a.localeCompare(b)));
    const next: Fixture = { suite, recorded: { ...fixture.recorded, [architecture]: { mlx: MLX_VERSION, cases } } };
    mkdirSync(dirname(path), { recursive: true });
    // One entry per line: [label, tensors, hash].
    const text = JSON.stringify(next, null, 1).replace(/\[\n\s+("(?:[^"\\]|\\.)*"),\n\s+(\d+),\n\s+("[0-9a-f]{16}")\n\s+\]/g, "[$1, $2, $3]");
    writeFileSync(path, `${text}\n`);
    console.log(`synthetic-digests ${suite}: recorded ${Object.keys(cases).length} cases for ${architecture} (MLX ${MLX_VERSION})`);
  });
  const skip = !recording && !recorded;
  return {
    /** Whether this architecture has a record (the suite's cases skip without one). */
    recorded: !skip,
    test(name: string, run: (digest: DigestCase) => void | Promise<void>, timeout?: number): void {
      test.skipIf(skip)(`${name} (digest)`, async () => {
        const recorder = new Recorder();
        await run(recorder);
        recorder.close();
        if (recording) { fresh[name] = recorder.entries; return; }
        const expected = recorded!.cases[name];
        expect(expected, `${suite}: no digest recorded for "${name}" on ${architecture}; ` +
          `record it with scripts/synthetic-digests.ts record ${suite}`).toBeDefined();
        const moved = recorder.entries.findIndex((entry, index) => entry.join() !== expected![index]?.join());
        const where = moved === -1 ? expected!.length === recorder.entries.length ? null : recorder.entries.length : moved;
        expect(where === null ? null : { at: where, now: recorder.entries[where] ?? null, recorded: expected![where] ?? null },
          `${suite} "${name}": the first label whose tensors moved from the record for ${architecture} ` +
          `(MLX ${recorded!.mlx}, this run ${MLX_VERSION})`).toBeNull();
      }, timeout);
    },
  };
}
