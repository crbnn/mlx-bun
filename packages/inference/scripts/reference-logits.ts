import { createHash } from "node:crypto";
import { strict as assert } from "node:assert";

/** Full-vector evidence, including exact maximum membership rather than a tolerance. */
export interface LogitsEvidence {
  sha256: string; vocab: number; finite: true; argmax: number; maxima: number[];
}
export interface PhaseEvidence {
  phase: "prompt" | "verify" | "decode"; ids: number[];
  before: (number | null)[]; after: (number | null)[]; afterRollback?: (number | null)[];
}
export interface TeacherforceEvidence {
  plain: LogitsEvidence[]; windows: LogitsEvidence[]; phases: PhaseEvidence[];
}
export function logitsEvidence(row: Float32Array): LogitsEvidence {
  assert(row.length > 0 && row.every(Number.isFinite), "reference logits must be nonempty and finite");
  let argmax = 0;
  for (let i = 1; i < row.length; i++) if (row[i]! > row[argmax]!) argmax = i;
  const maxima = [];
  for (let i = 0; i < row.length; i++) if (row[i] === row[argmax]) maxima.push(i);
  return { sha256: createHash("sha256").update(new Uint8Array(row.buffer, row.byteOffset, row.byteLength)).digest("hex"),
    vocab: row.length, finite: true, argmax, maxima };
}
export function cachePositions(cache: readonly { offset?: number }[]): (number | null)[] {
  return cache.map(c => c.offset ?? null);
}
export function validateLogitsEvidence(row: LogitsEvidence, vocab: number, what: string): void {
  assert(row && /^[a-f0-9]{64}$/.test(row.sha256), `${what}: invalid logits SHA-256`);
  assert.equal(row.vocab, vocab, `${what}: vocabulary`);
  assert.equal(row.finite, true, `${what}: logits were not all finite`);
  assert(Array.isArray(row.maxima) && row.maxima.length > 0 && row.maxima.every((id, i) =>
    Number.isSafeInteger(id) && id >= 0 && id < vocab && (i === 0 || id > row.maxima[i - 1]!)), `${what}: exact maxima`);
  assert.equal(row.argmax, row.maxima[0], `${what}: first maximum`);
}
export function acceptedRounds(forwards: readonly (readonly number[])[], tokens: readonly number[]) {
  let start = 0;
  return forwards.slice(1).map(window => {
    assert.equal(window[0], tokens[start], "teacherforce pending token");
    let accepted = 0;
    while (accepted + 1 < window.length && window[accepted + 1] === tokens[start + accepted + 1]) accepted++;
    const round = { start, kept: accepted + 1, window: [...window] };
    start += round.kept;
    return round;
  });
}
/** Validate the measured control's shape against the native run's stored inputs. */
export function validateTeacherforce(control: TeacherforceEvidence, forwards: number[][], tokens: number[], vocab: number): void {
  const rounds = acceptedRounds(forwards, tokens), kept = rounds.reduce((n, r) => n + r.kept, 0);
  assert.equal(control?.plain?.length, kept + 1, "teacherforce plain vector count");
  assert.equal(control?.windows?.length, kept, "teacherforce window vector count");
  control.plain.forEach((row, i) => validateLogitsEvidence(row, vocab, `teacherforce plain ${i}`));
  control.windows.forEach((row, i) => validateLogitsEvidence(row, vocab, `teacherforce window ${i}`));
  const expected = [{ phase: "prompt", ids: forwards[0]! }, ...rounds.flatMap(r => [
    { phase: "verify", ids: r.window }, ...Array.from({ length: r.kept }, (_, j) => ({phase: "decode", ids: [tokens[r.start + j]!] }))])];
  assert.equal(control.phases?.length, expected.length, "teacherforce phase count");
  const count = control.phases[0]!.before.length;
  assert(count > 0, "teacherforce cache positions missing");
  let current = control.phases[0]!.before;
  assert(current.every(p => p === null || p === 0), "teacherforce initial cache positions");
  control.phases.forEach((phase, i) => {
    assert.equal(phase.phase, expected[i]!.phase, `teacherforce phase ${i}: operation`);
    assert.deepEqual(phase.ids, expected[i]!.ids, `teacherforce phase ${i}: inputs`);
    for (const positions of [phase.before, phase.after, ...(phase.phase === "verify" ? [phase.afterRollback] : [])])
      assert(Array.isArray(positions) && positions.length === count && positions.every(p => p === null || (Number.isSafeInteger(p) && p >= 0)),
        `teacherforce phase ${i}: cache positions`);
    assert.deepEqual(phase.before, current, `teacherforce phase ${i}: predecessor positions`);
    assert.deepEqual(phase.after, phase.before.map(p => p === null ? null : p + phase.ids.length),
      `teacherforce phase ${i}: appended positions`);
    current = phase.phase === "verify" ? phase.afterRollback! : phase.after;
    if (phase.phase === "verify") assert.deepEqual(phase.afterRollback, phase.before, `teacherforce phase ${i}: rollback positions`);
  });
}
/** Every emitted native token must be an exact maximum of its actual target head. */
export function validateNativeHeads(heads: LogitsEvidence[][], forwards: number[][], tokens: number[], vocab: number): void {
  assert.equal(heads?.length, forwards.length, "native head call count");
  heads.forEach((rows, i) => {
    assert.equal(rows.length, i === 0 ? 1 : forwards[i]!.length, `native head ${i}: row count`);
    rows.forEach((row, j) => validateLogitsEvidence(row, vocab, `native head ${i} row ${j}`));
  });
  assert(heads[0]![0]!.maxima.includes(tokens[0]!), "native token 0 is not an exact maximum");
  acceptedRounds(forwards, tokens).forEach((round, r) => {
    for (let j = 0; j < round.kept && round.start + j + 1 < tokens.length; j++)
      assert(heads[r + 1]![j]!.maxima.includes(tokens[round.start + j + 1]!),
        `native token ${round.start + j + 1} is not an exact maximum`);
  });
}
