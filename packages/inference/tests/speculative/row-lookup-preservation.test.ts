import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { rowOperationFor, rowTableWithDefault } from "../../src/models/row-table";
import type { MlxRowTable } from "../../src/contracts/mlx/row-table";
import { mapOperations } from "../../src/models/speculative/dflash2-operations";

const transpiler = new Bun.Transpiler({ loader: "ts", target: "bun" });
const source = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

// Execute the actual declarations with opaque host handles and private numerical
// methods replaced by spies. Importing the numerical models would initialize MLX.
function contextFixture() {
  const text = source("../../src/models/speculative/deepspec.ts");
  const fields = text.match(/readonly projectContext(?:KVRows)?:[\s\S]*?;\n/g)!;
  expect(fields).toHaveLength(2);
  const calls: unknown[][] = [], projected = {}, kv = [{}];
  const code = `class ContextDeclarations {
    #projectContext(input) { calls.push(["project", input]); return projected; }
    #projectContextKVRows(input, position) { calls.push(["kv", input, position]); return kv; }
    ${fields.join("\n")}
  }
  return new ContextDeclarations();`;
  const model = new Function("rowTableWithDefault", "calls", "projected", "kv",
    transpiler.transformSync(code))(rowTableWithDefault, calls, projected, kv) as {
      projectContext: MlxRowTable<(input: unknown) => unknown>;
      projectContextKVRows: MlxRowTable<(input: unknown, position: unknown) => unknown>;
    };
  return { model, calls, projected, kv };
}

test("DeepSpec context declarations preserve unbounded B×L rows with one immutable operation", () => {
  const f = contextFixture(), input = {}, positions = {};
  for (const rows of [1, 16_384, 16_385, 16 * 2048, 262_144]) {
    expect(rowOperationFor(f.model.projectContext, rows)(input)).toBe(f.projected);
    expect(rowOperationFor(f.model.projectContextKVRows, rows)(f.projected, positions)).toBe(f.kv);
    expect(f.model.projectContext[rows]).toBe(f.model.projectContext[1]);
    expect(f.model.projectContextKVRows[rows]).toBe(f.model.projectContextKVRows[1]);
  }
  expect(f.calls).toEqual(Array.from({ length: 5 }, () => [["project", input], ["kv", f.projected, positions]]).flat());
  for (const table of [f.model.projectContext, f.model.projectContextKVRows]) {
    expect(Object.keys(table)).toEqual([]);
    expect(Object.isFrozen(table)).toBe(true);
    expect(Reflect.set(table, "16385", () => null)).toBe(false);
    for (const key of ["0", "-1", "1.5", "then", "Infinity", Symbol.iterator])
      expect(Reflect.get(table, key)).toBeUndefined();
  }
});

function selectionFixture() {
  const text = source("../../src/generation/speculative/bindings/qwen-mtp-rows.ts");
  const declarations = text.slice(text.indexOf("function bindMtpSelection("), text.indexOf("/** The same draft graph"));
  const calls: string[] = [];
  const tensor = (name: string, shape = [1, 1, 7]) => ({ name, shape,
    [Symbol.dispose]() { calls.push(`dispose ${name}`); } });
  const ids = tensor("ids", [1]);
  class Vocabulary {
    readonly size = 3;
    constructor(readonly head: unknown, readonly tokens: readonly number[]) { calls.push("bind listed"); }
    project() { calls.push("listed"); return tensor("subset"); }
  }
  const ops = { reshape(logits: { name: string }, shape: number[]) { calls.push(`reshape ${logits.name}`); return tensor("flat", shape); } };
  const select = new Function("rowTableWithDefault", "mapOperations", "DraftVocabularyHead", "ops", "normalizedArgmax", "toLogprobs",
    transpiler.transformSync(`${declarations}\nreturn {bindMtpSelection, prefillOnly};`))(
      rowTableWithDefault, mapOperations, Vocabulary, ops,
      () => { calls.push("argmax"); return ids; },
      () => { calls.push("logprobs"); return tensor("logprobs"); }) as {
        bindMtpSelection(target: unknown, sampling: unknown, owned: { vocabulary: unknown }):
          MlxRowTable<(output: unknown, steps: { length: number }) => unknown>;
        prefillOnly: MlxRowTable<() => unknown>;
      };
  const head = (name: string) => () => { calls.push(name); return tensor(name); };
  const target = { logits: rowTableWithDefault({ 1: head("one"), 3: head("three") }, head("default")), vocabularyHead: {} };
  return { select, calls, tensor, ids, target };
}

for (const kind of ["greedy", "sampled"] as const) {
  test(`Qwen MTP ${kind} selection retains exact narrow heads and unbounded defaults`, () => {
    const f = selectionFixture(), owned = { vocabulary: null };
    const sampling = { selection: { kind }, sample() { f.calls.push("sample"); return f.ids; } };
    const table = f.select.bindMtpSelection(f.target, sampling, owned);
    expect(Object.keys(table)).toEqual(["1", "3"]);
    expect(Object.isFrozen(table)).toBe(true);
    expect(Reflect.set(table, "16385", () => null)).toBe(false);
    for (const [rows, expected] of [[1, "one"], [3, "three"], [65, "default"], [16_385, "default"]] as const) {
      const operation = rowOperationFor(table, rows);
      expect(rowOperationFor(table, rows)).toBe(operation);
      f.calls.length = 0;
      expect(operation(f.tensor("output"), { length: rows })).toBe(f.ids);
      expect(f.calls).toEqual(kind === "greedy"
        ? [expected, `reshape ${expected}`, "argmax", "dispose flat", `dispose ${expected}`]
        : [expected, `reshape ${expected}`, "logprobs", "sample", "dispose logprobs", "dispose flat", `dispose ${expected}`]);
    }
    expect(owned.vocabulary).toBeNull();
    for (const key of ["0", "1.5", "then", Symbol.iterator]) expect(Reflect.get(table, key)).toBeUndefined();
    const partial = f.select.bindMtpSelection({ logits: { 1: f.target.logits[1] } }, sampling, { vocabulary: null });
    expect(typeof rowOperationFor(partial, 1)).toBe("function");
    expect(() => rowOperationFor(partial, 2)).toThrow("no declared operation for 2 rows");
  });
}

test("listed and prefill-only MTP selection use compact constant positive-row defaults", () => {
  const f = selectionFixture(), owned = { vocabulary: null };
  const table = f.select.bindMtpSelection(f.target, { selection: { kind: "listed", ids: [1, 3, 7],
    sample() { f.calls.push("sample listed"); return f.ids; } } }, owned);
  expect(Object.keys(table)).toEqual([]);
  expect(table[16_385]).toBe(table[1]);
  expect(owned.vocabulary).not.toBeNull();
  expect(table[16_385]!(f.tensor("output"), { length: 16_385 })).toBe(f.ids);
  expect(f.calls).toEqual(["bind listed", "listed", "reshape subset", "sample listed", "dispose flat", "dispose subset"]);
  expect(Object.keys(f.select.prefillOnly)).toEqual([]);
  expect(f.select.prefillOnly[16_385]).toBe(f.select.prefillOnly[1]);
  expect(() => f.select.prefillOnly[16_385]!()).toThrow("Qwen MTP prefill rows do not draft");
});
