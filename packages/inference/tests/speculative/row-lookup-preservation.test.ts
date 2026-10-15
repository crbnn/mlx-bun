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
    expect(rowOperationFor(f.model.projectContext, rows)).toBe(f.model.projectContext.default!);
    expect(rowOperationFor(f.model.projectContextKVRows, rows)).toBe(f.model.projectContextKVRows.default!);
  }
  expect(f.calls).toEqual(Array.from({ length: 5 }, () => [["project", input], ["kv", f.projected, positions]]).flat());
  for (const table of [f.model.projectContext, f.model.projectContextKVRows]) {
    expect(Object.keys(table)).toEqual(["default"]);
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
    expect(Object.keys(table)).toEqual(["1", "3", "default"]);
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
  expect(Object.keys(table)).toEqual(["default"]);
  expect(rowOperationFor(table,16_385)).toBe(table.default!);
  expect(owned.vocabulary).not.toBeNull();
  expect(rowOperationFor(table,16_385)(f.tensor("output"), { length: 16_385 })).toBe(f.ids);
  expect(f.calls).toEqual(["bind listed", "listed", "reshape subset", "sample listed", "dispose flat", "dispose subset"]);
  expect(Object.keys(f.select.prefillOnly)).toEqual(["default"]);
  expect(rowOperationFor(f.select.prefillOnly,16_385)).toBe(f.select.prefillOnly.default!);
  expect(() => rowOperationFor(f.select.prefillOnly,16_385)()).toThrow("Qwen MTP prefill rows do not draft");
});

function dflashBindingFixture(missingDefault = false) {
  const text = source("../../src/models/speculative/dflash2.ts");
  const declaration = text.slice(text.indexOf("  #bindOperations("), text.indexOf("\n  #contextKV("));
  const calls: string[] = [];
  const projection = (name: string) => ({ name, forward(input: unknown) {
    calls.push(name);
    return { input, [Symbol.dispose]() { calls.push(`dispose ${name}`); } };
  } });
  const one = projection("one"), fallback = projection("default"), checkpoint = projection("checkpoint");
  const affine = { at: Object.freeze(missingDefault ? {1:one} : {1:one,default:fallback}) };
  const original = { at: [{from:1,to:Infinity,operation:checkpoint}], defaultProjection: checkpoint };
  const fc = affine, layers = [{q:affine,k:original,v:affine,o:affine,gate:affine,up:affine,down:affine}];
  const output = {}, context = [{}];
  const code = `class Bindings {
    #fc; #layers; #hiddenNorm={};
    constructor(fc,layers) {this.#fc=fc;this.#layers=layers;}
    #rms() {return output;}
    #contextKV(_input,_position,fixed) {calls.push(fixed[0].q.name+" "+fixed[0].k.name);return context;}
    #block(_context,_projection,_anchors,_position,_columns,fixed) {calls.push(fixed[0].gate.name);return output;}
    bind(rows) {this.#bindOperations(rows);}
    ${declaration}
  }
  return new Bindings(fc,layers);`;
  const model = new Function("rowOperationFor","mapOperations","fc","layers","calls","output","context",
    transpiler.transformSync(code))(rowOperationFor,mapOperations,fc,layers,calls,output,context) as {
      bind(rows: MlxRowTable<unknown> | readonly {from:number;to:number;operation:unknown}[]):void;
      projectContext: MlxRowTable<(input:unknown)=>unknown>;
      projectContextKVRows: MlxRowTable<(input:unknown,position:unknown)=>unknown>;
      draft: MlxRowTable<(...args:unknown[])=>unknown>;
    };
  return {model,calls,output,context};
}

test("DFlash binds authored exact and default component forms without inventing a row count",()=>{
  const f=dflashBindingFixture();
  f.model.bind({1:true,default:true});
  expect(Object.keys(f.model.projectContext)).toEqual(["1","default"]);
  expect(Object.isFrozen(f.model.projectContext)).toBe(true);
  const input={},position={};
  for(const rows of [1,65,262144]) {
    expect(rowOperationFor(f.model.projectContext,rows)(input)).toBe(f.output);
    expect(rowOperationFor(f.model.projectContextKVRows,rows)(input,position)).toBe(f.context);
    expect(rowOperationFor(f.model.draft,rows)(null,null,[],position,4)).toBe(f.output);
  }
  expect(f.calls).toEqual(["one","dispose one","one checkpoint","one",
    "default","dispose default","default checkpoint","default",
    "default","dispose default","default checkpoint","default"]);
  expect(rowOperationFor(f.model.projectContext,65)).toBe(f.model.projectContext.default!);
  expect(f.model.projectContext[65]).toBeUndefined();
});

test("DFlash refuses a missing component default at binding and never adds a default to exact-only declarations",()=>{
  const incomplete=dflashBindingFixture(true);
  expect(()=>incomplete.model.bind({1:true,default:true})).toThrow("DFlash2 projection has no declared default operation");
  const exact=dflashBindingFixture(true);
  exact.model.bind({1:true});
  expect(Object.keys(exact.model.projectContext)).toEqual(["1"]);
  expect(rowOperationFor(exact.model.projectContext,1)({})).toBe(exact.output);
  expect(()=>rowOperationFor(exact.model.projectContext,65)).toThrow("no declared operation for 65 rows");
});
