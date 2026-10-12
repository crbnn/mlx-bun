// The app-path digest's hashing and arguments. No model and no native library: the
// measurement itself needs real weights and runs in the gates that consume the tool.
import { expect, test } from "bun:test";
import { resolve } from "node:path";
import { cellFlags, cellLabel, compareDigests, DEFAULT_CELLS, digestRows, parseDigestArgs, parseDigestOutput, REPOSITORY, sha16 } from "../../scripts/digest";

const SCRIPT = resolve(import.meta.dir, "../../scripts/digest.ts");

test("a one-row digest hashes as the Trellis checks did: ids joined by ',', one JSON line per token joined by '\\n'", () => {
  // Reference values from `shasum -a 256`, cut to 16 hex digits.
  expect(sha16("1,2,3")).toBe("8a6ae15122001229");
  const one = digestRows([{ tokens: [1, 2, 3], logprobs: ['{"logprob":-0.5}', "null"] }]);
  expect(one).toEqual({ tokens: "8a6ae15122001229", logprobs: "56b6bfdc4b416989",
    rows: [{ tokens: "8a6ae15122001229", logprobs: "56b6bfdc4b416989", n: 3 }] });
});

test("an R-row digest hashes the rows in order and lists each row's own pair", () => {
  const rows = [{ tokens: [1, 2, 3], logprobs: ['{"logprob":-0.5}', "null"] }, { tokens: [4, 5], logprobs: ["a", "b"] }];
  const four = digestRows(rows);
  expect(four.tokens).toBe("c73246d3b521aa83");
  expect(four.logprobs).toBe(sha16('{"logprob":-0.5}\nnull\n\na\nb'));
  // Row 0 of a multi-row cell compares directly with the one-row cell.
  expect(four.rows[0]).toEqual(digestRows([rows[0]!]).rows[0]!);
  expect(four.rows.map(row => row.n)).toEqual([3, 2]);
  // Order matters: swapping rows changes the cell's hashes.
  expect(digestRows([rows[1]!, rows[0]!]).tokens).not.toBe(four.tokens);
});

test("cells are the saved digest's flag lists; this tool's own options are refused inside a cell", () => {
  expect(cellFlags("bf16")).toEqual({});
  expect(cellFlags("kv-quant=4,fused-sdpa=on")).toEqual({ "kv-quant": "4", "fused-sdpa": "on" });
  expect(cellFlags("kv-quant=turbo:k8v3")).toEqual({ "kv-quant": "turbo:k8v3" });
  expect(() => cellFlags("kv-quant")).toThrow("is not flag=value");
  expect(() => cellFlags("kv-quant=")).toThrow("is not flag=value");
  expect(() => cellFlags("max-tokens=8")).toThrow("--tokens");
  expect(() => cellFlags("kv-quant=4,rows=4")).toThrow("this tool's option");
  expect(DEFAULT_CELLS).toEqual(["bf16", "kv-quant=turbo", "kv-quant=4", "kv-quant=4,fused-sdpa=on", "kv-quant=4,fused-sdpa=off",
    "kv-quant=8", "kv-quant=8,fused-sdpa=on", "kv-quant=8,fused-sdpa=off", "kv-quant=config", "kv-quant=config,fused-sdpa=on",
    "kv-quant=config,fused-sdpa=off"]);
});

test("arguments: defaults, and every cell crossed with each row count and chunk", () => {
  const defaults = parseDigestArgs(["--model", "/models/x"])!;
  expect(defaults).toMatchObject({ root: REPOSITORY, model: "/models/x", tokens: 32 });
  expect(defaults.cells.map(cell => cell.label)).toEqual(DEFAULT_CELLS);
  expect(defaults.cells.every(cell => cell.rows === 1 && cell.prefillChunk === null)).toBe(true);
  const matrix = parseDigestArgs(["--model", "/models/x", "--root", "/trees/main", "--tokens", "16", "--rows", "1,4",
    "--prefill-chunk", "default,64", "bf16", "kv-quant=4"])!;
  expect(matrix.root).toBe("/trees/main");
  expect(matrix.tokens).toBe(16);
  expect(matrix.cells.map(cell => cell.label)).toEqual(["bf16", "bf16,prefill-chunk=64", "bf16,rows=4", "bf16,prefill-chunk=64,rows=4",
    "kv-quant=4", "kv-quant=4,prefill-chunk=64", "kv-quant=4,rows=4", "kv-quant=4,prefill-chunk=64,rows=4"]);
  expect(matrix.cells[3]).toEqual({ spec: "bf16", rows: 4, prefillChunk: 64, label: "bf16,prefill-chunk=64,rows=4" });
  expect(cellLabel("kv-quant=8,fused-sdpa=off", 1, null)).toBe("kv-quant=8,fused-sdpa=off");
  expect(parseDigestArgs(["--help"])).toBeNull();
  expect(() => parseDigestArgs([])).toThrow("--model must be an absolute snapshot directory");
  expect(() => parseDigestArgs(["--model", "models/x"])).toThrow("--model must be an absolute");
  expect(() => parseDigestArgs(["--model", "/m", "--root", "trees/main"])).toThrow("--root must be an absolute path");
  expect(() => parseDigestArgs(["--model", "/m", "--rows", "0"])).toThrow("--rows: 0 is not a positive integer");
  expect(() => parseDigestArgs(["--model", "/m", "--prefill-chunk", "2.5"])).toThrow("--prefill-chunk: 2.5");
  expect(() => parseDigestArgs(["--model", "/m", "--tokens", "x"])).toThrow("--tokens: x");
  expect(() => parseDigestArgs(["--model", "/m", "--kv", "4"])).toThrow();
  expect(() => parseDigestArgs(["--model", "/m", "kv-quant"])).toThrow("is not flag=value");
});

test("compare: two saved outputs cell by cell, other output skipped, an error never identical", () => {
  const baseline = ["loading model…", 'bf16: tokens aaaa logprobs bbbb n=32 | graph=Qwen35Model | "x"',
    'bf16,rows=4: tokens cccc logprobs dddd n=32,32,32,32 | rows a/b c/d e/f g/h | graph=Qwen35Model | "y"',
    "kv-quant=4: ERROR the drafter did not load | graph=?"].join("\n");
  const candidate = ['bf16: tokens aaaa logprobs bbbb n=32 | graph=M1MaxGraph | "x"',
    'bf16,rows=4: tokens cccc logprobs eeee n=32,32,32,32 | rows a/b c/d e/f g/h | graph=M1MaxGraph | "y"',
    'kv-quant=4: tokens ffff logprobs gggg n=32 | graph=M1MaxGraph | "z"', 'kv-quant=8: tokens 1111 logprobs 2222 n=32 | graph=M1MaxGraph | "w"'].join("\n");
  expect(parseDigestOutput(baseline).map(line => [line.label, line.hashes, line.graph])).toEqual([
    ["bf16", "tokens aaaa logprobs bbbb", "Qwen35Model"], ["bf16,rows=4", "tokens cccc logprobs dddd", "Qwen35Model"],
    ["kv-quant=4", "ERROR the drafter did not load", "?"]]);
  const result = compareDigests(baseline, candidate);
  expect(result.identical).toBe(false);
  expect(result.table.split("\n")).toEqual(["| cell | baseline | candidate | candidate graph | |", "|---|---|---|---|---|",
    "| bf16 | tokens aaaa logprobs bbbb | tokens aaaa logprobs bbbb | M1MaxGraph | identical |",
    "| bf16,rows=4 | tokens cccc logprobs dddd | tokens cccc logprobs eeee | M1MaxGraph | **differs** |",
    "| kv-quant=4 | ERROR the drafter did not load | tokens ffff logprobs gggg | M1MaxGraph | **differs** |",
    "| kv-quant=8 | — | tokens 1111 logprobs 2222 | M1MaxGraph | **differs** |"]);
  expect(compareDigests(candidate, candidate).identical).toBe(true);
  expect(compareDigests("", "").identical).toBe(false);
  const error = "kv-quant=4: ERROR boom | graph=?";
  expect(compareDigests(error, error).identical).toBe(false);
});

test("--help documents the cells and the line format and loads no native library", () => {
  const help = Bun.spawnSync([process.execPath, SCRIPT, "--help"], { env: { ...process.env, MLX_BUN_LIBMLXC: "/nonexistent" } });
  expect(help.exitCode).toBe(0);
  const text = help.stdout.toString();
  for (const part of ["--model /abs/snapshot", "--root", "--rows", "--prefill-chunk", "MLX_BUN_RD_PREFILL_CHUNK",
    "CELL: tokens <16 hex> logprobs <16 hex>", "kv-quant=config,fused-sdpa=off"]) expect(text).toContain(part);
});
