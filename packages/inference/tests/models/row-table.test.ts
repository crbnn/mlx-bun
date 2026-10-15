import { expect, test } from "bun:test";
import { rowOperationFor, rowTableWithDefault } from "../../src/models/row-table";

const odd=()=>"odd",even=()=>"even",wide=()=>"wide";

test("literal exact rows preserve odd/even operation identity and a constant unbounded default",()=>{
  const table=rowTableWithDefault({1:odd,2:even,3:odd,4:even,5:odd,6:even,7:odd,8:even,256:even},wide);
  expect([1,2,3,4,5,6,7,8].map(row=>table[row])).toEqual([odd,even,odd,even,odd,even,odd,even]);
  expect(table[1]).toBe(odd);expect(table[256]).toBe(even);
  for(const rows of [9,255,257,1_000_000,1_000_000_000_000,Number.MAX_SAFE_INTEGER]) expect(rowOperationFor(table, rows)).toBe(wide);
  expect(Object.isFrozen(table)).toBe(true);
  expect(Reflect.set(table,"1",wide)).toBe(false);
  expect(Reflect.set(table,"257",odd)).toBe(false);
  expect(table[1]).toBe(odd);expect(rowOperationFor(table, 257)).toBe(wide);
  expect(Object.keys(table)).toEqual(["1","2","3","4","5","6","7","8","256","default"]);
});

test("metadata, promise probes, symbols and nonpositive or fractional rows have no operation",()=>{
  const table=rowTableWithDefault({1:odd},wide);
  for(const key of ["0","-1","1.5","01","NaN","Infinity","then","missing",Symbol.iterator])
    expect(Reflect.get(table,key)).toBeUndefined();
  expect(rowOperationFor(table,257)).toBe(wide);
  expect(()=>rowOperationFor(table,0)).toThrow("no declared operation for 0 rows");
});

test("whole-context ranges remain available beyond any serving table's finite entries",()=>{
  const ranges=[{from:1,to:1,operation:odd},{from:2,to:3,operation:even},{from:4,to:Infinity,operation:wide}];
  expect(rowOperationFor(ranges,1)).toBe(odd);expect(rowOperationFor(ranges,3)).toBe(even);
  expect(rowOperationFor(ranges,4)).toBe(wide);expect(rowOperationFor(ranges,262144)).toBe(wide);
  expect(()=>rowOperationFor(ranges,0)).toThrow("no declared operation for 0 rows");
});
