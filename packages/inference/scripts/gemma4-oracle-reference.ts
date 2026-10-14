/** Reuse the unchanged 02d723a Gemma oracle calculation in an explicit external
 * Python environment, then pin its capture, artifact and installed source files.
 *
 * bun gemma4-oracle-reference.ts --source /abs/git-checkout --python /abs/venv/bin/python
 *   --model /abs/snapshot --out /abs/empty-reference-directory
 *
 * The original producer's embedded Python runs unchanged. Only its explicit
 * snapshot and output arguments replace the old machine-specific TS paths.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { isAbsolute, join, relative, resolve } from "node:path";
import { parseArgs } from "node:util";

export const ORIGINAL_REVISION = "02d723a2875153196f8c6c10bce2daf6f0044655";
export const ORIGINAL_SCRIPT = "scripts/regen/parity.ts";
export const ORIGINAL_SHA256 = "e8f9ff086ea839362e2e806577e0db07a13972073ccf8e880bdcaf5a34522787";
const sha = (bytes: Uint8Array | string) => createHash("sha256").update(bytes).digest("hex");

export function originalOraclePython(source: string): string {
  if (sha(source) !== ORIGINAL_SHA256) throw new Error("original Gemma producer source differs from its Git pin");
  const match = /const py = `([\s\S]*?)`;/u.exec(source);
  if (!match) throw new Error("original Gemma producer has no embedded Python");
  return match[1]!;
}

export function externalOutput(source: string, out: string): string {
  if (!isAbsolute(out)) throw new Error("--out must be an absolute external directory");
  const path = resolve(out);
  for (const root of [resolve(source),resolve(import.meta.dir,"../../..")]) {
    const rel = relative(root,path);
    if (!rel || (!rel.startsWith("..") && !isAbsolute(rel))) throw new Error("reference output must be outside source checkouts");
  }
  if (existsSync(path) && readdirSync(path).length) throw new Error("reference output must be empty or absent");
  return path;
}

async function produce(argv: string[]): Promise<void> {
  const {values} = parseArgs({args:argv,options:{help:{type:"boolean"},source:{type:"string"},python:{type:"string"},model:{type:"string"},out:{type:"string"}}});
  if (values.help) {
    console.log("bun gemma4-oracle-reference.ts --source <git-checkout> --python <external-venv/bin/python> --model <snapshot> --out <empty-external-dir>");
    return;
  }
  for (const key of ["source","python","model","out"] as const)
    if (!values[key] || !isAbsolute(values[key]!)) throw new Error(`--${key} must be an absolute path`);
  const source = resolve(values.source!), model = resolve(values.model!), out = externalOutput(source,values.out!);
  const original = Bun.spawnSync(["git","-C",source,"show",`${ORIGINAL_REVISION}:${ORIGINAL_SCRIPT}`]);
  if (original.exitCode) throw new Error(`cannot read original Gemma producer: ${original.stderr.toString()}`);
  const python = originalOraclePython(original.stdout.toString());
  mkdirSync(out,{recursive:true});
  const proc = Bun.spawn([values.python!,"-c",python,model,"The capital of France is","100","4",out],
    {env:{...process.env,HF_HUB_OFFLINE:"1"},stdout:"pipe",stderr:"pipe"});
  const [stdout,stderr,code] = await Promise.all([new Response(proc.stdout).text(),new Response(proc.stderr).text(),proc.exited]);
  writeFileSync(join(out,"oracle.stderr.txt"),stderr);
  if (code) throw new Error(`original Python oracle failed (${code}): ${stderr}`);
  const capture = JSON.parse(stdout);
  const captureBytes = JSON.stringify(capture,null,1);
  writeFileSync(join(out,"parity.json"),captureBytes);

  // Metadata inspection uses no MLX imports or GPU work. These are the actual
  // installed package files, not a claimed upstream commit or inferred version.
  const pins = `
import sys,json,hashlib,importlib.metadata
from pathlib import Path
model=Path(sys.argv[1])
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()
files={name:sha(model/name) for name in ['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json']}
index=json.loads((model/'model.safetensors.index.json').read_text())
weights={name:sha(model/name) for name in sorted(set(index['weight_map'].values()))}
packages={}
for name in ['mlx','mlx-lm','mlx-optiq']:
    package=importlib.metadata.distribution(name)
    sources={str(file):sha(Path(package.locate_file(file))) for file in package.files
        if str(file).endswith(('.py','.so','.dylib','.metallib')) and Path(package.locate_file(file)).is_file()}
    if not sources: raise RuntimeError('no installed source files for '+name)
    packages[name]={'version':package.version,'files_sha256':sources}
print(json.dumps({'files':files,'weights':weights,'packages':packages}))
`;
  const inspection = Bun.spawn([values.python!,"-c",pins,model],{stdout:"pipe",stderr:"pipe"});
  const [measured,inspectionError,inspectionCode] = await Promise.all([new Response(inspection.stdout).text(),new Response(inspection.stderr).text(),inspection.exited]);
  if (inspectionCode) throw new Error(`provenance inspection failed (${inspectionCode}): ${inspectionError}`);
  const measuredPins = JSON.parse(measured);
  const wrapperRevision = Bun.spawnSync(["git","-C",resolve(import.meta.dir,"../../.."),"rev-parse","HEAD"]).stdout.toString().trim();
  const manifest = {...capture,oracle:{...capture.oracle,
    model_files_sha256:measuredPins.files,weights_sha256:measuredPins.weights,
    provenance:{producer:{script:ORIGINAL_SCRIPT,script_sha256:ORIGINAL_SHA256,revision:ORIGINAL_REVISION},
      wrapper:{script:"packages/inference/scripts/gemma4-oracle-reference.ts",script_sha256:sha(readFileSync(import.meta.path)),revision:wrapperRevision},
      packages:measuredPins.packages,capture:{"parity.json":sha(captureBytes)}}}};
  writeFileSync(join(out,"gemma4-parity.json"),JSON.stringify(manifest,null,1)+"\n");
  console.log(JSON.stringify({reference:join(out,"gemma4-parity.json"),sha256:sha(readFileSync(join(out,"gemma4-parity.json"))),oracle:manifest.oracle.device,packages:Object.fromEntries(Object.entries(measuredPins.packages).map(([name,entry])=>[name,(entry as {version:string}).version]))}));
}

if (import.meta.main) await produce(process.argv.slice(2));
