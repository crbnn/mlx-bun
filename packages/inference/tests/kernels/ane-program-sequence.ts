// Child process of ane-program.test.ts, which kills it when it does not finish in time.
// ANE programs of different shapes created, run and disposed in sequence in one process,
// in the pattern of D1's merge gate (2026-10-11): the Qwen3.8-27B M4 Pro graph's Neural
// Engine layers at the artifact's widths and 2048 rows, then more graphs' layers at
// synthetic widths and several chunk sizes, all still held, then a lone synthetic program.
// Each graph's layers share programs as Qwen38TrellisM4Pro builds them: each MLP slice
// shares the previous layer's, each projection the previous one of its shape. Every
// program runs once. 13 graphs are 2288 layer programs, more than twice the ANE driver's
// direct-path clients on the whole machine (about a thousand on an M4 Pro), which a loaded
// program holds each: one per layer exhausts them, one per shape per graph holds 39. Each
// distinct program is compiled first (the install step, which holds no client), so every
// creation only loads.
import { AneProgram, mlpSliceProgram, streamedMatmulProgram, type AneProgramSpec } from "../../src/kernels/ane/linear";

const LAYERS = 64, FULL_ATTENTION_INTERVAL = 4;
const started = performance.now();
const progress = (text: string) => console.error(`[${Math.round(performance.now() - started)} ms] ${text}`);

function run(program: AneProgram): void { program.input(0); program.evalAsync(); program.wait(); }

/** One graph's layer programs: per layer an MLP slice, and two projections (in_proj_z and
 *  out_proj) in DeltaNet layers or one (o_proj) in every fourth, full-attention layer. */
function graph(slice: AneProgramSpec, z: AneProgramSpec, out: AneProgramSpec): AneProgram[] {
  for (const spec of [slice, z, out]) progress(`${spec.label}: ${AneProgram.compile(spec)}`);
  const programs: AneProgram[] = [];
  let mlpShare: AneProgram | undefined, zShare: AneProgram | undefined, outShare: AneProgram | undefined;
  for (let layer = 0; layer < LAYERS; layer++) {
    programs.push(mlpShare = AneProgram.create(slice, mlpShare));
    if ((layer + 1) % FULL_ATTENTION_INTERVAL) programs.push(zShare = AneProgram.create(z, zShare));
    programs.push(outShare = AneProgram.create(out, outShare));
  }
  for (const program of programs) run(program);
  return programs;
}

const held: AneProgram[] = [];
// The artifact: hidden 5120, intermediate 17408 (ANE channels 9600), DeltaNet value width 6144.
held.push(...graph(mlpSliceProgram(5120, 9600, 2048), streamedMatmulProgram(5120, 3392, 2048), streamedMatmulProgram(6144, 2816, 2048)));
progress(`artifact-width graph built and run (${held.length} layer programs held)`);
for (const rows of [2048, 1024, 512, 256])
  for (let copy = 0; copy < 3; copy++) {
    held.push(...graph(mlpSliceProgram(512, 128, rows), streamedMatmulProgram(512, 320, rows), streamedMatmulProgram(512, 192, rows)));
    progress(`synthetic graph at ${rows} rows built and run (${held.length} layer programs held)`);
  }
const lone = AneProgram.create(mlpSliceProgram(512, 128, 2048));
run(lone);
lone.dispose();
progress("lone synthetic program created and run");
for (const program of held.splice(0).reverse()) program.dispose();
progress("disposed");
console.log("done");
