// The install step for the Neural Engine: every program the shipped compositions
// load on this machine, compiled once into the system compile cache, the way
// Apple's own pipeline compiles a model for the Neural Engine on the device and
// keeps it. A graph never compiles at load (kernels/ane/linear); it loads what
// this step left in the cache, about 4 ms a program.
//
// The table is the loader's: each graph table entry whose served device is this
// machine's (or any) names the programs it loads for a composition
// (`GraphEntry.anePrograms`), a function of the composition alone, so the step
// needs no downloaded model. Programs identical in MIL and weights are one
// program and compile once. Compiling loads nothing and holds none of the ANE
// driver's clients. The cache is per code-signing identity, so the step runs
// under the executable that serves (the app's `compile-ane` verb).
import type { Composition } from "../contracts/portable/composition";
import { AneProgram, aneProgramKey, type AneCompileStatus, type AneProgramSpec } from "../kernels/ane/linear";
import { machineComposition } from "./composition";
import { MLX_GRAPH_TABLES } from "./factory";
import type { GraphTable } from "./implementation";

/** One distinct program of the install step and the graph entries that load it. */
export interface AneInstallProgram {
  readonly spec: AneProgramSpec;
  readonly graphs: readonly string[];
}

/** An install step program and what compiling it did. */
export interface AneInstallResult extends AneInstallProgram {
  readonly status: AneCompileStatus;
  readonly milliseconds: number;
}

export interface AneInstallOptions {
  /** The record compiled for. Default: {@link machineComposition}, what a load with no flags resolves here. */
  readonly composition?: Composition;
  /** The graph tables read. Default: the loader's ({@link MLX_GRAPH_TABLES}). */
  readonly tables?: Readonly<Record<string, GraphTable<unknown, unknown>>>;
}

/** Every distinct Neural Engine program the entries of `tables` load for `composition`, among the entries whose
 * served device is `composition.device` or unconstrained, in table order. */
export function aneInstallPrograms(options: AneInstallOptions = {}): AneInstallProgram[] {
  const composition = options.composition ?? machineComposition();
  const programs = new Map<string, { spec: AneProgramSpec; graphs: string[] }>();
  for (const table of Object.values(options.tables ?? MLX_GRAPH_TABLES))
    for (const entry of table.entries) {
      if (!entry.anePrograms || (entry.serves.device !== undefined && entry.serves.device !== composition.device)) continue;
      for (const spec of entry.anePrograms(composition)) {
        const key = aneProgramKey(spec), known = programs.get(key);
        if (known) { if (!known.graphs.includes(entry.id)) known.graphs.push(entry.id); }
        else programs.set(key, { spec, graphs: [entry.id] });
      }
    }
  return [...programs.values()];
}

/** The install step: compiles every program {@link aneInstallPrograms} lists into this Mac's Neural Engine
 * compile cache, skipping those already there, and returns what it did for each, in order (`onProgram` hears each
 * as it finishes). Throws `AneBridgeError` when the bridge does not load or the compiler refuses a program. */
export function compileAnePrograms(options: AneInstallOptions & { readonly onProgram?: (result: AneInstallResult) => void } = {}):
  AneInstallResult[] {
  return aneInstallPrograms(options).map(program => {
    const started = performance.now();
    const status = AneProgram.compile(program.spec);
    const result = { ...program, status, milliseconds: performance.now() - started };
    options.onProgram?.(result);
    return result;
  });
}
