// The generated Gemma 4 graphs' entries in the loader's Gemma 4 table
// (`models/factory.ts`). The graph files come from scripts/gen-gemma4.ts, one per
// composition; these entries are kept by hand. Each serves the composition its
// file states (`COMPOSITION`: the artifact's config fingerprint and the KV scheme
// its caches serve, the artifact's kv_config as `--kv-quant config` composes it)
// and builds its graph from the composition the loader resolved. Any other
// composition gets the generic graph.

import type { ModelConfig } from "../../../artifacts/config";
import type { Weights } from "../../../artifacts/weights";
import type { Composition } from "../../../contracts/portable/composition";
import type { GraphEntry, ServedComposition } from "../../implementation";
import * as g12b from "./gemma4-12b";
import * as ge4b from "./gemma4-e4b";
import * as g26b from "./gemma4-26b";

/** A generated Gemma 4 graph. */
export type GeneratedGemma4Graph = g12b.GeneratedGemma4 | ge4b.GeneratedGemma4 | g26b.GeneratedGemma4;

function generated(id: string, module: {
  readonly COMPOSITION: ServedComposition;
  readonly GeneratedGemma4: new (weights: Weights, config: ModelConfig, composition: Composition) => GeneratedGemma4Graph;
}): GraphEntry<Weights, GeneratedGemma4Graph> {
  return Object.freeze({
    id, serves: module.COMPOSITION,
    create: (weights: Weights, config: ModelConfig, composition: Composition) => new module.GeneratedGemma4(weights, config, composition),
  });
}

/** Each generated graph, named by its file's stem. */
export const GENERATED_GRAPHS: readonly GraphEntry<Weights, GeneratedGemma4Graph>[] = Object.freeze([
  generated("gemma4-12b", g12b),
  generated("gemma4-e4b", ge4b),
  generated("gemma4-26b", g26b),
]);
