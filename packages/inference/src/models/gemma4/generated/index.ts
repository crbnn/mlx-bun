// The generated Gemma 4 graphs (scripts/gen-gemma4.ts), one per composition:
// what each was generated for, its `COMPOSITION` (the artifact's config
// fingerprint and the KV scheme its caches serve), and its class. The loader's
// selection table keys on these. Hand-kept; the graph files are generated.

import type { ModelConfig } from "../../../artifacts/config";
import type { Weights } from "../../../artifacts/weights";
import type { Gemma4Model } from "../model";
import * as g12b from "./gemma4-12b";
import * as ge4b from "./gemma4-e4b";
import * as g26b from "./gemma4-26b";

/** A generated Gemma 4 graph. */
export type GeneratedGemma4Graph = g12b.GeneratedGemma4 | ge4b.GeneratedGemma4 | g26b.GeneratedGemma4;

/** Each generated graph with the composition it was generated for. */
export const GEMMA4_GENERATED_GRAPHS = Object.freeze([
  Object.freeze({ stem: "gemma4-12b", composition: g12b.COMPOSITION, Graph: g12b.GeneratedGemma4 }),
  Object.freeze({ stem: "gemma4-e4b", composition: ge4b.COMPOSITION, Graph: ge4b.GeneratedGemma4 }),
  Object.freeze({ stem: "gemma4-26b", composition: g26b.COMPOSITION, Graph: g26b.GeneratedGemma4 }),
]);

/** @deprecated E1: the factory's current lookup by config fingerprint, deleted
 * when its selection table keys on `GEMMA4_GENERATED_GRAPHS` and its
 * `RuntimeModel` union lists `GeneratedGemma4Graph`. The generated classes extend
 * nothing since D2, so until then this map states the union member the factory
 * returns; every consumer of that union reads only members the generated classes
 * implement (the workspace typechecks with them in the union). */
export const GENERATED: ReadonlyMap<string, new (weights: Weights, config: ModelConfig) => Gemma4Model> = new Map(
  GEMMA4_GENERATED_GRAPHS.map(({ composition, Graph }) =>
    [composition.fingerprint, Graph as unknown as new (weights: Weights, config: ModelConfig) => Gemma4Model]));
