import type { Weights } from "../../artifacts/weights";
import type { GraphEntry } from "../implementation";
import * as e4b from "./gemma4-e4b";
import * as twelve from "./gemma4-12b";
import * as moe from "./gemma4-26b";

export type Gemma4Graph = e4b.GeneratedGemma4 | twelve.GeneratedGemma4 | moe.GeneratedGemma4;
export type GeneratedGemma4Graph = Gemma4Graph;

/** Defaults selected by the loader. Callers can construct any exported graph. */
export const GEMMA_GRAPHS: readonly GraphEntry<Weights, Gemma4Graph>[] = Object.freeze([
  { id: "gemma4-e4b", serves: e4b.COMPOSITION, create: (weights, config) => new e4b.GeneratedGemma4(weights, config) },
  { id: "gemma4-12b", serves: twelve.COMPOSITION, create: (weights, config) => new twelve.GeneratedGemma4(weights, config) },
  { id: "gemma4-26b", serves: moe.COMPOSITION, create: (weights, config) => new moe.GeneratedGemma4(weights, config) },
]);
export const GENERATED_GRAPHS = GEMMA_GRAPHS;
export { GeneratedGemma4 as Gemma4E4b } from "./gemma4-e4b";
export { GeneratedGemma4 as Gemma4TwelveB } from "./gemma4-12b";
export { GeneratedGemma4 as Gemma4Moe } from "./gemma4-26b";
export * from "./written-blocks";
