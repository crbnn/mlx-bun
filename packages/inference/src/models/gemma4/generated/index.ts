// The generated Gemma 4 graphs' entries in the loader's Gemma 4 table
// (`models/factory.ts`). The files come from scripts/gen-gemma4.ts; these entries
// are kept by hand. Each names the composition its file was generated for: the
// config fingerprint and the per-layer affine KV of the artifact's kv_config.json
// (`--kv-quant config`), the literals its cache-signature guard checks. Any other
// composition gets the generic graph.

import type { ModelConfig } from "../../../artifacts/config";
import type { Weights } from "../../../artifacts/weights";
import type { KvStorageScheme } from "../../../contracts/portable/composition";
import type { GraphEntry } from "../../implementation";
import type { Gemma4Model } from "../model";
import * as g12b from "./gemma4-12b";
import * as ge4b from "./gemma4-e4b";
import * as g26b from "./gemma4-26b";

/** Affine KV at group 64, one width per cache in cache order. */
function perLayerKv(bits: readonly number[]): KvStorageScheme {
  return Object.freeze({ kind: "affine-layers",
    layers: Object.freeze(bits.map((width, layer) => Object.freeze({ layer, bits: width, groupSize: 64 }))) });
}

function generated(id: string, module: { readonly FINGERPRINT: string; readonly GeneratedGemma4: new (weights: Weights, config: ModelConfig) => Gemma4Model },
  kvBits: readonly number[]): GraphEntry<Weights, Gemma4Model> {
  return Object.freeze({
    id, serves: Object.freeze({ fingerprint: module.FINGERPRINT, kv: Object.freeze([perLayerKv(kvBits)]) }),
    create: (weights: Weights, config: ModelConfig) => new module.GeneratedGemma4(weights, config),
  });
}

export const GENERATED_GRAPHS: readonly GraphEntry<Weights, Gemma4Model>[] = Object.freeze([
  generated("gemma4-12b", g12b, [
    4, 4, 4, 8, 4, 4, 8, 8, 8, 4, 8, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  ]),
  generated("gemma4-e4b", ge4b, [4, 4, 4, 8, 8, 4, 4, 4, 4, 8, 8, 8, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4]),
  generated("gemma4-26b", g26b, [8, 4, 4, 8, 8, 8, 4, 4, 8, 8, 4, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]),
]);
