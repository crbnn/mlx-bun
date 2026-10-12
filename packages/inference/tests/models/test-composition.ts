// The composition record of a graph built directly in a test: bf16 KV unless a
// scheme is named, and every other field at its default. It never asks the
// Neural Engine bridge (no program loads); graphs read only the fields their
// family uses.
import { deviceArchitecture } from "@mlx-bun/mlx/ffi";
import { COMPOSITION_DEFAULTS, type AffineKernelChoice, type Composition, type KvStorageScheme } from "../../src/contracts/portable/composition";
import { PREFILL_CHUNK_TOKENS } from "../../src/models/composition";
import { KvScheme } from "../../src/state/kv-scheme";
import { decodesCompiled } from "../../src/state/kv-maintenance";

export function testComposition(kv: KvStorageScheme = { kind: "bf16" }, fields: Partial<Composition> = {}): Composition {
  // Compiled decode resolves as the loader resolves it: off for storage with no compiled form.
  const compiledDecode = (fields.compiledDecode ?? COMPOSITION_DEFAULTS.compiledDecode) &&
    decodesCompiled(kv, (fields.pagedKv ?? null) !== null);
  return Object.freeze({
    device: deviceArchitecture(), aneBridge: false, kv, draftDepth: 0, prefillChunkTokens: PREFILL_CHUNK_TOKENS,
    adapters: false, maxRows: 8, ...COMPOSITION_DEFAULTS, ...fields, compiledDecode,
  });
}

/** The KV flags a request carried before E3. */
export interface KvOptions {
  readonly kvBits?: number;
  readonly kvGroupSize?: number;
  readonly kvConfig?: readonly { readonly layerIdx: number; readonly bits: number; readonly groupSize: number }[];
  readonly quantizedKvStart?: number;
  readonly turboQuant?: { readonly kBits: number; readonly vBits: number };
}

/** The composition's KV scheme for those flags, with the starts they had: a
 * uniform width converts at 5000 tokens unless a start is named, a per-layer
 * config and TurboQuant at 0. Affine reads take `kernels`, else the scheme's
 * default. A test that ran without the app before E3 read affine KV with the
 * tiled-causal set (MLX_BUN_NO_FUSED_SDPA unset); one that compares against
 * those numbers names it. */
export function kvOf(options: KvOptions, kernels?: AffineKernelChoice): KvStorageScheme {
  if (options.turboQuant)
    return new KvScheme("turbo", { turboQuant: { ...options.turboQuant }, quantizedKvStart: options.quantizedKvStart ?? 0 }).storage;
  if (options.kvConfig?.length)
    return new KvScheme("affine-config", { kvConfig: options.kvConfig.map(entry => ({ ...entry })),
      quantizedKvStart: options.quantizedKvStart }, kernels).storage;
  if (options.kvBits !== undefined)
    return new KvScheme("affine-uniform", { kvBits: options.kvBits, kvGroupSize: options.kvGroupSize,
      quantizedKvStart: options.quantizedKvStart }, kernels).storage;
  return { kind: "bf16" };
}
