// The composition record of a graph built directly in a test: bf16 KV unless a
// scheme is named, and every other field at its default. It never asks the
// Neural Engine bridge (no program loads); graphs read only the fields their
// family uses.
import { deviceArchitecture } from "@mlx-bun/mlx/ffi";
import { COMPOSITION_DEFAULTS, type AffineKernelChoice, type Composition, type KvStorageScheme } from "../../src/contracts/portable/composition";
import { PREFILL_CHUNK_TOKENS } from "../../src/models/composition";
import { KvScheme } from "../../src/state/kv-scheme";

export function testComposition(kv: KvStorageScheme = { kind: "bf16" }, fields: Partial<Composition> = {}): Composition {
  return Object.freeze({
    device: deviceArchitecture(), aneBridge: false, kv, draftDepth: 0, prefillChunkTokens: PREFILL_CHUNK_TOKENS,
    adapters: false, maxRows: 8, ...COMPOSITION_DEFAULTS, ...fields,
  });
}

/** A test's KV scheme in the options `KvScheme` takes (what `--kv-quant` and
 * `--quantized-kv-start` resolve to). */
export interface KvOptions {
  readonly kvBits?: number;
  readonly kvGroupSize?: number;
  readonly kvConfig?: readonly { readonly layerIdx: number; readonly bits: number; readonly groupSize: number }[];
  readonly quantizedKvStart?: number;
  readonly turboQuant?: { readonly kBits: number; readonly vBits: number };
}

/** The composition's KV scheme for those options, with the schemes' default
 * starts: a uniform width converts at 5000 tokens unless a start is named, a
 * per-layer config and TurboQuant at 0. Affine reads take `kernels`, else the
 * scheme's default; a test that compares against the tiled-causal set names it. */
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
