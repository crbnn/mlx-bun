import type { GenerateOptions } from "../index";

/** Snapshot the request's configuration when it is bound. The KV scheme is
 * the composition's, shared by every request, so nothing here copies it. */
export function captureSpeculativeOptions(options: GenerateOptions): GenerateOptions {
  return { ...options };
}

/** The prompt-cache namespace of a speculative method's entries: its method
 * namespace and adapters. Execution and persistence consume the same captured
 * settings; the store holding the entries is keyed by the composition's KV
 * scheme. */
export function speculativePrefixNamespace(method: string | undefined, adapters: string | string[],
  _options: GenerateOptions): string {
  return JSON.stringify({ method, adapters });
}
