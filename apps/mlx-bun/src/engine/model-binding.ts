import type { MlxGatewayBinding } from "@mlx-bun/inference/execution";
import type { embedMany } from "@mlx-bun/inference/embeddings";

/** Numerical services supplied by one loaded model. HTTP requests and prompt
 * policy are not part of this boundary; replacement models supply it directly.
 * Saved state restores through the cache service, into the caches the loaded
 * graph composes. */
export interface ModelBinding {
  readonly stateCompatibility: string;
  readonly gateway: MlxGatewayBinding;
  signal(promptIds: number[], bins: number, minimum: number): Promise<{ bins: number[]; vocab: number }>;
  diagnostics(): Record<string, unknown>;
  readonly discovery: { readonly adapters: boolean; readonly training: boolean;
    readonly dsa: unknown; readonly embeddings: boolean };
  embed?(inputs: string[], instruction?: string): ReturnType<typeof embedMany>;
}
