/** Describes an implementation's ABI, not a claim of oracle parity or speed. */
export interface GraphDescriptor {
  readonly id: string;
  readonly backend: string;
  readonly graphAbi: string;
  readonly stateAbi: string;
  readonly artifact: string;
}

/** How prepared media reaches the graph.
 * - `embeddings`: spliced prompt embeddings (image/audio soft tokens, optional
 *   bidirectional and multimodal masks) enter the ordinary forward.
 * - `embeddings+positions`: spliced embeddings with request-owned multi-axis
 *   rotary positions and a decode delta.
 * - `pixels`: raw image pixels are the denoising method's own prefill input. */
export type GraphMediaInput = "embeddings" | "embeddings+positions" | "pixels";

/** Facts a graph declares once, when it is constructed. Composition binds the
 * operations these facts require and refuses what is genuinely missing;
 * scheduling reads only these declarations plus request and dynamic state
 * (batch membership, cache contents, cancellation), never a model's class or
 * name. A declaration that promises an operation is checked where it is bound. */
export interface GraphCapabilities {
  /** The method the graph's forward serves. Token-level methods (speculation,
   * grammar proposals, fill) bind only to `autoregressive`. */
  readonly method: "autoregressive" | "denoising";
  /** Prepared media the graph accepts, or null. Requests carrying prepared
   * embeddings need a `bindMediaInput` operation; `video` says video frames
   * are accepted input. */
  readonly media: { readonly input: GraphMediaInput; readonly video: boolean } | null;
  /** The graph owns a compiled decode step. The composition keeps it on only
   * where its storage has a compiled form (`Composition.compiledDecode`). */
  readonly compiledDecode: boolean;
  /** Attention selects a sparse subset of positions (surfaced to clients). */
  readonly sparseAttention: boolean;
  /** The graph exposes a pooled text-embedding path. */
  readonly embeddings: boolean;
  /** Hidden-layer outputs can be tapped for drafts that read them. */
  readonly hiddenLayerTaps: boolean;
  /** A draft head the graph's own checkpoint carries and has loaded, or null.
   * `kind` names the native draft provider that binds to this graph; the
   * provider registry resolves it, so no consumer names the graph or its
   * provider. `numDraftTokens` is the head's round width. */
  readonly nativeDraft: { readonly kind: string; readonly numDraftTokens: number } | null;
  /** How prefill is sized for this graph. */
  readonly prefill: {
    /** Long prompts prefill in smaller chunks so the attention score matrices
     * a fallback (unfused) attention materializes stay within a fixed
     * workspace. It changes work size, never admission. */
    readonly boundedWorkspace: boolean;
  };
  /** What the graph's target verification qualifies beyond ordinary rounds.
   * Each is a property of the graph's verifier, independent of the draft that
   * supplies proposals and of how the composition's caches store keys and
   * values (every cache lego holds the speculative transaction). */
  readonly speculation: {
    /** Verification under mounted adapters. */
    readonly adapters: boolean;
    /** Verification of requests that return logprobs. */
    readonly logprobs: boolean;
    /** Verification of externally supplied tokens (echo of a known continuation). */
    readonly externalTokens: boolean;
    /** Verified grammar-constraint proposals. */
    readonly grammarProposals: boolean;
  };
}

/** A fused InferenceMethod can bypass this graph interface entirely. It need
 * only satisfy the method's output, cancellation, and ownership contracts. */
export interface GraphFactory<Artifact, Binding, Graph> {
  open(artifact: Artifact, binding: Binding): Promise<{ graph: Graph; close(): Promise<void> }>;
}
