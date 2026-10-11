import type { ModelConfig } from "../artifacts/config";
import type { Composition, DraftDepth, KvStorageScheme } from "../contracts/portable/composition";
import {
  assertResolvedModelProfile,
  type GenerationLoop,
  type ModelExecutionComposition,
  type ModelGraph,
  type ModelLoader,
  type ResolvedModelProfile,
} from "./profile";

/** Engine code implements this port. Source and Model remain backend-specific;
 * selection introduces no tensor conversion or dispatch in the token loop.
 * create borrows source; a failed construction must release its own resources.
 * `composition` is the record the loader resolved; a graph registry receives a
 * {@link Composition}. A complete implementation that owns its loader takes
 * `void` and composes itself from its source. */
export interface ModelImplementation<Source, Model, Composed = Composition> {
  readonly id: string;
  readonly graph: ModelGraph;
  readonly loader: ModelLoader;
  readonly loop: GenerationLoop;
  create(source: Source, config: ModelConfig, profile: ResolvedModelProfile, composition: Composed): Model;
}

export interface ModelImplementationProvider<Source, Model, Composed = Composition> {
  /** Validate identity and composition, then return one compatible binding.
   * A declared implementation must never silently fall back to another. */
  select(config: ModelConfig, profile: ResolvedModelProfile): ModelImplementation<Source, Model, Composed>;
}

/** The implementation a profile runs: the one it names, else its graph's own.
 * Which of the family's graphs that builds is its {@link GraphTable}'s choice,
 * made from the composition, never the profile's. */
export function implementationIdFor(execution: ModelExecutionComposition): string {
  return execution.implementation ?? execution.graph;
}

/** Immutable composition, with identity matching delegated to model profiles.
 * Exact declarations are requirements: absent/incompatible code is an error,
 * never an invitation to retry with a different numerical implementation. */
export class ModelImplementationRegistry<Source, Model> implements ModelImplementationProvider<Source, Model> {
  private readonly entries: ReadonlyMap<string, ModelImplementation<Source, Model>>;

  constructor(implementations: readonly ModelImplementation<Source, Model>[]) {
    const entries = new Map<string, ModelImplementation<Source, Model>>();
    for (const implementation of implementations) {
      if (!implementation.id.trim()) throw new Error("model implementation id must not be empty");
      if (entries.has(implementation.id))
        throw new Error(`duplicate model implementation ${implementation.id}`);
      entries.set(implementation.id, Object.freeze({
        id: implementation.id,
        graph: implementation.graph,
        loader: implementation.loader,
        loop: implementation.loop,
        create: implementation.create.bind(implementation),
      }));
    }
    this.entries = entries;
  }

  /** Return a new composition; active models retain their existing binding. */
  with(...implementations: readonly ModelImplementation<Source, Model>[]): ModelImplementationRegistry<Source, Model> {
    return new ModelImplementationRegistry([...this.entries.values(), ...implementations]);
  }

  select(config: ModelConfig, profile: ResolvedModelProfile): ModelImplementation<Source, Model> {
    assertResolvedModelProfile(config, profile);
    const execution = profile.profile.execution;
    const id = implementationIdFor(execution);
    const implementation = this.entries.get(id);
    if (!implementation)
      throw new Error(`model profile ${profile.profile.id} requires unavailable implementation ${id}; refusing to fall back`);
    if (implementation.graph !== execution.graph || implementation.loader !== execution.loader ||
        implementation.loop !== execution.loop)
      throw new Error(`model implementation ${id} is incompatible with profile ${profile.profile.id}; refusing to fall back`);
    return implementation;
  }
}

/** The composition a graph was built and qualified for, as its {@link GraphTable}
 * entry states it. Each field present constrains the composition; an absent
 * field accepts every value. */
export interface ServedComposition {
  /** The artifact's fingerprint, as the table's `fingerprint` computes it from the config. */
  readonly fingerprint?: string;
  /** The GPU architecture (`Composition.device`). */
  readonly device?: string;
  readonly aneBridge?: boolean;
  /** The KV schemes served. A composition matches one with the same kind and
   * widths (every layer's, for a per-layer scheme). The delayed conversion start
   * (`quantizedKvStart`) is the cache's own timing and constrains nothing. */
  readonly kv?: readonly KvStorageScheme[];
  readonly draftDepth?: DraftDepth;
  readonly prefillChunkTokens?: number;
  readonly adapters?: boolean;
  /** The most rows per forward the graph was built for: a composition with at most this many matches. */
  readonly maxRows?: number;
}

/** One graph of a family and the composition it serves. `create` borrows the
 * source and releases what it built if construction fails, as
 * {@link ModelImplementation.create} does. */
export interface GraphEntry<Source, Model> {
  /** Names the graph in diagnostics and tests. */
  readonly id: string;
  readonly serves: ServedComposition;
  create(source: Source, config: ModelConfig, composition: Composition): Model;
}

const kvKey = (scheme: KvStorageScheme): string => {
  switch (scheme.kind) {
    case "bf16": return "bf16";
    case "affine": return `affine ${scheme.bits}/${scheme.groupSize}`;
    case "affine-layers":
      return `layers ${[...scheme.layers].sort((a, b) => a.layer - b.layer)
        .map(layer => `${layer.layer}:${layer.bits}/${layer.groupSize}`).join(",")}`;
    case "turbo": return `turbo k${scheme.kBits}v${scheme.vBits}`;
  }
};
const depthKey = (depth: DraftDepth): string => typeof depth === "number" ? `${depth}` : `adaptive ${depth.max}`;
const SERVED_FIELDS: readonly (keyof ServedComposition)[] =
  ["fingerprint", "device", "aneBridge", "kv", "draftDepth", "prefillChunkTokens", "adapters", "maxRows"];
const constraintCount = (serves: ServedComposition): number =>
  SERVED_FIELDS.filter(field => serves[field] !== undefined).length;

/** Whether `composition`, on an artifact with `fingerprint`, meets every constraint of `serves`. */
function meets(serves: ServedComposition, fingerprint: string | undefined, composition: Composition): boolean {
  return (serves.fingerprint === undefined || serves.fingerprint === fingerprint) &&
    (serves.device === undefined || serves.device === composition.device) &&
    (serves.aneBridge === undefined || serves.aneBridge === composition.aneBridge) &&
    (serves.kv === undefined || serves.kv.some(scheme => kvKey(scheme) === kvKey(composition.kv))) &&
    (serves.draftDepth === undefined || depthKey(serves.draftDepth) === depthKey(composition.draftDepth)) &&
    (serves.prefillChunkTokens === undefined || serves.prefillChunkTokens === composition.prefillChunkTokens) &&
    (serves.adapters === undefined || serves.adapters === composition.adapters) &&
    (serves.maxRows === undefined || composition.maxRows <= serves.maxRows);
}

/** Whether one composition could meet both: every field both constrain admits a common value. */
function overlap(a: ServedComposition, b: ServedComposition): boolean {
  const both = <T>(x: T | undefined, y: T | undefined, joint: (x: T, y: T) => boolean) =>
    x === undefined || y === undefined || joint(x, y);
  return both(a.fingerprint, b.fingerprint, (x, y) => x === y) && both(a.device, b.device, (x, y) => x === y) &&
    both(a.aneBridge, b.aneBridge, (x, y) => x === y) &&
    both(a.kv, b.kv, (x, y) => x.some(scheme => y.some(other => kvKey(scheme) === kvKey(other)))) &&
    both(a.draftDepth, b.draftDepth, (x, y) => depthKey(x) === depthKey(y)) &&
    both(a.prefillChunkTokens, b.prefillChunkTokens, (x, y) => x === y) &&
    both(a.adapters, b.adapters, (x, y) => x === y);
}

/**
 * A family's graphs, selected by composition. An entry is a composition its
 * graph was built and qualified for; adding a graph means adding an entry. The
 * family's generic graph is the one entry that constrains nothing.
 *
 * Selection runs once, when the loader builds the graph, and its result is the
 * graph: among the entries whose every constraint the composition meets, the one
 * constraining the most fields wins. The generic entry meets every composition,
 * so the table never refuses; a request the built graph cannot serve is refused
 * before it runs, by the gateway's planning from the graph's declared
 * capabilities. Construction rejects a table without exactly one generic entry,
 * or with two entries of equal constraint count that one composition could both
 * meet, so no winner depends on the order of entries.
 */
export class GraphTable<Source, Model> {
  readonly entries: readonly GraphEntry<Source, Model>[];
  /** The unconstrained entry: it serves what nothing more specific matches, and
   * is how a bench constructs the generic graph explicitly as an A/B control. */
  readonly generic: GraphEntry<Source, Model>;
  readonly #fingerprint: ((config: ModelConfig) => string) | undefined;

  /** `fingerprint` computes the artifact fingerprint the entries name; required when one names it. */
  constructor(entries: readonly GraphEntry<Source, Model>[], fingerprint?: (config: ModelConfig) => string) {
    const generic = entries.filter(entry => constraintCount(entry.serves) === 0);
    if (generic.length !== 1)
      throw new Error(`graph table needs exactly one generic entry (constraining nothing); found ${generic.length}`);
    for (const [index, entry] of entries.entries()) {
      if (entries.findIndex(other => other.id === entry.id) !== index) throw new Error(`graph table: duplicate entry ${entry.id}`);
      if (entry.serves.fingerprint !== undefined && !fingerprint)
        throw new Error(`graph table: ${entry.id} names a fingerprint, and the table computes none`);
      if (entry.serves.kv?.length === 0) throw new Error(`graph table: ${entry.id} serves no KV scheme`);
      for (const other of entries.slice(index + 1))
        if (constraintCount(entry.serves) === constraintCount(other.serves) && overlap(entry.serves, other.serves))
          throw new Error(`graph table: ${entry.id} and ${other.id} constrain as many fields and one composition ` +
            "could meet both; constrain one of them further");
    }
    this.entries = Object.freeze([...entries]);
    this.generic = generic[0]!;
    this.#fingerprint = fingerprint;
  }

  /** The entry serving `composition` for the artifact `config` describes. */
  select(config: ModelConfig, composition: Composition): GraphEntry<Source, Model> {
    return this.match(this.#fingerprint?.(config), composition);
  }

  /** The entry serving `composition` on an artifact with `fingerprint` (as the table computes it). */
  match(fingerprint: string | undefined, composition: Composition): GraphEntry<Source, Model> {
    let chosen = this.generic, count = 0;
    for (const entry of this.entries) {
      const constrained = constraintCount(entry.serves);
      if (constrained > count && meets(entry.serves, fingerprint, composition)) { chosen = entry; count = constrained; }
    }
    return chosen;
  }
}
