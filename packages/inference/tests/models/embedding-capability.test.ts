import { describe, expect, test } from "bun:test";
import type { MlxArray } from "@mlx-bun/mlx/array";
import type { MlxDeclaredGraph } from "../../src/contracts/mlx/graph";
import type { GraphCapabilities, GraphDeclaration } from "../../src/contracts/portable/graph";
import { declareGraph, declaredGraph, graphCapabilitiesOf } from "../../src/models/capabilities";

describe("pooled embedding capability", () => {
  test("operation presence supplies the portable description without invoking it", () => {
    let calls = 0;
    const graph: MlxDeclaredGraph = {
      graphCapabilities: declareGraph({ sparseAttention: true }),
      dispose() {},
      embedPooled(ids) { calls++; return ids; },
    };

    const validated = declaredGraph(graph);
    const capabilities = graphCapabilitiesOf(validated);
    expect(validated).toBe(graph);
    expect(capabilities.embeddings).toBe(true);
    expect(capabilities.sparseAttention).toBe(true);
    expect(Object.isFrozen(capabilities)).toBe(true);
    expect("embeddings" in graph.graphCapabilities).toBe(false);
    expect(calls).toBe(0);
  });

  test("a graph without the operation reports no embedding support", () => {
    const graph: MlxDeclaredGraph = { graphCapabilities: declareGraph(), dispose() {} };
    expect(graphCapabilitiesOf(declaredGraph(graph)).embeddings).toBe(false);
    expect(graphCapabilitiesOf(declaredGraph({ ...graph, embedPooled: undefined })).embeddings).toBe(false);
  });

  test("a prototype operation retains its graph receiver after validation and reporting", () => {
    class PoolingGraph implements MlxDeclaredGraph {
      readonly graphCapabilities = declareGraph();
      calls = 0;
      dispose() {}
      embedPooled(ids: MlxArray): MlxArray { this.calls++; return ids; }
    }
    const graph = new PoolingGraph();
    const validated = declaredGraph(graph);
    expect(Object.hasOwn(graph, "embedPooled")).toBe(false);
    expect(graphCapabilitiesOf(validated).embeddings).toBe(true);
    expect(graph.calls).toBe(0);

    // An opaque identity token is enough to exercise dispatch; no native tensor is created.
    const ids = {} as MlxArray;
    expect(validated.embedPooled?.(ids)).toBe(ids);
    expect(graph.calls).toBe(1);
  });

  test.each([null, true, false])("rejects the untyped operation %j at binding", embedPooled => {
    expect(() => declaredGraph({ graphCapabilities: declareGraph(), embedPooled, dispose() {} }))
      .toThrow("embedPooled operation must be a function");
  });

  test.each([true, false, undefined])("rejects the obsolete declaration field, including %j", embeddings => {
    // Both constructors reject the field itself, even when a JS caller supplies undefined.
    const declaration = { embeddings } as unknown as Parameters<typeof declareGraph>[0];
    expect(() => declareGraph(declaration)).toThrow("not an embeddings flag");
    expect(() => declaredGraph({
      graphCapabilities: { ...declareGraph(), embeddings },
      dispose() {},
      embedPooled: (ids: MlxArray) => ids,
    })).toThrow("not an embeddings flag");
  });
});

// Compiled by the package typecheck; never executed by the CPU behavior tests.
function embeddingDeclarationTypes(report: GraphCapabilities): void {
  // @ts-expect-error Embedding support cannot be supplied independently of its operation.
  declareGraph({ embeddings: true });
  // @ts-expect-error Disabling an existing operation with a separate flag is also invalid.
  declareGraph({ embeddings: false });

  const flagged = { ...declareGraph(), embeddings: true };
  // @ts-expect-error Reject existing variables too, rather than only excess properties on literals.
  declareGraph(flagged);
  // @ts-expect-error The public declaration type rejects a separately authored flag.
  const invalidDeclaration: GraphDeclaration = flagged;
  // @ts-expect-error A portable report is an output, not a graph construction declaration.
  const reportAsDeclaration: GraphDeclaration = report;
  // @ts-expect-error A portable report cannot be passed back as constructor input either.
  declareGraph(report);

  const invalidOperation: MlxDeclaredGraph = {
    graphCapabilities: declareGraph(),
    dispose() {},
    // @ts-expect-error An embedding capability must be callable.
    embedPooled: true,
  };
  void invalidDeclaration;
  void reportAsDeclaration;
  void invalidOperation;
}
void embeddingDeclarationTypes;
