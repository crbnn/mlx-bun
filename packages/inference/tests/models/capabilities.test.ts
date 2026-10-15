import { describe, expect, test } from "bun:test";
import { declareGraph, declaredGraph, ORDINARY_VERIFICATION } from "../../src/models/capabilities";
import { testComposition } from "./test-composition";
import { Gemma4Model } from "../../src/models/gemma4/model";
import { UniversalDenseModel } from "../../src/models/universal/dense";
import { DiffusionGemmaModel } from "../../src/models/diffusion-gemma/model";
import { Glm52Model } from "../../src/models/glm52/model";
import { MiniCPM5Model } from "../../src/models/minicpm5/model";
import { Qwen3Model } from "../../src/models/qwen/qwen3";
import { Qwen3MoeModel } from "../../src/models/qwen/qwen3-moe";
import { Qwen35Model } from "../../src/models/qwen/qwen3_5";

// Declarations are read from graphs built without weights: what a graph declares
// follows from its own construction facts, so prototype stand-ins with those facts suffice.
const graph = (prototype: object, facts: object = {}) => declaredGraph(Object.assign(Object.create(prototype), facts));
const plainCaps = declareGraph();

describe("every graph declares its capabilities", () => {
  test("Gemma4: prepared image/audio embeddings, taps, compiled decode except through experts", () => {
    // Taps are every phase's capture parameter, so the graph always declares them.
    const text = (enableMoeBlock: boolean, modelType = "gemma4") => ({ config: { modelType, text: { enableMoeBlock } } });
    expect(graph(Gemma4Model.prototype, text(false)).graphCapabilities).toEqual(declareGraph({
      media: { input: "embeddings", video: false }, hiddenLayerTaps: true, compiledDecode: true }));
    expect(graph(Gemma4Model.prototype, text(true)).graphCapabilities.compiledDecode).toBe(false);
    expect(graph(Gemma4Model.prototype, text(false, "gemma3n")).graphCapabilities.compiledDecode).toBe(false);
  });

  test("Qwen3.5: prepared embeddings with positions, video, bounded prefill workspace", () => {
    // The written M1 Max Trellis graph declares the same (tests/models/qwen38-m1max-identity.test.ts).
    expect(graph(Qwen35Model.prototype, { composition: testComposition() }).graphCapabilities).toEqual(declareGraph({
      media: { input: "embeddings+positions", video: true }, hiddenLayerTaps: true, prefill: { boundedWorkspace: true } }));
  });

  test("ordinary autoregressive graphs declare the defaults: every method over every cache lego", () => {
    expect(graph(MiniCPM5Model.prototype).graphCapabilities).toEqual(plainCaps);
    expect(graph(Qwen3MoeModel.prototype).graphCapabilities).toEqual(plainCaps);
    expect(graph(Qwen3Model.prototype).graphCapabilities).toEqual(declareGraph({ embeddings: true }));
  });

  test("a universal graph reads its own attention: softcapped layers verify ordinary rounds only", () => {
    expect(graph(UniversalDenseModel.prototype, { encodedKvAttention: true }).graphCapabilities).toEqual(plainCaps);
    expect(graph(UniversalDenseModel.prototype, { encodedKvAttention: false }).graphCapabilities)
      .toEqual(declareGraph({ speculation: ORDINARY_VERIFICATION }));
    // Before the attention facts are bound, neither is declared.
    expect(graph(UniversalDenseModel.prototype).graphCapabilities).toEqual(plainCaps);
  });

  test("DiffusionGemma denoises canvases and takes pixels; GLM streams experts, mounts no adapters, reports sparse attention", () => {
    expect(graph(DiffusionGemmaModel.prototype).graphCapabilities)
      .toEqual(declareGraph({ method: "denoising", media: { input: "pixels", video: false } }));
    expect(graph(Glm52Model.prototype, { capabilities: { dsa: true } }).graphCapabilities)
      .toEqual(declareGraph({ sparseAttention: true }));
    expect(graph(Glm52Model.prototype, { capabilities: { dsa: false } }).graphCapabilities.sparseAttention).toBe(false);
  });
});

describe("a graph's operations match what it declares", () => {
  test("a graph that declares nothing is refused, never read as having none", () => {
    expect(() => declaredGraph({})).toThrow("must declare its capabilities");
  });

  test("declared media input needs its operation", () => {
    expect(() => declaredGraph({ graphCapabilities: declareGraph({ media: { input: "embeddings", video: false } }) }))
      .toThrow("provides no bindMediaInput");
    expect(() => declaredGraph({ graphCapabilities: declareGraph({ media: { input: "pixels", video: false } }) }))
      .toThrow("provides no pixelInput");
    expect(declaredGraph({ graphCapabilities: declareGraph({ media: { input: "embeddings", video: false } }),
      bindMediaInput: () => ({}) }).graphCapabilities.media?.input).toBe("embeddings");
  });

  test("the class that declares media supplies the operations that serve it", () => {
    for (const [prototype, input] of [[Gemma4Model.prototype, "embeddings"], [Qwen35Model.prototype, "embeddings+positions"]] as const) {
      expect(typeof (prototype as { bindMediaInput?: unknown }).bindMediaInput).toBe("function");
      expect(graph(prototype, { composition: testComposition() }).graphCapabilities.media?.input).toBe(input);
    }
    expect(typeof (DiffusionGemmaModel.prototype as { pixelInput?: unknown }).pixelInput).toBe("function");
  });
});
