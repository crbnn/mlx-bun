import { quantFor, type ModelConfig, type QuantSpec } from "../../artifacts/config";

/** Artifact facts of main's committed-span qualification. The loader supplies
 * its device recommendation separately; this reads no tensors or device. */
export function qwen35KnownSpansEligible(config: ModelConfig): boolean {
  const t = config.text;
  if (t.hiddenSize !== 5120 || t.intermediateSize !== 17408 || t.numHiddenLayers !== 64 ||
      t.headDim !== 256 || t.numAttentionHeads !== 24 || t.numKeyValueHeads !== 4)
    return false;
  const qualified = (spec: QuantSpec | null) => spec?.mode === "affine" &&
    [2, 3, 4, 6, 8].includes(spec.bits) && [32, 64, 128].includes(spec.groupSize);
  for (let layer = 0; layer < t.numHiddenLayers; layer++) {
    const prefix = `language_model.model.layers.${layer}`;
    const roles = (layer + 1) % t.fullAttentionInterval === 0
      ? ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
      : ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.out_proj"];
    if (!roles.every(role => qualified(quantFor(config.quantization, `${prefix}.${role}`)))) return false;
    const mlp = ["gate_proj", "up_proj", "down_proj"].map(role => quantFor(config.quantization, `${prefix}.mlp.${role}`));
    if (!mlp.every(spec => spec?.mode === "trellis") && !mlp.every(qualified)) return false;
  }
  return true;
}
