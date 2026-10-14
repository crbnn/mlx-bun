import { MlxArray } from "@mlx-bun/mlx/array";
import { Dtype } from "@mlx-bun/mlx/ffi";
import * as ops from "@mlx-bun/mlx/ops";

/** Per-adapter LoRA weights for one linear (mlx-lm LoRALinear shapes:
 *  a [in_features, rank], b [rank, out_features], typically f32). */
export interface LoraWeights {
  a: MlxArray;
  b: MlxArray;
  scale: number;
  rank: number;
}

/** Active-adapter state, shared by every mounted linear of one model.
 *  A plain field, NOT a ContextVar port: under continuous batching requests
 *  interleave, so the field is only valid for the forward being built. Each
 *  execution group binds its rows' adapter list (`bindAdapterContext`) and
 *  sets/restores it around every forward; `adapterScoped` does the same for a
 *  single generation. */
export class LoraState {
  active: string[] = [];
  /** Training-only LoRA-input dropout. `rate` is the drop probability; `seed` is
   *  set per micro-step by the trainer and is CONSTANT across that step's forward
   *  and any recompute (segmented / gradient-checkpoint), so each layer's mask —
   *  keyed by (seed, the linear's dropoutId) — is reproduced exactly in the
   *  backward. `seed === null` (the default, and for inference) disables it. */
  dropoutRate = 0;
  dropoutSeed: number | null = null;
}

/** Inverted LoRA-input dropout, keyed by (seed, id) so it is deterministic —
 *  the same (seed, id, shape) reproduces the mask, which is what makes the
 *  segmented / gradient-checkpoint recompute correct. kept ⇒ x/(1-p),
 *  dropped ⇒ 0 (preserves the expectation). Caller owns the result. */
export function loraInputDropout(x: MlxArray, p: number, seed: number, id: number): MlxArray {
  const key = ops.randomKey(BigInt(seed) * 100003n + BigInt(id));
  const u = ops.randomUniform(x.shape, Dtype.float32, 0, 1, key); // [shape] in [0,1)
  key.dispose();
  const pArr = ops.scalarLike(p, u);
  const keep = ops.less(pArr, u); // u > p ⇒ keep (bool [shape])
  const scaleX = ops.scalarLike(1 / (1 - p), x);
  const zeroX = ops.scalarLike(0, x);
  const mask = ops.where(keep, scaleX, zeroX); // x.dtype
  const xd = ops.mul(x, mask);
  for (const a of [u, pArr, keep, scaleX, zeroX, mask]) a.dispose();
  return xd;
}

/** Adapter delta after any projection. Borrows x and consumes the projection
 * output, returning its owned replacement (or the same handle with no delta).
 * Request routing sets the shared active state before the operation runs. */
export class LoraDelta {
  adapters: Map<string, LoraWeights> | null = null;
  loraState: LoraState | null = null;
  dropoutId = 0;

  forward(x: MlxArray, out: MlxArray): MlxArray {
    // LoRA residual — composition is mlx-lm LoRALinear / optiq apply.py:
    //   y + (scale · ((x @ A) @ B)).astype(x.dtype)
    // (optiq mount.py omits the astype, leaking the f32 residual into the
    // bf16 stream; the
    // cast form is what the adapters were trained behind.)
    const st = this.loraState;
    if (st && st.active.length > 0 && this.adapters && this.adapters.size > 0) {
      // Training-only LoRA-input dropout (PEFT applies dropout to x before A;
      // the base quantized path is untouched). Keyed by (seed, dropoutId) so the
      // backward recompute reproduces the exact mask.
      let xLora = x;
      let xDrop: MlxArray | null = null;
      if (st.dropoutSeed !== null && st.dropoutRate > 0) {
        xDrop = loraInputDropout(x, st.dropoutRate, st.dropoutSeed, this.dropoutId);
        xLora = xDrop;
      }
      for (const id of st.active) {
        const lw = this.adapters.get(id);
        if (!lw) continue;
        const xa = ops.matmul(xLora, lw.a);
        const z = ops.matmul(xa, lw.b);
        xa.dispose();
        const zs = ops.mulScalar(z, lw.scale);
        z.dispose();
        const zc = zs.astype(x.dtype);
        zs.dispose();
        const previous = out; out = ops.add(out, zc); previous.dispose();
        zc.dispose();
      }
      xDrop?.dispose();
    }
    return out;
  }
}
