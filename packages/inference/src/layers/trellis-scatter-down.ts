import type { MlxArray } from "@mlx-bun/mlx/array";
import * as ops from "@mlx-bun/mlx/ops";
import { balancedScatterRow, balancedScatterRows, scatterRow, scatterRows } from "../kernels/trellis/scatter";
import type { ScatterRepresentation } from "../kernels/trellis/qualified";
import type { TrellisLinear } from "./trellis-linear";

// Packed down-projection layers over a Qwen MLP's axis-0 Trellis codes on the
// scatter kernels (lane per code word, split-K partials folded by one sum;
// kernels/trellis/index.ts). Each layer runs one kernel for one row form and one
// code layout (row-major 2- or 4-bit codes, or the 3-bit block interleave, which
// the balanced kernels decode) and takes everything that kernel varies on at
// construction: the projection's decode variant, the single-row representations
// its GPU family qualifies and whether its shared-row codebook was composed
// (`TrellisLinear.useSharedScatterCodebook`). forward(x) keeps x's leading dimensions.

function project(lin: TrellisLinear, x: MlxArray, kernel: (x2: MlxArray) => MlxArray): MlxArray {
  const g = lin.geometry, lead = x.shape.slice(0, -1);
  using x2 = ops.reshape(x, [lead.reduce((a, b) => a * b, 1), g.inFeatures]);
  using y = kernel(x2);
  return lin.delta.forward(x, ops.reshape(y, [...lead, g.outFeatures]));
}
function rowMajor(name: string, lin: TrellisLinear): void {
  const g = lin.geometry;
  if (g.axis !== 0 || g.blockInterleave || lin.codes.ndim !== 2 || (g.k === 3 && g.T === 256 && g.L <= 12))
    throw new Error(`${name}: needs packed row-major axis-0 codes the balanced kernels do not decode (3-bit codes take the K3i layers)`);
}
function k3Interleaved(name: string, lin: TrellisLinear): void {
  const g = lin.geometry;
  if (g.axis !== 0 || g.blockInterleave !== 2 || g.k !== 3 || g.T !== 256 || g.L !== 12)
    throw new Error(`${name}: needs 3-bit block-interleaved axis-0 codes`);
}

/** One activation row; row-major codes. */
export class ScatterDownRow {
  readonly #variant: number;
  readonly #representation: ScatterRepresentation;

  constructor(readonly down: TrellisLinear) {
    rowMajor("ScatterDownRow", down);
    this.#variant = down.variant;
    this.#representation = down.representations.scatter;
  }

  forward(x: MlxArray): MlxArray {
    const { codes, scales, geometry } = this.down;
    return project(this.down, x, x2 => scatterRow(x2, codes, scales, geometry, this.#variant, this.#representation));
  }
}

/** 2..4 activation rows sharing each decoded weight; row-major codes. */
export class ScatterDownRows {
  readonly #variant: number;

  constructor(readonly down: TrellisLinear) {
    rowMajor("ScatterDownRows", down);
    this.#variant = down.variant;
  }

  forward(x: MlxArray): MlxArray {
    const { codes, scales, geometry } = this.down;
    return project(this.down, x, x2 => scatterRows(x2, codes, scales, geometry, this.#variant));
  }
}

/** One activation row; 3-bit block-interleaved codes. */
export class ScatterDownK3iRow {
  readonly #variant: number;
  readonly #representation: ScatterRepresentation;

  constructor(readonly down: TrellisLinear) {
    k3Interleaved("ScatterDownK3iRow", down);
    this.#variant = down.variant;
    this.#representation = down.representations.scatter;
  }

  forward(x: MlxArray): MlxArray {
    const { codes, scales, geometry } = this.down;
    return project(this.down, x, x2 => balancedScatterRow(x2, codes, scales, geometry, this.#variant, this.#representation));
  }
}

/** 2..4 activation rows sharing each decoded weight; 3-bit block-interleaved codes, with the shared-row codebook at
 *  three and four rows when the projection composed it. */
export class ScatterDownK3iRows {
  readonly #variant: number;
  readonly #sharedCodebook: boolean;

  constructor(readonly down: TrellisLinear) {
    k3Interleaved("ScatterDownK3iRows", down);
    this.#variant = down.variant;
    this.#sharedCodebook = down.useSharedScatterCodebook;
  }

  forward(x: MlxArray): MlxArray {
    const { codes, scales, geometry } = this.down;
    return project(this.down, x, x2 => balancedScatterRows(x2, codes, scales, geometry, this.#variant, this.#sharedCodebook));
  }
}
