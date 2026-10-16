---
title: Using the libraries
description: Choose the component you need and start from an executable example.
---

Use the `@mlx-bun/` TypeScript libraries to run inference inside your Bun app.
They require **Bun 1.4.2 or later on an Apple Silicon Mac with macOS 14 or later**;
native libraries ship inside the package archives.

```sh
bun add @mlx-bun/inference
```

Start from the tested
[Qwen generation example](https://github.com/joshuarossi/mlx-bun/blob/main/packages/inference/examples/qwen3-generate.ts)
and its [setup instructions](https://github.com/joshuarossi/mlx-bun/blob/main/packages/inference/README.md#direct-library-use).
The caller owns the model, inputs, and resource lifetime. Public component
subpaths also expose kernels, layers, graphs, state, and generation independently.

## Choose a package

- [MLX bindings](https://github.com/joshuarossi/mlx-bun/blob/main/packages/mlx/README.md): arrays, operations, native resource ownership.
- [Inference](https://github.com/joshuarossi/mlx-bun/blob/main/packages/inference/README.md): graphs, loading, preprocessing, generation, reusable execution, and memory fit estimates.
- [Quantization](https://github.com/joshuarossi/mlx-bun/blob/main/packages/quantize/README.md): checkpoint conversion and allocation.
- [Training](https://github.com/joshuarossi/mlx-bun/blob/main/packages/training/README.md): adapter training and production.
- [Hub](https://github.com/joshuarossi/mlx-bun/blob/main/packages/hub/README.md): model acquisition and discovery.

The [array example](https://github.com/joshuarossi/mlx-bun/blob/main/packages/mlx/examples/arrays.ts)
introduces the MLX bindings. Package READMEs explain how to run the examples;
the linked source is tested in CI.

For an application in another runtime, use the
[local HTTP APIs](/getting-started/quickstart/#call-the-local-api), including
[Jev and OpenAI Decisions](/getting-started/quickstart/#local-decisions).

The [generated library API](/api/) lists every public package subpath, resolved
exports, signatures, and API comments. Start at a package root for high-level
entry points, or choose a component subpath for lower-level composition. Source
links lead to the owning implementation. Ownership and disposal obligations
belong in those API comments; the reference includes only what the code documents.
