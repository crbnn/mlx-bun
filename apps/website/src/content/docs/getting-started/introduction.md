---
title: Introduction
description: What mlx-bun is, why it runs locally, and where to start.
---

mlx-bun is local AI for Apple Silicon Macs: an MLX inference engine, an
OpenAI and Anthropic compatible server, a browser app, and TypeScript
libraries you can embed in your own Bun applications. Model execution runs on
the Mac's GPU through MLX and Metal; the Neural Engine programs a model uses
are compiled by the installer, or before the first model load that needs them.
Models download from Hugging Face; inference needs no Python service and no
hosted-model API key.

## What you get

- **The app.** `mlx-bun` downloads models, serves them, generates once,
  embeds, quantizes, fine-tunes and transcribes. The
  [CLI reference](/reference/cli/) lists every command and flag.
- **The server.** Chat and text completions, Anthropic Messages, OpenAI
  Responses, embeddings, audio transcription and the Decisions API on
  `localhost:8080`, over one continuous batching scheduler. The
  [HTTP API reference](/reference/server-api/) lists every route.
- **The browser app.** Chat, model management, transcription, quantization,
  training, dataset generation, metrics and benchmarks, served by the same
  process.
- **The libraries.** The `@mlx-bun/` packages offer convenient loading and
  generation and the lower-level kernels, layers, graphs, state, sampling,
  quantization and training they compose. A library never selects a global
  model or starts a server for you.

## Why mlx-bun

Validated configurations are bit-exact with mlx-lm, and with mlx-optiq for
mixed-precision KV; the [correctness page](/about/correctness/) states the
contract, its evidence and what it does not cover. Connect existing OpenAI and
Anthropic clients through a local base URL for the supported interfaces.
Downloaded models stay in the Hugging Face cache on your Mac, and the app's
own data lives under `~/.mlx-bun`.

Use **Jev or OpenAI Decisions-compatible endpoints with any supported local
language model**. `POST /v1/decisions` and `POST /v1/systemone` classify and score
with the same loaded weights as generation; you don't need a separate decision
model. See [local decisions](/getting-started/quickstart/#local-decisions) for
the SDK base URLs, supported questions and limits.

## Where next

The engine requires an Apple Silicon Mac running macOS 14 or later. Models must
be supported by mlx-bun; a Hugging Face repository alone does not guarantee
compatibility. Available memory, the model, and context length determine what
fits. Model downloads require internet access; inference runs locally once the
model is available. Image questions require a supported vision model.

Start with [installation](/getting-started/installation/) and the
[quickstart](/getting-started/quickstart/), or go straight to the
[library guide](/guides/library/).
