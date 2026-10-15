---
title: Quickstart
description: Start the app, choose a model, and find its interfaces.
---

## Start browser chat

After [installation](/getting-started/installation/), run:

```sh
mlx-bun
```

The app selects a cached model and opens **http://localhost:8080** in your
browser. If no supported model is cached, it downloads a starter from Hugging Face
and may download a recommended model in the background. Downloads require internet access;
download time and memory use depend on the model. Pass `--no-open` to skip
opening the browser. Press **Ctrl+C** in the terminal to stop the server.

## Call the local API

Leave the server running and send this request from another terminal:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 128}'
```

Use `http://localhost:8080/v1` as an OpenAI client's base URL. The app supports
Chat Completions, text completions, Responses, Anthropic Messages, and embeddings
where the model supports them. See the [HTTP API reference](/reference/server-api/)
for the available routes and the
[app guide](https://github.com/joshuarossi/mlx-bun/blob/main/apps/mlx-bun/README.md)
for their limits.

## Local decisions

Use **Jev or OpenAI Decisions-compatible endpoints with any supported local
language model**, sharing the same loaded weights as generation. Supply candidate
answers to classify input, choose an option, or score against named levels.
mlx-bun scores those answers directly with the model; it doesn't generate a reply
and parse it into a decision.

| Interface | Endpoint | SDK base URL |
| --- | --- | --- |
| OpenAI Decisions | `POST /v1/decisions` | `http://localhost:8080/v1` |
| Jev SystemOne | `POST /v1/systemone` | `http://localhost:8080` |

Set the SDK API key to `local` and use a local model ID from `GET /v1/models`. The
[HTTP examples](https://github.com/joshuarossi/mlx-bun/blob/main/apps/mlx-bun/scripts/decision-example.ts)
cover both protocols and their three question types; the
[SDK examples](https://github.com/joshuarossi/mlx-bun/blob/main/apps/mlx-bun/scripts/decision-website-examples.mjs)
require Node 22+ and the OpenAI and TypeSafe SDKs. Image questions require a supported vision
model. Box questions and denoising models are unsupported. Scores describe the
local model's answer likelihoods; they are not calibrated probabilities or a
promise of the same answers as a hosted model. See the
[decision contract](https://github.com/joshuarossi/mlx-bun/blob/main/apps/mlx-bun/README.md#local-decisions)
for details.

## Choose a model

To select explicitly, download a supported model and serve it by name:

```sh
mlx-bun get mlx-community/gemma-4-e4b-it-OptiQ-4bit
mlx-bun serve e4b
```

Stop an existing server before starting another on the same port. `mlx-bun ls`
lists cached models; `mlx-bun fit e4b` estimates memory fit. You can also pass a
local model directory to `serve`. The [CLI reference](/reference/cli/) lists the options.

For a single generation without a server:

```sh
mlx-bun generate e4b "Hello!"
```

For code you can run and adapt, use the [library examples](/guides/library/).
