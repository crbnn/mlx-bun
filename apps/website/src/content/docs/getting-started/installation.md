---
title: Installation
description: Install the mlx-bun app, or build it from source.
---

mlx-bun requires an **Apple Silicon Mac running macOS 14 or later**. The
standalone installer is the simplest way to start; it includes the runtime and
native libraries, so you don't need to install Bun or Python separately.

## Standalone bundle

```sh
curl -fsSL https://mlx-bun.dev/install.sh | sh
```

The [installer](/install.sh) downloads the release bundle and links `mlx-bun` at
`~/.local/bin/mlx-bun`. Keep `~/.local/bin` on your `PATH`, then run:

```sh
mlx-bun
```

The app opens browser chat at **http://localhost:8080**. If no supported model
is cached, it downloads a starter and may download a recommended model in the
background. See the [quickstart](/getting-started/quickstart/) for model selection
and API usage.

Rerun the installer to upgrade, then restart any running app. Keep the executable
and its native sidecars together if you move the bundle.

### Installation options

`MLX_BUN_VERSION` selects a release tag (default `latest`); `MLX_BUN_INSTALL_DIR` changes the bundle
installation root (default `~/.mlx-bun`), not the app's data location. The app writes
its data (converted models, adapters, datasets, job history, chats) under `~/.mlx-bun`,
or `MLX_BUN_HOME` when set; the Hugging Face cache holds only downloads. The command
is linked at `~/.local/bin/mlx-bun`, with `mlx-bun.server`, `mlx-bun.generate`,
`mlx-bun.convert`, `mlx-bun.fuse`, `mlx-bun.lora` and `mlx-bun.upload` beside it: mlx-lm's
commands under mlx-bun's names, accepting `mlx_lm`'s arguments (see the app's
[mlx-lm compatibility notes](https://github.com/joshuarossi/mlx-bun/blob/main/apps/mlx-bun/README.md#mlx-lm-compatibility-mlx-buncmd)
for the flag mapping and what is not supported). That directory must precede older installations
on `PATH`.

## Homebrew

```sh
brew install joshuarossi/tap/mlx-bun
```

Homebrew installs the same bundle and links the `mlx-bun` command. Run
`brew upgrade mlx-bun` to upgrade, then restart the app.

## npm package

Requires **Bun 1.4.2 or later**. Install the command globally:

```sh
bun install -g mlx-bun
```

Or use `bunx mlx-bun` without a permanent installation. Use Bun rather than
Node's `npx`. Native libraries ship inside the package archives.

## Neural Engine preparation

The standalone installer runs `mlx-bun compile-ane` to prepare Neural Engine
programs in the system cache. Where compilation is unavailable, it warns and
keeps the installation. The app checks this cache before model loading for all
installation methods and compiles missing programs. Use
`mlx-bun compile-ane --check` to inspect their status.

## Build from source

Clone `main` from the [repository](https://github.com/joshuarossi/mlx-bun) and follow its
[development instructions](https://github.com/joshuarossi/mlx-bun/blob/main/README.md#development)
for the Bun version, dependency installation, and native build or staging steps.
Unlike an installed release, a source checkout needs those native inputs.

The app's [source entry and verification instructions](https://github.com/joshuarossi/mlx-bun/blob/main/apps/mlx-bun/README.md)
are kept beside the implementation. Run `mlx-bun --help` to see the commands
it supports. Native libraries are bundled in package artifacts, not committed to
Git or downloaded by library imports.
