# Refactor plan

Open work only; delete a block when its exit criteria are met. Josh has authorized the library and application migration through a first full
draft; keep changes focused and reviewed. Standalone Pi integration is deferred. Ownership and documentation rules live in [ARCHITECTURE.md](ARCHITECTURE.md).

Decisions that govern all work:

- One engine: the continuous batching scheduler. A single request is a batch of one; there is no second one-at-a-time engine or fallback path. A shape the scheduler cannot run yet gets a typed error.
- Main is reference-only, not an oracle to reproduce: an improvement over main is fine when bit parity (where the numerical contract requires it), performance and test gates hold.
- mlx-bun is a drop-in superset of mlx-lm: the `mlx-bun.<cmd>` aliases accept mlx-lm arguments (landed in #241; the flags they refuse are listed in the
  [app README](apps/mlx-bun/README.md#mlx-lm-compatibility-mlx-buncmd)).
- Refactor acceptance requires numerical parity, preserved functionality and no
  performance regressions. Knowledge benchmark scores characterize the candidate
  quant for its Hugging Face model card; that publication work has its own criteria.
- Legos compose at load. A graph never contains a fallback to another graph, the cache owns its read, and the
  scheduler calls named phases. The composition refactor below is the current structural work.

## Composition refactor

Status: draft, 2026-10-10; Josh approved this plan. One PR per numbered item, targeting `main`,
not merged without Josh's instruction. Delete a block when its exit criteria are met.

### Target

Every served configuration is one composition, built once at load from the user's model choice,
their flags and the machine. The loader builds the graph, the cache and the drafter for exactly
that configuration. The scheduler decides what work runs and calls the phase it is in. The graph
runs the layers it was built with and never inspects its input or its cache. The cache owns
storage, the read, prefix reuse, paging, SSD offload and eviction behind one interface. Each piece
implements its contract and never checks what another piece is doing, because nothing below the
scheduler decides anything at run time. Vision, adapters and KV scheme are compositions, never
refusals. A generic graph for a family is a complete structure the loader may select when no
specialized one matches; a specialized graph never contains a path back to it.

Batching is not a mode. The scheduler is the one engine and a single request is a batch of one,
as the first refactor established. One request or four requests run the same composition: the
scheduler batches the rows it has, up to `--batch-size`, and calls the phase operation for that
row count. The graph builds one operation per phase per width class its kernels define (one row,
2 to 4, 5 to 8, wider) and the scheduler picks by the rows it is loading, never the graph by
inspecting its input. Row-wise layers take M rows and do not know whether they are one sequence
or M sequences. Only `attend` and the recurrence see the row layout.

Why: the pairing of storage and read kernel, of row count and MLP kernel, of device and graph, is
known before a request exists. Re-checking it on every forward costs CPU on decode's critical path,
multiplies the numerical paths a parity test has to cover, and is how batching came to drop the
optimized kernels: the M4 Pro graph is organized as whole-forward plans keyed on one sequence, so
any batch fails the plan and the whole forward goes generic, MLP included. Row-wise layers do not
care whether M rows are one sequence or M sequences; only attention and the recurrence do, and
those read per-sequence state the cache owns.

### Rules the gate holds when this plan closes

- A graph extends nothing and never calls another graph's forward.
- No runtime flag, env var, `deviceArchitecture()` call, `globalThis` read or `instanceof` of a
  layer or cache class inside a forward, attend or kernel-dispatch body. Constructors, `build`,
  `load`, graph acceptance functions and the loader may read them. Development profiling hooks
  (`__deltaProf`, `MLX_BUN_SPEC_*` diagnostics) are exempt and listed by name.
- The cache contract has no representation-specific escape hatch (`updateAndFetch`,
  `quantizedAttention`, `rotatedValueAttention`, `affineConversion`, `turboConversion`).
- The graph contract names its phases; the scheduler calls them by name.
- Legos compose additively, never multiplicatively (Josh, 2026-10-11): five modules with three
  configurations each are fifteen implementations against five contracts, not 243 plans. No
  module names another module's configuration: a graph's declared capabilities say what it
  computes, never which cache, drafter, sampler or batch width it tolerates; a cache lego
  implements the whole cache contract for every model; the user never hears "you can't use
  that cache with this model". The served-surface inventory lists no cross-module exclusion.
- Existing violators ride the seam ratchet in `packages/inference/tests/architecture.test.ts`:
  counts only fall. The ratchet is a transition device, not a resting state: every entry is
  owned by a numbered item in this plan, and when the plan closes no example of a banned pattern
  remains in the tree. Agents copy what they read, so a catalogued bad example is still a bad
  example.
- Every gate failure message names the paved road: what to do instead and where it lives (the
  generator script, the contract method, the layer directory), so the mistake teaches its fix at
  the moment it is made.
- One supported way per pattern, written down in ARCHITECTURE: add a model by generating its
  graph; add a kernel as one file in `kernels/` bound by one layer; read a cache through
  `attend`; add a phase through the graph contract. A second way to do any of these is a gate
  failure, not a style note.

### Decisions

Closed by Josh, 2026-10-10:

- **Off-composition requests** keep the existing typed refusal from declared capabilities for
  now. This applies to the graph only; legos still compose at load, and a composition that lacks
  a lego is a different composition, not a refusal.
- **Draft depth.** `--num-draft-tokens N` is fixed depth; `adaptive` means the scheduler chooses
  each round among the verify widths the composition built. The drafter is chosen by
  `--draft-kind` (`dflash2`, `mtp`, ...).
- **Quantized KV is a cache lego whose first append is bf16.** Corrected 2026-10-10 by A3's
  equivalence test, which overturned the premise recorded earlier: today's served path and
  mlx-lm's loop both run the first forward against unquantized keys and convert afterwards,
  even with `quantized_kv_start` 0. So the lego `--kv-quant N` composes keeps that behavior
  inside itself (bf16 through its first append, quantized storage after; `--quantized-kv-start
  M` extends the bf16 span to M tokens), which preserves today's numbers and mlx-lm parity with
  no check outside the cache. A cache quantized from token zero is a different composition with
  different first-chunk numbers (up to 0.125 at 4 bits on a test tensor), no mlx-lm oracle, and
  a cheaper first chunk; it exists as its own lego, opt-in, gated by KL. Confirmed by Josh
  2026-10-11: the parity-preserving lego is the default `--kv-quant N` composes.
- **KV scheme is a serve flag.** `--kv-quant` composes the cache lego at load. The per-request
  `kvBits`, `kvConfig` and `quantizedKvStart` options are removed; mlx-lm's server takes these as
  server flags too.
- **The TQ graph** is broken into its pieces, which join the lego box where they are reusable,
  and the file is deleted when the M1 Max composition is generated.
- **`--batch-size`** is the cap on rows per forward the scheduler may batch. Nothing else.

- **Bit-stability across batch width** (closed by Josh, 2026-10-11): keep the measured-best
  kernels per width class (factored for 1 to 4 rows, matrix-unit for 5 to 8), so a request's last
  bits can depend on how many rows were batched with it; this is what the old graph served and it
  is documented and KL-gated. There is no separate switch: choosing `--batch-size N` above one opts
  into batching, and batching may move the last bits; `--batch-size 1` says nothing can be batched,
  so nothing can change. The batched paged read follows the same rule (joined rows, as served today).

### Phase A: contracts (no behavior change)

#### A1. The cache owns the read

`contracts/mlx/cache.ts` has four read shapes today: `updateAndFetch` (bf16 hands tensors back),
`quantizedAttention.updateAndFetchQuantized` (affine hands packed tensors back),
`rotatedValueAttention` (TurboQuant), and `attentionState.appendAndFetch(k, v)` returning a view
with `attend(q)` (paged, already owns its read). Make the paged shape the only one.

- [ ] Define the single attention interface: append k and v, then named reads. One operation for
      the single-row decode read and one for the window read, so the calling phase picks and
      nothing branches. The cache builds its own mask from the offsets and row layout it owns;
      `makeMask` becomes internal.
- [ ] Define the SSM cache's recurrence operation the same way: the cache owns conv state and the
      gated-delta state update; the DeltaNet block hands it the projected inputs.
- [ ] Mark `updateAndFetch`, `quantizedAttention`, `rotatedValueAttention`, `affineConversion` and
      `turboConversion` deprecated on the contract, removed in B1.
- [ ] Keep `RowBatchCache` and `BatchableCache` (`rowOffsets`, `filterRows`, `extractRow`): the row
      layout is the one thing the scheduler passes through to the stateful layers.

Exit: the new interface compiles beside the old members; no caller moved yet; `bun run typecheck`
and `bun run test` pass.

#### A2. The graph names its phases

- [ ] Add `prefillChunk`, `prefillTail`, `decode` and `verify` to `contracts/mlx/graph.ts`, with
      the capture callback a parameter of `verify`. Vision prefill is prefill with embeddings and
      positions, which `forwardEmbeddingsAtPositions` already is.
- [ ] `decode` and `verify` exist per width class, declared by the graph as a table of
      operations the scheduler indexes by the rows it is loading. The graph never derives the
      class from its input. D1 set the classes to the row-count boundaries the shipped kernels
      were measured at: one, two to three, four, five to eight, wider (the old M4 Pro graph had a
      separate four-row plan with the matrix-unit down; lumping four with two and three cost
      0.078 on real verify logits). `logits` is a width table too (the head is part of the
      phase), and `prefillChunk` is per execution unit, `gpu` and, when composed, `ane`.
- [ ] Keep `forwardHidden` and `forwardHiddenMixed` until E2 moves every caller, then delete.
- [ ] `verifyRoundCosts` stays as declared data; the scheduler's adaptive gate reads it.

Exit: every existing graph implements the four by delegating to its one forward, one line each;
no scheduler call site changed yet; tests pass.

#### A3. Composition facts reach the loader

- [ ] `createModel` takes the composition: device, ANE bridge present, KV scheme, draft depth,
      prefill chunk size, adapters, max rows per forward (`--batch-size`, default 8, which bounds
      the width classes the graph builds). The app's `loadContext` in
      `apps/mlx-bun/src/engine/model-host.ts` already holds all of these when it creates the model.
- [ ] `resolveKvScheme` in `apps/mlx-bun/src/engine/cache-services.ts` runs before model creation
      and feeds the composition instead of becoming per-request options afterward.
- [ ] `makeCache()` takes the composition's KV scheme. Building the lego it names is B1's work:
      A3's equivalence test showed that a cache built quantized from token zero differs from the
      convert-after path in its first forward's numbers (see Decisions) and, past the offset, in
      the preallocated padding's scales, so the lego must reproduce the convert-after semantics
      inside itself rather than being the plain quantized cache.

Exit: a loaded graph can print its composition; behavior unchanged; tests pass.

### Phase B: state

#### B1. Every cache implements the one read

`state/` may import `layers/` and `kernels/`, so the attention kernels move under the caches
without a gate change.

- [ ] bf16: `kv.ts`, `batched-kv.ts`, `rotating-kv.ts` and their row storage implement append plus
      the two reads over the stock SDPA kernels.
- [ ] Affine: `quantized-kv.ts`, `batched-quantized-kv.ts`, `rotating-quantized-kv.ts` over
      `layers/quantized-attention.ts` kernels. The M4 Pro composition's 4-bit group-64 head-dim-256
      cache is its own class: KV4 decode kernel for the row read, folded GQA for the window read.
- [ ] TurboQuant: `turboquant-kv.ts`, `batched-turboquant-kv.ts` over their codec. B1c found the
      codec still picks its decode kernel per call (`tryDecodePackedKv` returns null and the
      eager decode runs): the kernel is fixed at construction with the rest of the scheme, and
      the null path goes. `appendBidirectional` on the TurboQuant caches (Gemma 4 image prefill)
      is still to implement. B1e: two kernel sets at construction, `denseTurboQuantKernels`
      (Qwen's read before C1: decode keys and values, then SDPA) and
      `deferredTurboQuantKernels` (Gemma 4 and MiniCPM5: SDPA on rotated values, un-rotate the
      output); the graph puts its set on the bf16 caches `makeCache` builds and every
      conversion, copy and restore takes it from the cache it converts. A graph that passes no
      set leaves a placeholder whose named reads throw; D3 gives every graph its set and the
      placeholder goes.
- [ ] Transition reads, all marked E3 and deleted there: the conversion site in
      `kv-maintenance.ts` and the two SSD restore sites choose the affine kernel set from the
      flag the app writes today through `state/served-affine-kernels.ts` (`1` → unfused,
      otherwise tiled-causal where the configuration tiles). E3 composes the set from the KV
      scheme at load and the app stops writing the flag.
- [x] B1f, landed (`refactor/composition-b1f`): served numbers equal the pre-D3 base on e4b and
      12B at every stage of two-row cohorts under `--kv-quant config` and `4`. (1) The delayed row layouts (`delayed-quantized-kv.ts`,
      `delayed-rotating-quantized-kv.ts`, `bf16-first-kv.ts`, `batched-rotating-quant.ts`) take
      their affine kernel set when `kv-maintenance`'s `preparePrefill` builds them, from the same
      transition rule, instead of a hardcoded unfused set: D3 Gemma 4 measured that rows converted
      inside a multi-row prefill cohort under `--kv-quant config` read windows unfused where the
      old graph tiled them (0.77 on the window read; the e4b assistant-drafter cohort's draft
      counts moved 37/5 → 28/11), and this likely reaches Qwen 3.5 since C1. (2) The batched paged
      cache's window read joins its rows left-padded into one SDPA, as the old `PagedKvRows` path
      did, so `gemma4-paged`'s "gathered bf16 equals plain KV" check at three rows holds again
      (0.03 today); per-row reads are a change of served numbers and wait for the bit-stability
      decision.
- [ ] The quantized lego `--kv-quant N` composes is bf16 through its first append and quantized
      storage after, inside its own `appendWindow`/`appendDecode` (the state transition the
      three `delayed-*-kv.ts` caches do today, owned by one lego built at load), so the first
      forward's numbers equal today's and mlx-lm's. A separate quantized-from-token-zero lego is
      opt-in and KL-gated. Equivalence against the convert-after path is checked below the
      offset bit for bit and by first-forward logits; the padding past the offset is the lego's
      own and is not compared.
- [ ] Paged: `state/paged/cache.ts` already conforms; align names. B1a found its direct
      kernel was chosen by query count (8 or fewer); the window kernel becomes a construction
      choice of the paged lego (direct up to the composition's verify width, SDPA otherwise).
      B1f restored the batched paged lego's reads to the old joined-rows computation for both
      the window and the decode read (per-row decode differed by up to 3.9e-3 at head dim 512),
      so `gemma4-paged` holds at three rows; per-row reads wait for the bit-stability decision.
- [ ] B1a's first divergence: a single-position append on a rotating cache keeps today's
      numbers only through the in-place write, which is `appendDecode`; a one-row window
      concatenates and reorders the keys after the ring wraps (up to 3.9e-3). E2 maps the
      planner's last-token-alone step to `appendDecode`, or the planner stops producing it.
- [ ] Wire one `AttentionMasks` memo per model into every `makeCache()` (one line each) when
      C1 and D3 move the graphs onto the new reads; until then each cache builds its own mask
      and no graph calls the new reads.
- [ ] One mask build per forward: the caches of one forward derive their mask from the row
      layout once (the scheduler's `state/layout` object), not once per layer.
- [ ] The three reads A1 found that the two phases do not cover, each already its own named
      interface, implemented only by the caches composed for them: `appendCommitted` (fill
      spans, today's independent-rows path through `quantizedAppendAttention`),
      `appendBidirectional` (Gemma 4 vision prefill), `readBlock` (DiffusionGemma's canvas pass,
      DFlash 2 block attention over the context), and `appendEncoderWindow` (DiffusionGemma's
      vision encoder, whose first-window mask ignores the sliding window; D3 found
      `appendBidirectional` differs by 0.68 past the window and gave the encoder its own read).
- [ ] `state/ssm.ts` implements `recurDecode` and `recurWindow`, and the training cache sits
      behind the same calls, which removes the `instanceof TrainingSSMCache` branch in the block.
- [ ] Delete the device check (`applegpu_g16s` head-shape case) in `quantized-attention.ts`
      (B1b: grouped heads give the same bits as the plain kernel on the M4 Pro at every size
      tried, so the case is a speed choice for the depth-2 verify composition, kept as its own
      lego). The env flag `MLX_BUN_NO_FUSED_SDPA` is a composition rule the app writes today:
      uniform `--kv-quant N` serves the unfused kernel (mlx-lm's port, parity) and
      `--kv-quant config` the tiled one (OptiQ parity). The affine lego takes its kernels
      explicitly at construction: `unfusedAffineKernels` (mlx-lm parity),
      `tiledCausalAffineKernels` (OptiQ's own wrapper rule: tiled while the cache's mask is
      plain causal, unfused once the cache's own state makes it an array, so `config` composes
      this one), and `tiledAffineKernels` (always tiled, no oracle, KL-gated, opt-in); the
      tiled constructors refuse a configuration that cannot tile; the loader composes by the
      scheme's kind;
      the flag read stays in the deprecated `quantizedSdpa` path, untouched, until the graphs
      read through `attend` (D3) and the app stops writing it (E3).
- [ ] Out of this step, each needing its own contract before its graph can move: GLM-5.2 MLA
      compressed attention (`models/glm52/mla.ts`), the Gemma assistant drafter's donor reads,
      and the segmented backward's donor checkpoints in `models/gemma4/trainable.ts`. The
      universal graph's softcap attention landed in D3 as `state/softcap-attention.ts`, a named
      read chosen per layer at construction.
- [ ] Training-side reads (owner: the training item): `TrainingCache` now takes its layer's
      sliding window and a mask memo (D3 Gemma 4), which closed the perplexity divergence D3
      dense had pinned (the dense gate's scoring case reads maxDiff 0 on the merged tree).
      `PrefixSharedCache` has no read and the flash choice is a trainer global, so MiniCPM5's and
      Gemma 4's training forwards stay in their graphs, deprecated, and their remaining
      `cache-escape` ratchet entries (two each) belong here with the segmented backward's donor
      checkpoints.
- [ ] The deprecated members leave the contract in D3, once every graph reads through `attend`.

Exit: `quantizedSdpaUnfused` and `kv4DecodeAttention` have no callers outside `state/`; the
kernel each cache dispatches per read is fixed at construction; bit-identical logits for every
cache class against its pre-refactor read on the existing parity inputs; `grep` for
`quantizedAttention` in `models/` and `layers/` is empty.

#### B2. Eviction, prefix reuse and SSD inside the cache

- [ ] Compose `prefix-cache.ts`, `tiered-prefix-cache.ts`, `ssd-cache.ts`, `persistence.ts` and
      the budget (`execution/kv-budget.ts`) into one cache service constructed with its budget.
- [ ] The scheduler's surface is acquire(request) returning state plus reused-token count, or a
      miss, and release(request). The scheduler never learns what is resident or when it was
      evicted.
- [ ] Delete `state/kv-maintenance.ts`; its batch preparation moves to `state/layout.ts`.

Exit: saved KV under `MLX_BUN_HOME/kv` round-trips as before; prefix hit rate on the existing
warm-reuse test is unchanged; no `toQuantized` caller outside `state/`.

#### B3. The cache is swappable: nothing else knows or cares

Josh, 2026-10-11: "I can swap between paged KV, quantized KV, regular KV, TurboQuant KV; nothing
else should know or care." Today the planner and the gateway do know: `execution/plan.ts` refuses
speculation when the composition is paged (`!features.pagedKv`) and gates it on
`speculativeKvQuant` / `speculativeTurboQuant`; batching is gated on `quantizedBatch`,
`turboQuantBatch` and `pagedBatch`; shared fill is off under paging; `gateway-binding.ts` derives
`kvBatchCapabilities`, `delayedAffine`, `affineKv` and `cachesBatchable()` from the cache format
and `state/capabilities.ts` answers `isPlainKvCache` / `isRotatingPlainCache` / `isRecurrentCache`
to callers outside `state/`. Every one of these is a method learning the representation.

- [ ] Every cache lego implements the whole contract: append and `attend` for each phase, the
      row layout for batching, the speculative transaction (`specRoundCommit` /
      `specRoundRollback` to the accepted length, and the known-token append), the prefix
      snapshot and restore, and the SSD form. The paged lego and the TurboQuant legos implement
      the transaction and the batched layout they lack today; a lego that cannot implement an
      operation is a bug in the lego, not a capability flag.
- [ ] Delete the cache-format terms from the planner and the gateway: `pagedKv`, `kvQuant` and
      `turboQuant` conditions in `plan.ts`; `quantizedBatch`, `turboQuantBatch`, `pagedBatch`,
      `speculativeKvQuant`, `speculativeTurboQuant` from `GraphCapabilities`; `kvBatchCapabilities`,
      `delayedAffine`, `affineKv`, `cachesBatchable` from `gateway-binding.ts`; the `state/capabilities.ts`
      predicates have no caller outside `state/`. Media and adapter rows no longer bypass paging
      (`paged-kv-bypassed-for-media-or-adapters`); the adapter namespace is the cache's.
- [ ] Order: after the merge of E1 + D2 + E3. The lego work (state/) runs beside E2; the planner
      and gateway deletions share E2's files and land with it.
- [x] Lego side landed (`refactor/composition-b3-legos`, 33c7f5be, integration branch
      fast-forwarded). One model-free conformance suite (`tests/state/cache-contract.test.ts`)
      runs every lego through append, attend per phase, decode/prefill/verification rows, the
      speculative transaction, the committed, bidirectional, encoder-window and block reads,
      donor rows, snapshot and SSD: 190 of 190 cells, against 135 on d71467c4. Each storage now
      declares its three row layouts in one table in `state/layout.ts` (three `instanceof`
      chains gone; the file's ratchet entry deleted); `SpeculativeRowLayout` is the transaction
      as a contract. The paged lego gained the transaction, verification rows, the committed,
      bidirectional, encoder and block reads and donor rows; ring legos a committed read; the
      quantized and TurboQuant legos the encoder-window and block reads; KV4 verification rows
      read donors with their own kernels. Correction to the brief: TurboQuant already had the
      transaction and the batched layout. Every served path is bit-identical to the head
      (gemma4-paged, speculative-group under four schemes, continuation under six, 20 app-path
      digests, SSD digests). Newly served paths, planner bypassed, on e4b: speculation over
      paged bf16 gives the bf16 lego's tokens at one and four rows; teacher-forced rounds under
      paged 4 / paged 8 / TurboQuant 8/3 sit at KL 0.016 / 0.010 / 0.016 (max 0.17 / 0.064 /
      0.18) against the tiled null's 0.10 / 1.31. Not whole and not in the suite: the
      grouped-heads KV4 cache, a verification-rows form whose window kernel serves exactly 6
      query heads per KV head at 3 queries (a width-specialized verify lego; its conformance
      is at its width, or the kernel generalizes in `layers/quantized-attention.ts`). Still
      owed on the lego side: compiled decode on the paged and TurboQuant legos; DiffusionGemma's
      `makeCache` ignores the scheme (models/); pages under turbo or a per-layer config are
      still bf16. The planner side's callers, for the item that lands with E2:
      `gateway-binding.ts` (`convertible`, `isRecurrentCache`, `batchable`/`targetRows`,
      `kvBatchCapabilities`, `delayedAffine`, `affineKv`, `cachesBatchable`, `kvBatchable`),
      `batch-group.ts` (`isPlainKvCache`, `isRotatingPlainCache`, `isBatchableCache`),
      `plan.ts` (paged, kvQuant, turbo terms), `generation/autoregressive.ts`
      (`isRotatingPlainCache`), `models/gemma4/compiled-decode.ts` (`isCompiledDecodeCache`),
      `scoring/full-sequence.ts` (`isRecurrentCache`), and `state/kv-scheme.ts`'s `batchable`.

Exit: on e4b, speculative-group (assistant drafter), grammar-group, fill, prefix reuse and the SSD
round-trip each run under every lego the composition can name (bf16, affine uniform, affine
config, TurboQuant, paged, bf16-first KV4 on the M4 Pro composition) with per-line output
identical to the same lego on `30412523` where that lego served the method today, and KL-gated
against bf16 where it did not; `plan.ts` and `gateway-binding.ts` contain no cache-format term;
the served-surface inventory lists no cache × method exclusion.

### Phase C: layers

#### C1. Qwen blocks become legos

- [ ] Move `GatedDeltaNet`, `Qwen3Attention`, `Qwen3MLP` and `Qwen3Layer` from
      `models/qwen/qwen3_5.ts` to `models/qwen/blocks.ts`. The projection loader injection stays.
- [ ] The attention block calls `cache.attend`; its three-way branch on cache representation and
      `independentRows` goes.
- [ ] Follow-up, after the DeltaNet block takes the SSM cache contract instead of the concrete
      class: move the blocks to `layers/`.

- [ ] Landed in C1 (#337) with B1e (#338): the attention block holds one read strategy taken at
      construction (the cache's own read; a deprecated strategy reproduces the Trellis graphs'
      injected cores until D1); the DeltaNet block calls `recurDecode`/`recurWindow`; the MLP
      is `TrellisMlp` or `AffineMlp`, chosen by `loadQwenMlp` from the quant table at load; the
      phase is a value passed down `forwardLayers`, derived from a shape only in
      `deprecatedPhase`, the one adapter E2 deletes. Still inside the generic blocks and owned
      by E2's width tables: `TrellisMlp`'s "4 rows or fewer → fused gate/up" choice.
      `MLX_BUN_MIXED_PACKED_MLP` is read once at construction and becomes a composition fact in
      E3. `verify` on the TQ graph now runs the TQ plan where A2's delegation ran the generic
      mixed loop: E2's capture-path decision.

Exit: `Qwen35Model` and the M4 Pro graph both import from `blocks.ts`; no `extends` between
them; bit-identical generic forward.

#### C2. `TrellisLinear` stops choosing

- [ ] Remove the per-call variant read and the five-way kernel dispatch from `forward`. The layer
      keeps the one prefill kernel the default variant selects; row-count kernels already have
      their own layers (`trellis-gate-up.ts`, `trellis-down.ts`).
- [ ] `setTrellisVariant` survives only as explicit bench scaffolding; `MLX_BUN_TRELLIS_VARIANT`,
      `MLX_BUN_TRELLIS`, `MLX_BUN_TRELLIS_ASYNC_EXPAND` are not read on the hot path.
- [ ] PR #311's five `MLX_BUN_TRELLIS_*` flags in `kernels/trellis/scatter.ts` and
      `mixed-gate-up.ts` become separate kernels or bench-only scaffolding.

- [ ] Kernels take their representation choices as required arguments; the device-dependent
      choice is made once in the layer's constructor and stored, never inside a kernel body.
- [ ] After the C2 PR merges and the paired serve benchmark has passed on both machines (the
      M1 Max parity test needs that host and the q4b artifact), delete the frozen pre-C2 layer
      `tests/kernels/trellis-linear-reference.ts` and the identity test that uses it; the PR is
      the record of the maxDiff-0 result.

Exit: `grep runtimeFlag\|runtimeValue\|runtimeNumber packages/inference/src/layers/trellis-linear.ts`
is empty; no `deviceArchitecture()` call inside a kernel function; bit-identical outputs against
the default variant on the M4 Pro and the M1 Max; paired serve benchmark shows no regression.

#### C3. The ANE split owns its programs

- [x] `layers/ane-prefill-split.ts`: programs are instance-owned and disposed with the layer; no
      module-level maps.
- [x] Programs are built for exactly the chunk size; no buckets, no minimum-row floor. The
      leftover chunk is `prefillTail` on the GPU.
- [ ] ANE programs compile at install, the way Apple's own pipeline compiles a model for the
      Neural Engine on the device and caches it (Josh, 2026-10-10). The installer, or the app's
      first launch, compiles every program the shipped compositions need and leaves them in the
      framework's cache, so no prompt ever pays the compile and the serving process never
      compiles. The MIL source ships in the bundle as the compiler's input. Shipping the compiled
      form itself, one per Neural Engine generation built on a host of that generation, stays
      possible later; it is not required.

- [x] Program creation blocked (D1's merge gate, 2026-10-11, over 20 minutes). Cause found on
      `refactor/composition-c3-hang` (e3cc7f29, merged into the integration branch at
      d71467c4 with the ANE, drift and architecture gates passing on the merged tree): every
      loaded ANE model holds one direct-path client of the ANE driver, the driver serves about
      a thousand across the whole machine, and the M4 Pro graph loaded 176 programs per graph,
      one per layer, all identical MIL and weights because the weights stream through the
      buffers. Past the cap the client open inside the load fails and the framework retries
      every 15 s for as long as the clients are held, in our process and in every other ANE
      user's (`mediaanalysisd` restarted every 120 s during the gate). The earlier suspects
      (aned's reply, client state, buffer release order) were not involved and the 2048-row
      shape did not matter, only the count. Fix in the bridge: a program created with `share`
      whose MIL and weights are identical to the shared program's holds that same loaded model
      (holder count; the last `mbane_free` unloads), so a graph holds 3 loaded models instead
      of 176. Outputs bit-identical to the old bridge; the real-artifact gate passes at maxDiff
      0 holding 6 clients. Regression tests opt in with `MLX_BUN_TEST_ANE=1`: a child process
      creates 2288 programs of four chunk sizes then one more, killed at 240 s; splits of two
      chunk sizes coexist and dispose in either order. Found: the identity test and the frozen
      reference graph never dispose their graphs (3 leaked clients per graph now; dispose them
      when the frozen files go). Found: `mbane_available()` / `aneAvailable()` is a soft
      runtime check for whether the framework loaded; under the composition the bridge is a
      load-time fact (`aneBridge`), so the loader composes it or fails the load with a typed
      error and the predicate goes (E2).

Exit: two graphs can coexist in one process without evicting each other's programs; KL gate on
ANE prefill unchanged; after install, the first ANE prefill on a fresh machine loads from the
cache and compiles nothing, measured by the bridge's own `compiledModelExists`; a process that
runs several ANE programs of different shapes in sequence never blocks in program creation.

### Phase D: graphs

#### D1. `Qwen38TrellisM4Pro` as an entire graph

- [ ] `implements MlxDeclaredGraph`, extends nothing. Constructor takes the composition and builds
      embedding, norms, head, and per layer the block set each phase needs: decode blocks, one
      verify block set at width depth+1 (or the set the adaptive decision names), chunk blocks with
      the ANE split when the bridge is present, tail blocks.
- [ ] Vision prefill phase included because the model has vision; mRoPE in the attention block.
- [ ] Adapters: the LoRA-applying linear in the composition that serves adapter rows.
- [ ] The 40 interleaved and 24 row-major down projections are read from the quant table once at
      construction.
- [ ] `draftTarget` declares the hidden-layer taps and the residual basis (R1 seed, final-norm
      gain).
- [ ] Removed: the plan table (`#planFor`), the guard in `forwardLayers`, every `super` call, the
      capture hook, the `instanceof TrellisLinear` checks, the head's row switch, the KV bits read.

Exit: `grep "extends\|super\.\|instanceof\|globalThis" qwen38-27b-trellis-m4pro.ts` is empty
except the listed profiling hook; bit-identical logits per phase against the stack's graph on the
same phase; the Qwen generator (D2) produces it.

#### D2. Generators are the asset

- [x] Fix the template in `packages/inference/scripts/gen-gemma4.ts`: no `extends`, no
      `#matches`, named phases. Regenerate `gemma4-e4b.ts`, `gemma4-12b.ts`, `gemma4-26b.ts`.
      Landed (`refactor/composition-d2`): the blocks live in `models/gemma4/blocks.ts`; each
      generated class extends nothing, folds its per-layer KV scheme as literals, declares
      `COMPOSITION`, and is checked for drift by `gemma4-generated-sync.test.ts`; 840
      comparisons at maxDiff 0 and the e4b and 12B digests identical. Seams for the E1 merge:
      the generated classes join the `RuntimeModel` union (then the transitional `GENERATED`
      map and its cast go), and training on those three fingerprints must load the generic
      graph, since a generated graph declares no `trainable` (a composition field the train
      verbs set, or the train module selecting the generic entry explicitly).
- [ ] Write `packages/inference/scripts/gen-qwen38.ts` taking the composition as input; the M4 Pro
      graph and the M1 Max graph are its outputs. The hand-written TQ graph is broken into its
      reusable pieces and deleted. D1 landed the generator and the M4 Pro output (`--describe`
      reads a snapshot's config and shard headers; the committed file regenerates with an empty
      diff). The M1 Max output needs, first: layer classes for the fused gate/up reduce kernels
      and the packed down (today functions), the TQ graph's three env A/B flags turned into
      composition facts or separate kernels, and an identity gate run on that host.

Exit: `git diff` of a regenerated file against its committed version is empty; the generated Gemma
parity test passes on e4b and 12B; no generated graph extends a graph; no hand-written
specialized graph remains, so the only example of a specialized graph an agent can read is a
generated one.

#### D3. Generic graphs

- [ ] `Qwen35Model`, `Gemma4Model`, `MiniCPM5Model`, `Qwen3Model`, `Qwen3MoeModel`, GLM and the
      universal graph implement the four phases by delegation and take their cache through
      `attend`. Their other internal branches are out of scope here.
- [ ] `createAppend` in `Qwen35Model` no longer checks the device; the loader composed it.

Exit: no graph reads `quantizedAttention`; bit-identical generic forwards.

### Phase E: loader, scheduler, app, drafter

#### E1. Loader selection table

- [ ] The `qwen3.5` entry in `models/factory.ts` becomes a table keyed by fingerprint, device,
      bridge, KV scheme, depth, chunk size, adapters and max rows. The generic graph is the entry
      yielded when nothing specialized matches. The same shape for Gemma's `GENERATED` map.

- [ ] The key includes the KV scheme and max rows because the M4 Pro composition serves bf16 and
      affine 4-bit group 64 only (D1: the old graph went generic for 8-bit and per-layer config;
      the new one would run its width sets over whatever cache it is handed) and its
      `Kv4Head256Cache` lego serves one sequence until E3's batched form lands.

- [x] Landed (`refactor/composition-e1`): `GraphTable` in `models/implementation.ts`, one table
      per resident family in `MLX_GRAPH_TABLES`; building a table fails unless it has exactly
      one generic entry and no two equally constrained entries can both match. The generated
      Gemma graphs serve their kv_config scheme, so the default bf16 composition selects the
      generic Gemma 4 graph, which is what the old `#matches` guard fell back to. Loose ends:
      `models/profile.ts` still labels the generated-fingerprint Gemma artifacts
      `gemma4-generated` regardless of composition (make the label the selected graph's), and
      two README sentences describe the old selection (F2).
- [x] Merged with D2 and E3 (`refactor/composition-merge`, c691ba23). One list of generated
      graphs (`GENERATED_GRAPHS`, each entry's composition from its file); D2's transitional map
      gone; generated classes in `RuntimeModel`; the table keys KV on `ServedKv` (storage kind
      and widths only: read kernels and the conversion start are the cache's); the profile names
      the family and the table labels the graph (`gemma4-generated` profile, its specialization
      and the `generated-graph` capability deleted; family-registry goldens for e4b and 26B
      change to the family's labels; saved generation checkpoints for those artifacts miss once
      because their identity's specialization changed). Found: on the Trellis artifact the
      table builds the generic Qwen 3.5 graph for `--kv-quant 8` and TurboQuant, where the old
      M4 Pro graph ran its own internal fallback; the app-path digests differ for those two
      schemes (every other artifact and scheme equals base). The generic graph is the loader's
      selection by design; the old fallback was the path this refactor deletes, and the generic
      graph's numbers are what main served before the M4 Pro graph existed. Measured
      (teacher-forced decode KL, 16 default prompts cut to 256 tokens, last 32 teacher-forced,
      487 positions, base 30412523 → d71467c4): kv8 mean 9.4e-4 (median 5.8e-4, p95 2.4e-3,
      argmax agrees 476/487), turbo mean 1.7e-3 (median 1.2e-3, p95 5.5e-3, 473/487); bf16
      control byte-identical. Scale: kv8's own cost over bf16 on base is 9.8e-4 and turbo's
      6.1e-3, so the graph change costs about one kv8 quantization and a quarter of a turbo
      one. The tiled null is 0 at the method's own settings (no multi-query read ever touches
      a quantized cache for prompts under one chunk); with a 2-token chunk it is 1.6e-4 (the
      four prompts that get such reads: null 6.6e-4 vs 7.2e-4 for the change). Limit: the
      default prompts give contexts of at most about 48 tokens. Accepted.

Exit: selection is data; `qwen38TrellisM4ProAccepts` and the nested ternary are gone.

#### E2. Scheduler calls phases

- [ ] `execution/batch-group.ts` (decode and verify call sites), `prefill-rows.ts`,
      `speculative-group.ts`, `grammar-group.ts` and `fill-group.ts` call the named operation for
      their phase.
- [ ] `AdaptiveDraftGate` stays in the scheduler and chooses among the verify widths the
      composition built when `adaptive` was requested.
- [ ] The scheduler indexes the graph's operation table by the rows it batched: one request is
      one row and calls the one-row decode; four requests call the 2-to-4 decode; verify width is
      rows times depth+1. No flag and no mode; `--batch-size` is only the cap.
- [ ] The Neural Engine is a compute unit the scheduler manages (Josh, 2026-10-11). The graph
      exposes its prompt-chunk operation per execution unit, `prefillChunk.gpu` and, when the
      composition built it, `prefillChunk.ane` (the ANE+GPU split for exactly the chunk size);
      which unit a chunk runs on is the scheduler's dispatch decision, read once at bind as a
      declared fact and chosen per chunk by the scheduler, which owns the ANE as a resource and
      can overlap one request's ANE chunk with another's GPU work. No graph routes to the ANE by
      itself. Until E2 dispatches to it, the deprecated entry points call the GPU operation.
- [ ] Batching sends M rows through the row-wise layers; only `attend` and the recurrence see
      `rowOffsets`.
- [ ] Call sites A2 found that do not fit the four phases, each resolved explicitly, none by
      inspecting shape:
      - Tapped prefill: the DFlash 2 and DSpark drafters capture hidden layers during prefill.
        `prefillChunk` and `prefillTail` take the capture the way `verify` does; no mutable
        `hiddenTap` field survives.
      - Two capture paths exist today (`legacyForwardWithTaps` setting `hiddenTap`, and the
        mixed lane's callback) and they differ per graph: the TQ graph runs the generic loop
        under the mixed path, and generated Gemma falls to the monolith under `hiddenTap`. E2
        keeps one, gated by bit-identity against the other.
      - Mixed iteration: `advanceMixed` packs a decode group and a prefill group into one
        forward. Either two phase calls, or a composition that declares it serves mixed rows.
      - The compiled decode step is chosen at run time from one row's state with a fallback to
        the ordinary forward. The composition decides compiled or not at load; the fallback goes.
      - Media rows: decode through `forwardHiddenAtPositions` with a positions grid, prefill
        through `promptInput.forward`. The vision prefill phase takes embeddings and positions.
      - Known-token appends (`createAppend`, grammar spans, fill's `verifyKnown`): a fifth named
        phase over the committed read, `appendKnown`.
      - The prefill planner's snapshot-boundary cuts and its last-token-alone step map onto
        chunk and tail explicitly, or the planner changes; nothing infers "whole chunk" from a
        length.
      - GLM verify kernel pinning (`pinVerify`): the composition pins at load.
      - The DFlash 2 drafter's draft block picks its kernel once per block from anchors × width
        (E4): the draft operation becomes a width-classed table like the graph's `verify`, and
        the scheduler indexes it by the rows it is drafting for.
      - Prefill cohorts split a row's planned chunk when a sibling's chunk is shorter (B1d).
        A quantized lego converts by its own stored length, so a split chunk converts at the
        split and reads after it change (0.24 at 4 bits measured); today's rows layouts convert
        at the planned boundary instead. Decide: the planner never splits a row's planned
        chunk, or conversion per forward is accepted under the bit-stability decision. Until
        then the `delayed-*` rows caches keep the planned-boundary timing.
      - A verify window that crosses a lego's start converts even if its tail is rejected; the
        lego converts on commit while a speculative round is armed (the cache's own state).
- [ ] Delete `forwardHiddenMixed` and `forwardHidden` from the contract.

Exit: one request and four concurrent requests on the M4 Pro composition both dispatch the
Trellis row kernels for their width, shown by the op inventory diagnostic, with no generic path
entered; a request's output across batch widths satisfies the bit-stability decision above;
paired batched-lane benchmark records the gain at one and four rows.

Baseline for that benchmark (2026-10-11, M4 Pro, the ladder's command: the Trellis artifact,
`--kv-quant 4`, the DFlash 2 drafter, `--num-draft-tokens adaptive`; six chat prompts × two
reps, greedy, 256 tokens, HTTP, medians; main and the integration head alternated, one server
at a time): decode main 29.16 / 30.54 tok/s, d71467c4 29.40 / 28.95 tok/s (rounds 1 / 2);
time to first token in the quiet round 183 / 176 ms at about 100-token prompts. Equal within
the per-prompt spread (20–41 tok/s on both). Two things the reproduction found: on main the
DFlash 2 provider needs `MLX_BUN_DFLASH2_TARGET_BASIS` pointing at the basis JSON on the M1 Max
share, otherwise the drafter runs unrotated and decode is plain (10.8 tok/s with the drafter
bound; the integration head declares the basis from the artifact's sidecar, E4); and the
repo's paired serve benchmark (`scripts/bench/plan.ts`) has no DFlash 2 / adaptive
configuration, so this was measured with a harness outside the repo. F's perf item adds that
configuration so the served composition is what the check measures.

#### E3. App

- [x] `--kv-quant` composes the cache lego before the model loads. D1 measured why this cannot
      wait: the scheduler still calls `makeCache()` with no scheme and `kv-maintenance` converts
      to the plain affine cache, so the entire M4 Pro graph (which reads through the cache
      instead of injecting its KV4 and folded kernels) serves `--kv-quant 4` through the generic
      read until the lego is composed (logits 0.028 at one row, up to 0.069 at eight). The M4 Pro
      composition's lego is a bf16-first `Kv4Head256Cache` (first append bf16 as the old graph's
      conversion-after-first-forward did, then the KV4 decode and folded window reads), and its
      batched form reads with the KV4 kernels while its own row layout holds one row and with the
      plain affine read when it holds more, which is what the old graph served (one sequence on
      the KV4 cores, several on the generic path); the choice keys on the cache's own layout, as
      the tiled-causal set does. E3's gate: the composed lego against the old graph's injected
      cores at maxDiff 0 on the real artifact.
- [x] `MLX_BUN_QWEN_SPEC_KV4` (read once in the M4 Pro constructor since D1) and
      `MLX_BUN_MIXED_PACKED_MLP` become composition facts.
- [x] The per-request `kvBits`, `kvConfig` and `quantizedKvStart` options leave `generation/types.ts` and the server request plan; the cache lego is composed at load from `--kv-quant` and `--quantized-kv-start`.
- [ ] `--num-draft-tokens adaptive` is served by the scheduler choosing among the built widths; a number is fixed depth.
- [x] Landed (`refactor/composition-e3`, five commits on `30412523`). Every graph's `makeCache()`
      composes from `composition.kv`: plain bf16; bf16-first affine (uniform, or per listed
      layer under `config`), read kernels from `KvStorageScheme.kernels` (`--fused-sdpa` on =
      tiled-causal, off = unfused, unset = by scheme); bf16-first TurboQuant dense (Qwen) or
      deferred (Gemma 4, MiniCPM5); `Bf16FirstKv4Head256Cache` for the M4 Pro composition,
      reading with the KV4 kernels while its own layout holds one row and with the plain affine
      kernels when it holds more. Gate (a) passed on the real artifact at maxDiff 0 (chunk,
      conversion point, tail, decodes, verify 3 and 8); 24 app-path digests on e4b and MiniCPM5
      unchanged; speculative-group and gemma4-paged lines identical. Deleted: the per-request KV
      options, kv-maintenance's serial conversion branch, `served-affine-kernels.ts`, every
      `MLX_BUN_NO_FUSED_SDPA` read and write. `speculativeAffineKv` and `packedMixedMlp` are
      composition fields, default on, with no env override. Behavior that changed, accepted:
      prefix and checkpoint entries stored at the conversion start are saved converted (reuse
      is identical; a request diverging before the start recomputes instead of reusing);
      speculative prefix namespaces and method keys dropped their KV fields, so entries written
      before E3 miss once. Found and assigned: under a quantized scheme the generated Gemma 4
      graphs matched on raw cache signatures and fell to the monolith on every forward (numbers
      unchanged, fast path lost); the merge of E1 + D2 + E3 deletes that run-time match and the
      table selects them on the scheme (gate: `gemma4-generated` asserts the generated forward,
      digests equal `30412523`). The deprecated attention views and the donor reads still pick
      the decode kernel by query count (E2 deletes the views; the donor reads ride the Gemma
      assistant-drafter item). Real-weight tests E3 ported but did not run are the merge's to
      run: `gemma4-generated`, `compiled-decode`, `ordinary-continuation`,
      `affine-grammar-spans`, `gemma4-parity`, `minicpm5-parity`, the Trellis parity tests.
      Merged (c691ba23): all of those ran on both trees and match line for line, except
      `affine-grammar-spans` on e4b, which fails identically on base (the plan routes e4b's
      grammar jump through the speculative method, so e4b is not a model that test accepts).
      Under `--kv-quant 4` the Trellis app-path digests differ from base: base served the
      generic read (D1's regression, logits 0.028), merged serves the composed KV4 lego, which
      E3's gate proved equal to the old graph's injected cores. Closed: the app-path digests on
      `main` (9593d881) under kv4 with `--fused-sdpa` unset, on and off, and bf16, equal
      d71467c4's token for token and logprob for logprob (`69e8cc07…` / `ee58a136…` for kv4),
      so the loop is closed through the scheduler. The generated Gemma graphs needed no new read: with D2 merged they have no
      run-time match, the template builds the bf16-first lego on every listed cache, and the
      gate records every cache read (bf16 first, quantized after) with digests equal to base.

Exit: `serve-cli.test.ts` and the draft-flags test pass; the CLI inventory regenerates with no
new flag.

#### E4. DFlash 2

- [ ] `Dflash2Provider` loads the drafter unrotated and folds `fc` when it binds, from the
      target's declared basis. `MLX_BUN_DFLASH2_TARGET_BASIS` is deleted. (PR #334, draft.)
- [ ] The basis needs a recorded source before #334 can merge: the quantizer folds the
      final-norm gain γ into `lm_head` and writes the norm as zeros, and nothing in the artifact
      records γ, so no graph can declare it today and the drafter would run unrotated
      (acceptance collapses, output stays exact). Fix: `packages/quantize` records `final_gain`
      (γ = stored + 1 of the source's final norm) in `turboquant_fold.json` beside the seed;
      the published `mlx-bun/Qwen3.8-27B-Trellis-3.2bpw` sidecar gets the values computed
      from the bf16 source (a one-off, where that source is available); each Trellis graph
      reads the sidecar once at construction and declares `residualBasis` through
      `draftTarget`. Resolved 2026-10-11: the basis JSON was on the M1 Max share
      (`mlx-bun-perf/models/Qwen3.8-27B-Trellis-3.2bpw-dflash2-basis.json`, gain taken from
      the bf16 final norm of a sibling quant of the same source, effective 1+w), and the
      published artifact's `turboquant_fold.json` now carries `final_gain`, its convention and
      its source (hub commit `d5a4fc64`). E4b makes the graphs read it and the quantizer write it.
- [ ] The drafter builds its matrix-unit layer for its fixed block width at construction and uses
      the ordinary quantized matmul for context projection; no per-call row switch in `matmulW`.

Exit: acceptance rate on the paired MTP bench unchanged; no env read in `dflash2-source.ts`.

#### E5. Drafters compose like graphs

Found by review on 2026-10-11 against the integration tree.

- [ ] The MTP draft loop (`generation/speculative/bindings/qwen-mtp-rows.ts`) decides per token
      whether a listed draft vocabulary exists, tries that projection, falls back to the full
      head, then chooses greedy or sampled selection. The draft group binds one selection
      operation at composition (projection over the listed vocabulary or the full head; greedy
      argmax or the sampler's draw) and the loop calls it on the hidden output; no branch in the
      loop. The same for the DeepSpec and assistant sources where they repeat the pattern.
- [ ] `DraftProvider.open()`, the code's per-request draft source, is deleted from the contract.
      There is one lane and a single request is a batch of one (Josh, 2026-10-11: "we only have
      batched, and serial should just be a batch of one"); `Dflash2Provider.open()` throwing
      "runs only in the batched speculative lane" is the evidence that the method is a promise
      no composition keeps.
- [ ] The MTP companion's forward calls its named phase, not `deprecatedPhase` (rides E2's
      deletion); its draft operation is width-classed like the graph's `verify` (the DFlash 2
      entry under E2 covers both drafters).

Exit: no `sampling!.vocabulary` / `greedy` branch inside a draft loop; no `open()` on
`DraftProvider`; for fixed inputs under teacher forcing the drafted tokens on e4b (assistant) and
on the M4 Pro composition (MTP, DFlash 2) equal today's bit for bit; paired MTP bench acceptance
unchanged.

### Phase F: gate and documentation

- [ ] Add the rules from the top of this file to `architecture.test.ts`, each proven on the
      synthetic workspace, with a ratchet table for files that still break them on landing, each
      entry naming the plan item that removes it. Entries F1 found with no owner, now assigned:
      - `execution/batch-group.ts` hot-path reads of `MLX_BUN_BATCH_VEC_SAMPLE` (a sampling-path
        choice: a composition fact, E2) and `MLX_BUN_GRAMMAR_DEBUG` (a diagnostic: add it to the
        development-hook allowlist by name, F1).
      - `models/glm52/model.ts` `instanceof MLACache`: the GLM-5.2 MLA contract item listed under
        B1's out-of-step work owns it; until that item has a number, it rides the ratchet.
      - `models/minicpm5/model.ts` `MLX_BUN_COMPILED_SWIGLU`: D3, where a generic graph's env
        switches become constructor choices.
      - `models/qwen/qwen3_5.ts` `Qwen3MLP.forward` `instanceof TrellisLinear`/`QuantizedLinear`:
        C1, where the MLP block takes its projection type at construction.
      - `state/layout.ts` `prefillCacheLayout` branching on the cache class: B2, where the row
        layout reads the cache's own declaration.
      - `models/qwen/mtp.ts` `makeMask`: D3, the MTP companion reads through `attend`.
      - `packages/training/src/kernels/flash-cce.ts` per-dispatch `runtimeValue`: C2's rule,
        the choice becomes a constructor argument of the training kernel's owner.
- [ ] Each new rule's failure message names the paved road, in the style of the existing
      "scheduling reaches storage through the row-layout port" message.
- [x] Make the library checks a required status check on `main`, so a PR with a failing gate is
      not mergeable (#312 merged with its checks still pending).
- [ ] Automatic performance-regression check. GitHub's runners have no Apple GPU, so this needs
      a self-hosted runner on a quiet named Mac (the M4 Pro, or the M1 Max over SSH). Runner
      installed 2026-10-11 on the M1 Max (Josh's go): name `m1max`, labels `self-hosted, macOS,
      ARM64, m1max, gpu`, a launch agent in the logged-in session under `~/actions-runner`. No
      prior M1 Max numbers exist (the refactor came before measurement moved there); the check
      is paired, head against base on the same host in one run, so its first run on the
      integration PR is the first M1 Max measurement of both. Landed on the integration branch
      (918a6afb): `.github/workflows/perf.yml` (pull requests to main touching kernels, layers,
      models, state, execution, mlx or native sources; job-level guard so a fork's PR schedules
      nothing on the runner; read-only token; one measurement at a time, never cancelled) and
      `scripts/bench/perf-check.ts` (main's preflight thresholds; the paired delta as the
      geometric mean of per-request ratios with a 95% Student-t interval; a metric fails only
      when the whole interval is on the worse side; exit 0 pass / 1 regression / 2 no qualified
      numbers). First run, 2026-10-12 on #339 (918a6afb vs main 9593d881), preflight clear
      before and after, default configuration (bf16 KV, no drafter): Trellis 27B on the TQ
      graph decode 14.4 → 14.3 tok/s (0.994 [0.989, 0.999], flagged), prefill 1k 123.4 →
      125.5, cold TTFT 1k 6136 → 6032 ms; e4b on the generic graph decode 64.4 → 64.4, prefill
      993.6 → 1001.1, TTFT 674 → 670 ms, all within noise. The flag is the per-step cost E2
      removes (the TQ graph still runs through `deprecatedPhase` and the bf16-first wrapper on
      every decode step); rerun after E2. Follow-ups: each tree runs in one server session, so
      the interval covers request noise but not session drift; alternate sessions (ABAB) so
      the band includes it. Add the DFlash 2 adaptive configuration (`--kv-quant 4`,
      `--num-draft-tokens adaptive`) to `scripts/bench/plan.ts` so the served composition is
      what the check measures; the drafter is in the M1 Max's hub cache since 2026-10-11. For a PR
      that touches `kernels/`, `layers/` or `models/`, run the existing paired serve benchmark
      against `main` with the preflight gate, and fail on a regression outside the paired noise
      band. Unqualified numbers never gate; the check reports the paired delta and its interval.
- [ ] Agent briefs for this plan point at the generator and the contract, never at an example
      file to copy; copying the nearest example is how the M4 Pro graph inherited the fallback.
- [ ] The app-path digest (token ids plus top-8 logprobs over 32 greedy tokens through
      `generate`, per KV scheme and `--fused-sdpa` setting) and the teacher-forced decode KL
      (the 16 default prompts, last 32 tokens forced, `klPerToken`) become
      `packages/inference/scripts/` tools with `--help`: three agents reconstructed each from
      earlier transcripts (E3, the merge, the Trellis checks). Their consumer is every gate in
      this plan and the perf workflow's correctness check.
- [x] ARCHITECTURE.md: the cache owns the read; phases are named; a specialized graph extends
      nothing; the exemplar sentence points at the regenerated graphs; the loader-level generic
      fallback is stated and the in-graph one forbidden; the one-way-per-pattern list above.
- [ ] PLAN.md: first decision updated per the off-composition decision; this file's closed phases
      deleted.
- [ ] `packages/inference/README.md`: the M4 Pro graph paragraph describes phases, not plans.

Exit: `bun test packages/inference/tests/architecture.test.ts` passes with the ratchet table
empty; documentation generators run clean.

### Verification, every phase

- Kernels and their order do not change in any step, so each graph and cache step is gated by
  bit-identical logits per phase against the implementation it replaces, on the existing parity
  inputs. Folded attention and ANE keep their KL gates.
- `bun run typecheck`, `bun run test`, and the architecture gate on every PR.
- Speed only through `mlx-bun serve` in the batched lane with the paired benchmarks, ms per round
  for MTP, on a quiet named machine.
- One PR per numbered item, targeting `main`, not merged without Josh's instruction.

### Untouched

The 13 single-purpose kernels from PR #312, the four one-kernel layers, the ANE native bridge,
the DFlash 2 numerics, the bench scripts, modules, web, training.

## Verify the migrated library

- [ ] Establish real-weight parity for Qwen Trellis and Gemma, and extend MiniCPM
  beyond the [verified basic decode path](packages/inference/README.md#external-parity-evidence),
  against the applicable external oracle and main: logits, state continuation and
  specialized paths under identical artifacts and settings. Python environments and
  generated references stay outside the repository; reusable runners are in Git
  ([runtime comparison](packages/inference/README.md#repeatable-runtime-comparison)).
  Exit: opt-in comparisons meet the numerical contracts against pinned published
  golden revisions recording source revisions and oracle versions; synthetic tests
  alone do not close this. Done so far: the
  [generated Gemma4 graph test](packages/inference/tests/parity/gemma4-generated.test.ts)
  passed on e4b and 12B (identity within this tree; 26B-A4B not run). Packed Trellis now matches main in 96 full greedy logit vectors, three shared
  B1/depth-two MTP runs, and the 18-case plain-KV runtime grid with restored
  continuation ([evidence](packages/inference/README.md#packed-qwen-trellis-parity)).
  Open: published references and broader Trellis shapes;
  [runtime-oracle](packages/inference/tests/parity/runtime-oracle.test.ts) references
  for Gemma and extended MiniCPM (MiniCPM mixed KV waits on the next item);
  real Gemma2 B2 rows and a different prefill chunk geometry differ from main's B1 in
  logits and valid KV, which needs same-shaped oracle or main comparisons, not a tolerance.
- [ ] Extend speech and embedding evidence. Whisper large-v3-turbo, Silero VAD and
  Qwen3-Embedding-4B-4bit-DWQ match the external oracle and main bit for bit (Whisper fast
  path within main's two-token bound; VAD within 5e-3 of torch) on the
  [opt-in comparisons](packages/inference/README.md#speech-and-embedding-parity). Open: other
  Whisper sizes and non-WAV input, other embedding checkpoints, batched embedding, and
  performance. GLM-5.2 (declared L1) has no local weights and no current numerics evidence, so
  its graph, streamed experts and MTP are unverified against the oracle or main.
- [ ] Confirm the long-term mixed-KV reference contract for single-query decode:
  the documented stock mlx-lm path or OptiQ serve's fused default. The opt-in
  state comparison verifies the existing composition; keep that implementation
  unchanged during this refactor and do not imply blanket OptiQ serve compatibility.
- [ ] Extend training preservation. MiniCPM5 SFT, ORPO and DPO match main exactly
  through the app's [fine-tune preservation test](apps/mlx-bun/tests/engine/finetune-preservation.test.ts),
  as do short Gemma4 e4b SFT pairs and Qwen3.5-0.8B SFT with one LoRA layer (main trains
  Qwen3.5 only when no DeltaNet layer is on the gradient path; here the DeltaNet recurrence
  has a backward, so more layers, DPO and batches train with no main reference);
  DiffusionGemma save, reload and seeded
  generation and its HTTP adapter serving are consistent within this tree (not parity).
  Open: other model families and main-only paths (by_bits/by_kl, warm start); synthetic
  native tests do not close this.

## Improvements identified during migration

Concrete improvements with the current limitation and the owning domain. Preserve
shipped behavior first; an improvement is not permission to redesign during migration.
Migration gaps stay required work in the feature table.

- [ ] (audio) `transcribe --vad-trim` is accepted but not applied, as in main, though the
  service implements `vad.trim`.
- [ ] (memory) The entity seed gold is personal data read from a repository path; seed from
  the vault's aliases (`EntityResolver.fromStore`) and drop the file.
- [ ] (memory) `--since` and `--model` are parsed and never consumed (main reserved them
  too); consuming them is a decision.
- [ ] (isolation) The worker's `/health` reports only `ready`/`draining` because the socket
  binds after the model loads; a pre-bind stub would let the parent observe `loading`.
- [ ] (isolation) Pi's SDK retries a 502 refused before generation (three attempts, 2/4/8 s)
  without a browser frame; mapping `auto_retry_start` to a frame would show the wait.
- [ ] (isolation) The worker keeps its own bounded Responses store for requests the parent
  already records; honoring `x-mlx-bun-response-owner: parent` would skip the duplicate.
- [ ] (isolation) The parent's `/health` answers 200 while the engine is exhausted; a distinct
  status for load balancers is a policy decision.
- [ ] (capability reporting) `/v1/models` reports DiffusionGemma with `vision: false` (as main)
  though image requests succeed.

## Split the app into modules

- [ ] Split `apps/mlx-bun` into modules on core services, as the repository split into apps and
  libraries (design: [Modular application](ARCHITECTURE.md#modular-application); contracts:
  `packages/app-core`). Modules: transcription, chat, models, train, quantize, datasets, memory,
  benchmarks (answer quality, `scripts/eval-serve.ts`), metrics and performance (live tokens/s, TTFT,
  batch occupancy, queue, KV/prefix usage and hit rate, memory per loaded model, swap and load
  times; launching `bench-serve` profiles with history). An agentic workflow engine is a later module,
  not planned here. Every step keeps existing paths, verbs and behavior: moved tests keep their
  expectations, and steps that move an execution path rerun real weights before and after.
  - [ ] (d) Memory module migration under validation (moved as is; the memory feature stays deferred).
    Datasets, quantize, benchmarks, train, models and chat landed. Datasets', quantize's and train's
    browser pages are still legacy pages that become panels the web shell mounts. Memory reaches chat
    through `registry` (`chat.tool`, `chat.guidance`); its panel owns the memory entry, consent card and
    provenance chips, wired through the chat panel's `host`. The chat panel owns its markdown
    renderer, hold-to-talk mic (calling transcription over HTTP), and the assistant's app catalog, which
    should come from contributions once the shell lists them. Models owns hub, library, cleanup, adapter
    administration and its panel; chat owns selection of adapters for a turn. Exit per module: its domain
    leaves `apps/mlx-bun/src`, its browser page becomes its panel, the app domain map shrinks, and
    served-surface inventories are unchanged.
  - [ ] (e) Model host events reach every consumer. Residency, live switching, saved state and the model router landed
    (`serve` runs one isolated worker per resident model by default, held by memory fit in the parent; `--in-process`
    loads them in the serving process; the two share one residency manager), the host publishes `model.load` (with
    `resumed` when saved state was found), `model.unload` (reason `evicted`, flushed) and `model.memory`, and each worker's
    own events are relayed onto the parent's bus. The in-process composition now shares one `modelHost` across
    persistent and model-scoped modules for `generate`, `adapters` and the borrowed Whisper `transcribe` operation.
    Isolated HTTP voice discovery and idle unloading preserve the configured companion's identity and policy.
    Remaining: the isolated parent's typed module host also leases `transcribe` from its companion worker.
    Exit: a module acquires
    `generate` and `transcribe` from one service (the models module already leases `adapters` and switches the served model through the state's host).

## Remaining features by layer

One row per feature, one column per layer it crosses. A feature closes only when every cell is
done or a decision marks it dropped; delete the row then. Cells: `done` · `partial` · `missing` ·
`501` (route stubbed) · `n/a` · `held`. Evidence means an opt-in real-weight test that passes against
a pinned published golden or regenerates it; synthetic tests never fill it. This repository carries
code, tests and documentation; goldens, records and benchmark data are published datasets, and
experiments and benchmarks live outside it. Package READMEs hold the evidence detail; keep cells short.

| Feature | Library | Engine / host | Server | CLI | Web | Evidence | Open work or decision |
|---|---|---|---|---|---|---|---|
| Audio: Whisper transcription, Gemma4 audio input, voice sessions | done | done (lazy Whisper companion, exclusive-lock takes) | done (`/v1/audio/*`, `/admin/transcription/unload`, served by `@mlx-bun/module-transcription`) | done (`transcribe`, `dictate` from the module; `--whisper-*` on serve; `apps/transcribe` hosts the module alone) | done (hold-to-talk composer dictation; no upload or translate controls, as main) | partial: Whisper large-v3-turbo and Silero VAD match the external oracle and main ([speech parity](packages/inference/README.md#speech-and-embedding-parity)); Gemma4 e4b audio-only HTTP (B1/B2) matches main; real-weight voice, companion and mixed-media consumers pass; [details](packages/module-transcription/README.md#audio-transcription) | Open: physical-microphone `dictate` (mic, key tap, clipboard, typing); Gemma4 audio with KV quantization and other variants; main parity for mixed image+audio and e2b (app-behavior runs only); Gemma4 audio external-oracle parity; other Whisper checkpoints; performance |
| Memory synthesis (nightly pipeline) | n/a (app-owned; pipeline ported op-for-op) | done (task model on the continuous gateway under the engine's execution lease; isolated: the current worker's `/admin/memory/complete`) | done (`GET /v1/memory/synthesize`, `schedule` in `/api/memory/status`) | done (`memory` verbs incl. `init`/`setup`, `schedule`) | read panel done; no synthesize control (as main) | partial: stages match main token for token and in full logits, width-3 batch, isolated synthesis; [details](apps/mlx-bun/README.md#memory-synthesis) | Decision: the memory feature is incomplete and deferred; the ingest source is unchanged (Pi's global sessions directory, as main) |
| Live model switching, residency, isolation | n/a | done (one crash-isolated worker per resident model by default, held by the parent by memory fit; `--in-process` loads them in the serving process; either way the LRU model is drained with its saved state) | done (`POST /api/hub/serve`, routing by request `model`, `/v1/models` and `/library` with residency; worker socket: `/health`, `/admin/lease`, `/admin/drain`, `/admin/memory/complete`, `/admin/events`, `/admin/adapters`; isolated parent: `/engine`, `/health`, `/stats`) | done (`--model-budget`, `--ssd-cache`, `--in-process`; `--isolate` is accepted and ignored) | the Models panel switches live | partial: two small models resident and serving concurrently, a swap through saved state that resumes the prior conversation, and a pinned Whisper, in the default isolated mode and with `--in-process` (Qwen2.5-0.5B, MiniCPM5-1B, Whisper large-v3-turbo), and a worker killed mid-request answers 502 and reloads and resumes; no oracle parity; [details](apps/mlx-bun/README.md#model-host-residency-by-memory-fit) | Decisions (Josh): residency is by memory fit, saved state makes a swap feel like two models running, isolation is the default (one worker per resident model; `--in-process` opts out), and web chat works under isolation. Internal worker routes on TCP (`/admin/lease`, `/admin/drain`, `/admin/memory/complete`, `/admin/events`, `/admin/adapters`) and `/engine` under `--in-process` answer 404, not 501. Small items: Improvements |
| Speculative decoding: draft models, n-gram, MTP, DSpark/DFlash | done | partial (grouped draft required; ungrouped shapes get the typed error) | rides completions | done (`--draft-model`, `--draft-kind`, `--num-draft-tokens`, `--ngram-*`, `--mtp`; `draft regen|train|calibrate|quantize` produce drafters) | n/a | partial: B1 main preservation for n-gram (MiniCPM), Gemma4 two-model and Qwen3.8 Trellis MTP; [grouped consumer](packages/inference/README.md#speculative-generation) ran for n-gram, two-model and assistant | Open: the grouped consumer for MTP, DSpark, DeepSpec (no local drafts fit 32 GB) and GLM-5.2 native MTP (no local weights); other main-supported provider/target/cache combinations; equality with main or an oracle; HTTP; performance |
| Paged KV | done | partial (Gemma4 only; others get the typed error; media and adapter rows bypass paging) | rides completions | done (`--paged-kv`, `--paged-kv-block-size`) | n/a | done for Gemma4 E4B/12B: gathered pages equal plain KV bit for bit, direct reader within tolerance, KV4/KV8 pages, HTTP cancellation per reader, cold prefill; [details](packages/inference/README.md#state-and-attention) | Open: warm-prefix (prompt cache on) identity over HTTP (main's `paged-cache-http` test is not ported); broader model support is the cache lego in the Composition refactor (B1) |
| Unsupported request shapes (typed error) | done | typed `UnsupportedExecutionError` per shape | typed 501 envelope | `generate` prints it, exit 1 | n/a | per shape once supported | Shapes become compositions under the Composition refactor (Phase E); the typed error stays for a request the composition was not built for |
| DiffusionGemma denoising (shared execution) | done (interleaved rows, B=1 graph calls) | done (placed continuously; grammar, draft, logprobs, logits processors, fill, encoded and paged KV refused with typed reasons) | rides completions | rides `generate` | n/a | partial: text and one-image requests match main's measured trajectories (direct, grouped, B2/B4 join and cancel, HTTP), image cancellation and disconnect recovery pass; [details](packages/inference/README.md#optional-execution-and-persistence) | Open: stacked (B>1) canvases and token-chunked prefill (separate optimizations); concurrent HTTP rows, full logits and KV planes, external-oracle parity, performance. Main's `goldens/diffusion/gen*.json` no longer match main on MLX 0.32.2, so references are measured |
| Evaluation and benchmarking: `bench`, `evals`, `perplexity`, EvalDB, eval tasks | n/a (experiments; the in-package bench harness moves out) | n/a | `/fit` measured fields stay null | not app verbs | status page shows dashes | n/a | Runs consume the published packages and drive the app through its public HTTP and CLI surface; results are published as datasets or quoted as text in docs. The benchmarks and metrics modules ([split](#split-the-app-into-modules)) launch these runners and show history; they add no second runner |
| Pi terminal: `pi`, `harness pi` | n/a | chat backend reusable | n/a | held (`pi-terminal`, `harness-pi` not ported) | n/a | n/a | Held by Josh |
| Managed jobs: quantize, fine-tune, `convert`, `fuse`, datasets | n/a | done (each job child leads its own process group; stop is SIGTERM then SIGKILL after 3 s; the child stops its group when the parent's stdin pipe closes) | done (job, dataset and adapter routes) | done (`train`, `convert`, `fuse`) | job views done | partial: finished and merged fine-tune adapters mount, unmount and reproduce in a fresh process, and `fuse` output loads and generates (MiniCPM5-1B, opt-in `managed-jobs.test.ts`); [details](apps/mlx-bun/README.md#jobs-quantization-and-fine-tuning) | Decision: a managed job keeps running when its terminal closes (quantize's synchronous sensitivity sweep cannot observe the closed pipe): accepted. Verified on M1 Max: all four managed-job cases (including BF16 sensitivity cancellation and uniform conversion), isolated memory synthesis/cancellation, base memory stages and width-3 batching, and standalone library-host build. The trained memory-chunk adapter case still skips without its artifact. Relocated compiled app, installed current/previous bundles, worker and managed-child reentry passed (`verify:binary`; #295). |
| Existing-user data compatibility: prior-format sessions, jobs, settings, vault, caches; memory Reference symlinks into the main checkout | n/a | partial (Pi sessions, browser preferences and active-job prior schemas read across versions; the web chat sidebar lists every recorded chat; old `~/.cache/mlx-bun*` job history, memory and registry data are left in place, not migrated; adapter stores are listed read-only) | n/a | n/a | n/a | partial: synthetic old/new round trips (Pi sessions, preferences, jobs) and an acceptance on an isolated copy of real data pass (53 sessions, 10 adapters); the credential and schedule paths only synthetically (no saved token or nightly plist in the real data) | Decisions: everything mlx-bun writes by default lives under `MLX_BUN_HOME` (`~/.mlx-bun`); the HF hub cache holds downloads only; explicit paths win (#224); the opt-in `--expert-offload` still builds inside the model directory. Before deleting the old checkout Josh selects what to preserve outside it: the memory Reference symlink targets (7 point into it; 2 already dangle) and ignored adapters, checkpoints and other local artifacts that Git cannot recover. No automatic retargeting. Open: opening a chat appends SDK entries and opening a v1/v2 chat rewrites it, as in main |
| Docs surface and gates | n/a | n/a | HTTP and configuration inventories generated from source | CLI inventory generated from source | site and legacy redirects implemented | `apps/website/tests/server-api.test.ts`, `apps/website/tests/server-config.test.ts` | Per ARCHITECTURE: generate inventories from source as build-only output with a coverage gate, and write the explanations by hand (models, environment, training, memory, distribution, troubleshooting; quotable numbers as text with provenance in a benchmarks page). No STATUS file, docs map, or ledgers are restored. |
| Installation and release | n/a | n/a | n/a | bundle, launchers, safe installer, formula and staged release preparation implemented | n/a | n/a | Complete the release acceptance checks below; public installer delivery must follow a compatible bundle release. Decision: the installer links only `~/.local/bin` (accepted). No actual release. |

## Candidate quant and Hugging Face model card

This work characterizes the first quant published under the `mlx-bun` organization.
Use a validated engine to evaluate the artifact; duplicating quality evaluations
on main and the refactor is only useful when investigating a behavior difference.

- [ ] Audit the completed M4 Pro campaign for reusable scores: exact artifact and
  source revisions, dataset splits, prompts/templates, thinking mode, sampling,
  token budget, answer extraction, scoring and truncations. Retain scores whose
  protocol can be reproduced and accurately named; identify missing evidence.
- [ ] Select established benchmark protocols that support comparison with the base
  model or popular quants. Use the upstream runners, task definitions and scorers;
  record versions and full dataset coverage. Compare matching protocols, including
  thinking mode and generation budget. Label externally reported results and their
  settings; unresolved protocol differences cannot establish relative quant quality.
  Frozen subsets and different MMLU task variants must be named explicitly.
- [ ] Prepare model-card performance numbers from qualified measurements on M4 Pro,
  M1 Max or both. Record machine, engine/build, context, batch, KV configuration,
  cache state and MTP alongside prefill/decode, request latency and memory as measured.
- [ ] Document the quant's creation from the actual manifests: base model revision,
  quantizer revision and commands, rotation/folding, calibration and mixed-precision
  allocation, packing, final size and runtime requirements. Replace the inherited
  base-model card with the candidate's own measured results and reproduction details.
  Exit: the artifact and model card are ready to review and publish under `mlx-bun`.

## Release acceptance

- [ ] With Josh's release instruction, verify real Developer ID signing and
  notarization, including loading the signed native libraries from a relocated
  signed bundle. The native-blocked version check and mocked commands do not
  establish this acceptance.
- [ ] Reproduce the remaining notices of code embedded in prebundled dependencies.
  `babel-plugin-parameter-decorator@1.0.16` in Jiti's prebundle has no license text
  upstream: only its manifest's `"license": "MIT"` and `"author": "Warner"`, its npm
  `gitHead` is not in its repository, and no commit there adds a license file, so a
  full text must come from its author. The toolchain that built XGrammar's WASM is
  unknown (no producers section; a local build): the inference notice reproduces
  Emscripten 3.1.56's texts because the runtime glue is a release-era match for 3.1.56,
  inferred from two fingerprints (WebAssembly-check absence and `Module["ready"]`
  presence), not exact provenance. Whether the stripped WASM links LLVM compiler-rt is
  also unknown; if it does, its notice needs that Emscripten's
  `system/lib/compiler-rt/LICENSE.TXT`.

## Completion

- [ ] Josh could delete the pre-refactor reference branch (`origin/pre-monorepo`, main at `02d723a`) without losing a capability it shipped.
  Exit: every row above is all-done or carries a recorded decision and is deleted;
  the verify items above close against pinned published goldens or are recorded
  decisions; main's user flows (CLI verbs, HTTP protocols, web chat, jobs,
  installed and compiled artifacts, existing user data) are accepted with real
  weights; documentation follows ARCHITECTURE (generated inventories, handwritten
  explanations, no ledgers); no capability main shipped is retired silently.

## Ideas and research

One line per idea: the question and the domains it touches. Picking one up makes it an item
above with an exit; an answered one moves its finding to the owning documentation and is deleted.

- **SkillOpt skill optimization** (chat, memory, a new optimization domain; [research](https://github.com/joshuarossi/mlx-bun/pull/247)):
  can SkillOpt's gated skill-edit loop (Microsoft, MIT, arXiv:2605.23904) improve the default
  assistant's prompt and skills and the memory stages' `Meta` policies with local models as target
  and optimizer, and should the loop run inside mlx-bun? First experiment: a `scripts/` runner using
  SkillOpt's `openai_compatible` backend against `mlx-bun serve` on one verifier-scored task set,
  measuring the optimizer's edit-JSON parse rate and the held-out gain with a 4B–27B optimizer.
