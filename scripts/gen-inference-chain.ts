// Generates docs/inference-chain.svg: the modules of the inference chain and the features each one
// owns, as ARCHITECTURE.md and PLAN.md (Composition refactor) assign them. Edit the tables below and
// rerun; scripts/gen-inference-chain.test.ts fails when the committed figure is stale.
//
//   bun scripts/gen-inference-chain.ts [--out <path>]
import { parseArgs } from "node:util";
import { join } from "node:path";

type Box = {
  id: string; title: string; path: string; role: string; items: string[];
  row: number; col: number; span?: number; fill: string; stroke: string;
};

const COL_W = 352, GAP = 132, MARGIN = 30, ROW_GAP = 76, TOP = 96;
const LINE = 17, CHAR_W = 6.35, PAD = 14;

const BOXES: Box[] = [
  { id: "server", title: "Server", path: "apps/mlx-bun/src/server", row: 1, col: 1, fill: "#eef3fb", stroke: "#4a6fa5",
    role: "Protocols in, streams out. Decides nothing about the model.",
    items: [
      "OpenAI: /v1/chat/completions, /v1/completions, /v1/responses, /v1/embeddings, /v1/models",
      "Anthropic: /v1/messages",
      "Audio: /v1/audio/transcriptions, /v1/audio/translations, streaming audio sessions",
      "Adapters: /v1/adapters (list, mount, remove, available)",
      "Request shapes, prompt policy, SSE streaming, JSON",
      "Tool-call and reasoning markers surfaced as API fields (declared by the family record)",
      "Borrows the engine; serve startup owns it (serve-state / serve-host)",
      "Web UI and /api routes belong to app modules, not the chain",
    ] },
  { id: "input", title: "Input", path: "inference/input", row: 1, col: 2, fill: "#eef3fb", stroke: "#4a6fa5",
    role: "Text and media become tokens.",
    items: [
      "Tokenizer, special tokens, token groups",
      "Chat templates and the family's chat defaults",
      "Tool-call formats: parse and emit",
      "Vision preprocessing: images to patches and soft tokens; media fetch",
      "Audio features for Whisper",
      "Prompt assembly through encoder interfaces",
    ] },
  { id: "loader", title: "Loader", path: "inference/models (factory, implementation, composition), artifacts, adapters, @mlx-bun/hub",
    row: 1, col: 3, fill: "#fdf3e0", stroke: "#b8860b",
    role: "Resolves the composition once from model + flags + machine, then builds everything the scheduler runs.",
    items: [
      "Composition: device, ANE bridge, KV scheme (--kv-quant), draft depth (--num-draft-tokens), prefill chunk, adapters, max rows (--batch-size)",
      "Graph table: the most-constrained graph wins; exactly one generic graph per family",
      "Artifacts: config dialects, safetensors (mmap), Trellis codebooks, TurboQuant fold, expert offload, fingerprints and identity",
      "Family records: marker tokens, chat defaults, media soft tokens, memory plan",
      "Drafter loading: MTP head, DFlash 2 basis, assistant model (--draft-model)",
      "LoRA adapters mounted onto named targets (--adapter-path)",
      "Fit estimate and memory plan",
      "ANE program compile at install time (planned)",
      "Model download, search and upload: @mlx-bun/hub",
    ] },

  { id: "sampler", title: "Sampler", path: "inference/sampling", row: 2, col: 1, fill: "#f3eefb", stroke: "#7b5ea7",
    role: "Logits become tokens.",
    items: [
      "Temperature, top-k, top-p, min-p, XTC",
      "Processors: repetition, presence and frequency penalties, logit bias",
      "Grammar constraints: JSON schema or grammar to token masks",
      "Draft acceptance sampling for speculative decoding",
      "Curve sampler (log-prob transfer curve), window filters",
      "Seeded draws: key derived from (seed, step); only the token id crosses to JS",
    ] },
  { id: "scheduler", title: "Scheduler", path: "inference/execution (+ generation/ methods)", row: 2, col: 2, fill: "#e9f6ea", stroke: "#2e7d32",
    role: "The only component that decides at run time.",
    items: [
      "Admission, queue, cancellation, sessions and continuations",
      "Continuous batching: rows up to --batch-size; one request is a batch of one",
      "Names the phase and width: prefillChunk, prefillTail, decode, verify, logits (width classes 1, 2-3, 4, 5-8, wider)",
      "Dispatches prefill chunks to the GPU or the ANE: the ANE is a compute unit it manages",
      "Mixed iteration: prefill rows and decode rows in one round",
      "Speculative rounds: draft, verify, accept; adaptive depth over the built widths",
      "Methods: autoregressive, speculative, denoising, fill, grammar groups",
      "Asks the cache for prefix hit or miss; KV budget; fit",
      "Row sampling: hands each row's logits to the sampler",
    ] },
  { id: "drafter", title: "Drafter", path: "inference/generation/speculative/sources, models/speculative", row: 2, col: 3, fill: "#e6f4f4", stroke: "#1f7a78",
    role: "A second graph proposing tokens; the target verifies them.",
    items: [
      "Qwen MTP head, GLM-5.2 MTP",
      "DFlash and DFlash 2 (residual basis captured from the target)",
      "DeepSpec drafter, n-gram, assistant model (two-model)",
      "Built by the loader; borrows the target's KV and hidden states through the DraftSource seam",
      "draft() returns 1..n tokens; the scheduler prunes and verifies",
    ] },

  { id: "graph", title: "Graph", path: "inference/models", row: 3, col: 1, span: 2, fill: "#fdeee3", stroke: "#c25e1c",
    role: "One entire graph per model x quant x machine. Extends nothing; never inspects its input or its cache.",
    items: [
      "Generated: Qwen3.8-27B Trellis 3.2bpw (M4 Pro), Gemma 4 e4b / 12B / 26B",
      "Generic, one per family: Qwen3 / 3.5 / MoE, Gemma 4, MiniCPM5, GLM-5.2, Llama (universal dense), DiffusionGemma, Whisper",
      "Phases: prefillChunk { gpu, ane }, prefillTail, decode / verify / logits width tables",
      "makeCache(): composes the cache lego its composition names",
      "Layer captures: residual basis and hidden taps for drafters",
      "Declared capabilities: vision, audio, draft head, pooled embeddings; vision and audio towers",
      "Forward for LoRA training (consumed by @mlx-bun/training)",
    ] },
  { id: "cache", title: "Cache", path: "inference/state", row: 3, col: 3, fill: "#fbeaf0", stroke: "#b03060",
    role: "Owns storage and the read. Answers hit or miss.",
    items: [
      "Storage legos: bf16; affine uniform (--kv-quant N, mlx-lm parity), tiled-causal (OptiQ parity), tiled; TurboQuant dense / deferred; KV4 head-256 (M4 Pro); bf16-first (first append bf16, then quantized: the default)",
      "The read: attend(q, scale); masks memoized once per forward",
      "Rotating / sliding-window KV; recurrent state (GatedDelta, SSM)",
      "Paged rows and row layouts for batching; row transactions",
      "Prefix reuse (prefix cache, tiered); SSD offload (--ssd-cache-max, persistence worker)",
      "Eviction and retention policy, KV budget, leases",
      "Snapshots and rollback for speculative verify; donor reads for drafters",
      "Adapter-namespaced caches; encoder cache",
      "Delayed quantization legos take their kernel set at build",
    ] },

  { id: "layers", title: "Layers", path: "inference/layers, models/qwen/blocks, models/gemma4/blocks", row: 4, col: 1, span: 2, fill: "#eef1f5", stroke: "#53627a",
    role: "The blocks a graph is made of. Each imports the kernels it needs.",
    items: [
      "Attention: Qwen3, Gemma, KV4 decode, quantized-append; the read strategy is cache.attend",
      "GatedDeltaNet recurrence; MLP: SwiGLU, GeGLU, Trellis gate-up / down, affine",
      "Linears: dense, quantized, Trellis (variant 13), switch (MoE experts), affine-verify",
      "Embeddings (dense, quantized); norms (RMS, layer norm); softcap",
      "RoPE, mRoPE, 2D RoPE; LoRA application on named targets",
      "ANE prefill split: instance-owned programs for exactly the chunk size",
      "Per-layer MLX compile choice (measured, never an app option)",
    ] },
  { id: "kernels", title: "Kernels", path: "inference/kernels, native/", row: 4, col: 3, fill: "#eef1f5", stroke: "#53627a",
    role: "One op, one way, measured.",
    items: [
      "attention: fused SDPA, KV4 decode attention, softcap",
      "quantization: affine quant / dequant, qmv, steel qmm",
      "trellis: codebook decode, fused gate-up / down, qualified representations",
      "turboquant: codec (rotate, encode, decode)",
      "delta: gated delta rule; glm52: MLA",
      "ane: bridge (private AppleNeuralEngine, MIL compile), linear",
      "logits, sampling kernels (categorical = inverse CDF)",
    ] },

  { id: "outside", title: "Outside the chain", path: "packages, app modules", row: 5, col: 1, span: 3, fill: "#fafafa", stroke: "#999999",
    role: "Produce or consume the chain; they are not on the request path.",
    items: [
      "@mlx-bun/quantize: the Trellis quantizer, publishes the one quant per tier (Qwen3.8-27B-Trellis-3.2bpw)   |   @mlx-bun/training: LoRA and ORPO on the graphs and state contracts   |   @mlx-bun/mlx: the native binding",
      "App modules: chat (Pi, sessions, tool approvals), web, jobs, hub, memory (vault, synthesis), datasets, quantize and train jobs, metrics, models, transcription",
    ] },
];

type Arrow = { from: string; to: string; label: string[]; dashed?: boolean; place: "right" | "down" | "left" | "diagonal" | "right-upper" | "right-lower" };
const ARROWS: Arrow[] = [
  { from: "server", to: "input", label: ["messages, media"], place: "right" },
  { from: "input", to: "scheduler", label: ["tokens, prompt"], place: "down" },
  { from: "scheduler", to: "sampler", label: ["logits per row", "token back"], place: "left" },
  { from: "scheduler", to: "drafter", label: ["draft(n)", "verify, accept"], place: "right" },
  { from: "scheduler", to: "graph", label: ["phase(rows): the width the scheduler is loading"], place: "down" },
  { from: "graph", to: "cache", label: ["attend(q, scale), append"], place: "right-upper" },
  { from: "graph", to: "cache", label: ["makeCache() at load"], dashed: true, place: "right-lower" },
  { from: "graph", to: "layers", label: ["the layers it was built with"], place: "down" },
  { from: "layers", to: "kernels", label: ["imports"], place: "right" },
  { from: "cache", to: "kernels", label: ["quant, codec and attention kernels"], place: "down" },
  { from: "loader", to: "scheduler", label: ["composition, built once:", "graph, cache lego, drafter, widths"], dashed: true, place: "diagonal" },
  { from: "loader", to: "drafter", label: ["builds"], dashed: true, place: "down" },
];

const TITLE = "mlx-bun inference chain: the modules and the features each one owns";
const SUBTITLE = [
  "Solid arrows: the request path at run time. Dashed arrows: built once at load from the model, the flags and the machine. Nothing below the scheduler decides anything at run time.",
  "Each module implements its contract and never checks what another is doing: composition is the guarantee. Swap any box for another implementation of the same contract and the rest does not care.",
];

function wrap(text: string, maxChars: number): string[] {
  const lines: string[] = [];
  let current = "";
  for (const word of text.split(" ")) {
    const next = current ? `${current} ${word}` : word;
    if (next.length > maxChars && current) { lines.push(current); current = word; } else current = next;
  }
  if (current) lines.push(current);
  return lines;
}
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

type Laid = Box & { x: number; y: number; w: number; h: number; lines: string[][]; pathLines: string[]; roleLines: string[] };

function layout(): { laid: Laid[]; width: number; height: number } {
  const laid: Laid[] = [];
  let y = TOP;
  for (const row of [...new Set(BOXES.map((b) => b.row))].sort((a, b) => a - b)) {
    const prepared = BOXES.filter((b) => b.row === row).map((b) => {
      const span = b.span ?? 1;
      const w = span * COL_W + (span - 1) * GAP;
      const lines = b.items.map((item) => wrap(item, Math.floor((w - 2 * PAD - 14) / CHAR_W)));
      const pathLines = wrap(b.path, Math.floor((w - 2 * PAD) / 6.9));
      const roleLines = wrap(b.role, Math.floor((w - 2 * PAD) / 6.1));
      const head = 40 + pathLines.length * 14 + roleLines.length * 15;
      const h = head + lines.reduce((n, l) => n + l.length, 0) * LINE + PAD;
      return { ...b, x: MARGIN + (b.col - 1) * (COL_W + GAP), y, w, h, lines, pathLines, roleLines };
    });
    const rowHeight = Math.max(...prepared.map((p) => p.h));
    for (const p of prepared) laid.push({ ...p, h: rowHeight });
    y += rowHeight + ROW_GAP;
  }
  return { laid, width: MARGIN * 2 + 3 * COL_W + 2 * GAP, height: y - ROW_GAP + MARGIN };
}

export function renderInferenceChain(): string {
  const { laid, width, height } = layout();
  const byId = Object.fromEntries(laid.map((b) => [b.id, b])) as Record<string, Laid>;
  const parts: string[] = [];
  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="-apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif">`);
  parts.push(`<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>`);
  parts.push(`<rect width="${width}" height="${height}" fill="#ffffff"/>`);
  parts.push(`<text x="${MARGIN}" y="34" font-size="22" font-weight="700" fill="#222">${esc(TITLE)}</text>`);
  SUBTITLE.forEach((line, i) => parts.push(`<text x="${MARGIN}" y="${58 + i * 18}" font-size="13" fill="#444">${esc(line)}</text>`));

  for (const b of laid) {
    parts.push(`<g><rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="10" fill="${b.fill}" stroke="${b.stroke}" stroke-width="2"/>`);
    let ty = b.y + 24;
    parts.push(`<text x="${b.x + PAD}" y="${ty}" font-size="18" font-weight="700" fill="#222">${esc(b.title)}</text>`);
    ty += 16;
    for (const l of b.pathLines) { parts.push(`<text x="${b.x + PAD}" y="${ty}" font-size="11" font-family="'SF Mono', Menlo, Consolas, monospace" fill="#555">${esc(l)}</text>`); ty += 14; }
    for (const l of b.roleLines) { parts.push(`<text x="${b.x + PAD}" y="${ty}" font-size="12.5" font-style="italic" fill="#333">${esc(l)}</text>`); ty += 15; }
    ty += 10;
    for (const item of b.lines) {
      item.forEach((line, i) => {
        parts.push(`<text x="${b.x + PAD + (i === 0 ? 0 : 11)}" y="${ty}" font-size="12.5" fill="#222">${esc((i === 0 ? "• " : "") + line)}</text>`);
        ty += LINE;
      });
    }
    parts.push(`</g>`);
  }

  const midX = (b: Laid) => b.x + b.w / 2;
  const midY = (b: Laid) => b.y + b.h / 2;
  for (const a of ARROWS) {
    const from = byId[a.from]!, to = byId[a.to]!;
    let x1: number, y1: number, x2: number, y2: number, lx: number, ly: number, anchor = "middle";
    switch (a.place) {
      case "right": x1 = from.x + from.w; y1 = midY(from); x2 = to.x; y2 = midY(to); lx = (x1 + x2) / 2; ly = y1 - 8 - (a.label.length - 1) * 14; break;
      case "left": x1 = from.x; y1 = midY(from); x2 = to.x + to.w; y2 = midY(to); lx = (x1 + x2) / 2; ly = y1 - 8 - (a.label.length - 1) * 14; break;
      case "right-upper": x1 = from.x + from.w; y1 = midY(from) - 20; x2 = to.x; y2 = y1; lx = (x1 + x2) / 2; ly = y1 - 8; break;
      case "right-lower": x1 = from.x + from.w; y1 = midY(from) + 20; x2 = to.x; y2 = y1; lx = (x1 + x2) / 2; ly = y1 + 20; break;
      case "down": x1 = midX(from); y1 = from.y + from.h; x2 = x1; y2 = to.y; lx = x1 + 10; ly = (y1 + y2) / 2 + 4; anchor = "start"; break;
      case "diagonal": x1 = from.x + 60; y1 = from.y + from.h; x2 = to.x + to.w - 40; y2 = to.y; lx = (x1 + x2) / 2 + 16; ly = (y1 + y2) / 2 - 2; anchor = "start"; break;
    }
    parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#333" stroke-width="2"${a.dashed ? ` stroke-dasharray="7,5"` : ""} marker-end="url(#arrow)"/>`);
    a.label.forEach((l, i) => parts.push(`<text x="${lx}" y="${ly + i * 14}" font-size="12" fill="#222" text-anchor="${anchor}" paint-order="stroke" stroke="#ffffff" stroke-width="5" stroke-linejoin="round">${esc(l)}</text>`));
  }
  parts.push(`</svg>`);
  return parts.join("\n") + "\n";
}

export const DEFAULT_OUT = join(import.meta.dir, "..", "docs", "inference-chain.svg");

if (import.meta.main) {
  const { values } = parseArgs({ args: process.argv.slice(2), options: { out: { type: "string" }, help: { type: "boolean" } } });
  if (values.help) {
    console.log("Usage: bun scripts/gen-inference-chain.ts [--out <path>]\nWrites the inference-chain figure (default docs/inference-chain.svg).");
    process.exit(0);
  }
  const outPath = values.out ?? DEFAULT_OUT;
  await Bun.write(outPath, renderInferenceChain());
  console.log(`wrote ${outPath}`);
}
