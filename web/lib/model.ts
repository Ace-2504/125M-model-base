/**
 * Single source of truth for every number shown in the UI.
 * All values are pulled from this repo's real artifacts:
 *   - config.py                     (model + training hyperparameters)
 *   - data/tokens/index.json        (final token / window counts)
 *   - data/corpus/phase2_report.json (dedup + decontamination stats)
 * Reference numbers (e.g. the stated 40/40/20 mix) are intentionally
 * replaced with the internally-consistent figures this pipeline produced.
 */

export const HEADLINE = {
  params: "125.8M",
  vocab: "16,384",
  context: "1,024",
  trainTokens: "2.04B",
  valPerplexity: "11.35", // final val perplexity, step 3889 (full epoch)
  cost: "~5.1 h", // A100 wall-clock, 1 epoch
} as const;

export type Stat = { label: string; value: string; sub?: string; pending?: boolean };

export const HEADLINE_STATS: Stat[] = [
  { label: "parameters", value: "125.8M", sub: "tied embeddings" },
  { label: "vocabulary", value: "16,384", sub: "byte-level BPE" },
  { label: "context window", value: "1,024", sub: "tokens" },
  { label: "training tokens", value: "2.04B", sub: "+20.6M held-out val" },
  { label: "val perplexity", value: "11.35", sub: "val loss 2.43 · 1 epoch" },
  { label: "trained in", value: "5.1 h", sub: "A100 · ~111k tok/s" },
];

export const ARCHITECTURE: { k: string; v: string }[] = [
  { k: "Architecture", v: "LlamaForCausalLM (from scratch)" },
  { k: "Layers", v: "12" },
  { k: "Hidden size", v: "768" },
  { k: "Attention heads", v: "12 (head dim 64)" },
  { k: "KV heads", v: "12 — full MHA" },
  { k: "MLP", v: "SwiGLU, inner 3,072" },
  { k: "Positional", v: "RoPE, θ = 10,000" },
  { k: "Normalization", v: "RMSNorm, ε = 1e-5" },
  { k: "Embeddings", v: "tied input/output" },
  { k: "Attention bias", v: "none" },
];

export const TRAINING: { k: string; v: string }[] = [
  { k: "Global batch", v: "524,288 tokens / step" },
  { k: "Windows per step", v: "512 (micro-batch 32 × grad-accum 16)" },
  { k: "Training steps", v: "3,889 (1 epoch)" },
  { k: "Learning rate", v: "6e-4 → 6e-5, cosine" },
  { k: "Warmup", v: "200M tokens (~382 steps)" },
  { k: "Optimizer", v: "AdamW β = (0.9, 0.95), wd 0.1" },
  { k: "Grad clip", v: "1.0" },
  { k: "Precision", v: "bfloat16 · SDPA attention" },
  { k: "Seed", v: "1337" },
];

/** Pipeline stages, matching config.STAGES. */
export const PIPELINE = [
  { id: "stream", title: "Stream", desc: "3 HF datasets streamed to token budgets" },
  { id: "clean", title: "Clean", desc: "rule chain: length, symbol ratio, OCR gate, repetition" },
  { id: "dedup", title: "Dedup + decontaminate", desc: "near-dup MinHash · strip LexGLUE / CaseHOLD leakage" },
  { id: "tokenizer", title: "Tokenizer", desc: "train 16K byte-level BPE" },
  { id: "tokenize", title: "Pack", desc: "encode → pack into 1,024-token uint16 windows" },
  { id: "pretrain", title: "Pretrain", desc: "125M Llama, 1 epoch on A100" },
];

/** Real corpus composition from data/tokens/index.json (train tokens). */
export const CORPUS = [
  {
    name: "US case law",
    hf: "HFforLegal/case-law",
    tokens: 714_936_320,
    pct: 35.1,
    color: "var(--accent)",
    note: "us split · strict OCR gate",
  },
  {
    name: "SEC filings",
    hf: "PleIAs/SEC",
    tokens: 860_036_096,
    pct: 42.2,
    color: "var(--accent-2)",
    note: "financial / legal register",
  },
  {
    name: "Educational web",
    hf: "HuggingFaceFW/fineweb-edu",
    tokens: 464_188_416,
    pct: 22.8,
    color: "var(--accent-3)",
    note: "sample-10BT config",
  },
];

/** Real dedup / decontamination outcomes from data/corpus/phase2_report.json. */
export const DEDUP = [
  { source: "case-law", kept: 207_041, nearDup: 1_606, exactDup: 0, contaminated: 24_000 },
  { source: "sec", kept: 45_035, nearDup: 0, exactDup: 1_989, contaminated: 175 },
  { source: "fineweb-edu", kept: 418_405, nearDup: 0, exactDup: 62, contaminated: 0 },
];

export const TOTAL_KEPT = DEDUP.reduce((a, d) => a + d.kept, 0);
export const TOTAL_REMOVED = DEDUP.reduce(
  (a, d) => a + d.nearDup + d.exactDup + d.contaminated,
  0
);

/** Fixed eval prompts from eval.py — used as the demo's example prompts. */
export const EXAMPLE_PROMPTS = [
  "The plaintiff alleges that the defendant",
  "The court held that",
  "In this agreement, the parties",
  "A useful way to think about language models is",
];

export const DEFAULT_GEN = {
  temperature: 0.8,
  maxTokens: 90,
  topP: 0.95,
  topK: 50,
};
