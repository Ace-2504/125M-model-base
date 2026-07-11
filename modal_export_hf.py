"""One-off: convert the trained ckpt.pt on the Modal Volume into a Hugging Face
model repo (safetensors + config + tokenizer) and push it to the Hub.

Runs the conversion on Modal (torch + the checkpoint + bandwidth all live there),
so nothing large touches the local disk. The HF token is read from the local
machine's HF login and passed to the remote function over Modal's encrypted
channel — it is never stored as a named secret or printed.

    ./.venv/Scripts/modal.exe run modal_export_hf.py --repo-id Ace-2504/slm-125m-base
"""

from __future__ import annotations

import modal

app = modal.App("slm-125m-export")
volume = modal.Volume.from_name("slm-125m")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "numpy>=1.26,<2.0",
        "huggingface_hub>=0.24",
    )
    .add_local_python_source("config", "train_core")
)

MODEL_CARD = """---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
  - llama
  - legal
  - from-scratch
  - small-language-model
---

# SLM-125M — a legal language model trained from scratch

A 125.8M-parameter Llama-style base model pretrained **from scratch** on ~2.04B
tokens of US case law, SEC filings, and educational web text. Trained for one
epoch on a single Modal A100 (~5.1 h, ~111k tokens/sec).

This is a **base model, not a chatbot** — it continues text rather than answering
questions. It has no instruction tuning and no factual grounding.

## Results
- Held-out validation perplexity: **11.35** (val loss 2.43, 1% split, 20.6M tokens)
- Perplexity trajectory: 17.2 → 13.0 → 11.6 → 11.35 over the epoch

## Architecture
- 12 layers, hidden 768, 12 heads (head dim 64), full MHA
- SwiGLU (inner 3072), RoPE (theta 10000), RMSNorm, tied embeddings
- Vocab 16,384 (byte-level BPE), context 1,024

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Ace-2504/slm-125m-base")
model = AutoModelForCausalLM.from_pretrained("Ace-2504/slm-125m-base")
ids = tok("The court held that", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
```
"""


@app.function(image=image, volumes={"/data": volume}, timeout=60 * 30)
def export(hf_token: str, repo_id: str) -> str:
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    from huggingface_hub import create_repo, upload_folder

    import train_core

    out_dir = "/tmp/hf_export"

    model = train_core._build_model("cpu", attn_implementation="sdpa")
    ckpt = torch.load("/data/checkpoints/base/ckpt.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Sensible default generation config baked into the repo.
    model.generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        max_new_tokens=90,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )

    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained("/data/tokenizer").save_pretrained(out_dir)

    with open(f"{out_dir}/README.md", "w", encoding="utf-8") as f:
        f.write(MODEL_CARD)

    create_repo(repo_id, token=hf_token, repo_type="model", exist_ok=True)
    upload_folder(folder_path=out_dir, repo_id=repo_id, token=hf_token)

    step = int(ckpt.get("step", 0))
    return f"pushed {repo_id} (from step {step})"


@app.local_entrypoint()
def main(repo_id: str = "Ace-2504/slm-125m-base"):
    from huggingface_hub import HfFolder

    token = HfFolder.get_token()
    if not token:
        raise SystemExit("No local HF token found — run `huggingface-cli login` first.")
    print(export.remote(token, repo_id))
