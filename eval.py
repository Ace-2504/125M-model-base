"""Evaluate a Phase 5 checkpoint on validation bins and sample fixed prompts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import config
import train_core


PROMPTS = (
    "The court held that",
    "In this agreement, the parties",
    "A useful way to think about language models is",
)


def load_model(ckpt_path: str, device: str):
    import torch

    model = train_core._build_model(device, attn_implementation="sdpa")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def generate_samples(model, tokenizer_dir: str, device: str, max_new_tokens: int) -> None:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\nPROMPT: {prompt}\n{text}")


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to ckpt.pt")
    parser.add_argument("--data-root", default=config.DATA_ROOT)
    parser.add_argument("--tokenizer", default=config.TOKENIZER_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)
    val_data = train_core.BinWindowDataset(args.data_root, "val", config.SEQ_LEN)
    val_loss = train_core.evaluate_loss(model, val_data, args.batch_size, device, max_batches=None)
    print(f"val_loss={val_loss:.4f}")
    print(f"perplexity={math.exp(min(20.0, val_loss)):.4f}")

    tokenizer_dir = Path(args.tokenizer)
    if tokenizer_dir.exists():
        generate_samples(model, str(tokenizer_dir), device, args.max_new_tokens)
    else:
        print(f"tokenizer not found at {tokenizer_dir}; skipped samples")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
