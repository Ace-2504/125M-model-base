---
title: SLM-125M
emoji: ⚖️
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
short_description: A 125M legal language model trained from scratch (CPU demo).
---

# SLM-125M — Gradio demo

Interactive text-completion demo for
[`Ace-2504/slm-125m-base`](https://huggingface.co/Ace-2504/slm-125m-base),
a 125.8M-parameter Llama-style legal language model trained from scratch
(val perplexity 11.35). The model weights live in that model repo; this Space
only holds the demo code and loads the model at startup.

This is a **base model** — it continues text, it does not answer questions.

> Note: `sdk_version` above must be a Gradio version HF currently supports. If you
> create the Space through the HF web UI with the Gradio SDK, keep the version HF
> fills in automatically rather than this one.
