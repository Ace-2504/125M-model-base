---
title: SLM-125M Inference
emoji: ⚖️
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SLM-125M inference

FastAPI inference server for [`Ace-2504/slm-125m-base`](https://huggingface.co/Ace-2504/slm-125m-base),
a 125.8M-parameter Llama-style legal language model trained from scratch.

`POST /generate` with `{ "prompt": "...", "temperature": 0.8, "maxTokens": 90, "topP": 0.95, "topK": 50 }`
and an `Authorization: Bearer <API_SECRET>` header. Returns `{ "completion": "..." }`.

This is a base model — it continues text, it does not answer questions.
