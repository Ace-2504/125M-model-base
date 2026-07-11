"""FastAPI inference server for SLM-125M, deployed as a Hugging Face Docker Space.

Loads Ace-2504/slm-125m-base once at startup and serves POST /generate, gated by
a shared-secret bearer token (set as the Space secret API_SECRET). The frontend's
own /api/generate route proxies here so the secret and the Space URL stay
server-side.
"""

from __future__ import annotations

import os
import threading

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Ace-2504/slm-125m-base")
API_SECRET = os.environ.get("API_SECRET", "")

MAX_NEW_TOKENS_CAP = 256
MAX_PROMPT_CHARS = 2000

app = FastAPI(title="SLM-125M inference")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = threading.Lock()
print(f"loading {MODEL_ID} …", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()
torch.set_num_threads(max(1, os.cpu_count() or 1))
print("model ready", flush=True)


class GenerateRequest(BaseModel):
    prompt: str = Field(default="")
    temperature: float = 0.8
    maxTokens: int = 90
    topP: float = 0.95
    topK: int = 50


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_ID, "ready": True}


@app.post("/generate")
def generate(req: GenerateRequest, authorization: str = Header(default="")):
    if API_SECRET and authorization != f"Bearer {API_SECRET}":
        raise HTTPException(status_code=401, detail="unauthorized")

    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")
    prompt = prompt[:MAX_PROMPT_CHARS]

    max_new = int(max(1, min(MAX_NEW_TOKENS_CAP, req.maxTokens)))
    temperature = float(max(0.0, min(2.0, req.temperature)))
    top_p = float(max(0.0, min(1.0, req.topP)))
    top_k = int(max(0, min(200, req.topK)))

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)  # Llama generate() rejects this key
    input_len = inputs["input_ids"].shape[1]

    with _lock, torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {"ready": True, "completion": completion}
