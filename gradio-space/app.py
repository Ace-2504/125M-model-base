"""Gradio (ZeroGPU) demo for SLM-125M — a 125.8M-parameter Llama-style legal LM.

Loads Ace-2504/slm-125m-base once at startup and serves text completions on a
ZeroGPU-allocated GPU. This is a base model: it continues text, not a chatbot.
"""

import os

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Ace-2504/slm-125m-base")
MAX_NEW_TOKENS_CAP = 256
# rebuild trigger: model repo restored 2026-07-11

# Load once at module level. On ZeroGPU the model is placed on CUDA here (a CUDA
# emulation is active outside @spaces.GPU); the real GPU is attached per call.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
model.eval()


@spaces.GPU(duration=60)
def generate(
    prompt: str, temperature: float, max_new_tokens: int, top_p: float, top_k: int
) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return "Enter a prompt to continue."

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    inputs.pop("token_type_ids", None)  # Llama generate() rejects this key

    do_sample = temperature is not None and temperature > 0
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(min(MAX_NEW_TOKENS_CAP, max_new_tokens)),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            top_p=float(top_p),
            top_k=int(top_k),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


EXAMPLES = [
    ["The plaintiff alleges that the defendant", 0.8, 90, 0.95, 50],
    ["The court held that", 0.8, 90, 0.95, 50],
    ["In this agreement, the parties", 0.8, 90, 0.95, 50],
]

with gr.Blocks(title="SLM-125M") as demo:
    gr.Markdown(
        "# SLM-125M — a legal language model trained from scratch\n"
        "A 125.8M-parameter Llama-style **base model** pretrained from scratch on "
        "2.04B tokens of US case law, SEC filings, and educational web text "
        "(val perplexity 11.35). It **continues text** — it is not a chatbot and has "
        "no factual grounding, so treat every completion as stylistic, not authoritative."
    )
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Prompt (a prefix to continue)",
                lines=4,
                value="The plaintiff alleges that the defendant",
            )
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature")
            max_new_tokens = gr.Slider(10, MAX_NEW_TOKENS_CAP, value=90, step=1, label="Max new tokens")
            top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p")
            top_k = gr.Slider(0, 100, value=50, step=1, label="Top-k")

    output = gr.Textbox(label="Completion", lines=8)

    generate_btn.click(
        generate,
        inputs=[prompt, temperature, max_new_tokens, top_p, top_k],
        outputs=output,
    )
    gr.Examples(EXAMPLES, inputs=[prompt, temperature, max_new_tokens, top_p, top_k])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
