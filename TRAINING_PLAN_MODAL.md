# Phase 5 Training Plan — MODAL (A100)

**Job:** train the fresh 125M model **from scratch to 1 epoch** on a Modal A100, run **detached**
(immune to your local power/internet dropping) and **resumable** via Volume checkpoints.
Data pipeline (Phases 0–4) is already done locally; the token bins get uploaded to a Modal Volume first.

Adapts the reference brief's Modal conventions (App / Image / Volume at `/data` / `local_entrypoint` /
`volume.commit()`) to a **GPU training** function.

---

## 1. Training spec (already in `config.py` — nothing to edit)

- **Model**: `LlamaForCausalLM(LlamaConfig(**MODEL.to_llama_kwargs()))` — 125.8M params, vocab 16384, 12L/768d/12h, ctx 1024, RoPE θ=10000, RMSNorm, SwiGLU, tied embeddings.
- **Data**: `data/tokens/train/*.bin` (1,991,368 windows), `val/*.bin` (20,119 windows), uint16, packed 1024-token windows.
- **TrainConfig** (`config.TRAIN`): global batch **524,288 tokens = 512 windows/step**, LR 6e-4 → 6e-5 cosine, warmup 200M tokens, AdamW (0.9/0.95), wd 0.1, grad-clip 1.0, seed 1337.

**Derived (1 epoch):**
- **Steps/epoch = 1,991,368 ÷ 512 = 3,889 steps**
- **Warmup ≈ 382 steps**
- Precision: **bf16** + `attn_implementation="sdpa"`.
- On A100-40GB: `micro_batch_size=32` fits → grad-accum = 16. Expected wall-clock **~3–5 h**, cost **~$6–10** (possibly **$0** within Modal's monthly free credits).

---

## 2. Files to create

1. **`train_core.py`** — device-agnostic engine (shared with the local plan): build model, memmap bin data loader, cosine-LR loop, **atomic checkpoint save/load + auto-resume**, eval on val, metrics logging.
2. **`modal_train.py`** — Modal app: image, Volume, A100 GPU function, `local_entrypoint`, reusing `train_core`.
3. **`eval.py`** — perplexity on val bins + fixed-prompt samples.

None touch the 4 canonical pipeline files.

---

## 3. One-time account + CLI setup  *(the browser step is yours — OAuth can't run in an agent session)*

```bash
pip install modal
modal token new                 # opens browser to authorize → writes ~/.modal.toml
modal profile current           # verify
```
Non-interactive alt: create a token in Settings → API Tokens, then
`modal token set --token-id ak-XXXX --token-secret as-XXXX`.

---

## 4. Volume

```bash
modal volume create slm-125m          # durable artifacts, mounted at /data
```
(No HuggingFace secret needed — training is from scratch on your own local data. Only add one later if
you decide to push the finished model to HF.)

---

## 5. Upload your local artifacts to the Volume (the key new step — data is local, not on Modal)

```bash
modal volume put slm-125m ./data/tokens     /tokens       # ~4 GB train/val bins + index.json
modal volume put slm-125m ./data/tokenizer  /tokenizer
```
One-time upload from your connection; Modal reuses it for every run.

---

## 6. The Modal training function (`modal_train.py`, skeleton)

```python
import modal, config

app = modal.App("slm-125m-pretrain")
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", "transformers==4.46.3", "numpy")
    .add_local_python_source("config", "train_core")     # MUST come after pip (Modal rule)
)
volume = modal.Volume.from_name("slm-125m", create_if_missing=True)

@app.function(
    image=gpu_image, gpu="A100-40GB",
    volumes={"/data": volume},
    timeout=60 * 60 * 8,                                  # 8 h > expected ~5 h
)
def pretrain(resume: bool = True):
    import train_core
    train_core.run(
        data_root="/data",
        ckpt_dir="/data/checkpoints/base",
        init_from_hf="",              # fresh init (from scratch)
        epochs=1,                     # ONE pass over the corpus = 1st epoch
        lr_mode="scratch",            # config's 6e-4 → cosine → 6e-5, warmup ~382 steps
        commit_fn=volume.commit,      # persist checkpoints to the Volume each save
        resume=resume,
    )

@app.local_entrypoint()
def main(resume: bool = True):
    pretrain.remote(resume=resume)
```

Notes:
- **`volume.commit()` after every checkpoint** — Modal Volume writes aren't durable until committed.
- **Timeout > run time.** If it hits the cap or Modal preempts, the checkpoint/resume logic restarts from the last save (rerun the command; `resume=True` picks it up).
- Same atomic-save + rotate-2 checkpoint code as the local plan, writing to `/data/checkpoints/base`.
- **Checkpoint cadence:** every ~200 steps (~15 min on A100) or 20 min — bounds any restart loss.

---

## 7. Launch detached + monitor

```bash
modal run --detach modal_train.py            # runs server-side; survives your power/net dropping
modal app list                               # confirm it's running
modal app logs slm-125m-pretrain             # tail loss / lr / tokens-per-sec
```
`--detach` is what makes **your local power cut irrelevant** — training continues in Modal's cloud, and
you reconnect later just to read logs or pull the result.

---

## 8. Download the result

```bash
modal volume get slm-125m /checkpoints/base ./model_epoch1
./.venv/Scripts/python.exe eval.py --ckpt ./model_epoch1/ckpt.pt
```

---

## 9. Observations to log

To `metrics.jsonl`: `step, lr, train_loss, grad_norm, tokens_seen, wall_time`; periodic `val_loss / perplexity`.
At the end capture: **total wall-clock, avg tokens/sec, MFU%, final val perplexity, $ cost**, plus a few
fixed-prompt sample generations. (If you also run the local plan, these same fields give a clean
cloud-vs-local time/cost comparison for the identical job.)

---

## 10. Roles

- **I (Claude Code) will:** write `train_core.py`, `modal_train.py`, `eval.py`; run the `modal volume put` uploads and `modal run --detach`; tail logs, verify checkpoints, pull and eval the model.
- **You do (I can't):** the `modal token new` **browser authorization**. Once done, I drive everything else from the terminal.

---

## 11. Bottom line

~3–5 h, ~$6–10 (maybe free with credits), **immune to your local power/internet** via `--detach`, and
resumable from Volume checkpoints if Modal ever preempts or times out. Setup = authorize CLI once →
upload bins once → one `modal run --detach` command.
