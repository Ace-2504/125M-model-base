# Phase 5 Training Plan — LOCAL (RTX 3060 12 GB)

**Job:** train the fresh 125M model **from scratch to 1 epoch** on your local GPU,
hardened against power cuts / reboots / crashes.
Data pipeline (Phases 0–4) is already done; token bins are on local disk.

---

## 1. Training spec (already in `config.py` — nothing to edit)

- **Model**: `LlamaForCausalLM(LlamaConfig(**MODEL.to_llama_kwargs()))` — 125.8M params, vocab 16384, 12L/768d/12h, ctx 1024, RoPE θ=10000, RMSNorm, SwiGLU, tied embeddings.
- **Data**: `data/tokens/train/*.bin` (1,991,368 windows), `val/*.bin` (20,119 windows), uint16, packed 1024-token windows.
- **TrainConfig** (`config.TRAIN`): global batch **524,288 tokens = 512 windows/step**, LR 6e-4 → 6e-5 cosine, warmup 200M tokens, AdamW (0.9/0.95), wd 0.1, grad-clip 1.0, seed 1337.

**Derived (1 epoch):**
- **Steps/epoch = 1,991,368 ÷ 512 = 3,889 steps**
- **Warmup ≈ 382 steps**
- Precision: **bf16** (Ampere native) + `attn_implementation="sdpa"` (no flash-attn install needed).
- Expected wall-clock: **~2 days** (~8–12k tokens/sec on a 3060; benchmark to confirm). Cost ≈ **$2 electricity**.

---

## 2. Files to create

1. **`train_core.py`** — device-agnostic engine (shared with the Modal plan): build model, memmap bin data loader, cosine-LR loop, **atomic checkpoint save/load + auto-resume**, eval on val, metrics to `config.METRICS_PATH`.
2. **`train_local.py`** — thin wrapper: local paths, checkpoints to `C:\slm_ckpts`, small micro-batch for 12 GB, fresh init.
3. **`run_local.ps1`** — auto-relaunch loop (restarts after any crash/reboot; resumes from last checkpoint).
4. **`eval.py`** — perplexity on the val bins + fixed-prompt samples (for your observations).

None of these touch the 4 canonical pipeline files.

---

## 3. Make it fit 12 GB (output-preserving)

`micro_batch_size=32` will likely OOM at seq 1024. Fit it by trading micro-batch for grad-accumulation
while keeping the **global batch identical** (same math, same result):

| micro-batch | grad-accum | global batch |
|---|---|---|
| **8 (safe start)** | 64 | 512 windows ✅ |
| 16 (if it fits) | 32 | 512 windows ✅ |

Enable **gradient checkpointing** only if needed to reach a usable micro-batch.
**Benchmark 100 steps first**, read real tokens/sec, then commit to the full run.

---

## 4. Checkpoint / resume — the core of power-cut safety

Each checkpoint is one `.pt` holding **everything** needed to resume bit-exactly:
- model `state_dict`
- optimizer `state_dict` (Adam m, v) ← without this, resume corrupts training
- **step number + tokens seen** (LR recomputed from step, so the schedule survives)
- RNG states (torch / numpy / python) **and the data cursor** (next window index) → no data repeated or skipped
- running val-loss / metrics

Rules:
- **Atomic write**: save to `ckpt.tmp` → `os.replace()` → `ckpt.pt`. A power cut mid-save can't corrupt the good file.
- **Rotate 2**: keep `ckpt.pt` + `ckpt.prev.pt`; if the newest is corrupt, fall back one.
- **Cadence override (critical):** `config.ckpt_every_steps=500` was tuned for 8×H100 where 500 steps = minutes. On a 3060, 500 steps ≈ **6 hours** → far too sparse. **Checkpoint every ~25 steps OR every 20 minutes, whichever first.** Worst-case loss on interruption ≈ 20 min.
- **Auto-resume:** on launch, if `ckpt.pt` exists → load and continue. Recovery = rerun one command.

---

## 5. Disk (your drive is ~98% full — this bites)

- Each checkpoint ≈ **1.5–2 GB** (fp32 weights + Adam states). Keep **only 2** (~4 GB).
- Store at **`C:\slm_ckpts\`**, NOT under `OneDrive\Desktop` — otherwise every save triggers a 2 GB cloud sync (wasted I/O + space churn).
- Confirm **≥8 GB free** before starting.

---

## 6. Windows interruption sources (more likely than a power cut)

```powershell
# Pause auto-reboot for updates (Settings → Windows Update → Pause), and:
powercfg /change standby-timeout-ac 0     # never sleep on AC
powercfg /change hibernate-timeout-ac 0
```
- Optional **UPS** (~₹3–5k): rides out flickers or triggers a graceful checkpoint-and-shutdown — the only real defense against a *hard* cut mid-GPU-op.

---

## 7. Auto-relaunch wrapper (`run_local.ps1`, sketch)

```powershell
while ($true) {
  ./.venv/Scripts/python.exe -u train_local.py   # auto-resumes from C:\slm_ckpts\ckpt.pt
  if ($LASTEXITCODE -eq 0) { break }             # 0 = training finished
  Start-Sleep -Seconds 15                         # crash/OOM → wait, relaunch, resume
}
```
Pair with a Windows "on startup" Scheduled Task running the same script, so even a reboot self-heals.

---

## 8. Run it

```powershell
# 1) one-time fit check + real throughput
./.venv/Scripts/python.exe train_local.py --benchmark-steps 100
# 2) full resumable run
./run_local.ps1
# 3) after it finishes
./.venv/Scripts/python.exe eval.py --ckpt C:\slm_ckpts\ckpt.pt
```

---

## 9. Observations to log

To `metrics.jsonl`: `step, lr, train_loss, grad_norm, tokens_seen, wall_time`; periodic `val_loss / perplexity`.
At the end capture: **total wall-clock, avg tokens/sec, MFU%, final val perplexity**, plus a few
fixed-prompt sample generations. (If you also run the Modal plan, these same fields give you a clean
local-vs-cloud time/cost comparison for the identical job.)

---

## 10. Roles

- **I (Claude Code) will:** write `train_core.py`, `train_local.py`, `run_local.ps1`, `eval.py`; run the 100-step benchmark; monitor logs; verify checkpoints resume correctly.
- **You do:** pause Windows Update before the run; make sure `C:\slm_ckpts` has ≥8 GB free.

---

## 11. Bottom line

~2 days, ~$2, and **safe to interrupt** (≤20-min loss) via atomic checkpoints every ~25 steps + an
auto-relaunch wrapper, with checkpoints on `C:\slm_ckpts` off OneDrive and Windows-update paused.
