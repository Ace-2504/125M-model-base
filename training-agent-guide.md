# Training Agent Guide — continue pretraining SLM-125M to epoch N and evaluate it

**Audience:** a fresh Claude Code thread with **no prior context**. Follow this
top to bottom to (1) continue pretraining the 125M SLM for one more epoch, (2)
record research parameters from Modal, (3) preserve that epoch's model, and (4)
reproduce the hardened evaluation + report. One thread = one epoch.

> **Read this whole file first.** Then read `debugging/` (gitignored, on this
> machine) for known gotchas before running anything.

---

## 0. What this project is (30-second orientation)

A 125.8M-parameter Llama-style causal LM pretrained from scratch on ~2.04B tokens
of US case law + SEC filings + FineWeb-Edu. Epoch 1 finished at **validation
perplexity 11.35** (step 3,889). Training runs on **Modal** (A100); the model
weights live on a **Modal Volume**, not in git. Each epoch is archived as its own
**Hugging Face model repo** and evaluated with a fixed, hardened benchmark.

## 1. The single most important mental model

**Cloning the repo does NOT copy the model.** The model checkpoint lives on the
shared Modal Volume `slm-125m` at `/data/checkpoints/base/ckpt.pt`, and training
*continues* that same checkpoint forward. So:

- The **git clone per epoch** preserves that epoch's **code, config, and eval
  outputs** — a lab notebook, not the weights.
- The **weights for epoch N are preserved by exporting them to a new HF repo**
  (`Ace-2504/slm-125m-e{N}`) *right after epoch N finishes and before epoch N+1
  starts* — because epoch N+1 overwrites `ckpt.pt` on the volume.
- Optionally also snapshot the raw checkpoint on the volume per epoch.

If you skip the export before the next epoch, that epoch's exact weights are lost.

## 2. Fixed facts / paths (verify, don't assume)

| Thing | Value |
| --- | --- |
| Main repo | `C:\Users\harma\OneDrive\Desktop\python\vizuara\SLM-course\Replicate-the-125M-SLM-Data-Pipeline` |
| Eval clone (reused across epochs) | `C:\Users\harma\slm-125m-eval` (has its own `.venv`, `.hf-cache`, `experiment/`) |
| Modal app | `slm-125m-pretrain` · Modal CLI: `./.venv/Scripts/modal.exe` |
| Modal Volume | `slm-125m`, mounted at `/data` |
| Checkpoint | `/data/checkpoints/base/ckpt.pt` · metrics: `/data/checkpoints/base/metrics.jsonl` |
| Tokenizer / tokens on volume | `/data/tokenizer`, `/data/tokens/{train,val}` |
| GPU | A100-40GB · `micro_batch_size=32`, global batch 524,288 tokens = **512 windows/step** |
| Steps per epoch | **3,889** (1,991,368 train windows ÷ 512). Cumulative *target* `total_steps = 3889 × N`, but with `resume=True` the epoch-N run only trains the last **3,889 new steps** (it skips the `3889×(N-1)` already done) — so **each epoch ≈ one epoch of compute (~5 h), not N×**. |
| Seed | 1337 · LR 6e-4 → 6e-5 cosine · warmup 200M tokens (~382 steps) |
| HF model repos | epoch 1 = `Ace-2504/slm-125m-base` · epoch N≥2 = `Ace-2504/slm-125m-e{N}` |
| Windows / auth | Modal + HF already authenticated as `ace-2504`. Prefix Python/Modal/HF commands with `PYTHONIOENCODING=utf-8` (cp1252 crashes on ✓/emoji output — see `debugging/11`). |

Sanity-check auth before doing anything:
```bash
cd "<MAIN_REPO>"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/modal.exe profile current      # -> ace-2504
./.venv/Scripts/python.exe -c "from huggingface_hub import whoami; print(whoami()['name'])"  # -> Ace-2504
PYTHONIOENCODING=utf-8 ./.venv/Scripts/modal.exe volume ls slm-125m checkpoints/base  # ckpt.pt exists
```

---

## STEP A — Clone the repo for this epoch (lab notebook)

Pick `N` (the epoch you are about to train — e.g. 2). Clone **off OneDrive** to
avoid cloud-sync churn:

```bash
git clone "C:/Users/harma/OneDrive/Desktop/python/vizuara/SLM-course/Replicate-the-125M-SLM-Data-Pipeline" "C:/Users/harma/slm-125m-epoch{N}"
```

Work in that folder for the training + config edits. (The eval reuses the single
`slm-125m-eval` clone — see Step E.)

---

## AUTOMATED SAFEGUARD — never overwrite an un-exported epoch (the agent does this itself)

The volume holds **only one, continuously-overwritten checkpoint**. Training epoch
N overwrites epoch N-1's weights on the volume. So **before launching any epoch,
the agent MUST automatically guarantee the epoch currently on the volume is already
saved to its own HF repo** — and export it if not. Do this itself; never ask the
user to remember it, and never launch training until this passes.

Run this **single non-interactive preflight** (from the main repo) and act on its
last line:
```bash
cd "<MAIN_REPO>"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/modal.exe volume get slm-125m checkpoints/base/metrics.jsonl ./_m.jsonl
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -c "
import json
from huggingface_hub import list_repo_files
step=max(json.loads(l).get('step',0) for l in open('_m.jsonl') if l.strip())
E=round(step/3889)                        # completed epoch currently on the volume
repo='Ace-2504/slm-125m-base' if E==1 else f'Ace-2504/slm-125m-e{E}'
try: ok='model.safetensors' in list_repo_files(repo)
except Exception: ok=False
print(f'volume_step={step} epoch_on_volume={E} repo={repo}')
print('OK - already exported, safe to continue' if ok else f'ACTION REQUIRED - export first: modal run modal_export_hf.py --repo-id {repo}')
"
```
- If the last line says **OK**, proceed to Step B.
- If it says **ACTION REQUIRED**, the agent runs the printed
  `modal run modal_export_hf.py --repo-id <repo>` (Step D), verifies the repo has
  `model.safetensors`, then proceeds. **Never launch training while it says ACTION
  REQUIRED.** (If a prior run was interrupted mid-epoch, `epoch_on_volume` is the
  last *completed* epoch — export that one.)

> The agent also runs Step D automatically at the **end** of every training run, so
> the chain is protected from both sides (end-of-run export + start-of-run
> re-check). Belt and suspenders — the model's weights are never left only on the
> soon-to-be-overwritten volume.

---

## STEP B — Continue pretraining for epoch N on Modal

The engine (`train_core.run`) resumes from the checkpoint and trains until the
cumulative target `total_steps = 3889 × epochs`. Setting `epochs=N` with
`resume=True` trains **only the Nth pass** — the loop is `range(start_step,
total_steps)` where `start_step` is the checkpoint's step, so for epoch 2 it runs
`range(3889, 7778)` = **3,889 new steps**, not 7,778. Each epoch is therefore ~one
epoch of compute (~5 h / ~one epoch of A100 cost), *not* N× — the `×N` is just the
finish line, and resume skips everything already done. Each epoch uses a fresh
data permutation (`seed + epoch`).

> **This cheapness depends entirely on `resume=True` + the prior checkpoint being
> present on the volume.** If resume is off or `ckpt.pt` is missing, it retrains
> all `3889×N` steps from scratch (the expensive case) — so never clear the
> checkpoint, and confirm `modal volume ls slm-125m checkpoints/base` shows
> `ckpt.pt` before launching.

1. In the epoch-N clone, edit **`modal_train.py`** → set `epochs=N` in the
   `pretrain()` call (change the hardcoded `epochs=1`). Keep `resume=True`.

2. Launch **detached** (survives your machine sleeping) and monitor. The clone has
   no `.venv`, so use the **main repo's** Modal CLI + Python via absolute paths:
```bash
MODAL="C:/Users/harma/OneDrive/Desktop/python/vizuara/SLM-course/Replicate-the-125M-SLM-Data-Pipeline/.venv/Scripts/modal.exe"
PY="C:/Users/harma/OneDrive/Desktop/python/vizuara/SLM-course/Replicate-the-125M-SLM-Data-Pipeline/.venv/Scripts/python.exe"
cd "C:/Users/harma/slm-125m-epoch{N}"
PYTHONIOENCODING=utf-8 "$MODAL" run --detach modal_train.py
PYTHONIOENCODING=utf-8 "$MODAL" app logs slm-125m-pretrain   # tail loss / lr / tok/s
```

3. **Confirm it resumed** (do not skip): the startup logs must show
   `resumed from step {3889×(N-1):,} with … tokens seen`. If it shows step 0 or the
   line is absent, it is retraining from scratch — stop, fix `resume=True` / the
   checkpoint, and relaunch.

4. Expect **~3–5 h** on A100. It is **done only when metrics reach step `3889×N`**:
```bash
PYTHONIOENCODING=utf-8 "$MODAL" volume get slm-125m checkpoints/base/metrics.jsonl ./_chk.jsonl
"$PY" -c "import json; print('max step:', max(json.loads(l).get('step',0) for l in open('_chk.jsonl') if l.strip()))"
```

5. **If the run stops early** (8 h timeout or preemption — the app shows `stopped`
   in `modal app list` before reaching `3889×N`): simply re-run the same
   `modal run --detach modal_train.py`. `resume=True` continues from the last
   checkpoint (saved every 500 steps). Repeat until the max step hits `3889×N`.

> **⚠️ LR-schedule research note.** `epochs=N` stretches the cosine over N epochs,
> so when epoch N resumes, its LR is *mid-schedule* (higher than the min the prior
> epoch ended at) and decays to `min_lr` by step `3889×N`. This is a valid warm
> continuation but **not** a fresh per-epoch cosine. If your research intends a
> different LR policy (constant low LR, or a re-warm), that is a code change to
> `_lr_at_step` in `train_core.py` — decide and document it. Default = leave as is.

> **⚠️ Overfitting watch.** Epochs 2+ re-see the *same* 2.04B tokens. The key
> research question is whether val perplexity keeps improving or starts to worsen
> (memorization). The Step E eval answers this; expect diminishing returns.

---

## STEP C — Capture Modal research parameters

Pull the training telemetry from the volume and record it. Do this **per epoch**.

```bash
cd "C:/Users/harma/slm-125m-epoch{N}"
PYTHONIOENCODING=utf-8 ../<MAIN_REPO>/.venv/Scripts/modal.exe volume get slm-125m checkpoints/base/metrics.jsonl ./metrics_e{N}.jsonl
```

Then compute and write these into **`training_params_e{N}.md`** in the clone
(more than epoch 1 recorded — capture all of them):

**From `metrics.jsonl`** (filter to this epoch's step range `3889×(N-1)`→`3889×N`):
- final `train_loss` (note: it is ~`grad_accum`×true loss ≈ 16× — divide by 16 for per-token)
- final `val_loss` and `perplexity` (last eval row of the epoch)
- perplexity **trajectory** within the epoch (every eval row)
- LR at the final step; warmup already elapsed
- avg / min / max `grad_norm` (training stability; flag any spikes)
- avg `tokens_per_sec` (throughput)
- `tokens_seen` (cumulative) and step count
- per-epoch **wall-clock** = `wall_time` at last step − `wall_time` at first step of the epoch

**From Modal (app logs / dashboard `modal.com/apps`)**:
- GPU type (A100-40GB), container start/stop, GPU-seconds
- **cost** for the run (Modal billing/dashboard, or estimate: A100-40GB ≈ $2–4/GPU-hr × wall-hours; may be $0 within monthly credits)
- **MFU%** ≈ `6 × 125.8e6 × tokens_per_sec ÷ (A100 bf16 peak ≈ 312e12 FLOP/s)`

**Cross-epoch table** (append to a master `EPOCHS.md`): epoch · final val ppl ·
final val loss · avg tok/s · wall-clock · cost · MFU% — so the progression is one
glance (epoch 1 baseline: val ppl **11.35**, val loss 2.43, ~111k tok/s, ~5.1 h).

---

## STEP D — Export epoch N's weights to a new HF repo (the agent does this automatically)

**The agent runs this automatically at the end of every training run — it is not
optional and must not wait for the user to ask.** It is the durable archive of
epoch N: it reads the current `ckpt.pt` (= epoch N right now) and pushes an
HF-format model, so the weights survive the next epoch overwriting the volume.
(The AUTOMATED SAFEGUARD above also re-checks it at the start of the next run.)
Run from the **main repo** (has `.venv` + `modal_export_hf.py`):

```bash
cd "<MAIN_REPO>"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/modal.exe run modal_export_hf.py --repo-id Ace-2504/slm-125m-e{N}
```
- For N=2 use `Ace-2504/slm-125m-e2`, etc. (epoch 1 already = `Ace-2504/slm-125m-base`).
- The `MODEL_CARD` string in `modal_export_hf.py` has epoch-1 numbers hardcoded —
  update the perplexity/epoch line for accuracy, or accept it's generic.
- Verify: `./.venv/Scripts/python.exe -c "from huggingface_hub import list_repo_files as l; print(l('Ace-2504/slm-125m-e{N}'))"` → should list `model.safetensors`, `config.json`, tokenizer files.

**Optional but recommended — snapshot the raw checkpoint on the volume** so epoch N
survives on Modal too:
```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/modal.exe volume cp slm-125m checkpoints/base/ckpt.pt checkpoints/epoch{N}/ckpt.pt
```

Only after the export + verify is safe to launch the next epoch (Step B for N+1).

---

## STEP E — Evaluate epoch N + build the report (end to end)

Reuse the hardened eval clone `C:\Users\harma\slm-125m-eval`. It compares the SLM
against fixed baselines (gpt2, distilgpt2, pythia-160m) on **independent,
decontaminated** held-out text with bits-per-byte + CIs + a CaseHOLD downstream
task, and renders a report + screenshots. Do the *exact same* pipeline as epoch 1,
just pointed at the epoch-N model.

**E1. Point the eval at epoch N.** Edit the model id in **both** harness scripts in
`slm-125m-eval/experiment/`:
- `run_experiment_v2.py` → in `MODELS`, change `("slm", "Ace-2504/slm-125m-base")`
  to `("slm", "Ace-2504/slm-125m-e{N}")`. (To compare epochs side by side, instead
  add extra entries like `("slm-e1","Ace-2504/slm-125m-base"), ("slm-e{N}","Ace-2504/slm-125m-e{N}")`
  — but then also add them to `CHART_MODELS`/`COLORS`/`ORDER` in `build_report_v2.py`
  and `build_report_v3.py`. Simplest per-epoch run: keep one `slm` entry pointing at epoch N.)
- `run_downstream.py` → same one-line change in its `MODELS`.

> **⚠️ Drift caution.** The eval clone is **reused** across epochs, so its `MODELS`
> lists still hold the *previous* epoch's id. Set BOTH scripts to epoch N and
> re-confirm the printed model id in the run logs before trusting results —
> otherwise you silently re-evaluate the old model. *Cleaner one-time fix:* change
> the `slm` entry in both scripts to
> `os.environ.get("SLM_MODEL_ID", "Ace-2504/slm-125m-base")`, then set
> `SLM_MODEL_ID=Ace-2504/slm-125m-e{N}` per run — no code edit each epoch.

**E2. Held-out eval data is fixed** (independent SCOTUS / LEDGAR / Wikipedia / C4,
already decontaminated and cached in `eval_data_v2/`). **Reuse it — do not
re-prepare** (keeps epochs comparable). Only re-run `prepare_data_v2.py` if
`eval_data_v2/` is missing.

**E3. Run the pipeline** (CPU, ~40 min for the density harness + ~45 min CaseHOLD;
run in background and watch the logs). Use the eval clone's venv:
```bash
cd "C:/Users/harma/slm-125m-eval/experiment"
PY="C:/Users/harma/slm-125m-eval/.venv/Scripts/python.exe"
HF="HF_HOME=C:/Users/harma/slm-125m-eval/.hf-cache"; PW="PLAYWRIGHT_BROWSERS_PATH=C:/Users/harma/slm-125m-eval/.pw-browsers"

# density metrics + generations (bits-per-byte, perplexity, CIs, BOS variant)
PYTHONIOENCODING=utf-8 $HF $PY -u run_experiment_v2.py > ../results/run_v2.log 2>&1
# downstream capability (CaseHOLD, zero-shot, chance 20%)
PYTHONIOENCODING=utf-8 $HF $PY -u run_downstream.py   > ../results/downstream.log 2>&1
# build the v3 report (adds compression ratio + word-perplexity + downstream + assumptions bullets + conclusion)
PYTHONIOENCODING=utf-8 $PY build_report_v3.py
# screenshots via headless Chromium (in-app preview is unreliable — see debugging/17)
PYTHONIOENCODING=utf-8 $PW $PY shoot_v3.py
```
Outputs land in `slm-125m-eval/results/` (`metrics_v2.json`, `downstream.json`,
`REPORT_v3.md`, `report_v3.html`) and `slm-125m-eval/screenshots_v3/` (full report
+ per-section PNGs).

**E4. What the report contains** (this is the "report" the researcher wants — same
as epoch 1): assumptions & parameters (bulleted), a **bits-per-byte** chart with
95% CI whiskers + plain-language interpretation, the **CaseHOLD** accuracy chart +
interpretation, an auditable table (bits/byte · compression× · perplexity ·
word-ppl · tokens · bytes), fairness caveats, side-by-side generations, and a
conclusion. Verify by viewing `screenshots_v3/report_full.png`.

**E5. Archive this epoch's eval** so epochs don't overwrite each other:
```bash
cp -r "C:/Users/harma/slm-125m-eval/results"          "C:/Users/harma/slm-125m-eval/results_e{N}"
cp -r "C:/Users/harma/slm-125m-eval/screenshots_v3"   "C:/Users/harma/slm-125m-eval/screenshots_e{N}"
```

---

## STEP F — Cross-epoch comparison (the research payoff)

**Automated trajectory chart.** After archiving this epoch (Step E5), render the
SLM's across-epoch trajectory with `compare_epochs.py` (in the eval clone). It
reads every `results_e{N}/` and plots bits-per-byte per domain across epochs +
CaseHOLD across epochs:
```bash
# one-time backfill of epoch 1's eval (skip if results_e1/ already exists):
[ -d "C:/Users/harma/slm-125m-eval/results_e1" ] || cp -r "C:/Users/harma/slm-125m-eval/results" "C:/Users/harma/slm-125m-eval/results_e1"
cd "C:/Users/harma/slm-125m-eval/experiment"
PYTHONIOENCODING=utf-8 PLAYWRIGHT_BROWSERS_PATH="C:/Users/harma/slm-125m-eval/.pw-browsers" "C:/Users/harma/slm-125m-eval/.venv/Scripts/python.exe" compare_epochs.py
```
Outputs: `results/EPOCH_COMPARISON.md`, `results/report_epochs.html`,
`results/epoch_comparison.png`. **Lines turning upward = overfitting** on the fixed
2.04B-token corpus (the key thing to watch for epochs 2+).

Also append this epoch's headline numbers to `EPOCHS.md` (in the epoch-N clone or
the eval clone) so the trajectory is explicit:

| Epoch | val ppl (own tok) | SEC bits/byte | Legal(SCOTUS) bits/byte | CaseHOLD acc | wall-clock | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 11.35 | 0.652 | 1.143 | 15.5% | 5.1 h | baseline |
| N | … | … | … | … | … | improved / plateaued / overfit? |

State plainly whether epoch N **improved, plateaued, or regressed** vs prior epochs
(watch for val perplexity worsening = overfitting on the fixed corpus), and whether
its position vs the baselines changed.

---

## STEP G — Publish a per-epoch frontend to its own Vercel URL

Each epoch gets its own live site that looks identical but shows that epoch's
numbers, at its own clean URL (`125m-slm-e{N}.vercel.app`). Epoch 1's site
(`125m-slm.vercel.app`) is a separate project and stays untouched.

1. **Update the numbers (one place).** In the epoch-N clone, edit
   **`web/lib/model.ts`** → the `RUN` object only:
   ```ts
   export const RUN = {
     epoch: N,
     valPerplexity: "<this epoch's val ppl>",
     valLoss: "<this epoch's val loss>",
     trainedHours: "<cumulative A100 wall-clock>",
     tokensPerSec: "<from Step C>",
   } as const;
   ```
   Everything else on the page (hero stat tile, the "what this is" paragraph, the
   pretrain chip) reads from `RUN`/`epochLabel` — no other edits needed. Sanity
   check: `npm --prefix web install && npm --prefix web run build`.

2. **(Optional) point the demo at this epoch's model.** The `/api/generate` route
   targets `INFERENCE_URL`. If you stand up a per-epoch inference endpoint, set that
   env var in the new Vercel project; otherwise the demo keeps its current target.

3. **Deploy to a NEW Vercel project** (the `vercel` CLI is already installed +
   logged in as this user). From the clone's `web/`:
   ```bash
   cd "C:/Users/harma/slm-125m-epoch{N}/web"
   vercel project add 125m-slm-e{N}
   vercel deploy --prod --yes --project 125m-slm-e{N}
   ```
   - `web/vercel.json` pins `framework: nextjs` and `package.json` uses a patched
     Next.js, so the CLI build works (see `debugging/12`, `debugging/13`). Deploying
     from inside `web/` means no Root-Directory setting is needed for a CLI deploy.
   - Production URL = `https://125m-slm-e{N}.vercel.app`. **Verify** it returns 200
     and shows the updated perplexity/epoch:
     ```bash
     curl -s -o /dev/null -w "%{http_code}\n" https://125m-slm-e{N}.vercel.app
     curl -s https://125m-slm-e{N}.vercel.app | grep -o "<this epoch's val ppl>"
     ```

---

## Ordering checklist (do not reorder)

1. `git clone` main repo → `slm-125m-epoch{N}` (Step A)
2. **AUTOMATED SAFEGUARD (agent-run, blocking):** detect the epoch on the volume and auto-export it to HF if not already there — never launch training until this passes (Safeguard section)
3. Edit `epochs=N` in the clone's `modal_train.py`; `modal run --detach` (Step B)
4. Wait for step `3889×N`; capture `metrics.jsonl` + Modal params → `training_params_e{N}.md` (Step C)
5. **Auto-export epoch N to `Ace-2504/slm-125m-e{N}` and verify** (+ optional volume snapshot) — the agent does this automatically at end of run (Step D)
6. Point the eval at epoch N; run v2 harness + downstream + build_report_v3 + shoot_v3; archive to `results_e{N}/` (Step E)
7. Run `compare_epochs.py`; update `EPOCHS.md` trajectory; write the verdict (Step F)
8. Update `web/lib/model.ts` `RUN` to epoch N; deploy per-epoch frontend → `125m-slm-e{N}.vercel.app` (Step G)

## Gotchas (see `debugging/` for full write-ups)

- **Windows cp1252** crashes Modal/HF CLIs on ✓/emoji → always `PYTHONIOENCODING=utf-8` (`debugging/11`).
- **In-app preview screenshots wedge** → use headless Playwright (`shoot_*.py`, `PLAYWRIGHT_BROWSERS_PATH`) (`debugging/17`).
- **Perplexity is not cross-tokenizer comparable** → bits-per-byte is the headline; the report already does this (`debugging/18`).
- **Do the export before the next epoch** — the volume `ckpt.pt` is overwritten by continued training (this file, Step 1).
- **Eval clone has its own `.venv`/`.hf-cache`** — don't mix with the main repo's `.venv`.
- **Off-OneDrive** for clones/caches to avoid multi-GB cloud-sync churn.
