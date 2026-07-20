# Plan — continue pretraining v1 on the v2 corpus ("v3")

**Status: NOT STARTED. This is a proposal to approve, nothing is executing.**

Replaces the "train v2 from scratch" arm of `pre-training-v2/v2-pre-training-roadmap.md`.
Instead of a fresh 125.8M model on the v2 corpus, take the **existing v1 weights** and
continue pretraining them on the **v2 corpus** for one full pass.

Throughout, **v3** = the resulting model. It is neither v1 nor v2; it needs its own
namespace everywhere (volume paths, HF repo, Vercel project) so it cannot overwrite either.

---

## 0. Verified facts (checked against the repos, not assumed)

| Fact | Evidence | Consequence |
|---|---|---|
| **v1 and v2 tokenizers are byte-identical** | sha256 match on `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` across `data_v1/tokenizer`, `data/tokenizer`, `pre-training-v2/data/tokenizer` (`4d048ad3604086af`) | Token IDs are compatible. **This is what makes the whole idea possible.** No re-tokenization, no embedding surgery. |
| **v2's SEC corpus is byte-identical to v1's** | all 5 shards identical size; `sec/shard-000.txt` sha256 identical (`e80469991130a525b6b8`) | ~35% of the v2 corpus is *exactly* what v1 already trained on. |
| **v2's fineweb shards 0–4 are prefixes of v1's** | v2 shard-000 = 332 MB vs v1 = 402 MB; same parquet sources, lower per-shard cap (82.5M vs 100M) | v2 fineweb 0–4 ⊂ v1 fineweb. Shards 5–9 are genuinely new. |
| **v2's case-law is larger and cleaner** | v1 kept 207,041 docs / 0.806B proxy; v2 kept 241,356 / 0.954B; OCR gate 0.20→0.10 | Mostly a superset of v1's, minus docs the tighter gate now rejects. |
| **v2 tokens do not exist yet** | `pre-training-v2/data/tokens/` is empty | Phase 4 is a hard prerequisite. |
| **v1's checkpoint is not on this machine** | no `*.pt` / `checkpoints/` anywhere in either repo | `ckpt.pt` (~720 MiB) exists only on the `ace-2504` Modal Volume at `/checkpoints/base/`. |
| **`_lr_at_step` rejects anything but `lr_mode="scratch"`** | `train_core.py:83` raises `ValueError` | A continuation LR policy is a required code change, not a config flag. |

### What this means for the experiment

Roughly **~79% of the v2 corpus is data v1 has already seen** (all of SEC, most of
case-law, half of fineweb), and **~21% is new** (~0.5B tokens: extra case-law, fineweb
shards 5–9).

So be precise about what this run does and does not answer:

- ✅ It produces a **better model**. v3 will have seen ~4.5B tokens total (v1's 2.04B +
  v2's 2.5B) for the same ~6 h / ~$13 as training v2 from scratch would have cost.
  There is no compute saving — one pass is one pass — but you keep v1's prior learning
  instead of throwing it away.
- ❌ It **destroys the controlled v1-vs-v2 comparison.** The v2 roadmap's entire premise
  was "data is the only changed variable." A model that trained on v1's corpus *and*
  v2's corpus cannot attribute any improvement to the v2 corpus. You will be measuring
  "does more training help", not "is the cleaner corpus better".
- ⚠️ The tighter OCR gate is **diluted**. v1 already absorbed the garbled case-law that
  v2's gate now excludes; continuing does not unlearn it.

If the clean data-quality answer matters, v2-from-scratch has to happen anyway — this
plan is additive to it, not a substitute. **Decide which question you are buying before
spending the GPU hours.**

---

## 1. Prerequisites

- [ ] **Phase 4 tokenize on v2** — `cd pre-training-v2 && python run_tokenize_resume.py 3 100`.
      ~2.5B train tokens, 14 shards, ~5 GB. Now resumable (markers in `data/tmp/tokenize_done/`).
      **Gate:** `index.json` shows ~2.50B train tokens, val ≈ 1.00%.
- [ ] **Confirm the realized step count** — `steps = train_windows ÷ 512`. Projected **~4,716**.
      Every number below that says 4,716 must be recomputed from the actual `index.json`.
- [ ] **Retrieve v1's checkpoint from Modal** (it exists nowhere else):
      ```bash
      MODAL_PROFILE=ace-2504 "$MODAL" volume get slm-125m checkpoints/base/ckpt.pt ./ckpt_v1.pt
      ```
      720 MiB; 54 GB free locally. Do this regardless of this plan — it is the only copy.
- [ ] **Pick the target Modal account** (see §4). If it is `aceaynon`, `ckpt_v1.pt` must be
      re-uploaded there; Volumes are not shared across accounts.
- [ ] **Disk** — v2 `data/clean` (12 GB) can be deleted after Phase 4 to make room. Not yet.

---

## 2. Required code changes

Two changes in `train_core.py`. Neither is optional; the current engine cannot express
this run correctly.

### 2a. The step arithmetic breaks with a different dataset

`train_core.run` computes `steps_per_epoch` from the **loaded** dataset and sets
`total_steps = steps_per_epoch × epochs`, then loops `range(start_step, total_steps)`
where `start_step` comes from the checkpoint (3,889).

With v2's ~4,716 steps/epoch that gives:

| `epochs=` | `total_steps` | new steps actually trained | coverage of v2 |
|---|---|---|---|
| 1 | 4,716 | **827** | 0.18 of a pass — silently "finishes" almost immediately |
| 2 | 9,432 | 5,543 | 1.18 passes |

Neither is one clean pass. Worse, at `epochs=2` the first stretch (steps 3,889→4,715)
slices `permutation[step_in_epoch × 512 : …]` starting at index 1,991,168 — so it only
touches the **tail** of that permutation and skips ~82% of the corpus before rolling to
the next epoch. The guide's rule of thumb ("epochs=N trains 3,889 new steps") assumed
`steps_per_epoch` never changes. It changes here.

**Fix — treat this as a fresh run whose weights are warm-started.** Add an
`init_from_ckpt` path that loads `model` (and optionally `optimizer`) state but **resets
`start_step` to 0**, and drive length with an explicit `total_steps` rather than
`epochs × steps_per_epoch`. Then `total_steps = 4,716` is exactly one full, fully-covered
pass over v2 with a coherent schedule.

### 2b. The LR schedule has no continuation mode

`_lr_at_step` raises on any `lr_mode != "scratch"` (`train_core.py:83`). Resuming
mid-cosine would put the LR at roughly **4.2e-4** — a ~7× jump above the 6e-5 v1 ended
at — which reliably produces a loss spike and partial forgetting.

**Fix — add `lr_mode="continue"`:** linear re-warm over the first ~150 steps from
`min_lr` to a **reduced peak of ~2e-4** (⅓ of the original 6e-4), then cosine decay to
`min_lr` at `total_steps`. Standard continued-pretraining practice: high enough to learn
the new ~21%, low enough not to wreck what v1 already knows.

> Both changes are additive — `lr_mode="scratch"` and the existing resume path must keep
> behaving exactly as they do today, so v1 and any future from-scratch v2 stay reproducible.
> Add a regression check that `_lr_at_step(step, 3889, "scratch")` is unchanged.

---

## 3. Does the v2 dataset need to move into this repo?

**No — not the way you'd expect.** What matters is what lands on the **Modal Volume**;
the local directory layout is cosmetic. `modal volume put` takes any local path, so you
can upload straight from the v2 folder:

```bash
"$MODAL" volume put slm-125m "../pre-training-v2/data/tokens" /v3/tokens
```

The thing that genuinely has to move is the **checkpoint**, not the dataset — and only if
you switch accounts (§4).

That said, **run the training from this repo**, because this is where `modal_export_hf.py`,
`eval.py`, `web/`, `debugging/` and `training-agent-guide.md` live. Copying ~5 GB of
`.bin` files here as well is optional and only costs disk.

---

## 4. Modal layout — v3 must not touch v1 or v2

| | v1 (existing) | v3 (this plan) |
|---|---|---|
| tokens | `/data/tokens` | `/data/v3/tokens` |
| checkpoints | `/data/checkpoints/base` | `/data/v3/checkpoints/base` |
| app | `slm-125m-pretrain` | `slm-125m-pretrain-v3` |

Seed the v3 checkpoint dir with v1's weights, then never write to v1's path:

```bash
"$MODAL" volume put slm-125m ./ckpt_v1.pt /v3/checkpoints/base/ckpt.pt
```

**Account choice.** v1's `ckpt.pt` lives in `ace-2504`. Running v3 in `aceaynon` means
downloading it (720 MiB) and re-uploading to the other account, plus re-uploading ~5 GB
of tokens there. Running v3 in `ace-2504` avoids both transfers but shares the GPU quota
with whatever is fine-tuning there. **Recommendation: `ace-2504`**, unless a concurrent-GPU
cap is actually blocking you — the transfers are pure overhead otherwise.

Pin the profile explicitly on every command; never rely on the active flag:
```bash
MODAL_PROFILE=<profile> "$MODAL" run --detach modal_train.py
```

---

## 5. Execution order

| # | Step | Gate before proceeding |
|---|---|---|
| 1 | Phase 4 tokenize in `pre-training-v2` | `index.json` ≈ 2.50B train tokens, val ≈ 1.00% |
| 2 | Recompute `total_steps` = train_windows ÷ 512 | matches ~4,716 ± 5% |
| 3 | `volume get` v1 `ckpt.pt` → `./ckpt_v1.pt` | file is ~720 MiB and `torch.load`s |
| 4 | Implement §2a + §2b in `train_core.py` | `lr_mode="scratch"` regression check passes |
| 5 | New `modal_train_v3.py` with the `/data/v3` namespace | dry-read: v1 paths appear nowhere |
| 6 | Upload v2 tokens + seeded ckpt to `/data/v3/...` | `volume ls` shows both |
| 7 | `modal run --detach` | logs show `total_steps=4716`, `start_step=0`, LR re-warming from 6e-5 |
| 8 | Export to `Ace-2504/slm-125m-v3` **immediately** on finish | `model.safetensors` present |

Step 8 is not optional — see the AUTOMATED SAFEGUARD section of
`training-agent-guide.md`. The volume holds one continuously-overwritten checkpoint.

~6.2 h on A100-40GB at v1's measured ~111k tok/s; ~$13–14.

---

## 6. Evaluation — one trap to avoid

**Do not headline v3's validation perplexity against v1's 11.35.** Two independent
reasons it would be misleading:

1. **Different val set.** v1's 11.35 was measured on v1's val split; v3 would be measured
   on v2's. Different windows, different distribution, not comparable.
2. **Leakage.** v2's val split is every 100th window of the v2 corpus — and the SEC
   portion of that corpus is byte-identical to v1's training data. So v2's val set
   contains windows **v1 already trained on**. Perplexity there is optimistically biased
   for any model descended from v1.

Use the existing hardened harness in `C:\Users\harma\slm-125m-eval` instead — independent,
decontaminated held-out text, bits-per-byte, fixed baselines (gpt2 / distilgpt2 /
pythia-160m), bootstrap CIs. That is already the comparable metric and it sidesteps both
problems. Steps E–F of `training-agent-guide.md` apply unchanged.

**Hypothesis:** legal-opinion BPB improves from 1.1434 (more + cleaner legal tokens, more
total training). **Guardrail:** SEC BPB must not regress from 0.6519 — SEC is the domain
v1 already wins and where v3 sees zero new data. **Do not expect** CaseHOLD movement;
all models sit at chance at this scale.

---

## 7. Open decisions

- [ ] **Is the clean v1-vs-v2 data comparison being abandoned, or just deferred?** (§0)
- [ ] Target account: `ace-2504` (no transfers) or `aceaynon` (isolated quota)? (§4)
- [ ] Carry v1's **optimizer state** into v3, or start AdamW fresh? Fresh moments pair more
      naturally with a re-warm; carrying them preserves more continuity. Recommend fresh.
- [ ] Re-warm peak LR: 2e-4 (recommended), or lower (1e-4) for a gentler continuation?
- [ ] Also copy v2's `data/tokens` into this repo, or upload straight from `pre-training-v2`? (§3)
