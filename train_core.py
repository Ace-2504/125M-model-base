"""Training engine for the 125M from-scratch SLM."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

import config


CommitFn = Callable[[], None] | None


class BinWindowDataset:
    """Random-access packed uint16 windows spread across multiple .bin shards."""

    def __init__(self, root: str | os.PathLike[str], split: str, seq_len: int):
        self.root = Path(root)
        self.split = split
        self.seq_len = seq_len
        self.dir = self.root / "tokens" / split
        self.paths = sorted(self.dir.glob("*.bin"))
        if not self.paths:
            raise FileNotFoundError(f"no token bin files found in {self.dir}")

        self.arrays: list[np.memmap] = []
        self.window_counts: list[int] = []
        for path in self.paths:
            size = path.stat().st_size
            bytes_per_window = seq_len * np.dtype(np.uint16).itemsize
            if size % bytes_per_window != 0:
                raise ValueError(f"{path} size is not a multiple of {seq_len} uint16 tokens")
            windows = size // bytes_per_window
            self.arrays.append(np.memmap(path, dtype=np.uint16, mode="r").reshape(windows, seq_len))
            self.window_counts.append(windows)

        self.offsets = np.cumsum([0, *self.window_counts], dtype=np.int64)
        self.total_windows = int(self.offsets[-1])

    def __len__(self) -> int:
        return self.total_windows

    def get_batch(self, indices: np.ndarray, device: str):
        import torch

        batch = np.empty((len(indices), self.seq_len), dtype=np.int64)
        shard_ids = np.searchsorted(self.offsets[1:], indices, side="right")
        for shard_id in np.unique(shard_ids):
            mask = shard_ids == shard_id
            local = indices[mask] - self.offsets[shard_id]
            batch[mask] = self.arrays[int(shard_id)][local]
        return torch.from_numpy(batch).to(device=device, non_blocking=True)


def _build_model(device: str, attn_implementation: str):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    llama_config = LlamaConfig(
        **config.MODEL.to_llama_kwargs(),
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        attn_implementation=attn_implementation,
    )
    model = LlamaForCausalLM(llama_config)
    model.to(device)
    if device == "cuda":
        model.to(dtype=torch.bfloat16)
    return model


def _lr_at_step(step: int, total_steps: int, lr_mode: str) -> float:
    train = config.TRAIN
    if lr_mode != "scratch":
        raise ValueError(f"unsupported lr_mode={lr_mode!r}")
    warmup_steps = max(1, math.ceil(train.warmup_tokens / train.global_batch_tokens))
    if step < warmup_steps:
        return train.lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return train.min_lr + cosine * (train.lr - train.min_lr)


def _atomic_save(obj: object, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    if path.exists():
        prev = path.with_name("ckpt_prev.pt")
        os.replace(path, prev)
    os.replace(tmp, path)


def _checkpoint_path(ckpt_dir: str | os.PathLike[str]) -> Path:
    return Path(ckpt_dir) / "ckpt.pt"


def _load_checkpoint(model, optimizer, ckpt_dir: str | os.PathLike[str], device: str) -> dict | None:
    import torch

    path = _checkpoint_path(ckpt_dir)
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if device == "cuda" and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    return checkpoint


def _save_checkpoint(
    *,
    model,
    optimizer,
    ckpt_dir: str | os.PathLike[str],
    step: int,
    epoch: int,
    tokens_seen: int,
    total_steps: int,
    commit_fn: CommitFn,
) -> None:
    import torch

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "tokens_seen": tokens_seen,
        "total_steps": total_steps,
        "model_config": asdict(config.MODEL),
        "train_config": asdict(config.TRAIN),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    _atomic_save(payload, _checkpoint_path(ckpt_dir))
    if commit_fn is not None:
        commit_fn()


def _append_metric(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def evaluate_loss(model, dataset: BinWindowDataset, batch_size: int, device: str, max_batches: int | None = None) -> float:
    import torch

    model.eval()
    losses: list[float] = []
    total_batches = math.ceil(len(dataset) / batch_size)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    with torch.no_grad():
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            stop = min(len(dataset), start + batch_size)
            indices = np.arange(start, stop, dtype=np.int64)
            input_ids = dataset.get_batch(indices, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=input_ids, labels=input_ids).loss
            losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses))


def run(
    *,
    data_root: str,
    ckpt_dir: str,
    init_from_hf: str = "",
    epochs: int = 1,
    lr_mode: str = "scratch",
    commit_fn: CommitFn = None,
    resume: bool = True,
    eval_batches: int = 50,
) -> None:
    import torch

    if init_from_hf:
        raise ValueError("init_from_hf is intentionally unused for this from-scratch phase")

    train_cfg = config.TRAIN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_data = BinWindowDataset(data_root, "train", train_cfg.seq_len)
    val_data = BinWindowDataset(data_root, "val", train_cfg.seq_len)
    windows_per_step = train_cfg.global_batch_tokens // train_cfg.seq_len
    grad_accum = windows_per_step // train_cfg.micro_batch_size
    if windows_per_step % train_cfg.micro_batch_size != 0:
        raise ValueError("micro_batch_size must evenly divide windows per global step")

    steps_per_epoch = len(train_data) // windows_per_step
    total_steps = steps_per_epoch * epochs
    model = _build_model(device, attn_implementation="sdpa")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
        fused=(device == "cuda"),
    )

    start_step = 0
    tokens_seen = 0
    if resume:
        checkpoint = _load_checkpoint(model, optimizer, ckpt_dir, device)
        if checkpoint is not None:
            start_step = int(checkpoint.get("step", 0))
            tokens_seen = int(checkpoint.get("tokens_seen", 0))
            print(f"resumed from step {start_step:,} with {tokens_seen:,} tokens seen", flush=True)

    metrics_path = Path(ckpt_dir) / "metrics.jsonl"
    started = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    current_epoch = -1
    permutation: np.ndarray | None = None
    for global_step in range(start_step, total_steps):
        epoch = global_step // steps_per_epoch
        step_in_epoch = global_step % steps_per_epoch
        if epoch != current_epoch or permutation is None:
            epoch_rng = np.random.default_rng(train_cfg.seed + epoch)
            permutation = epoch_rng.permutation(len(train_data))
            current_epoch = epoch
        batch_indices = permutation[
            step_in_epoch * windows_per_step : (step_in_epoch + 1) * windows_per_step
        ]

        lr = _lr_at_step(global_step, total_steps, lr_mode)
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_loss = 0.0
        step_start = time.time()
        for accum_idx in range(grad_accum):
            micro_indices = batch_indices[
                accum_idx * train_cfg.micro_batch_size : (accum_idx + 1) * train_cfg.micro_batch_size
            ]
            input_ids = train_data.get_batch(micro_indices, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=input_ids, labels=input_ids).loss / grad_accum
            loss.backward()
            step_loss += float(loss.detach().cpu()) * grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += train_cfg.global_batch_tokens
        completed_step = global_step + 1

        if completed_step % train_cfg.log_every_steps == 0 or completed_step == 1:
            elapsed = time.time() - started
            step_time = max(1e-6, time.time() - step_start)
            row = {
                "step": completed_step,
                "lr": lr,
                "train_loss": step_loss,
                "grad_norm": float(grad_norm.detach().cpu()),
                "tokens_seen": tokens_seen,
                "tokens_per_sec": train_cfg.global_batch_tokens / step_time,
                "wall_time": elapsed,
            }
            _append_metric(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)

        should_eval = completed_step % train_cfg.eval_every_steps == 0 or completed_step == total_steps
        if should_eval:
            val_loss = evaluate_loss(model, val_data, train_cfg.micro_batch_size, device, max_batches=eval_batches)
            row = {
                "step": completed_step,
                "val_loss": val_loss,
                "perplexity": math.exp(min(20.0, val_loss)),
                "tokens_seen": tokens_seen,
                "wall_time": time.time() - started,
            }
            _append_metric(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)

        should_ckpt = completed_step % train_cfg.ckpt_every_steps == 0 or completed_step == total_steps
        if should_ckpt:
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                ckpt_dir=ckpt_dir,
                step=completed_step,
                epoch=epoch,
                tokens_seen=tokens_seen,
                total_steps=total_steps,
                commit_fn=commit_fn,
            )
            print(f"checkpoint saved at step {completed_step:,}", flush=True)
