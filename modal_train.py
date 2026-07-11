"""Modal entrypoint for Phase 5 pretraining."""

from __future__ import annotations

import modal


app = modal.App("slm-125m-pretrain")
volume = modal.Volume.from_name("slm-125m", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "numpy>=1.26,<2.0",
    )
    .add_local_python_source("config", "train_core")
)


@app.function(
    image=gpu_image,
    gpu="A100-40GB",
    volumes={"/data": volume},
    timeout=60 * 60 * 8,
)
def pretrain(resume: bool = True):
    import train_core

    train_core.run(
        data_root="/data",
        ckpt_dir="/data/checkpoints/base",
        init_from_hf="",
        epochs=1,
        lr_mode="scratch",
        commit_fn=volume.commit,
        resume=resume,
    )


@app.local_entrypoint()
def main(resume: bool = True):
    pretrain.spawn(resume=resume)
