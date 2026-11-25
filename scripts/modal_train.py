"""
Train a model on a Modal GPU.

    pip install modal && modal setup     # one-time
    modal run modal_gpu_app.py           # train
    modal volume get training-vol /checkpoints ./checkpoints   # download results
"""

import modal
from pathlib import PurePosixPath, Path

app = modal.App("gpt2chat_train")
PROJECT_DIR = Path(__file__).parent.parent  # folder above this script
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch")
    .uv_sync(str(PROJECT_DIR))
    .env({"PYTHONPATH": "/root/project"}) # adds path for import to look for
    .add_local_dir(
        PROJECT_DIR,
        remote_path="/root/project",
        ignore=["**/.git", "**/__pycache__", "**/.venv", "**/venv", "**/*.pt"],
    )
)
volume = modal.Volume.from_name("gpt2chat-train", create_if_missing=True)
volume_path = PurePosixPath("/data")
dataset_path = volume_path / "raw_data" / "shakespeare.txt"
gpu = 'A100'

@app.function(gpu=gpu, image=image, volumes={volume_path: volume}, timeout=60 * 60 * 6)
def train():
    import os

    import torch
    import torch.nn as nn

    from scripts.train import Hyperparameters, train

    train(Hyperparameters(), modal=True)

    

@app.local_entrypoint()
def main():
    train.remote()