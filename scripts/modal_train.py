"""
Train a model on a Modal GPU.

    pip install modal && modal setup     # one-time
    modal run modal_gpu_app.py           # train
    modal volume get training-vol /checkpoints ./checkpoints   # download results
"""

from pathlib import Path, PurePosixPath

import modal

app = modal.App("gpt2chat_train")
PROJECT_DIR = Path(__file__).parent.parent  # folder above this script
MODAL_PROJECT_DIR = "/root/project"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch")
    .uv_sync(str(PROJECT_DIR))
    .env({"PYTHONPATH": MODAL_PROJECT_DIR})  # adds path for import to look for
    .add_local_dir(
        PROJECT_DIR,
        remote_path=MODAL_PROJECT_DIR,
        ignore=["**/.git", "**/__pycache__", "**/.venv", "**/venv", "**/*.pt"],
    )
)
volume = modal.Volume.from_name("gpt2chat-train", create_if_missing=True)
volume_path = PurePosixPath("/data")
dataset_path = volume_path / "raw_data" / "shakespeare.txt"
NUM_GPU = 8
gpu = f"A100:{NUM_GPU}"


@app.function(gpu=gpu, image=image, volumes={volume_path: volume}, timeout=60 * 60 * 6)
def train():
    import subprocess

    subprocess.run(
        ["torchrun", "--standalone", f"--nproc-per-node={NUM_GPU}", "-m", "scripts.train", "--modal"],
        check=True,
        cwd="/root/project"
    )


@app.local_entrypoint()
def main():
    train.remote()
