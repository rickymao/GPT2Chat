import os

import torch
from torch.distributed import init_process_group


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    return device


def use_dpp():

    # check for dpp
    dpp = os.environ.get("RANK") is not None
    if dpp:
        assert torch.cuda.is_available()
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device_name = f"cuda:{local_rank}"
        torch.cuda.set_device(device_name)
        device = device_name
        is_master = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = get_device()
        is_master = True
    return {
        "is_dpp": dpp,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "is_master": is_master,
    }
