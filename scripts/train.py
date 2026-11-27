import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import PurePosixPath
from time import perf_counter

import torch
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.shakespeare import ShakespeareDataset
from scripts.utils import get_device, use_dpp
from src.dataloader import DataLoader
from src.model import GPT2, ModelConfig
from src.scheduler import get_cosine_annealing_lr

volume_path = PurePosixPath("/data")


@dataclass
class Hyperparameters:
    steps = 50
    batch_size = 16
    context_length = 1024
    weight_decay = 0.1
    base_lr = 6e-4


device = get_device()


def log(
    loss: torch.Tensor | float,
    config: Hyperparameters,
    start_time: float,
    step: int,
    world_size: int,
    grad_acc_steps: int,
    grad_norm: torch.Tensor,
    lr: float,
):
    if device == "cuda":
        torch.cuda.synchronize()
    tn = perf_counter()
    tt = tn - start_time
    toks = (config.batch_size * config.context_length * grad_acc_steps * world_size) 
    tkps = toks / tt
    print(
        f"step {step} | lr {lr} | loss {loss:.4f} | {(tt * 1000):.3f} ms | grad_norm {grad_norm} | {tkps:.2f} tok/s"
    )


# get the param groups used for the optimizer
def get_param_groups(
    model: torch.nn.Module, weight_decay: float, is_master: bool = False
):
    decay = []
    no_decay = []
    for p in model.parameters():
        if p.requires_grad == False:
            continue
        if len(p.shape) >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    num_decay = sum(p.numel() for p in decay)
    num_non_decay = sum(p.numel() for p in no_decay)
    if is_master:
        print(f"{num_decay} decay parameters and {num_non_decay} non-decay parameters")
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train(config: Hyperparameters, modal: bool = False):
    dpp = use_dpp()
    is_dpp = dpp["is_dpp"]
    device = dpp["device"]
    rank = dpp["rank"]
    local_rank = dpp["local_rank"]
    world_size = dpp["world_size"]
    is_master = dpp["is_master"]
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    model = GPT2(ModelConfig()).to(device)
    model.compile()

    if is_dpp:
        model = DDP(model, device_ids=[local_rank])

    dataset_path = volume_path / "raw_data" / "shakespeare.txt" if modal else None
    data = ShakespeareDataset(path=dataset_path)
    torch.set_float32_matmul_precision("high")
    loader = DataLoader(
        data,
        batch_size=config.batch_size,
        context_length=config.context_length,
        rank=rank,
        world_size=world_size,
        is_master=is_master,
    )
    param_groups = get_param_groups(model, config.weight_decay, is_master=is_master)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
        fused=True,
    )

    # total batch size
    total_batch_size = 524_288
    # ensure that total batch size can be split into divisible chunks for the GPUs
    assert (
        total_batch_size % (config.batch_size * config.context_length * world_size) == 0
    )
    # we want to process total_batch_size in one training step collectively across all GPUs
    # As a result we take in consideration the number of processes/GPUs to get the grad_acc_steps for each GPU
    grad_acc_steps = total_batch_size // (
        config.batch_size * config.context_length * world_size
    )
    if is_master:
        print(f"total batch size: {total_batch_size} tokens")
        print(f"grad acc steps: {grad_acc_steps}")
    # training loop
    for i in range(config.steps):
        lr = get_cosine_annealing_lr(
            step=i,
            warmup_it=10,
            cosine_it=config.steps,
            max_lr=config.base_lr,
            min_lr=0.1 * config.base_lr,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        t0 = perf_counter()
        optimizer.zero_grad()
        loss_acc = torch.zeros((), device=device)  # the loss for the big batch
        for j in range(grad_acc_steps):
            is_last = j == (grad_acc_steps - 1)
            # load dataset
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)

            # set ctx to not sync grads thru DDP during acc steps until its finished acc
            ctx = nullcontext()
            if isinstance(model, DDP) and not is_last:
                ctx = model.no_sync()
            with ctx:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                if not loss:
                    continue
                loss = loss / grad_acc_steps
                loss.backward()
                loss_acc += loss.detach()
        if is_dpp:
            torch.distributed.all_reduce(loss_acc, op=torch.distributed.ReduceOp.AVG)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if is_master:
            log(
                loss=loss_acc,
                config=config,
                start_time=t0,
                step=i,
                world_size=world_size,
                grad_norm=grad_norm,
                grad_acc_steps=grad_acc_steps,
                lr=lr,
            )
    if is_dpp:
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modal", action="store_true", help="configure training for Modal"
    )
    args = parser.parse_args()
    is_modal = args.modal

    train(Hyperparameters(), modal=is_modal)
