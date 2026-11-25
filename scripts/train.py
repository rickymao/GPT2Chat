from dataclasses import dataclass
from pathlib import PurePosixPath
from time import perf_counter

import torch

from dataset.shakespeare_dataset import ShakespeareDataset
from scripts.utils import get_device
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

def log(loss: torch.Tensor | float, config: Hyperparameters, start_time: float, step: int, grad_acc_steps: int, grad_norm:torch.Tensor, lr: float):
    if device == 'cuda':
        torch.cuda.synchronize()
    tn = perf_counter()
    tt = tn - start_time
    toks = config.batch_size * config.context_length * grad_acc_steps / tt
    print(f"step {step} | lr {lr} | loss {loss:.4f} | {(tt * 1000):.3f} ms | grad_norm {grad_norm} | {toks:.2f} tok/s" )

# get the param groups used for the optimizer
def get_param_groups(model: torch.nn.Module, weight_decay: float):
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
    print(f"{num_decay} decay parameters and {num_non_decay} non-decay parameters")
    return [
        { "params": decay, "weight_decay": weight_decay },
        { "params": no_decay, "weight_decay": 0.0 }
    ]

def train(config: Hyperparameters, modal: bool = False):
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.manual_seed(1337)
    model = GPT2(ModelConfig()).to(device)
    model.compile()
    dataset_path =  volume_path / "raw_data" / "shakespeare.txt" if modal else None
    data = ShakespeareDataset(path=dataset_path)
    loader = DataLoader(data, batch_size=config.batch_size, context_length=config.context_length)
    torch.set_float32_matmul_precision("high")
    param_groups = get_param_groups(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.base_lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    # total batch size
    total_batch_size = 524_288
    grad_acc_steps = total_batch_size // (config.batch_size * config.context_length)
    print(f"grad acc steps: {grad_acc_steps}")

    # training loop
    for i in range(config.steps):
        lr = get_cosine_annealing_lr(step=i, warmup_it=10, cosine_it=config.steps, max_lr=config.base_lr, min_lr=0.1*config.base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        t0 = perf_counter()
        optimizer.zero_grad()
        loss_acc = 0.0 # the loss for the big batch
        for _ in range(grad_acc_steps):
            # load dataset
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device):
                _, loss = model.forward(x, y)
            if not loss:
                continue
            loss = loss / grad_acc_steps
            loss.backward()
            loss_acc += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        log(loss=loss_acc, config=config, start_time=t0, step=i, grad_norm=grad_norm, grad_acc_steps=grad_acc_steps, lr=lr)

