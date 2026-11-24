from dataclasses import dataclass
from time import perf_counter

import torch

from dataset.shakespeare import ShakespeareData
from scripts.utils import get_device
from src.dataloader import DataLoader
from src.model import GPT2, ModelConfig
from src.scheduler import get_cosine_annealing_lr


@dataclass
class Hyperparameters:
    steps = 50
    batch_size = 4
    context_length = 32
    weight_decay = 0.1
    base_lr = 6e-4

device = get_device()

def log(loss: torch.Tensor, config: Hyperparameters, start_time: float, step: int, grad_norm:torch.Tensor, lr: float):
    if device == 'cuda':
        torch.cuda.synchronize()
    tn = perf_counter()
    tt = tn - start_time
    toks = config.batch_size * config.context_length / tt
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

def train(config: Hyperparameters):
    model = GPT2(ModelConfig()).to(device)
    dataset = ShakespeareData()
    loader = DataLoader(dataset, batch_size=1, context_length=1024)
    torch.set_float32_matmul_precision("high")
    param_groups = get_param_groups(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.base_lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    # training loop
    for i in range(config.steps):
        lr = get_cosine_annealing_lr(step=i, warmup_it=10, cosine_it=config.steps, max_lr=config.base_lr, min_lr=0.1*config.base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        t0 = perf_counter()
        optimizer.zero_grad()
        # load dataset
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device):
            _, loss = model.forward(x, y)
        if not loss:
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        log(loss=loss, config=config, start_time=t0, step=i, grad_norm=grad_norm, lr=lr)







