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
    base_lr = 3e-4

device = get_device()

def log(loss: torch.Tensor, config: Hyperparameters, start_time: float, step: int, grad_norm:torch.Tensor, lr: float):
    if device == 'cuda':
        torch.cuda.synchronize()
    tn = perf_counter()
    tt = tn - start_time
    toks = config.batch_size * config.context_length / tt
    print(f"step {step} | lr {lr} | loss {loss:.4f} | {(tt * 1000):.3f} ms | grad_norm {grad_norm} | {toks:.2f} tok/s" )

def train(config: Hyperparameters):
    model = GPT2(ModelConfig()).to(device)
    dataset = ShakespeareData()
    loader = DataLoader(dataset, batch_size=1, context_length=1024)
    torch.set_float32_matmul_precision("high")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.base_lr)

    # training loop
    for i in range(config.steps):
        lr = get_cosine_annealing_lr(step=i, warmup_it=500, cosine_it=config.steps, max_lr=config.base_lr, min_lr=0.1*config.base_lr)
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


train(Hyperparameters())







