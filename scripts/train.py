from dataclasses import dataclass
from time import perf_counter

import tiktoken
import torch

from dataset.shakespeare import ShakespeareData
from scripts.utils import get_device
from src.dataloader import DataLoader
from src.model import GPT2, ModelConfig


@dataclass
class Hyperparameters:
    steps = 50
    batch_size = 4
    context_length = 32

tokenizer = tiktoken.encoding_for_model("gpt2")
device = get_device()
model = GPT2(ModelConfig()).to(device)
dataset = ShakespeareData()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loader = DataLoader(dataset, batch_size=1, context_length=1024)
torch.set_float32_matmul_precision("high")

def log(loss: torch.Tensor, config: Hyperparameters, start_time: float, step: int):
    if device == 'cuda':
        torch.cuda.synchronize()
    tn = perf_counter()
    tt = tn - start_time
    toks = config.batch_size * config.context_length / tt
    print(f"step {step} | loss {loss:.4f} | {(tt * 1000):.3f} ms | {toks:.2f} tok/s" )

def train(config: Hyperparameters):
    # training loop
    for i in range(config.steps):
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
        optimizer.step()
        log(loss=loss, config=config, start_time=t0, step=i)


train(Hyperparameters())







