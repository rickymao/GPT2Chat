from dataclasses import dataclass

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
loader = DataLoader(dataset, batch_size=4, context_length=32)


def log(loss: float, config: Hyperparameters):
    print

def train(config: Hyperparameters):
    # training loop
    for i in range(config.steps):
        optimizer.zero_grad()
        # load dataset
        x, y = loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        _, loss = model.forward(x, y)
        if not loss:
            continue
        loss.backward()
        optimizer.step()

        print(f"step{i}: {loss:.3f}")


train(Hyperparameters())







