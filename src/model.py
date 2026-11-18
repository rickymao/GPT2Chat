from dataclasses import dataclass
from typing import Any
from torch import nn

@dataclass
class ModelConfig:
    n_embd = 768
    n_layers = 12
    n_heads = 12
    context_length = 1024
    vocab_size = 50257 # from the GPT2 paper with <|endoftext|> token

class MultiHeadCausalAttention(nn.Module):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

class Block(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads




class GPT2(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        n_embd = self.n_embd = config.n_embd
        n_layers = self.n_layers = config.n_layers
        n_heads = self.n_heads = config.n_heads
        vocab_size = self.vocab_size = config.vocab_size
        context_length = self.context_length = config.context_length
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(context_length, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True)
        self.ln_f = nn.LayerNorm(n_embd)
        self.h = nn.ModuleList(Block() for _ in range(n_layers))
        


