from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn, softmax


@dataclass
class ModelConfig:
    n_embd = 768
    n_layers = 12
    init_std = 0.02
    n_heads = 12
    context_length = 1024
    vocab_size = 50257  # from the GPT2 paper with <|endoftext|> token



class MultiHeadCausalAttention(nn.Module):
    def __init__(self, n_heads: int, n_embd: int, context_length: int) -> None:
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, n_embd * 3)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.SCALED_INIT = 1 # type: ignore
        self.register_buffer(
            "mask", torch.tril(torch.ones((1, 1, context_length, context_length)))
        )
        self.mask: Tensor

    def forward(self, x: Tensor):
        _, ctx, _ = x.shape
        # get the Q, K, V for each head
        qkv = self.c_attn(x)
        # split into q, k, v
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = rearrange(q, " ... seq (h dim) -> ... h seq dim", h=self.n_heads)
        k = rearrange(k, " ... seq (h dim) -> ... h seq dim", h=self.n_heads)
        v = rearrange(v, " ... seq (h dim) -> ... h seq dim", h=self.n_heads)

        # dot product attention
        attn = einsum(q, k, "... seq1 dim, ... seq2 dim -> ... seq1 seq2") / (
            q.size(-1) ** 0.5
        )

        # causal mask
        attn = attn.masked_fill(self.mask[:, :, :ctx, :ctx] == 0, float("-inf"))
        # softmax
        attn = softmax(attn, dim=-1)
        # weighted sum
        weighted_sum = einsum(attn, v, "... seq1 seq2, ... seq2 dim -> ... seq1 dim")

        # concat the head dims
        final = rearrange(weighted_sum, "... h seq dim -> ... seq (h dim)")

        # map
        return self.c_proj(final)


class FeedForward(nn.Module):
    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.input_features = input_features
        self.c_fc = nn.Linear(input_features, 4 * input_features)
        self.gelu = nn.GELU(approximate="tanh")  # TODO: remove approx. after
        self.c_proj = nn.Linear(4 * input_features, input_features)

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_heads, context_length) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.mlp = FeedForward(n_embd)
        self.attn = MultiHeadCausalAttention(
            n_heads=n_heads, n_embd=n_embd, context_length=context_length
        )
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(self, x: Tensor):

        x_norm = self.ln_1(x)
        v = self.attn(x_norm)
        # residual
        x = v + x
        x_norm = self.ln_2(x)
        x_proj = self.mlp(x_norm)
        # residual
        x = x + x_proj

        return x


class GPT2(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        n_embd = self.n_embd = config.n_embd
        n_layers = self.n_layers = config.n_layers
        n_heads = self.n_heads = config.n_heads
        vocab_size = self.vocab_size = config.vocab_size
        context_length = self.context_length = config.context_length
        self.wpe = nn.Embedding(context_length, n_embd)
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.ln_f = nn.LayerNorm(n_embd)
        self.h = nn.ModuleList(
            Block(n_embd=n_embd, n_heads=n_heads, context_length=context_length)
            for _ in range(n_layers)
        )

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = self.config.init_std
            if hasattr(module, "SCALED_INIT") == 1:
                std = self.config.init_std / ((2 * self.n_layers)**0.5) # 2 for the 2 residual branches
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)



    def forward(self, x: Tensor, targets: Tensor | None=None):

        _, S = x.size()
        assert S <= self.config.context_length

        pos = torch.arange(0, S, device=x.device)
        pos_embd = self.wpe.forward(pos) # (S, E)
        tok_embd = self.wte.forward(x) # (B, S, E)
        embd = pos_embd + tok_embd # (B, S, E)
        for block in self.h:
            embd = block.forward(embd) # (B, S, E)
        embd = self.ln_f.forward(embd)
        logits = self.lm_head.forward(embd) # (B, S, vocab_size)

        loss = None
        if targets is not None:
            ce_logits = rearrange(logits, "B S vocab_size -> (B S) vocab_size")
            targets = rearrange(targets, "B S -> (B S)")
            loss = F.cross_entropy(ce_logits, targets)
        return (logits, loss)



