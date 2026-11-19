import tiktoken
import torch

from src.model import GPT2, ModelConfig

from .utils import get_device

device = get_device()
torch.manual_seed(42)
config = ModelConfig()
model = GPT2(config).to(device)
model.eval()
tokenizer = tiktoken.encoding_for_model("gpt2")

@torch.no_grad()
def generate(msg: str, max_length: int):
    toks = tokenizer.encode(msg)
    x = torch.tensor(toks, dtype=torch.long, device=device).view(1, -1) # (B, S)
    for _ in range(max_length):
        logits, _ = model.forward(x) # (B, S, vocab_size)
        logits = logits[:, -1, :] # (B, vocab_size) only look at the next token logits
        probs = torch.softmax(logits, dim=-1)
        probs, topk_idxs = torch.topk(probs, 50) # get top 50 probs and renormalize 
        idx = torch.multinomial(probs, 1)
        toks = torch.gather(topk_idxs, dim=-1, index=idx) # (B, 1)
        # concat
        x = torch.concat([x, toks], dim=-1)
    return x


resp = generate("Hello there, ", 50)
print(tokenizer.decode(resp.view(-1).tolist()))





