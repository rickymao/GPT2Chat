# gpt2chat

Implementation of a base GPT-2 to a chat and reasoning model in PyTorch.

## Structure

```
dataset/    datasets and dataset class
scripts/    Training, inference, data loader, and utilities
src/        GPT-2 model definition
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

Train the model:

```bash
uv run python -m scripts.train
```

Generate text (top-k sampling, k=50):

```bash
uv run python -m scripts.inference
```

## Tests

```bash
uv run -m pytest
```

## Stack

PyTorch, tiktoken (GPT-2 tokenizer), einops, transformers.

## Visualization

Check out a visualization of the model [here](https://claude.ai/artifact/UoxfiQAXB6ce99eNiZK1cA)
