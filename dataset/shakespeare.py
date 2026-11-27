from pathlib import Path

import numpy as np
from tiktoken import get_encoding

DATA_PATH = Path("dataset/raw_data") / "shakespeare.txt"
tokenizer = get_encoding("gpt2")
shakespeare_path = Path("dataset/shakespeare")
shakespeare_path.mkdir(exist_ok=True)

toks = []
with open(DATA_PATH) as f:
    for line in f:
        toks += tokenizer.encode(line)
    toks = np.array(toks, dtype=np.uint16)
    toks.tofile(shakespeare_path / "train.bin")


