import os
import multiprocessing as mp
from tiktoken import get_encoding
import datasets
from typing import Any
import numpy as np
import argparse
from pathlib import Path
import tqdm

DATASET_NAME = "fineweb_edu"
parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train", required=False)
args = parser.parse_args()
split = args.split
tokenizer = get_encoding("gpt2")
eot = tokenizer.eot_token
shard_size = int(1e8)
token_path = Path(f"dataset/{DATASET_NAME}")
token_path.mkdir(exist_ok=True)

def get_shard_path(split: str, shard_id: int):
    return token_path / f"{DATASET_NAME}_{split}_{shard_id:06d}.bin"


def tokenize(row: dict[str, Any]):
    data = row["text"]
    tokens = [eot]
    tokens += tokenizer.encode_ordinary(data)
    tokens = np.array(tokens)
    assert (tokens < 2**16).all(), "vocab size of data is larger than permitted for gpt2" 
    tokens = tokens.astype(np.uint16)
    return tokens


if __name__ == "__main__":
    fw = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=split, streaming=True)
    cpu_count = os.cpu_count() 
    if not isinstance(cpu_count, int): 
        raise TypeError("cpu_count should be not None")

    nproc = max(1, cpu_count)
    # iterate thru the dataset and tokenize
    with mp.Pool(nproc) as pool:
        shard = np.empty(shard_size, dtype=np.uint16)
        idx = 0
        shard_id = 0
        pbar = tqdm.tqdm(total=shard_size, unit="tok", desc=f"shard {shard_id}")
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if idx + len(tokens) <= len(shard):
                shard[idx: idx + len(tokens)] = tokens
                idx += len(tokens)
                pbar.update(len(tokens))
            else:
                # if there are remaining ones add them to the old one before writing
                remainder = shard_size - idx
                shard[idx: idx+remainder] = tokens[:remainder]
                pbar.update(remainder)
                # write the shard to file by doing to file 
                split = "val" if shard_id == 0 else "train"
                shard.tofile(get_shard_path(split=split, shard_id=shard_id))
                pbar.close()

                # create a new shard and prefill with leftover if needed 
                leftover = tokens[remainder:]
                idx = 0
                shard_id += 1
                # create new pbar
                pbar = tqdm.tqdm(total=shard_size, unit="tok", desc=f"shard {shard_id}")
                shard[idx: idx + len(leftover)] = leftover
                idx += len(leftover)
                pbar.update(len(leftover))

        # write last shard if shard is not full
        pbar.close()
        split = "val" if shard_id == 0 else "train"
        shard = shard[:idx]
        shard.tofile(get_shard_path(split=split, shard_id=shard_id))






                
                
        

                

