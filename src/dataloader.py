import os
from pathlib import Path

import numpy as np
import torch

volume_path = Path("/data")

# Dataloader that loads data sequentially from a dataset
class DataLoader:
    def __init__(self, path: str | Path, batch_size: int, context_length: int, rank: int, world_size: int, is_master: bool = True, split: str="train") -> None:
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.split = split
        assert split in ("train", "val")
        self.context_length = context_length
        self.idx = self.rank * self.context_length * self.batch_size  # track where are we in the dataset
        self.data_dir = Path("./dataset/fineweb_edu")
        shards = os.listdir(self.data_dir)
        assert len(shards) > 0
        self.shards = sorted(f for f in shards if f.endswith(".bin") and split in f)
        self.curr_shard = 0
    def next_batch(self):
        path = self.data_dir / self.shards[self.curr_shard]
        ds = np.memmap(path, dtype=np.uint16, mode="r")        
        # take batch_size * context_length + 1
        raw_data = ds[
            self.idx : self.idx
            + (self.batch_size
            * self.context_length)
            + 1
        ]
        raw_data = torch.tensor(raw_data, dtype=torch.long)

        x = raw_data[:-1].view(self.batch_size, self.context_length)
        y = raw_data[1:].view(self.batch_size, self.context_length)

        # next batch start from where the last target since we don't use the last target to predict and learn
        self.idx = (
            self.idx + self.batch_size * self.context_length * self.world_size # move to the next amount such that each process don't use the same batches 
        )

        # if next idx can't select enough examples go to next shard
        if self.idx  > (len(ds) - (self.batch_size * self.context_length * self.world_size + 1)):
            self.curr_shard = (self.curr_shard + 1) % len(self.shards)
            self.idx = self.rank * self.context_length * self.batch_size
        return (x, y)
