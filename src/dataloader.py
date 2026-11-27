from pathlib import Path
import numpy as np
import torch


# Dataloader that loads data sequentially from a dataset
class DataLoader:
    def __init__(self, path: str | Path, batch_size: int, context_length: int, rank: int, world_size: int, is_master: bool = True) -> None:
        self.path = path
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.context_length = context_length
        self.idx = self.rank * self.context_length * self.batch_size  # track where are we in the dataset

    def next_batch(self):

        ds = np.memmap(self.path, dtype=np.uint16, mode="r")        
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

        # if next idx can't select enough examples go back to zero
        if self.idx  > (len(ds) - (self.batch_size * self.context_length + 1)):
            self.idx = self.rank * self.context_length * self.batch_size
        return (x, y)
