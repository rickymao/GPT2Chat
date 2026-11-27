import tiktoken
import torch
from pathlib import Path

# Dataloader that loads data sequentially from a dataset
class DataLoader:
    def __init__(self, path: Path, batch_size: int, context_length: int, rank: int, world_size: int, is_master: bool = True) -> None:
        self.path = path
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.context_length = context_length
        self.idx = self.rank * self.context_length * self.batch_size  # track where are we in the dataset
        tokenizer = tiktoken.encoding_for_model("gpt2")
        # load and encode data from url
        
        self.tokenized_data = tokenizer.encode(self._data)

    def next_batch(self):
        # take batch_size * context_length + 1
        raw_data = self.tokenized_data[
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
        if self.idx  > (len(self.tokenized_data) - (self.batch_size * self.context_length + 1)):
            self.idx = self.rank * self.context_length * self.batch_size

        return (x, y)
