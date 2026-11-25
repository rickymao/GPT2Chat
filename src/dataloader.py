import tiktoken
import torch

from dataset.shakespeare_dataset import Dataset


# Dataloader that loads data sequentially from a dataset
class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, context_length: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.dataset_idx = 0  # track where are we in the dataset
        tokenizer = tiktoken.encoding_for_model("gpt2")
        # load and encode data from url
        self._data = dataset.data()
        self.tokenized_data = tokenizer.encode(self._data)

        print(
            f"{len(self.tokenized_data) // (batch_size * context_length)} steps per epoch"
        )

    def next_batch(self):
        # take batch_size * context_length + 1
        raw_data = self.tokenized_data[
            self.dataset_idx : self.dataset_idx
            + (self.batch_size
            * self.context_length)
            + 1
        ]
        raw_data = torch.tensor(raw_data, dtype=torch.long)

        x = raw_data[:-1].view(self.batch_size, self.context_length)
        y = raw_data[1:].view(self.batch_size, self.context_length)

        # next batch start from where the last target since we don't use the last target to predict and learn
        self.dataset_idx = (
            self.dataset_idx + self.batch_size * self.context_length
        )

        # if next idx can't select enough examples go back to zero
        if self.dataset_idx > (len(self.tokenized_data) - self.batch_size * self.context_length + 1):
            self.dataset_idx = 0

        return (x, y)
