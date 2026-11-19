from abc import abstractmethod
from collections.abc import Iterator


class Dataset:

    @abstractmethod
    def data(self, length: int | None=None) -> str:
        """Returns the full raw data of the dataset"""
    @abstractmethod
    def get_iterator(self) -> Iterator:
        """Returns a line iterator of the dataset"""

DATA_PATH = "./dataset/shakespeare.txt"

class ShakespeareData(Dataset):

    def __init__(self) -> None:
        super().__init__()

    def data(self, length: int | None=None):
        with open(DATA_PATH, "r") as f:
            if length is not None:
                return f.read(length)
            return f.read()

    def get_iterator(self):
        with open(DATA_PATH, "r") as f:
            yield from f  

