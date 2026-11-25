from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

DATA_PATH = Path("dataset/raw_data") / "shakespeare.txt"

class Dataset:

    @abstractmethod
    def data(self, length: int | None=None) -> str:
        """Returns the full raw data of the dataset"""
    @abstractmethod
    def get_iterator(self) -> Iterator:
        """Returns a line iterator of the dataset"""

class  ShakespeareDataset(Dataset):
    def __init__(self, path: Path | PurePosixPath | str | None) -> None:
        super().__init__()
        if path is None:
            self.path = DATA_PATH
        else:
            self.path = path

    def data(self, length: int | None=None):
        with open(self.path, "r") as f:
            if length is not None:
                return f.read(length)
            return f.read()

    def get_iterator(self):
        with open(self.path, "r") as f:
            yield from f  
