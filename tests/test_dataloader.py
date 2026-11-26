from pathlib import Path

import pytest

from dataset.shakespeare_dataset import ShakespeareDataset
from src.dataloader import DataLoader

SHAKESPEARE = Path(__file__).resolve().parent.parent / "dataset" / "raw_data" / "shakespeare.txt"
B, T = 2, 16
STEP = B * T
WORLD_SIZE = 1
RANK = 0


@pytest.fixture(scope="module")
def tokens():
    return DataLoader(ShakespeareDataset(SHAKESPEARE), batch_size=B, rank=RANK, world_size=WORLD_SIZE, context_length=T).tokenized_data


@pytest.fixture
def loader():
    return DataLoader(ShakespeareDataset(SHAKESPEARE), batch_size=B, context_length=T, rank=RANK, world_size=WORLD_SIZE)


@pytest.fixture
def limit(loader):
    # An index is valid for the *next* read only if idx <= N - (B*T + 1),
    # since a read takes B*T + 1 tokens (inputs plus the shifted target).
    return len(loader.tokenized_data) - (STEP + 1)


def start_for_next_idx(next_idx: int) -> int:
    """Starting position whose advanced index (start + B*T) equals next_idx."""
    return next_idx - STEP


def test_advances_by_one_step_normally(loader):
    loader.next_batch()
    assert loader.idx == STEP


def test_batch_shapes_and_target_shift(loader):
    x, y = loader.next_batch()
    assert x.shape == y.shape == (B, T)
    # targets are inputs shifted by one token
    assert (x.view(-1)[1:] == y.view(-1)[:-1]).all()


@pytest.mark.parametrize("past", [1, 2, 5, STEP])
def test_resets_to_zero_when_next_idx_past_limit(loader, limit, past):
    # next idx lands `past` tokens beyond the last valid start -> reset
    loader.idx = start_for_next_idx(limit + past)
    loader.next_batch()
    assert loader.idx == 0


def test_no_reset_when_next_idx_is_exactly_limit(loader, limit):
    loader.idx = start_for_next_idx(limit)
    loader.next_batch()
    assert loader.idx == limit


@pytest.mark.parametrize("below", [1, 2, STEP, STEP * 3])
def test_no_reset_when_next_idx_before_limit(loader, limit, below):
    loader.idx = start_for_next_idx(limit - below)
    loader.next_batch()
    assert loader.idx == limit - below


def test_read_at_limit_is_full_and_then_resets(loader, limit):
    # idx == limit is the last position with a full B*T + 1 token slice
    loader.idx = limit
    x, y = loader.next_batch()
    assert x.shape == y.shape == (B, T)
    assert loader.idx == 0


def test_reset_batch_restarts_from_beginning(loader, tokens, limit):
    loader.idx = limit
    loader.next_batch()
    x, _ = loader.next_batch()
    assert x.view(-1).tolist() == tokens[:STEP]


def test_full_epoch_never_produces_short_slice(loader, tokens):
    # walking the whole dataset twice must never raise from a short .view()
    n_steps = 2 * (len(tokens) // STEP) + 2
    for _ in range(n_steps):
        x, y = loader.next_batch()
        assert x.shape == y.shape == (B, T)
