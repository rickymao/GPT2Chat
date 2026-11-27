from pathlib import Path

import numpy as np
import pytest

from src.dataloader import DataLoader

B, T = 2, 16
STEP = B * T  # tokens per batch for one rank
N = 1000  # tokens per synthetic shard
SHARD_OFFSET = 10_000  # shard k holds tokens k*SHARD_OFFSET + [0, N)


def write_shard(directory: Path, name: str, tokens: np.ndarray) -> None:
    tokens.astype(np.uint16).tofile(directory / name)


def make_shards(directory: Path, n_shards: int = 1, n_tokens: int = N, split: str = "train") -> None:
    # each token's value encodes (shard, position) so we can tell exactly where a batch came from
    for k in range(n_shards):
        write_shard(directory, f"data_{split}_{k:06d}.bin", k * SHARD_OFFSET + np.arange(n_tokens))


def make_loader(directory: Path, rank: int = 0, world_size: int = 1, split: str = "train") -> DataLoader:
    return DataLoader(str(directory), batch_size=B, context_length=T, rank=rank, world_size=world_size, is_master=False, split=split)


def first_token(x) -> int:
    return int(x.view(-1)[0])


@pytest.fixture
def one_shard(tmp_path):
    make_shards(tmp_path, n_shards=1)
    return tmp_path


@pytest.fixture
def loader(one_shard):
    return make_loader(one_shard)


@pytest.fixture
def limit():
    # an index is valid for the *next* read only if idx <= N - (B*T + 1),
    # since a read takes B*T + 1 tokens (inputs plus the shifted target)
    return N - (STEP + 1)


def start_for_next_idx(next_idx: int) -> int:
    """Starting position whose advanced index (start + B*T) equals next_idx."""
    return next_idx - STEP


# --- single rank, single shard ---------------------------------------------


def test_advances_by_one_step_normally(loader):
    loader.next_batch()
    assert loader.idx == STEP


def test_batch_shapes_and_target_shift(loader):
    x, y = loader.next_batch()
    assert x.shape == y.shape == (B, T)
    # targets are inputs shifted by one token, from one contiguous slice
    assert (x.view(-1)[1:] == y.view(-1)[:-1]).all()
    assert (y.view(-1) == x.view(-1) + 1).all()


def test_first_batch_is_start_of_shard(loader):
    x, y = loader.next_batch()
    assert x.view(-1).tolist() == list(range(STEP))
    assert y.view(-1).tolist() == list(range(1, STEP + 1))


def test_consecutive_batches_are_contiguous(loader):
    x1, _ = loader.next_batch()
    x2, _ = loader.next_batch()
    assert first_token(x2) == first_token(x1) + STEP


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
    assert first_token(x) == limit
    assert loader.idx == 0


def test_single_shard_wraps_to_beginning(loader, limit):
    loader.idx = limit
    loader.next_batch()
    x, _ = loader.next_batch()
    assert loader.curr_shard == 0
    assert x.view(-1).tolist() == list(range(STEP))


def test_full_epoch_never_produces_short_slice(loader):
    # walking the whole shard twice must never raise from a short .view()
    n_steps = 2 * (N // STEP) + 2
    for _ in range(n_steps):
        x, y = loader.next_batch()
        assert x.shape == y.shape == (B, T)


def test_exact_fit_shard_repeats_same_batch(tmp_path):
    # a shard of exactly B*T + 1 tokens holds one batch, returned every step
    make_shards(tmp_path, n_tokens=STEP + 1)
    loader = make_loader(tmp_path)
    for _ in range(3):
        x, _ = loader.next_batch()
        assert x.view(-1).tolist() == list(range(STEP))


# --- multiple shards ---------------------------------------------------------


def test_moves_to_next_shard_then_wraps_to_first(tmp_path):
    make_shards(tmp_path, n_shards=3)
    loader = make_loader(tmp_path)
    shards_seen = []
    for _ in range(3 * (N // STEP) + 1):
        x, _ = loader.next_batch()
        shard = first_token(x) // SHARD_OFFSET
        if not shards_seen or shards_seen[-1] != shard:
            shards_seen.append(shard)
    assert shards_seen == [0, 1, 2, 0]


def test_new_shard_starts_at_its_beginning(tmp_path, limit):
    make_shards(tmp_path, n_shards=2)
    loader = make_loader(tmp_path)
    loader.idx = limit
    loader.next_batch()
    x, _ = loader.next_batch()
    assert loader.curr_shard == 1
    assert first_token(x) == SHARD_OFFSET


def test_shards_are_read_in_sorted_order(tmp_path):
    # written out of order on disk; loader must still read 0, 1, 2
    for k in [2, 0, 1]:
        write_shard(tmp_path, f"data_train_{k:06d}.bin", k * SHARD_OFFSET + np.arange(N))
    loader = make_loader(tmp_path)
    assert loader.shards == [f"data_train_{k:06d}.bin" for k in range(3)]


def test_only_loads_requested_split(tmp_path):
    make_shards(tmp_path, n_shards=2, split="train")
    write_shard(tmp_path, "data_val_000000.bin", np.arange(N))
    (tmp_path / "notes_train.txt").write_text("not a shard")
    assert make_loader(tmp_path, split="train").shards == ["data_train_000000.bin", "data_train_000001.bin"]
    assert make_loader(tmp_path, split="val").shards == ["data_val_000000.bin"]


def test_rejects_unknown_split(one_shard):
    with pytest.raises(AssertionError):
        make_loader(one_shard, split="test")


def test_no_matching_shards_fails_at_init(tmp_path):
    make_shards(tmp_path, split="train")
    with pytest.raises(AssertionError):
        make_loader(tmp_path, split="val")


# --- multiple ranks (DDP) ---------------------------------------------------


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ranks_start_at_staggered_offsets(one_shard, world_size):
    loaders = [make_loader(one_shard, rank=r, world_size=world_size) for r in range(world_size)]
    starts = [first_token(ld.next_batch()[0]) for ld in loaders]
    assert starts == [r * STEP for r in range(world_size)]


@pytest.mark.parametrize("world_size", [2, 4])
def test_ranks_stride_over_each_other(one_shard, world_size):
    loader = make_loader(one_shard, rank=1, world_size=world_size)
    x1, _ = loader.next_batch()
    x2, _ = loader.next_batch()
    assert first_token(x2) == first_token(x1) + STEP * world_size


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ranks_never_read_same_tokens_in_a_step(one_shard, world_size):
    loaders = [make_loader(one_shard, rank=r, world_size=world_size) for r in range(world_size)]
    for _ in range(3 * (N // (STEP * world_size)) + 2):
        batches = [set(ld.next_batch()[0].view(-1).tolist()) for ld in loaders]
        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert not batches[i] & batches[j]


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ranks_wrap_to_their_own_offset(one_shard, world_size):
    rank = world_size - 1
    loader = make_loader(one_shard, rank=rank, world_size=world_size)
    for _ in range(N // STEP):
        loader.next_batch()
        if loader.idx == rank * STEP and loader.curr_shard == 0:
            break
    else:
        pytest.fail("rank never wrapped")
    x, _ = loader.next_batch()
    assert first_token(x) == rank * STEP


# --- known gaps (remove xfail once fixed) ----------------------------------


@pytest.mark.xfail(strict=True, reason="no check that a shard holds one full step for every rank")
def test_shard_smaller_than_one_step_fails_at_init(tmp_path):
    world_size = 4
    make_shards(tmp_path, n_tokens=STEP * world_size)  # one token short
    with pytest.raises(AssertionError):
        make_loader(tmp_path, rank=world_size - 1, world_size=world_size)
