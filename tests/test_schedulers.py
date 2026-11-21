import pytest

from src.scheduler import get_cosine_annealing_lr

W, C, MAX, MIN = 10, 100, 1e-3, 1e-5


def lr(step):
    return get_cosine_annealing_lr(step, W, C, MAX, MIN)


def test_first_step_nonzero():
    assert lr(0) == pytest.approx(MAX / W)


def test_warmup_is_linear():
    for s in range(W):
        assert lr(s) == pytest.approx(MAX * (s + 1) / W)


def test_warmup_reaches_max():
    assert lr(W - 1) == pytest.approx(MAX)


def test_cosine_starts_at_max():
    assert lr(W) == pytest.approx(MAX)


def test_cosine_midpoint():
    mid = W + (C - W) // 2  # coeff = 0.5 -> cos(pi/2) = 0
    assert lr(mid) == pytest.approx((MAX + MIN) / 2)


def test_ends_at_min():
    assert lr(C) == pytest.approx(MIN)


def test_stays_at_min_after_decay():
    for s in [C, C + 1, C + 1000]:
        assert lr(s) == pytest.approx(MIN)


def test_min_equals_max_is_constant_after_warmup():
    for s in range(W, C + 10):
        assert get_cosine_annealing_lr(s, W, C, MAX, MAX) == pytest.approx(MAX)


@pytest.mark.parametrize("w,c", [(10, 10), (20, 10)])
def test_invalid_args(w, c):
    with pytest.raises(AssertionError):
        get_cosine_annealing_lr(0, w, c, MAX, MIN)