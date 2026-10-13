import pytest
import torch

from inference.simple_softmax import naive_softmax, stable_softmax


@pytest.mark.parametrize("shape", [(5,), (4, 5), (2, 3, 7), (1, 1)])
@pytest.mark.parametrize("dim", [-1, 0])
def test_stable_matches_torch(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape)
    torch.testing.assert_close(stable_softmax(x, dim=dim), torch.softmax(x, dim=dim))


@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_naive_matches_torch_on_small_inputs(dim):
    torch.manual_seed(0)
    x = torch.randn(3, 4, 5)
    torch.testing.assert_close(naive_softmax(x, dim=dim), torch.softmax(x, dim=dim))


def test_sums_to_one_and_nonnegative():
    torch.manual_seed(0)
    x = torch.randn(8, 100) * 10
    p = stable_softmax(x)
    assert (p >= 0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(8))


def test_shift_invariance():
    torch.manual_seed(0)
    x = torch.randn(4, 10)
    torch.testing.assert_close(stable_softmax(x), stable_softmax(x + 123.0))


def test_stable_on_large_logits_naive_overflows():
    x = torch.tensor([1000.0, 999.0, 0.0])
    p = stable_softmax(x)
    assert torch.isfinite(p).all()
    torch.testing.assert_close(p, torch.softmax(x, dim=-1))
    assert not torch.isfinite(naive_softmax(x)).all()  # exp(1000) = inf -> nan


def test_neg_inf_gets_zero_probability():
    x = torch.tensor([1.0, float("-inf"), 2.0])
    p = stable_softmax(x)
    assert p[1] == 0
    torch.testing.assert_close(p, torch.softmax(x, dim=-1))


def test_float64_preserved():
    x = torch.randn(3, 4, dtype=torch.float64)
    assert stable_softmax(x).dtype == torch.float64
