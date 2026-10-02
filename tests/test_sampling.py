import pytest
import torch

from simple_sampling import sample, topk_filter, topp_filter

NEG_INF = float("-inf")

# Вероятности 0.5, 0.3, 0.15, 0.05, перемешаны, чтобы проверить scatter обратно
PROBS = torch.tensor([0.15, 0.5, 0.05, 0.3])
LOGITS = PROBS.log()


def kept(x):
    return torch.isfinite(x).tolist()


# ---------- topk_filter ----------

def test_topk_keeps_k_largest():
    x = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    expected = torch.tensor([NEG_INF, 5.0, NEG_INF, 4.0, NEG_INF])
    assert torch.equal(topk_filter(x, 2), expected)


@pytest.mark.parametrize("k", [0, -1, 7, 100])
def test_topk_identity_cases(k):
    x = torch.randn(3, 7)
    assert torch.equal(topk_filter(x, k), x)


def test_topk_batched_rows_independent():
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [3.0, 2.0, 1.0]])
    expected = torch.tensor([[NEG_INF, NEG_INF, 3.0],
                             [3.0, NEG_INF, NEG_INF]])
    assert torch.equal(topk_filter(x, 1), expected)


def test_topk_random_matches_torch_topk():
    torch.manual_seed(0)
    x = torch.randn(8, 50)
    out = topk_filter(x, 5)
    assert (torch.isfinite(out).sum(-1) == 5).all()
    torch.testing.assert_close(torch.topk(out, 5).values, torch.topk(x, 5).values)


def test_topk_dim0():
    x = torch.tensor([[1.0, 9.0],
                      [5.0, 2.0],
                      [3.0, 4.0]])
    expected = torch.tensor([[NEG_INF, 9.0],
                             [5.0, NEG_INF],
                             [NEG_INF, NEG_INF]])
    assert torch.equal(topk_filter(x, 1, dim=0), expected)


def test_topk_custom_fill_value():
    x = torch.tensor([1.0, 3.0, 2.0])
    assert torch.equal(topk_filter(x, 1, fill_value=-100.0), torch.tensor([-100.0, 3.0, -100.0]))


def test_topk_does_not_modify_input():
    x = torch.randn(4, 10)
    x_copy = x.clone()
    topk_filter(x, 3)
    assert torch.equal(x, x_copy)


# ---------- topp_filter ----------

@pytest.mark.parametrize("p, expected", [
    (1.0, [True, True, True, True]),
    (0.0, [False, True, False, False]),
    (0.4, [False, True, False, False]),   # 0.5 >= 0.4
    (0.7, [False, True, False, True]),    # 0.5 < 0.7 <= 0.8
    (0.9, [True, True, False, True]),     # 0.8 < 0.9 <= 0.95
    (0.99, [True, True, True, True]),
])
def test_topp_minimal_prefix(p, expected):
    assert kept(topp_filter(LOGITS, p)) == expected


def test_topp_kept_values_unchanged():
    out = topp_filter(LOGITS, 0.9)
    keep = torch.isfinite(out)
    assert torch.equal(out[keep], LOGITS[keep])


def test_topp_batched_rows_independent():
    x = torch.stack([LOGITS, LOGITS.flip(0)])
    out = topp_filter(x, 0.7)
    assert kept(out) == [[False, True, False, True], [True, False, True, False]]


def test_topp_mass_property_random():
    # Оставленная масса >= p, а без самого слабого оставленного токена < p
    torch.manual_seed(1)
    x = torch.randn(16, 100)
    p = 0.8
    out = topp_filter(x, p)
    probs = torch.softmax(x, dim=-1)
    keep = torch.isfinite(out)
    kept_mass = (probs * keep).sum(-1)
    min_kept = probs.masked_fill(~keep, float("inf")).min(-1).values
    assert (kept_mass >= p - 1e-6).all()
    assert (kept_mass - min_kept < p).all()


@pytest.mark.parametrize("p, expected_rows", [
    (0.7, [[False, False], [True, True], [False, False], [True, True]]),
    (0.0, [[False, False], [True, True], [False, False], [False, False]]),
])
def test_topp_dim0(p, expected_rows):
    x = torch.stack([LOGITS, LOGITS], dim=1)  # [4, 2], распределение по dim=0
    assert kept(topp_filter(x, p, dim=0)) == expected_rows


def test_topp_p0_custom_fill_value():
    out = topp_filter(LOGITS, 0.0, fill_value=-100.0)
    assert out.tolist().count(-100.0) == 3


def test_topp_does_not_modify_input():
    x = torch.randn(4, 10)
    x_copy = x.clone()
    topp_filter(x, 0.5)
    assert torch.equal(x, x_copy)


# ---------- sample ----------

def test_sample_temperature0_is_argmax():
    torch.manual_seed(0)
    x = torch.randn(8, 50)
    assert torch.equal(sample(x, temperature=0), x.argmax(-1))


def test_sample_topk1_is_argmax():
    torch.manual_seed(0)
    x = torch.randn(8, 50)
    assert torch.equal(sample(x, topk=1), x.argmax(-1))


def test_sample_output_shape_and_dtype():
    out = sample(torch.randn(5, 11))
    assert out.shape == (5,)
    assert out.dtype == torch.long
    assert ((0 <= out) & (out < 11)).all()


def test_sample_1d_input():
    assert sample(torch.randn(11)).shape == ()


def test_sample_topk_only_from_topk():
    torch.manual_seed(0)
    x = torch.randn(1, 50).repeat(2000, 1)
    allowed = set(torch.topk(x[0], 3).indices.tolist())
    assert set(sample(x, topk=3).tolist()) == allowed


def test_sample_topp_only_from_nucleus():
    torch.manual_seed(0)
    x = LOGITS.repeat(2000, 1)
    assert set(sample(x, topp=0.7).tolist()) == {1, 3}


def freq(x, n, **kw):
    return torch.bincount(sample(x.repeat(n, 1), **kw), minlength=x.numel()).float() / n


def test_sample_matches_softmax_distribution():
    torch.manual_seed(0)
    torch.testing.assert_close(freq(LOGITS, 20000), PROBS, atol=0.02, rtol=0)


def test_sample_low_temperature_sharpens():
    torch.manual_seed(0)
    assert freq(LOGITS, 5000, temperature=0.1)[1] > 0.95  # (0.5/0.3)^10 ~ 165


def test_sample_high_temperature_flattens():
    torch.manual_seed(0)
    torch.testing.assert_close(freq(LOGITS, 20000, temperature=100.0),
                               torch.full((4,), 0.25), atol=0.02, rtol=0)


def test_sample_extreme_logits_no_nan():
    assert sample(torch.tensor([[1000.0, 0.0, -1000.0]]), temperature=0.5).item() == 0
