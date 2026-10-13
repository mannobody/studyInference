import pytest
import torch
import torch.nn.functional as F

from ref.simple_attn import my_attention


def rand_qkv(B, H, T, dh, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, H, T, dh, generator=g) for _ in range(3)]


@pytest.mark.parametrize("T", [1, 2, 7, 128])
@pytest.mark.parametrize("B,H,dh", [(1, 1, 8), (2, 4, 16), (3, 2, 64)])
def test_matches_sdpa_causal(B, H, T, dh):
    q, k, v = rand_qkv(B, H, T, dh)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(my_attention(q, k, v), ref, atol=1e-5, rtol=1e-5)


def test_output_shape():
    q, k, v = rand_qkv(2, 3, 5, 8)
    assert my_attention(q, k, v).shape == (2, 3, 5, 8)


def test_first_token_attends_only_to_itself():
    q, k, v = rand_qkv(2, 4, 10, 16)
    out = my_attention(q, k, v)
    torch.testing.assert_close(out[:, :, 0], v[:, :, 0])


def test_causality_future_does_not_leak():
    # Меняем k/v после позиции t -> выходы на позициях <= t не меняются
    T, t = 12, 5
    q, k, v = rand_qkv(2, 4, T, 16)
    out1 = my_attention(q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, t + 1:] = torch.randn_like(k2[:, :, t + 1:])
    v2[:, :, t + 1:] = torch.randn_like(v2[:, :, t + 1:])
    out2 = my_attention(q, k2, v2)
    torch.testing.assert_close(out1[:, :, : t + 1], out2[:, :, : t + 1])
    assert not torch.allclose(out1[:, :, t + 1:], out2[:, :, t + 1:])


def test_constant_values_give_same_output():
    # Веса внимания суммируются в 1 -> если все v одинаковые, выход = v
    q, k, _ = rand_qkv(1, 2, 6, 8)
    v = torch.ones(1, 2, 6, 8) * 3.0
    torch.testing.assert_close(my_attention(q, k, v), v)


def test_heads_independent():
    q, k, v = rand_qkv(1, 4, 7, 8)
    full = my_attention(q, k, v)
    for h in range(4):
        one = my_attention(q[:, h:h + 1], k[:, h:h + 1], v[:, h:h + 1])
        torch.testing.assert_close(full[:, h:h + 1], one)
