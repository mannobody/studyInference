import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from ref.gpt2 import forward, embed, layer_norm, mlp, GELU, GPT2Config

@pytest.fixture(scope="module")
def weights():
    return load_file("weights/gpt2/model.safetensors")

def test_embeddings_shape(weights):
    ids = torch.tensor([[15496, 995]])           # "Hello world"
    x = embed(ids, weights)
    assert x.shape == (1, 2, 768)

def test_embeddings_values(weights):
    ids = torch.tensor([[15496, 995]])
    x = embed(ids, weights)
    wte, wpe = weights["wte.weight"], weights["wpe.weight"]
    torch.testing.assert_close(x[0, 1], wte[995] + wpe[1])   # позиция 1, не 0

def test_embeddings_batch(weights):
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    x = embed(ids, weights)
    assert x.shape == (2, 3, 768)
    torch.testing.assert_close(x[1, 0], weights["wte.weight"][4] + weights["wpe.weight"][0])


def rand_ln(shape, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = shape[-1]
    x = torch.randn(*shape, generator=g) * scale
    w = torch.randn(C, generator=g)
    b = torch.randn(C, generator=g)
    return x, w, b


@pytest.mark.parametrize("scale", [1.0, 1e-3, 100.0])
@pytest.mark.parametrize("shape", [(1, 1, 768), (2, 5, 768), (3, 16)])
def test_layer_norm_matches_torch(shape, scale):
    x, w, b = rand_ln(shape, scale)
    ref = F.layer_norm(x, (shape[-1],), w, b, eps=1e-5)
    torch.testing.assert_close(layer_norm(x, w, b, 1e-5), ref)


def test_layer_norm_unit_affine_gives_zero_mean_unit_var():
    x, _, _ = rand_ln((2, 7, 768))
    y = layer_norm(x, torch.ones(768), torch.zeros(768), 1e-5)
    torch.testing.assert_close(y.mean(-1), torch.zeros(2, 7), atol=1e-5, rtol=0)
    torch.testing.assert_close(y.var(-1, unbiased=False), torch.ones(2, 7), atol=1e-3, rtol=0)


def test_layer_norm_constant_row_gives_bias():
    # var = 0 -> без eps было бы 0/0 = NaN
    x = torch.full((1, 1, 768), 3.0)
    _, w, b = rand_ln((768,))
    y = layer_norm(x, w, b, 1e-5)
    assert not torch.isnan(y).any()
    torch.testing.assert_close(y[0, 0], b)


def test_layer_norm_tokens_independent():
    # Нормировка по C: выход токена не зависит от других токенов
    x, w, b = rand_ln((1, 6, 768))
    full = layer_norm(x, w, b, 1e-5)
    for t in range(6):
        torch.testing.assert_close(full[:, t], layer_norm(x[:, t], w, b, 1e-5))


def test_layer_norm_real_weights(weights):
    x, _, _ = rand_ln((1, 4, 768))
    w, b = weights["h.0.ln_1.weight"], weights["h.0.ln_1.bias"]
    torch.testing.assert_close(layer_norm(x, w, b, 1e-5), F.layer_norm(x, (768,), w, b, eps=1e-5))


# ---------- GELU / MLP ----------

def test_gelu_matches_torch_tanh():
    x = torch.linspace(-5, 5, 1000)
    torch.testing.assert_close(GELU(x), F.gelu(x, approximate="tanh"))


def ref_mlp(x, w, i):
    p = f"h.{i}.mlp."
    h = F.gelu(x @ w[p + "c_fc.weight"] + w[p + "c_fc.bias"], approximate="tanh")
    return h @ w[p + "c_proj.weight"] + w[p + "c_proj.bias"]


@pytest.mark.parametrize("i", [0, 11])
def test_mlp_matches_reference(weights, i):
    x, _, _ = rand_ln((2, 5, 768))
    out = mlp(x, weights, i, GPT2Config())
    assert out.shape == (2, 5, 768)
    torch.testing.assert_close(out, ref_mlp(x, weights, i))


def test_mlp_tokens_independent(weights):
    # MLP обрабатывает каждый токен отдельно
    x, _, _ = rand_ln((1, 4, 768))
    full = mlp(x, weights, 0, GPT2Config())
    # Допуск 1e-4: matmul для 1 строки и для 4 строк суммирует в разном порядке
    for t in range(4):
        torch.testing.assert_close(full[:, t:t + 1], mlp(x[:, t:t + 1], weights, 0, GPT2Config()),
                                   atol=1e-4, rtol=1e-4)


# ---------- forward ----------

def ref_forward(ids, w, n_layer=12, n_head=12):
    # Эталон из готовых операций PyTorch
    B, T = ids.shape
    C = w["wte.weight"].shape[1]
    x = w["wte.weight"][ids] + w["wpe.weight"][:T]
    for i in range(n_layer):
        p = f"h.{i}."
        h = F.layer_norm(x, (C,), w[p + "ln_1.weight"], w[p + "ln_1.bias"], 1e-5)
        q, k, v = (h @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]).split(C, dim=-1)
        q, k, v = [t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in (q, k, v)]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, C)
        x = x + a @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
        h = F.layer_norm(x, (C,), w[p + "ln_2.weight"], w[p + "ln_2.bias"], 1e-5)
        x = x + ref_mlp(h, w, i)
    x = F.layer_norm(x, (C,), w["ln_f.weight"], w["ln_f.bias"], 1e-5)
    return x @ w["wte.weight"].T


IDS = torch.tensor([[15496, 11, 616, 1438, 318, 257, 3303, 290]])


def test_forward_shape(weights):
    assert forward(IDS, weights, GPT2Config()).shape == (1, 8, 50257)


def test_forward_single_token(weights):
    out = forward(IDS[:, :1], weights, GPT2Config())
    assert out.shape == (1, 1, 50257)
    torch.testing.assert_close(out, ref_forward(IDS[:, :1], weights), atol=1e-4, rtol=1e-4)


def test_forward_matches_reference(weights):
    out = forward(IDS, weights, GPT2Config())
    ref = ref_forward(IDS, weights)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    assert torch.equal(out[0, -1].argmax(), ref[0, -1].argmax())


def test_forward_causal(weights):
    # Замена будущих токенов не меняет logits на прошлых позициях
    ids2 = IDS.clone()
    ids2[0, 5:] = torch.tensor([100, 200, 300])
    out1 = forward(IDS, weights, GPT2Config())
    out2 = forward(ids2, weights, GPT2Config())
    torch.testing.assert_close(out1[:, :5], out2[:, :5], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(out1[:, 5:], out2[:, 5:], atol=1e-3)


def test_forward_batch_matches_single(weights):
    a = IDS[:, :4]
    b = torch.tensor([[464, 3290, 318, 922]])
    both = forward(torch.cat([a, b]), weights, GPT2Config())
    torch.testing.assert_close(both[0:1], forward(a, weights, GPT2Config()), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(both[1:2], forward(b, weights, GPT2Config()), atol=1e-4, rtol=1e-4)
