import pytest
import torch
from safetensors.torch import load_file

from ref.gpt2 import GPT2Config
from ref.generate import generate
from test_gpt2 import ref_forward


def tiny_model(V=50, C=16, H=2, L=2, n_pos=8, seed=0):
    # Маленькая случайная GPT-2: быстро и без файла весов
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.5
    w = {"wte.weight": r(V, C), "wpe.weight": r(n_pos, C),
         "ln_f.weight": torch.ones(C), "ln_f.bias": torch.zeros(C)}
    for i in range(L):
        p = f"h.{i}."
        w |= {p + "ln_1.weight": torch.ones(C), p + "ln_1.bias": torch.zeros(C),
              p + "ln_2.weight": torch.ones(C), p + "ln_2.bias": torch.zeros(C),
              p + "attn.c_attn.weight": r(C, 3 * C), p + "attn.c_attn.bias": r(3 * C),
              p + "attn.c_proj.weight": r(C, C), p + "attn.c_proj.bias": r(C),
              p + "mlp.c_fc.weight": r(C, 4 * C), p + "mlp.c_fc.bias": r(4 * C),
              p + "mlp.c_proj.weight": r(4 * C, C), p + "mlp.c_proj.bias": r(C)}
    cfg = GPT2Config(n_layer=L, n_head=H, d_model=C, n_positions=n_pos)
    return w, cfg


def ref_greedy(ids, w, n, n_layer, n_head, n_pos):
    for _ in range(n):
        logits = ref_forward(ids[:, -n_pos:], w, n_layer, n_head)
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return ids


# ---------- маленькая случайная модель ----------

def test_shape_and_prompt_kept():
    w, cfg = tiny_model()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = generate(ids, w, cfg, 4)
    assert out.shape == (2, 7)
    assert out.dtype == torch.long
    assert torch.equal(out[:, :3], ids)


def test_zero_new_tokens():
    w, cfg = tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    assert torch.equal(generate(ids, w, cfg, 0), ids)


def test_greedy_matches_reference_tiny():
    w, cfg = tiny_model()
    ids = torch.tensor([[7, 3, 9]])
    out = generate(ids, w, cfg, 4)
    assert torch.equal(out, ref_greedy(ids, w, 4, cfg.n_layer, cfg.n_head, cfg.n_positions))


def test_context_is_cropped():
    # n_positions = 8: prompt 6 + 10 новых > 8, без обрезки wpe упадёт с IndexError
    w, cfg = tiny_model(n_pos=8)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    out = generate(ids, w, cfg, 10)
    assert out.shape == (1, 16)
    assert torch.equal(out, ref_greedy(ids, w, 10, cfg.n_layer, cfg.n_head, cfg.n_positions))


def test_sampling_reproducible_with_seed():
    w, cfg = tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    torch.manual_seed(123)
    a = generate(ids, w, cfg, 5, temperature=1.0, topk=10)
    torch.manual_seed(123)
    b = generate(ids, w, cfg, 5, temperature=1.0, topk=10)
    assert torch.equal(a, b)
    assert ((a >= 0) & (a < 50)).all()


def test_batch_rows_independent_greedy():
    w, cfg = tiny_model()
    a, b = torch.tensor([[1, 2, 3]]), torch.tensor([[9, 8, 7]])
    both = generate(torch.cat([a, b]), w, cfg, 4)
    assert torch.equal(both[0:1], generate(a, w, cfg, 4))
    assert torch.equal(both[1:2], generate(b, w, cfg, 4))


# ---------- настоящая GPT-2 ----------

@pytest.fixture(scope="module")
def weights():
    return load_file("weights/gpt2/model.safetensors")


def test_greedy_matches_reference_gpt2(weights):
    ids = torch.tensor([[15496, 11, 616, 1438, 318]])
    out = generate(ids, weights, GPT2Config(), 5)
    assert torch.equal(out, ref_greedy(ids, weights, 5, 12, 12, 1024))
