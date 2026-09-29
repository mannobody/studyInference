import torch
import math
import torch.nn.functional as F

def my_attention(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor) -> torch.Tensor:
    B,H,T,dh = q.shape
    d = H * dh
    scores = q @ k.transpose(-2,-1) / math.sqrt(dh)
    mask = torch.triu(torch.ones(T,T,dtype=torch.bool),diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    p = torch.softmax(scores,dim=-1)
    out = p @ v
    return out

torch.manual_seed(0)

for T in [1, 7, 128]:
    B, H, dh = 2, 4, 16
    q = torch.randn(B, H, T, dh)
    k = torch.randn(B, H, T, dh)
    v = torch.randn(B, H, T, dh)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = my_attention(q, k, v)

    max_diff = (ref - out).abs().max().item()
    ok = torch.allclose(ref, out, atol=1e-5)
    print(f"T={T:4d}  max|diff|={max_diff:.2e}  {'OK' if ok else 'FAIL'}")