import torch
import math
import torch.nn.functional as F

def naive_softmax(p : torch.Tensor, dim : int = -1) ->  torch.Tensor:
    e = torch.exp(p)
    return e / e.sum(dim=dim,keepdim=True)
def stable_softmax(p : torch.Tensor, dim : int = -1) -> torch.Tensor:
    _max = p.max(dim=dim,keepdim=True).values
    e = torch.exp(p-_max)
    return e / e.sum(dim=dim,keepdim=True)

tensors = [torch.randn(4,5) for _ in range(5)]
for x in tensors:
    ref = torch.softmax(x, dim=-1)
    out = stable_softmax(x)
    max_diff = (ref - out).abs().max().item()
    ok = torch.allclose(ref, out, atol=1e-5)
    print(f"max|diff|={max_diff:.2e}  {'OK' if ok else 'FAIL'}")