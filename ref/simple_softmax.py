import torch

def naive_softmax(p : torch.Tensor, dim : int = -1) ->  torch.Tensor:
    e = torch.exp(p)
    return e / e.sum(dim=dim,keepdim=True)
def stable_softmax(p : torch.Tensor, dim : int = -1) -> torch.Tensor:
    _max = p.max(dim=dim,keepdim=True).values
    e = torch.exp(p-_max)
    return e / e.sum(dim=dim,keepdim=True)
