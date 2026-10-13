import torch
from inference.simple_softmax import stable_softmax

def topk_filter(x : torch.Tensor, k : int, dim : int = -1, fill_value : float = float('-inf')) -> torch.Tensor:
    if k <= 0:
        return x
    k = min(k,x.size(dim))
    topk_indices = torch.topk(x, k, dim=dim).indices
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(dim, topk_indices, True)
    return x.masked_fill(~mask,fill_value)

def topp_filter(x : torch.Tensor, p : float = 1., dim : int = -1, fill_value: float = float('-inf')) -> torch.Tensor:
    if p >= 1:
        return x
    if p <= 0:
        return topk_filter(x,1,dim=dim,fill_value=fill_value)
    sorted_logits, idx = torch.sort(x,descending=True,dim=dim)
    sorted_probs = stable_softmax(sorted_logits, dim=dim)
    cum = sorted_probs.cumsum(dim=dim)

    sorted_mask = (cum - sorted_probs) >= p
    
    sorted_logits = sorted_logits.masked_fill(sorted_mask,fill_value)
    out = torch.empty_like(sorted_logits)
    out.scatter_(dim,idx,sorted_logits)
    return out

def sample(logits : torch.Tensor, temperature : float = 1., topk : int = -1, topp : float = 1.) -> torch.Tensor:
    if temperature == 0 or topk == 1:
        return logits.argmax(dim=-1)
    logits_ = logits / temperature
    logits_ = topk_filter(logits_, topk)
    logits_ = topp_filter(logits_, topp)
    probs = stable_softmax(logits_)
    return torch.multinomial(probs, 1).squeeze(-1)
def repetition_penalty(logits : torch.Tensor, prev_ids : torch.Tensor, penalty : float = 1.) -> torch.Tensor:
    # CTRL penalty: for already seen tokens divide a positive logit by penalty,
    # multiply a negative one; either way the probability goes down
    if penalty == 1. or prev_ids.numel() == 0:
        return logits
    score = logits.gather(-1, prev_ids)
    score = torch.where(score > 0, score / penalty, score * penalty)
    return logits.scatter(-1, prev_ids, score)
