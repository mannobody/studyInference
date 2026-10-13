import torch
import math

def my_attention(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor) -> torch.Tensor:
    B,H,T,dh = q.shape
    scores = q @ k.transpose(-2,-1) / math.sqrt(dh)
    mask = torch.triu(torch.ones(T,T,dtype=torch.bool),diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    p = torch.softmax(scores,dim=-1)
    out = p @ v
    return out
