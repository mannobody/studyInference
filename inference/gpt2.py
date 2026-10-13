from dataclasses import dataclass
import torch
import math

from inference.simple_attn import my_attention

@dataclass
class GPT2Config:
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    n_positions: int = 1024
    eps: float = 1e-5

def embed(ids: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    wpe = weights["wpe.weight"]
    wte = weights["wte.weight"]
    B, T = ids.shape
    x = wte[ids] + wpe[torch.arange(T)]
    return x

def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    meanx = torch.mean(x,dim=-1,keepdim=True)
    varx = torch.var(x,dim=-1,keepdim=True, unbiased=False)
    y = (x-meanx) / torch.sqrt(varx + eps) * weight + bias
    return y

def attention(x : torch.Tensor, weights: dict[str, torch.Tensor], i: int, cfg: GPT2Config) -> torch.Tensor:
    B,T,C = x.shape
    H = cfg.n_head
    dh = C // H
    c_attn = weights[f"h.{i}.attn.c_attn.weight"]
    c_attn_bias = weights[f"h.{i}.attn.c_attn.bias"]
    qkv = x @ c_attn + c_attn_bias
    q,k,v = qkv.split(C,dim=-1)
    q = q.view(B,T,H,dh).transpose(1,2)
    k = k.view(B,T,H,dh).transpose(1,2)
    v = v.view(B,T,H,dh).transpose(1,2)
    
    y = my_attention(q,k,v)

    y = y.transpose(1,2).contiguous().view(B,T,C)

    c_proj = weights[f"h.{i}.attn.c_proj.weight"]   
    c_proj_bias = weights[f"h.{i}.attn.c_proj.bias"]
    y = y @ c_proj + c_proj_bias

    return y

def GELU(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def mlp(x : torch.Tensor, weights: dict[str, torch.Tensor], i: int, cfg: GPT2Config) -> torch.Tensor:
    c_fc = weights[f"h.{i}.mlp.c_fc.weight"]
    c_fc_bias = weights[f"h.{i}.mlp.c_fc.bias"]
    c_proj = weights[f"h.{i}.mlp.c_proj.weight"]
    c_proj_bias = weights[f"h.{i}.mlp.c_proj.bias"]
    x = x @ c_fc + c_fc_bias
    x = GELU(x)
    x = x @ c_proj + c_proj_bias
    return x

def forward(ids: torch.Tensor, weights: dict[str, torch.Tensor], cfg: GPT2Config) -> torch.Tensor:
    x = embed(ids, weights)
    for i in range(cfg.n_layer):
        ln_1_weight = weights[f"h.{i}.ln_1.weight"]
        ln_1_bias = weights[f"h.{i}.ln_1.bias"]
        ln_2_weight = weights[f"h.{i}.ln_2.weight"]
        ln_2_bias = weights[f"h.{i}.ln_2.bias"]


        # ln_1
        ln_1 = layer_norm(x, ln_1_weight, ln_1_bias, cfg.eps)

        # attention
        attention_out = attention(ln_1, weights, i, cfg)
        x = x + attention_out

        # ln_2
        ln_2 = layer_norm(x, ln_2_weight, ln_2_bias, cfg.eps)

        # mlp
        mlp_out = mlp(ln_2, weights, i, cfg)
        x = x + mlp_out

    # ln_f
    ln_f_weight = weights["ln_f.weight"]
    ln_f_bias = weights["ln_f.bias"]
    x = layer_norm(x, ln_f_weight, ln_f_bias, cfg.eps)

    
    wte = weights["wte.weight"]
    logits = x @ wte.T


    return logits
