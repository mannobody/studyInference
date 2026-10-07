import torch
from gpt2 import forward, GPT2Config
from simple_sampling import sample


@torch.no_grad()
def generate(ids: torch.Tensor, weights: dict[str, torch.Tensor],
             cfg: GPT2Config, max_new_tokens: int, temperature: float = 0., 
             topk: int = -1, topp: float = 1.) -> torch.Tensor:
    for i in range(max_new_tokens):
        logits = forward(ids[:, -cfg.n_positions:], weights, cfg)
        next_logits = logits[:, -1, :]
        next_id = sample(next_logits, temperature, topk, topp)
        ids = torch.cat([ids,next_id[:, None]], dim=-1)
    return ids