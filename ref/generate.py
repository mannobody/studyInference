from collections.abc import Iterator

import torch
from ref.gpt2 import forward, GPT2Config
from ref.simple_sampling import sample, repetition_penalty


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


@torch.no_grad()
def generate_stream(ids: torch.Tensor, weights: dict[str, torch.Tensor],
                    cfg: GPT2Config, max_new_tokens: int, temperature: float = 0.,
                    topk: int = -1, topp: float = 1., rep_penalty: float = 1.,
                    eos_id: int | None = None) -> Iterator[int]:
    # Batch 1: yield tokens one by one as soon as they are computed (for streaming).
    # EOS is not yielded: we just stop on it.
    assert ids.shape[0] == 1
    for _ in range(max_new_tokens):
        ctx = ids[:, -cfg.n_positions:]
        next_logits = forward(ctx, weights, cfg)[:, -1, :]
        next_logits = repetition_penalty(next_logits, ctx, rep_penalty)
        next_id = sample(next_logits, temperature, topk, topp)
        tok = next_id.item()
        if tok == eos_id:
            return
        yield tok
        ids = torch.cat([ids, next_id[:, None]], dim=-1)
