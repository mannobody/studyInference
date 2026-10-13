# studyInference

LLM inference study: a GPT-2 inference engine built from scratch, from a PyTorch reference
forward to a C++/CUDA engine with KV cache, batching and speculative decoding.

## Layout

```
ref/        PyTorch reference implementation (importable package)
  gpt2.py             GPT-2 forward from raw safetensors weights
  generate.py         autoregressive generation, streaming
  tokenizer.py        byte-level BPE tokenizer
  chat.py             console chat
  simple_attn.py      causal attention
  simple_softmax.py   naive and stable softmax
  simple_sampling.py  temperature, top-k, top-p, repetition penalty
bench/      benchmarks (run as modules)
  prefill.py          prefill time over prompt length (empirical roofline)
  generate.py         TTFT / TPOT of generation
  batch_decode.py     batched decode (B,1) vs prefill (1,B)
tests/      pytest tests
weights/    GPT-2 weights and tokenizer files (not in git)
```

## Run

All commands run from the repo root.

```
python -m pytest -q
python -m ref.chat
python -m bench.prefill --threads 6
python -m bench.generate --prompt-len 10 --new 200 --threads 6
python -m bench.batch_decode --threads 6
```

Weights are expected in `weights/gpt2/` (`model.safetensors`, `vocab.json`, `merges.txt`).
