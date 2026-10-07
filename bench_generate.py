"""Замер генерации без KV cache: TTFT, TPOT и как TPOT растёт с длиной.

Запуск из корня репозитория:
    python bench_generate.py --prompt-len 10 --new 200 --threads 6
"""
import argparse
import time

import torch
from safetensors.torch import load_file

from gpt2 import GPT2Config, forward
from simple_sampling import sample


@torch.no_grad()
def timed_generate(ids, weights, cfg, max_new_tokens):
    # Тот же цикл, что в generate, но с замером каждого шага
    times = []
    for _ in range(max_new_tokens):
        t0 = time.perf_counter()
        logits = forward(ids[:, -cfg.n_positions:], weights, cfg)
        next_id = sample(logits[:, -1, :], temperature=0.)
        ids = torch.cat([ids, next_id[:, None]], dim=1)
        times.append(time.perf_counter() - t0)
    return ids, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=10)
    ap.add_argument("--new", type=int, default=200)
    ap.add_argument("--threads", type=int, default=torch.get_num_threads())
    ap.add_argument("--weights", default="weights/gpt2/model.safetensors")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = GPT2Config()
    weights = load_file(args.weights)
    ids = torch.randint(0, weights["wte.weight"].shape[0], (1, args.prompt_len))

    # Прогрев: первые вызовы медленнее (аллокации, подгрузка страниц весов)
    timed_generate(ids, weights, cfg, 2)

    _, times = timed_generate(ids, weights, cfg, args.new)
    ttft = times[0]
    decode = times[1:]

    print(f"threads={args.threads}  prompt={args.prompt_len}  new={args.new}")
    print(f"TTFT (prefill {args.prompt_len} ток. + sample): {ttft * 1e3:8.1f} ms")
    if decode:
        avg = sum(decode) / len(decode)
        print(f"TPOT средний:                        {avg * 1e3:8.1f} ms  ({1 / avg:.1f} tok/s)")
        print(f"Всего: {sum(times):.2f} s на {args.new} токенов")
        print("\n  T (вход forward)   время шага, ms")
        n = len(times)
        for k in sorted({0, 1, n // 4, n // 2, 3 * n // 4, n - 1}):
            print(f"  {args.prompt_len + k:>16}   {times[k] * 1e3:10.1f}")


if __name__ == "__main__":
    main()
