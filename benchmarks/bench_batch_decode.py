"""Batched decode vs prefill: the same B rows through the weights, two shapes.

decode  (B, 1): B requests, one new token each -> one batched decode step.
prefill (1, B): one request, B prompt tokens   -> one prefill pass.

Linear layers see a (B x d) matrix in both cases, so the step times should be
close. There is no KV cache yet, so decode here models context length S ~ 0
(no cache reads): the optimistic case.

Run from the repo root (close VS Code first to avoid swapping):
    python -m benchmarks.bench_batch_decode --threads 6
"""
import argparse

import torch
from safetensors.torch import load_file

from benchmarks.bench_prefill import bench, count_params
from inference.gpt2 import GPT2Config, forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--weights", default="weights/gpt2/model.safetensors")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = GPT2Config()
    weights = load_file(args.weights)
    vocab = weights["wte.weight"].shape[0]
    _, n_matmul = count_params(weights)

    print(f"threads={args.threads}  matmul params {n_matmul / 1e6:.1f}M")
    print("decode = forward on (B, 1); prefill = forward on (1, B)\n")
    print(f"{'B':>4} {'decode,ms':>10} {'tok/s':>8} {'GFLOP/s':>8} {'speedup':>8} "
          f"{'prefill,ms':>11} {'dec/pre':>8}")

    base_tps = None
    with torch.inference_mode():
        for B in args.batch:
            ids_dec = torch.randint(0, vocab, (B, 1))
            ids_pre = torch.randint(0, vocab, (1, B))
            t_dec = bench(lambda: forward(ids_dec, weights, cfg), args.warmup, args.iters)
            t_pre = bench(lambda: forward(ids_pre, weights, cfg), args.warmup, args.iters)

            # TPOT = t_dec: every request gets exactly one token per step
            tps = B / t_dec
            base_tps = base_tps or tps
            gflops = 2 * n_matmul * B / t_dec / 1e9

            print(f"{B:4d} {t_dec * 1e3:10.1f} {tps:8.1f} {gflops:8.1f} {tps / base_tps:7.1f}x "
                  f"{t_pre * 1e3:11.1f} {t_dec / t_pre:8.2f}")

    print("\ndecode,ms: step time = TPOT seen by each of the B requests.")
    print("tok/s: system throughput B / TPOT; speedup is relative to B = 1.")
    print("dec/pre: ~1.0 means the two shapes cost the same through the weights.")


if __name__ == "__main__":
    main()
