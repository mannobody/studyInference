"""Эмпирический roofline: один forward на T токенах, T = 1 … 256.

Показывает, с какого T forward перестаёт упираться в память
и начинает упираться в вычисления.

Запуск из корня репозитория (VS Code лучше закрыть, чтобы не было swap):
    python bench_prefill.py --threads 6
"""
import argparse
import statistics
import time

import torch
from safetensors.torch import load_file

from gpt2 import GPT2Config, forward

def bench(fn, warmup: int = 3, iters: int = 5) -> float:
    """Медиана времени fn() в секундах."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def count_params(weights: dict[str, torch.Tensor]) -> tuple[int, int]:
    """(все параметры, параметры в матмулах).

    Матмулы: 2-D веса блоков + wte как LM head (tied). wpe и LayerNorm
    в матмулах не участвуют, но читаются из памяти.
    """
    n_total = sum(w.numel() for w in weights.values())
    n_matmul = sum(w.numel() for k, w in weights.items() if k.startswith("h.") and w.dim() == 2)
    n_matmul += weights["wte.weight"].numel()
    return n_total, n_matmul


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--weights", default="weights/gpt2/model.safetensors")
    # Пики ноутбука, замерены 2026-10-04 (Ryzen 5 5500U, 6 потоков)
    ap.add_argument("--peak-gflops", type=float, default=150.0, help="fp32 matmul, GFLOP/s")
    ap.add_argument("--peak-gbs", type=float, default=31.0, help="чтение памяти, GB/s")
    args = ap.parse_args()
    peak_flops, peak_bw = args.peak_gflops * 1e9, args.peak_gbs * 1e9

    torch.set_num_threads(args.threads)
    cfg = GPT2Config()
    weights = load_file(args.weights)
    vocab = weights["wte.weight"].shape[0]
    n_total, n_matmul = count_params(weights)
    weight_bytes = 4 * n_total  # fp32

    print(f"threads={args.threads}  N={n_total / 1e6:.1f}M (в матмулах {n_matmul / 1e6:.1f}M)  "
          f"веса {weight_bytes / 1e6:.0f} MB")
    print(f"ridge = {peak_flops / 1e9:.0f} GFLOP/s / {peak_bw / 1e9:.0f} GB/s "
          f"= {peak_flops / peak_bw:.1f} FLOP/байт  →  T ≈ {2 * peak_flops / peak_bw:.0f} для fp32")
    print()
    print(f"{'T':>4} {'time,ms':>9} {'GFLOP/s':>8} {'GB/s':>6} {'ms/tok':>7} "
          f"{'I,FLOP/B':>9} {'пол,ms':>7} {'×пола':>6}")

    with torch.inference_mode():
        for T in args.tokens:
            ids = torch.randint(0, vocab, (1, T))
            t = bench(lambda: forward(ids, weights, cfg), args.warmup, args.iters)

            # 2·N·T на матмулы + attention: QKᵀ и P·V, 2·2·T²·d на слой
            flops = 2 * n_matmul * T + 4 * cfg.n_layer * T * T * cfg.d_model
            intensity = flops / weight_bytes
            floor = max(flops / peak_flops, weight_bytes / peak_bw)

            print(f"{T:4d} {t * 1e3:9.1f} {flops / t / 1e9:8.1f} {weight_bytes / t / 1e9:6.1f} "
                  f"{t * 1e3 / T:7.2f} {intensity:9.1f} {floor * 1e3:7.1f} {t / floor:6.1f}")

    print("\nGB/s — «эффективная» пропускная: веса fp32 читаются один раз за forward.")
    print("пол = max(FLOPs / пик, байты / пропускная) — нижняя граница по roofline.")


if __name__ == "__main__":
    main()
