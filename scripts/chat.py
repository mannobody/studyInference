"""Console chat on top of our own GPT-2 forward.

GPT-2 is a base model (not instruct-tuned), so a dialogue is just text continuation
following a "User: ... / Assistant: ..." template. Replies will be mediocre; the point is the mechanics.

Run from the repo root:
    python -m scripts.chat
    python -m scripts.chat --temperature 0.8 --topk 50 --topp 0.95 --rep-penalty 1.2 --seed 0
Chat commands: /reset - clear history, /exit - quit.
"""
import argparse
import codecs
import time

import torch
from safetensors.torch import load_file

from inference.generate import generate_stream
from inference.gpt2 import GPT2Config
from inference.tokenizer import GPT2Tokenizer

SYSTEM = "The following is a conversation between a User and a helpful AI Assistant.\n\n"
STOP_STRINGS = ("\nUser:", "\nAssistant:")


def format_turn(role: str, text: str) -> str:
    return f"{role}: {text}\n"


def build_prompt(tok: GPT2Tokenizer, history: list[tuple[str, str]], user_msg: str,
                 max_prompt_tokens: int, system: str = SYSTEM) -> list[int]:
    # The 1024 window = prompt + reply. Slicing ids[:, -1024:] inside generate would cut
    # off the system line, so we drop the oldest turns as whole units instead.
    sys_ids = tok.encode(system)
    tail_ids = tok.encode(format_turn("User", user_msg) + "Assistant:")
    turns = [tok.encode(format_turn("User", u) + format_turn("Assistant", a)) for u, a in history]
    budget = max_prompt_tokens - len(sys_ids) - len(tail_ids)
    if budget < 0:
        # Even the single message does not fit: keep its tail
        return (sys_ids + tail_ids)[-max_prompt_tokens:]
    kept = []
    for t in reversed(turns):
        if len(t) > budget:
            break
        kept.append(t)
        budget -= len(t)
    return sys_ids + [i for t in reversed(kept) for i in t] + tail_ids


class StreamDecoder:
    """Tokens -> text piece by piece. One token can be half of a UTF-8 character
    (Cyrillic, emoji), so bytes are decoded incrementally: an incomplete
    character waits for the next token instead of being printed as '\ufffd'."""

    def __init__(self, tok: GPT2Tokenizer):
        self.tok = tok
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id: int) -> str:
        return self.dec.decode(self.tok.decode_bytes([token_id]))

    def flush(self) -> str:
        return self.dec.decode(b"", final=True)


class StopStrings:
    """Accumulates reply text and stops generation on a stop string.
    A tail that may turn out to be the start of a stop string ("\\nUs") is held
    back so it is not printed too early."""

    def __init__(self, stops=STOP_STRINGS):
        self.stops = stops
        self.text = ""      # the whole reply
        self.emitted = 0    # number of chars already emitted for printing
        self.stopped = False

    def push(self, piece: str) -> str:
        if self.stopped:
            return ""
        self.text += piece
        cut = min((self.text.find(s) for s in self.stops if s in self.text), default=-1)
        if cut >= 0:
            self.stopped = True
            self.text = self.text[:cut]
            return self._emit(cut)
        hold = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.text)), 0, -1):
                if self.text.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        return self._emit(len(self.text) - hold)

    def finish(self) -> str:
        return self._emit(len(self.text))

    def _emit(self, upto: int) -> str:
        upto = max(upto, self.emitted)
        out = self.text[self.emitted:upto]
        self.emitted = upto
        return out


def reply(prompt_ids, weights, cfg, tok, args, out=print):
    """Generate one reply with streaming output. Returns (text, metrics)."""
    ids = torch.tensor([prompt_ids])
    dec, stop = StreamDecoder(tok), StopStrings()
    t0 = time.perf_counter()
    times = []
    for token_id in generate_stream(ids, weights, cfg, args.max_new, args.temperature,
                                    args.topk, args.topp, args.rep_penalty, tok.eot_id):
        times.append(time.perf_counter())
        out(stop.push(dec.push(token_id)))
        if stop.stopped:
            break
    if not stop.stopped:
        stop.push(dec.flush())
        out(stop.finish())
    ttft = times[0] - t0 if times else 0.
    tpot = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 0.
    return stop.text.strip(), {"prompt": len(prompt_ids), "new": len(times),
                               "ttft": ttft, "tpot": tpot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="weights/gpt2")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--topp", type=float, default=0.95)
    ap.add_argument("--rep-penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=torch.get_num_threads())
    ap.add_argument("--no-stats", action="store_true")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cfg = GPT2Config()
    weights = load_file(f"{args.weights_dir}/model.safetensors")
    tok = GPT2Tokenizer.from_dir(args.weights_dir)
    max_prompt = cfg.n_positions - args.max_new
    history: list[tuple[str, str]] = []
    print("GPT-2 chat. /reset - clear history, /exit - quit.")

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history.clear()
            print("(history cleared)")
            continue

        prompt_ids = build_prompt(tok, history, user, max_prompt)
        print("Bot:", end="", flush=True)
        try:
            text, m = reply(prompt_ids, weights, cfg, tok, args,
                            out=lambda s: print(s, end="", flush=True))
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        print()
        history.append((user, text))
        if not args.no_stats and m["new"]:
            tps = 1 / m["tpot"] if m["tpot"] else float("inf")
            print(f"  [prompt {m['prompt']} tok, new {m['new']} tok, "
                  f"TTFT {m['ttft'] * 1e3:.0f} ms, TPOT {m['tpot'] * 1e3:.0f} ms ({tps:.1f} tok/s)]")


if __name__ == "__main__":
    main()
