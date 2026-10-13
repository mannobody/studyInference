import pytest
import torch

from scripts.chat import StopStrings, StreamDecoder, build_prompt, SYSTEM
from inference.generate import generate, generate_stream
from inference.simple_sampling import repetition_penalty
from test_generate import tiny_model
from inference.tokenizer import GPT2Tokenizer


@pytest.fixture(scope="module")
def tok():
    return GPT2Tokenizer.from_dir("weights/gpt2")


# ---------- generate_stream ----------

def test_stream_matches_generate_greedy():
    w, cfg = tiny_model(n_pos=8)
    ids = torch.tensor([[1, 2, 3]])
    full = generate(ids, w, cfg, 10)
    assert list(generate_stream(ids, w, cfg, 10)) == full[0, 3:].tolist()


def test_stream_stops_on_eos():
    w, cfg = tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    toks = list(generate_stream(ids, w, cfg, 10))
    eos = toks[2]
    stopped = list(generate_stream(ids, w, cfg, 10, eos_id=eos))
    assert stopped == toks[:toks.index(eos)]


def test_repetition_penalty():
    logits = torch.tensor([[2.0, -2.0, 1.0, 0.5]])
    out = repetition_penalty(logits, torch.tensor([[0, 1, 0]]), 2.0)
    assert torch.equal(out, torch.tensor([[1.0, -4.0, 1.0, 0.5]]))
    assert torch.equal(repetition_penalty(logits, torch.tensor([[0]]), 1.0), logits)


# ---------- stop strings ----------

def feed(pieces, stops=("\nUser:",)):
    s = StopStrings(stops)
    out = "".join(s.push(p) for p in pieces)
    if not s.stopped:
        out += s.finish()
    return out, s


def test_stop_string_cut():
    out, s = feed([" Hi", " there", "\n", "User", ":", " more"])
    assert out == " Hi there" and s.stopped and s.text == " Hi there"


def test_partial_stop_is_held_then_released():
    s = StopStrings(("\nUser:",))
    assert s.push("ok\nUs") == "ok"       # "\nUs" may turn out to be a stop
    assert s.push("ually") == "\nUsually"  # it was not: emit it
    assert not s.stopped


def test_no_stop_passthrough():
    out, s = feed(["a", "b", "\n"])
    assert out == "ab\n" and not s.stopped


# ---------- streaming decode ----------

@pytest.mark.parametrize("text", ["Привет, мир!", "emoji 😀🚀 ok", "plain text"])
def test_stream_decode_equals_full_decode(tok, text):
    ids = tok.encode(text)
    d = StreamDecoder(tok)
    pieces = [d.push(i) for i in ids] + [d.flush()]
    assert "".join(pieces) == tok.decode(ids)
    assert all("�" not in p for p in pieces)


# ---------- prompt building ----------

def test_prompt_format(tok):
    ids = build_prompt(tok, [("hi", "hello")], "how are you?", 1000)
    assert tok.decode(ids) == SYSTEM + "User: hi\nAssistant: hello\nUser: how are you?\nAssistant:"


def test_prompt_drops_oldest_turns_keeps_system(tok):
    history = [(f"question {i} " * 20, f"answer {i} " * 20) for i in range(30)]
    limit = 300
    ids = build_prompt(tok, history, "last", limit)
    text = tok.decode(ids)
    assert len(ids) <= limit
    assert text.startswith(SYSTEM)
    assert text.endswith("User: last\nAssistant:")
    assert "question 29" in text and "question 0 " not in text
