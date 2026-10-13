import pytest

from inference.tokenizer import GPT2Tokenizer, EOT_ID


@pytest.fixture(scope="module")
def tok():
    return GPT2Tokenizer.from_dir("weights/gpt2")


@pytest.mark.parametrize("text, ids", [
    ("The meaning of life is", [464, 3616, 286, 1204, 318]),
    ("Hello, my name is", [15496, 11, 616, 1438, 318]),
    ("<|endoftext|>", [EOT_ID]),
])
def test_known_ids(tok, text, ids):
    assert tok.encode(text) == ids


def test_special_token_inside_text(tok):
    ids = tok.encode("Hi<|endoftext|>Bye")
    assert EOT_ID in ids
    assert tok.decode(ids) == "Hi<|endoftext|>Bye"
    assert EOT_ID not in tok.encode("Hi<|endoftext|>Bye", allow_special=False)


@pytest.mark.parametrize("text", [
    "", " ", "Hello world", "  spaces   and\n\n\ttabs \n", "don't I'll we've 3.14",
    "Привет, мир!", "😀🚀 naïve café", "中文字符",
])
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_space_is_part_of_word(tok):
    # " life" is one token with the space inside (in vocab it is "Ġlife")
    assert len(tok.encode(" life")) == 1
    assert tok.decoder[tok.encode(" life")[0]] == "Ġlife"
