"""Byte-level BPE tokenizer for GPT-2 with no external dependencies (except regex).

decode: id -> string from vocab.json -> bytes via the byte<->unicode table -> utf-8.
encode: a regex splits text into "words" -> word bytes to symbols ->
        greedily merge the pair with the lowest rank from merges.txt -> id.
"""
import json
from functools import lru_cache

import regex

EOT = "<|endoftext|>"
EOT_ID = 50256

# Same regex as in the original GPT-2 encoder.py: the space is attached to the word
PAT = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def bytes_to_unicode() -> dict[int, str]:
    # Each of the 256 bytes -> a printable char. Printable bytes map to themselves,
    # the rest (space, \n, control chars) move to 256+n. Space -> 'Ġ'.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class GPT2Tokenizer:
    def __init__(self, vocab_path: str, merges_path: str):
        with open(vocab_path, encoding="utf-8") as f:
            self.encoder: dict[str, int] = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_path, encoding="utf-8") as f:
            lines = f.read().split("\n")[1:]  # first line is the version header
        self.ranks = {tuple(l.split()): i for i, l in enumerate(lines) if l}
        self.byte_enc = bytes_to_unicode()
        self.byte_dec = {v: k for k, v in self.byte_enc.items()}
        self.eot_id = self.encoder[EOT]
        self._bpe = lru_cache(maxsize=1 << 16)(self._bpe_uncached)

    @classmethod
    def from_dir(cls, path: str) -> "GPT2Tokenizer":
        return cls(f"{path}/vocab.json", f"{path}/merges.txt")

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def _bpe_uncached(self, word: str) -> tuple[str, ...]:
        parts = list(word)
        while len(parts) > 1:
            pairs = zip(parts, parts[1:])
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            # Merge all occurrences of best in a single pass
            merged, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == best:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return tuple(parts)

    def _encode_ordinary(self, text: str) -> list[int]:
        ids = []
        for word in PAT.findall(text):
            mapped = "".join(self.byte_enc[b] for b in word.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self._bpe(mapped))
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        # BPE never produces <|endoftext|>: split on it and insert its id directly
        if not allow_special:
            return self._encode_ordinary(text)
        ids = []
        for k, chunk in enumerate(text.split(EOT)):
            if k > 0:
                ids.append(self.eot_id)
            ids.extend(self._encode_ordinary(chunk))
        return ids

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(self.byte_dec[c] for i in ids for c in self.decoder[i])

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")
