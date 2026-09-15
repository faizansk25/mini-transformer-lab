"""A small, deterministic byte-pair encoding tokenizer."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Pair = Tuple[int, int]


class BPETokenizer:
    """Educational byte-level BPE with serializable merge rules."""

    def __init__(self, merges: Sequence[Pair] = ()) -> None:
        self.merges = [tuple(pair) for pair in merges]
        self.vocab: Dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        for index, (left, right) in enumerate(self.merges, start=256):
            if left not in self.vocab or right not in self.vocab:
                raise ValueError("merge rules must reference existing tokens")
            self.vocab[index] = self.vocab[left] + self.vocab[right]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, texts: Iterable[str], target_vocab_size: int = 512) -> "BPETokenizer":
        if target_vocab_size < 256:
            raise ValueError("target_vocab_size cannot be below 256")
        sequences = [list(text.encode("utf-8")) for text in texts]
        merges: List[Pair] = []
        while 256 + len(merges) < target_vocab_size:
            counts = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
            if not counts:
                break
            best_pair, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if count < 2:
                break
            new_id = 256 + len(merges)
            sequences = [cls._merge_sequence(seq, best_pair, new_id) for seq in sequences]
            merges.append(best_pair)
        return cls(merges)

    @staticmethod
    def _merge_sequence(tokens: Sequence[int], pair: Pair, new_id: int) -> List[int]:
        merged: List[int] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
                merged.append(new_id)
                index += 2
            else:
                merged.append(tokens[index])
                index += 1
        return merged

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode("utf-8"))
        for new_id, pair in enumerate(self.merges, start=256):
            tokens = self._merge_sequence(tokens, pair, new_id)
        return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        try:
            payload = b"".join(self.vocab[token] for token in tokens)
        except KeyError as exc:
            raise ValueError(f"unknown token id: {exc.args[0]}") from exc
        return payload.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(pair) for pair in payload["merges"]])
