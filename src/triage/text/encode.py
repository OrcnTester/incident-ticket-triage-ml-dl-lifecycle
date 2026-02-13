from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from src.triage.text.vocab import Vocab


@dataclass(frozen=True)
class EncodeStats:
    n_texts: int
    n_tokens_total: int
    n_unk_total: int
    oov_rate: float
    lengths: dict  # min/median/p95/max


def encode_tokens(
    toks: List[str],
    vocab: Vocab,
    *,
    max_len: int = 0,
    pad_to_max: bool = False,
) -> List[int]:
    ids = [vocab.encode_token(t) for t in toks]
    if max_len and max_len > 0:
        ids = ids[:max_len]
        if pad_to_max and len(ids) < max_len:
            ids = ids + [vocab.pad_id] * (max_len - len(ids))
    return ids


def summarize_lengths(lengths: List[int]) -> dict:
    if not lengths:
        return {"min": 0, "median": 0, "p95": 0, "max": 0}
    arr = np.array(lengths, dtype=np.int32)
    return {
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


def compute_encode_stats(encoded: List[List[int]], vocab: Vocab) -> EncodeStats:
    n_texts = len(encoded)
    lengths = [len(x) for x in encoded]
    n_tokens_total = int(sum(lengths))
    unk = vocab.unk_id
    n_unk_total = int(sum(sum(1 for i in seq if i == unk) for seq in encoded))
    oov_rate = (n_unk_total / n_tokens_total) if n_tokens_total else 0.0
    return EncodeStats(
        n_texts=n_texts,
        n_tokens_total=n_tokens_total,
        n_unk_total=n_unk_total,
        oov_rate=float(oov_rate),
        lengths=summarize_lengths(lengths),
    )
