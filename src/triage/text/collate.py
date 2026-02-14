from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict, Any, Literal, Optional

import numpy as np


PadTo = Literal["batch", "fixed"]
TruncationSide = Literal["head", "tail"]


@dataclass(frozen=True)
class CollateConfig:
    pad_id: int = 0
    pad_to: PadTo = "batch"                 # "batch" -> pad to max len in this batch
    max_len: int = 0                        # used when pad_to="fixed" or when truncating
    truncation: bool = True
    truncation_side: TruncationSide = "head"


def _truncate(seq: Sequence[int], max_len: int, side: TruncationSide) -> list[int]:
    if max_len <= 0 or len(seq) <= max_len:
        return list(seq)
    if side == "head":
        return list(seq[:max_len])
    if side == "tail":
        return list(seq[-max_len:])
    raise ValueError(f"Unknown truncation_side: {side}")


def collate_batch(
    sequences: Sequence[Sequence[int]],
    labels: Optional[Sequence[int]] = None,
    cfg: CollateConfig = CollateConfig(),
) -> Dict[str, Any]:
    """
    Convert variable-length sequences into a fixed-shape batch.

    Returns:
      - input_ids: np.int64 [B, T]
      - attention_mask: np.int64 [B, T]  (1 real, 0 pad)
      - labels: np.int64 [B] (optional)
      - lengths: list[int] original lengths after truncation (before padding)
      - seq_len: int T (final padded length)
    """
    if not sequences:
        raise ValueError("collate_batch: sequences is empty")

    truncated: list[list[int]] = []
    for s in sequences:
        s2 = _truncate(s, cfg.max_len, cfg.truncation_side) if cfg.truncation else list(s)
        truncated.append(s2)

    lengths = [len(s) for s in truncated]

    if cfg.pad_to == "batch":
        T = max(lengths) if lengths else 0
    elif cfg.pad_to == "fixed":
        if cfg.max_len <= 0:
            raise ValueError("pad_to='fixed' requires cfg.max_len > 0")
        T = cfg.max_len
    else:
        raise ValueError(f"Unknown pad_to: {cfg.pad_to}")

    if T <= 0:
        raise ValueError("Computed seq_len <= 0; check your inputs/max_len")

    B = len(truncated)
    input_ids = np.full((B, T), fill_value=cfg.pad_id, dtype=np.int64)
    attention_mask = np.zeros((B, T), dtype=np.int64)

    for i, s in enumerate(truncated):
        n = min(len(s), T)
        if n > 0:
            input_ids[i, :n] = np.asarray(s[:n], dtype=np.int64)
            attention_mask[i, :n] = 1

    out: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "seq_len": T,
    }

    if labels is not None:
        if len(labels) != B:
            raise ValueError(f"labels length {len(labels)} != batch size {B}")
        out["labels"] = np.asarray(labels, dtype=np.int64)

    return out
