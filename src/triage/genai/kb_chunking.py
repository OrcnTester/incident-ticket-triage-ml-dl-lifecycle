from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-./]+")


@dataclass(frozen=True)
class ChunkConfig:
    mode: str = "tokens"  # "tokens" | "chars"
    chunk_size: int = 200
    overlap: int = 40
    min_size: int = 20
    normalize_ws: bool = True


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _make_chunk_id(doc_id: str, start: int, end: int) -> str:
    h = hashlib.md5(f"{doc_id}:{start}:{end}".encode("utf-8")).hexdigest()[:12]
    return f"ch_{h}"


def chunk_text(text: str, *, doc_id: str, cfg: ChunkConfig) -> List[Dict]:
    raw = text or ""
    t = normalize(raw) if cfg.normalize_ws else raw

    out: List[Dict] = []

    if cfg.mode == "chars":
        n = len(t)
        step = max(1, cfg.chunk_size - cfg.overlap)
        start = 0
        while start < n:
            end = min(n, start + cfg.chunk_size)
            piece = t[start:end].strip()
            if len(piece) >= cfg.min_size and any(c.isalnum() for c in piece):
                out.append(
                    {
                        "chunk_id": _make_chunk_id(doc_id, start, end),
                        "doc_id": doc_id,
                        "text": piece,
                        "start": start,
                        "end": end,
                        "n_chars": len(piece),
                        "n_tokens": len(tokenize(piece)),
                    }
                )
            start += step
        return out

    if cfg.mode == "tokens":
        toks = tokenize(t)
        step = max(1, cfg.chunk_size - cfg.overlap)
        i = 0
        while i < len(toks):
            j = min(len(toks), i + cfg.chunk_size)
            piece = " ".join(toks[i:j]).strip()
            if len(piece) >= cfg.min_size and any(c.isalnum() for c in piece):
                out.append(
                    {
                        "chunk_id": _make_chunk_id(doc_id, i, j),
                        "doc_id": doc_id,
                        "text": piece,
                        "start": i,
                        "end": j,
                        "n_chars": len(piece),
                        "n_tokens": (j - i),
                    }
                )
            i += step
        return out

    raise ValueError(f"Unknown chunk mode: {cfg.mode}")
