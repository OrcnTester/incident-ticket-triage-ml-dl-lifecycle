from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-./]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

DEFAULT_SEPARATORS: Sequence[str] = ("\n\n", "\n", ". ", " ", "")


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _chunk_id(doc_id: str, start: int, end: int, kind: str) -> str:
    h = hashlib.md5(f"{doc_id}:{kind}:{start}:{end}".encode("utf-8")).hexdigest()[:12]
    return f"ch_{h}"


@dataclass(frozen=True)
class SplitterCfg:
    chunk_size: int = 200
    overlap: int = 40
    min_size: int = 20  # minimum tokens/chars to keep
    normalize: bool = True


def token_split(text: str, *, doc_id: str, cfg: SplitterCfg) -> List[Dict]:
    t = normalize_ws(text) if cfg.normalize else (text or "")
    toks = tokenize(t)

    step = max(1, cfg.chunk_size - cfg.overlap)
    out: List[Dict] = []
    i = 0
    while i < len(toks):
        j = min(len(toks), i + cfg.chunk_size)
        piece = " ".join(toks[i:j]).strip()
        if len(piece) >= cfg.min_size and any(c.isalnum() for c in piece):
            out.append(
                {
                    "chunk_id": _chunk_id(doc_id, i, j, "token"),
                    "doc_id": doc_id,
                    "strategy": "token",
                    "text": piece,
                    "start": i,
                    "end": j,
                    "n_tokens": int(j - i),
                    "n_chars": int(len(piece)),
                }
            )
        i += step
    return out


def sentence_split(text: str, *, doc_id: str, cfg: SplitterCfg) -> List[Dict]:
    raw = text or ""
    s = raw.strip()
    if not s:
        return []

    # Keep original text for nicer chunks; tokenize only for budgeting.
    sentences = _SENTENCE_RE.split(s)
    sentences = [x.strip() for x in sentences if x.strip()]

    out: List[Dict] = []
    cur: List[str] = []
    cur_tokens = 0
    start_sent = 0
    step_overlap_tokens = max(0, int(cfg.overlap))

    def flush(end_sent: int) -> None:
        nonlocal cur, cur_tokens, start_sent
        if not cur:
            return
        piece = " ".join(cur).strip()
        toks = tokenize(piece)
        if len(toks) >= cfg.min_size:
            out.append(
                {
                    "chunk_id": _chunk_id(doc_id, start_sent, end_sent, "sent"),
                    "doc_id": doc_id,
                    "strategy": "sentence",
                    "text": piece,
                    "start": start_sent,  # sentence index offsets
                    "end": end_sent,
                    "n_tokens": int(len(toks)),
                    "n_chars": int(len(piece)),
                }
            )

        # build overlap: keep last overlap tokens worth of sentences (approx)
        if step_overlap_tokens <= 0:
            cur = []
            cur_tokens = 0
            start_sent = end_sent
            return

        # Greedy: keep sentences from the end until token budget >= overlap
        keep: List[str] = []
        keep_tokens = 0
        k = len(cur) - 1
        while k >= 0 and keep_tokens < step_overlap_tokens:
            keep.insert(0, cur[k])
            keep_tokens += len(tokenize(cur[k]))
            k -= 1
        # new start is end_sent - len(keep) sentences
        cur = keep
        cur_tokens = keep_tokens
        start_sent = end_sent - len(keep)

    for idx, sent in enumerate(sentences):
        stoks = tokenize(sent)
        if not stoks:
            continue

        # If single sentence exceeds chunk budget, hard cut by tokens.
        if len(stoks) > cfg.chunk_size and not cur:
            hard = token_split(sent, doc_id=doc_id, cfg=cfg)
            for c in hard:
                c["strategy"] = "sentence"  # still under sentence strategy umbrella
                c["chunk_id"] = _chunk_id(doc_id, idx * 10_000 + c["start"], idx * 10_000 + c["end"], "sent_hard")
                c["start"] = idx  # sentence index (approx)
                c["end"] = idx + 1
            out.extend(hard)
            start_sent = idx + 1
            continue

        if not cur:
            start_sent = idx

        if cur_tokens + len(stoks) > cfg.chunk_size and cur:
            flush(idx)
        cur.append(sent)
        cur_tokens += len(stoks)

    flush(len(sentences))
    return out


def _split_by_sep(text: str, sep: str) -> List[str]:
    if sep == "":
        return list(text)
    return text.split(sep)


def _recursive_pieces(text: str, *, chunk_size_chars: int, seps: Sequence[str]) -> List[str]:
    """Return small-ish pieces (<= chunk_size_chars) using recursive separators."""
    if len(text) <= chunk_size_chars:
        return [text]

    for sep in seps:
        parts = _split_by_sep(text, sep)
        if len(parts) == 1:
            continue
        out: List[str] = []
        buf = ""
        joiner = sep
        for p in parts:
            candidate = (buf + (joiner if buf else "") + p) if buf else p
            if len(candidate) <= chunk_size_chars:
                buf = candidate
            else:
                if buf:
                    out.append(buf)
                buf = p
        if buf:
            out.append(buf)

        # If this separator didn't help, try next.
        if all(len(x) > chunk_size_chars for x in out):
            continue

        final: List[str] = []
        for o in out:
            if len(o) <= chunk_size_chars:
                final.append(o)
            else:
                final.extend(_recursive_pieces(o, chunk_size_chars=chunk_size_chars, seps=seps[seps.index(sep) + 1 :]))
        return final

    # fallback hard cut
    return [text[i : i + chunk_size_chars] for i in range(0, len(text), chunk_size_chars)]


def recursive_split(text: str, *, doc_id: str, cfg: SplitterCfg, separators: Sequence[str] = DEFAULT_SEPARATORS) -> List[Dict]:
    raw = (text or "").strip()
    if not raw:
        return []

    # Here cfg.chunk_size/overlap are interpreted as *chars* for recursive mode.
    chunk_size_chars = int(cfg.chunk_size)
    overlap_chars = int(cfg.overlap)

    pieces = _recursive_pieces(raw, chunk_size_chars=chunk_size_chars, seps=separators)

    # Merge pieces into final chunks with overlap
    out: List[Dict] = []
    cur = ""
    start_piece = 0

    def flush(end_piece: int) -> None:
        nonlocal cur, start_piece
        piece = cur.strip()
        if len(piece) >= cfg.min_size:
            out.append(
                {
                    "chunk_id": _chunk_id(doc_id, start_piece, end_piece, "recur"),
                    "doc_id": doc_id,
                    "strategy": "recursive",
                    "text": piece,
                    "start": start_piece,  # piece index offsets
                    "end": end_piece,
                    "n_tokens": int(len(tokenize(piece))),
                    "n_chars": int(len(piece)),
                }
            )

        if overlap_chars <= 0:
            cur = ""
            start_piece = end_piece
            return

        # keep last overlap_chars from current chunk
        keep = piece[-overlap_chars:] if len(piece) > overlap_chars else piece
        cur = keep
        start_piece = max(0, end_piece - 1)

    for idx, p in enumerate(pieces):
        p = p.strip()
        if not p:
            continue
        if not cur:
            start_piece = idx
            cur = p
            continue

        if len(cur) + 1 + len(p) <= chunk_size_chars:
            cur = cur + " " + p
        else:
            flush(idx)
            if cur:
                # cur is overlap; try to append
                if len(cur) + 1 + len(p) <= chunk_size_chars:
                    cur = (cur + " " + p).strip()
                else:
                    # if overlap itself is too big, reset
                    cur = p
                    start_piece = idx

    flush(len(pieces))
    return out
