from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


# Token pattern designed for incident text:
# keeps things like "AUTH_INVALID_TOKEN", "503", "/api/payments/charge", "db-01", "OOM"
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


@dataclass(frozen=True)
class TokenizerConfig:
    mode: str  # "word" | "char" | "subword"
    lowercase: bool = True
    # subword (char n-grams) settings
    ngram_min: int = 3
    ngram_max: int = 5


def tokenize_word(text: str, *, lowercase: bool = True) -> List[str]:
    t = text or ""
    if lowercase:
        t = t.lower()
    return TOKEN_RE.findall(t)


def tokenize_char(text: str, *, lowercase: bool = True) -> List[str]:
    t = text or ""
    if lowercase:
        t = t.lower()
    # keep spaces out to avoid massive padding signal; treat whitespace as separator
    t = t.replace("\n", " ").replace("\t", " ")
    chars = [c for c in t if not c.isspace()]
    return chars


def _word_char_ngrams(word: str, nmin: int, nmax: int) -> List[str]:
    # fastText-style boundary markers help capture prefixes/suffixes
    w = f"<{word}>"
    out: List[str] = []
    L = len(w)
    for n in range(nmin, nmax + 1):
        if n > L:
            continue
        for i in range(0, L - n + 1):
            out.append(w[i : i + n])
    return out


def tokenize_subword_char_ngrams(
    text: str,
    *,
    lowercase: bool = True,
    ngram_min: int = 3,
    ngram_max: int = 5,
) -> List[str]:
    """
    Subword approximation suitable for classical ML:
    - word split first
    - expand each word into character n-grams (fastText-like)

    This reduces OOV vs pure word tokens while keeping vocab bounded.
    """
    words = tokenize_word(text, lowercase=lowercase)
    out: List[str] = []
    for w in words:
        out.extend(_word_char_ngrams(w, ngram_min, ngram_max))
    return out


def tokenize(text: str, cfg: TokenizerConfig) -> List[str]:
    if cfg.mode == "word":
        return tokenize_word(text, lowercase=cfg.lowercase)
    if cfg.mode == "char":
        return tokenize_char(text, lowercase=cfg.lowercase)
    if cfg.mode == "subword":
        return tokenize_subword_char_ngrams(
            text,
            lowercase=cfg.lowercase,
            ngram_min=cfg.ngram_min,
            ngram_max=cfg.ngram_max,
        )
    raise ValueError(f"Unknown tokenizer mode: {cfg.mode}")
