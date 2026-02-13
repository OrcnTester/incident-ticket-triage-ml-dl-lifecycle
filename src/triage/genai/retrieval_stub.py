from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class RetrievedSnippet:
    doc_id: str
    path: str
    score: float
    text: str


def _chunk_text(text: str, *, max_chars: int = 600) -> List[str]:
    """
    Simple paragraph-based chunking.
    Keeps chunks reasonably small for safer citations.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buff: List[str] = []
    size = 0

    for p in paras:
        if size + len(p) + 2 <= max_chars:
            buff.append(p)
            size += len(p) + 2
        else:
            if buff:
                chunks.append("\n\n".join(buff).strip())
            buff = [p]
            size = len(p)

    if buff:
        chunks.append("\n\n".join(buff).strip())

    return [c for c in chunks if c]


class TfidfRetriever:
    """
    Minimal TF-IDF retriever for RAG demos.
    - Indexes markdown/txt files under kb_dir
    - Returns top-k chunks by cosine similarity
    """
    def __init__(self, kb_dir: str | Path):
        self.kb_dir = Path(kb_dir)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=100_000)
        self._chunks: List[Tuple[str, str, str]] = []  # (doc_id, path, chunk_text)
        self._matrix = None  # TF-IDF matrix

    def build(self) -> None:
        if not self.kb_dir.exists():
            raise FileNotFoundError(f"KB dir not found: {self.kb_dir}")

        files = sorted([p for p in self.kb_dir.rglob("*") if p.suffix.lower() in {".md", ".txt"}])
        if not files:
            raise ValueError(f"No KB files found under: {self.kb_dir}")

        chunks: List[Tuple[str, str, str]] = []
        for p in files:
            doc_id = p.stem
            text = p.read_text(encoding="utf-8", errors="ignore")
            for ch in _chunk_text(text):
                chunks.append((doc_id, str(p).replace("\\", "/"), ch))

        self._chunks = chunks
        texts = [c[2] for c in chunks]
        self._matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, *, top_k: int = 3) -> List[RetrievedSnippet]:
        if self._matrix is None:
            self.build()

        q = (query or "").strip()
        if not q:
            return []

        qv = self.vectorizer.transform([q])
        # cosine similarity for TF-IDF vectors
        scores = (self._matrix @ qv.T).toarray().ravel()

        if len(scores) == 0:
            return []

        top_k = max(1, min(int(top_k), 10))
        idxs = np.argsort(scores)[::-1][:top_k]

        out: List[RetrievedSnippet] = []
        for i in idxs:
            doc_id, path, text = self._chunks[int(i)]
            out.append(
                RetrievedSnippet(
                    doc_id=doc_id,
                    path=path,
                    score=float(scores[int(i)]),
                    text=text,
                )
            )
        return out
