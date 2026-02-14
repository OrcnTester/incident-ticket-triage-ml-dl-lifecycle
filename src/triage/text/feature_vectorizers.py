from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline


_TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-./]+")


def simple_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def subword_ngrams(token: str, n_min: int, n_max: int) -> Iterable[str]:
    t = token
    if len(t) < n_min:
        return []
    out = []
    L = len(t)
    for n in range(n_min, n_max + 1):
        if n > L:
            break
        for i in range(0, L - n + 1):
            out.append(t[i : i + n])
    return out


@dataclass(frozen=True)
class SparseVectorizerConfig:
    min_df: int = 2
    max_features: int = 200_000
    ngram_min: int = 1
    ngram_max: int = 2


def make_onehot(cfg: SparseVectorizerConfig) -> CountVectorizer:
    # one-hot = binary presence
    return CountVectorizer(
        lowercase=True,
        binary=True,
        min_df=int(cfg.min_df),
        max_features=int(cfg.max_features),
        ngram_range=(int(cfg.ngram_min), int(cfg.ngram_max)),
    )


def make_bow(cfg: SparseVectorizerConfig) -> CountVectorizer:
    # BoW = counts
    return CountVectorizer(
        lowercase=True,
        binary=False,
        min_df=int(cfg.min_df),
        max_features=int(cfg.max_features),
        ngram_range=(int(cfg.ngram_min), int(cfg.ngram_max)),
    )


def make_tfidf(cfg: SparseVectorizerConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        min_df=int(cfg.min_df),
        max_features=int(cfg.max_features),
        ngram_range=(int(cfg.ngram_min), int(cfg.ngram_max)),
    )


def make_svd_embedding_pipeline(
    *,
    tfidf: TfidfVectorizer,
    svd_dim: int,
    seed: int,
) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", tfidf),
            ("svd", TruncatedSVD(n_components=int(svd_dim), random_state=int(seed))),
        ]
    )


class EmbeddingBagVectorizer(BaseEstimator, TransformerMixin):
    """
    A lightweight, dependency-free "EmbeddingBag-style" feature:
    - tokens -> hash bucket -> embedding vector
    - document vector = mean(pool) of token vectors (dense)

    Notes:
    - deterministic via (seed + token hash)
    - collisions are expected; increase buckets if needed
    - optional subword ngrams improves OOV robustness
    """

    def __init__(
        self,
        emb_dim: int = 128,
        buckets: int = 50_000,
        seed: int = 42,
        use_subwords: bool = False,
        subword_min: int = 3,
        subword_max: int = 5,
        max_tokens: int = 512,
    ) -> None:
        self.emb_dim = int(emb_dim)
        self.buckets = int(buckets)
        self.seed = int(seed)
        self.use_subwords = bool(use_subwords)
        self.subword_min = int(subword_min)
        self.subword_max = int(subword_max)
        self.max_tokens = int(max_tokens)

    def fit(self, X: List[str], y=None):  # noqa: N803
        rng = np.random.default_rng(self.seed)
        # Normal init is fine for a fixed random table; the classifier learns on top.
        self._E = rng.normal(0.0, 1.0, size=(self.buckets, self.emb_dim)).astype(np.float32)
        return self

    @staticmethod
    def _bucket(token: str, buckets: int) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()  # stable across runs
        return int(h[:8], 16) % buckets

    def transform(self, X: List[str]):  # noqa: N803
        if not hasattr(self, "_E"):
            raise RuntimeError("EmbeddingBagVectorizer is not fitted yet. Call fit() first.")

        E: np.ndarray = self._E  # (buckets, emb_dim)
        out = np.zeros((len(X), self.emb_dim), dtype=np.float32)

        for i, text in enumerate(X):
            toks = simple_tokenize(text)
            if self.use_subwords:
                expanded = []
                for t in toks:
                    expanded.append(t)
                    expanded.extend(list(subword_ngrams(t, self.subword_min, self.subword_max)))
                toks = expanded

            if self.max_tokens and len(toks) > self.max_tokens:
                toks = toks[: self.max_tokens]

            if not toks:
                continue

            idxs = [self._bucket(t, self.buckets) for t in toks]
            vec = E[idxs].mean(axis=0)
            out[i] = vec

        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"emb_{i}" for i in range(self.emb_dim)], dtype=object)
