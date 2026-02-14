from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ModelName = Literal["logreg", "nb", "svm", "rf"]


@dataclass(frozen=True)
class VectorizerConfig:
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    max_features: int = 200_000


def build_tfidf(cfg: VectorizerConfig = VectorizerConfig()) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        min_df=int(cfg.min_df),
        max_features=int(cfg.max_features),
    )


def build_pipeline(
    model: ModelName,
    *,
    vec_cfg: VectorizerConfig = VectorizerConfig(),
    seed: int = 42,
    # RF-only knobs
    svd_dim: int = 256,
    rf_estimators: int = 300,
    rf_max_depth: Optional[int] = None,
) -> Pipeline:
    """
    Standard pipeline: tfidf -> clf
    RF variant: tfidf -> svd -> clf
    """
    tfidf = build_tfidf(vec_cfg)

    if model == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None,
        )
        return Pipeline([("tfidf", tfidf), ("clf", clf)])

    if model == "nb":
        clf = MultinomialNB(alpha=1.0)
        return Pipeline([("tfidf", tfidf), ("clf", clf)])

    if model == "svm":
        clf = LinearSVC(class_weight="balanced")
        return Pipeline([("tfidf", tfidf), ("clf", clf)])

    if model == "rf":
        # Trees + huge sparse TF-IDF is a rough combo → reduce dimensions first.
        svd = TruncatedSVD(n_components=int(svd_dim), random_state=int(seed))
        clf = RandomForestClassifier(
            n_estimators=int(rf_estimators),
            random_state=int(seed),
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_depth=rf_max_depth,
        )
        return Pipeline([("tfidf", tfidf), ("svd", svd), ("clf", clf)])

    raise ValueError(f"Unknown model: {model}")
