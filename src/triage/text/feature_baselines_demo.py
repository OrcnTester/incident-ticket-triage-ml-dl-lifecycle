from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.triage.text.feature_vectorizers import (
    EmbeddingBagVectorizer,
    SparseVectorizerConfig,
    make_bow,
    make_onehot,
    make_tfidf,
    make_svd_embedding_pipeline,
)


@dataclass(frozen=True)
class RunResult:
    mode: str
    metrics: Dict[str, Any]
    footprint: Dict[str, Any]
    timing: Dict[str, Any]
    out_dir: str


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def approx_sparse_mb(X) -> float:
    # CSR: data + indices + indptr
    n = 0
    for attr in ("data", "indices", "indptr"):
        arr = getattr(X, attr, None)
        if arr is not None:
            n += arr.nbytes
    return float(n) / (1024 * 1024)


def sparsity_ratio(X) -> Optional[float]:
    try:
        nnz = float(X.nnz)
        total = float(X.shape[0] * X.shape[1])
        return 1.0 - (nnz / total) if total > 0 else None
    except Exception:
        return None


def make_classifier(seed: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
        random_state=int(seed),
    )


def build_pipeline_for_mode(
    mode: str,
    *,
    seed: int,
    cfg: SparseVectorizerConfig,
    svd_dim: int,
    emb_dim: int,
    buckets: int,
) -> Pipeline:
    clf = make_classifier(seed)

    if mode == "onehot":
        vec = make_onehot(cfg)
        return Pipeline([("vec", vec), ("clf", clf)])

    if mode == "bow":
        vec = make_bow(cfg)
        return Pipeline([("vec", vec), ("clf", clf)])

    if mode == "tfidf":
        vec = make_tfidf(cfg)
        return Pipeline([("vec", vec), ("clf", clf)])

    if mode == "svd":
        # Dense embeddings via LSA: TF-IDF -> SVD -> classifier
        tfidf = make_tfidf(cfg)
        emb = make_svd_embedding_pipeline(tfidf=tfidf, svd_dim=int(svd_dim), seed=int(seed))
        # emb outputs dense; scaling helps logistic regression
        return Pipeline([("emb", emb), ("scaler", StandardScaler()), ("clf", clf)])

    if mode == "embbag":
        emb = EmbeddingBagVectorizer(emb_dim=int(emb_dim), buckets=int(buckets), seed=int(seed), use_subwords=True)
        return Pipeline([("emb", emb), ("scaler", StandardScaler()), ("clf", clf)])

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_priority_binary(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    # treat P0/P1 as "high" vs others
    yt = pd.Series(y_true).isin(["P0", "P1"]).astype(int)
    yp = pd.Series(y_pred).isin(["P0", "P1"]).astype(int)
    return {
        "p0p1_recall": float(recall_score(yt, yp)),
        "p0p1_precision": float(precision_score(yt, yp)),
        "confusion_matrix": [[int(((yt == 0) & (yp == 0)).sum()), int(((yt == 0) & (yp == 1)).sum())],
                             [int(((yt == 1) & (yp == 0)).sum()), int(((yt == 1) & (yp == 1)).sum())]],
    }


def run_one(
    mode: str,
    *,
    X_train: List[str],
    y_train: List[str],
    X_test: List[str],
    y_test: List[str],
    target: str,
    seed: int,
    cfg: SparseVectorizerConfig,
    svd_dim: int,
    emb_dim: int,
    buckets: int,
    out_base: Path,
) -> RunResult:
    pipe = build_pipeline_for_mode(
        mode, seed=seed, cfg=cfg, svd_dim=svd_dim, emb_dim=emb_dim, buckets=buckets
    )

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = pipe.predict(X_test)
    pred_s = time.perf_counter() - t1

    # metrics
    metrics: Dict[str, Any] = {
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "accuracy": float(accuracy_score(y_test, pred)),
    }
    if target == "priority":
        metrics.update(evaluate_priority_binary(y_test, list(pred)))

    # footprint (estimate from a transformed batch)
    feat_dim = None
    sparse_mb = None
    dense_mb = None
    sparsity = None

    # try to compute feature matrix from the vectorizer/embedding stage
    try:
        if "vec" in pipe.named_steps:
            Xt = pipe.named_steps["vec"].transform(X_train[:512])
            feat_dim = int(Xt.shape[1])
            sparsity = float(sparsity_ratio(Xt)) if hasattr(Xt, "nnz") else None
            sparse_mb = float(approx_sparse_mb(Xt)) if hasattr(Xt, "nnz") else None
        elif "emb" in pipe.named_steps:
            # emb may be a Pipeline (svd) or EmbeddingBagVectorizer
            emb_step = pipe.named_steps["emb"]
            Xt = emb_step.transform(X_train[:512])
            feat_dim = int(Xt.shape[1])
            dense_mb = float(Xt.nbytes) / (1024 * 1024)
    except Exception:
        pass

    footprint = {
        "feature_dim": feat_dim,
        "sparsity_ratio_est": sparsity,
        "sparse_mb_est": sparse_mb,
        "dense_mb_est": dense_mb,
    }

    timing = {"fit_s": float(fit_s), "predict_s": float(pred_s)}

    out_dir = out_base / target / mode
    ensure_dir(out_dir)

    joblib.dump(pipe, out_dir / "model.joblib")
    meta = {
        "target": target,
        "mode": mode,
        "seed": seed,
        "vectorizer_cfg": cfg.__dict__,
        "svd_dim": svd_dim if mode == "svd" else None,
        "emb_dim": emb_dim if mode == "embbag" else None,
        "buckets": buckets if mode == "embbag" else None,
        "metrics": metrics,
        "footprint": footprint,
        "timing": timing,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return RunResult(mode=mode, metrics=metrics, footprint=footprint, timing=timing, out_dir=str(out_dir))


def render_report(target: str, results: List[RunResult], out_md: Path, out_json: Path) -> None:
    ensure_dir(out_md.parent)
    ensure_dir(out_json.parent)

    payload = {
        "target": target,
        "results": [
            {"mode": r.mode, "metrics": r.metrics, "footprint": r.footprint, "timing": r.timing, "out_dir": r.out_dir}
            for r in results
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown table
    lines = []
    lines.append(f"# Feature Baselines Report — {target}")
    lines.append("")
    lines.append("| mode | macro_f1 | weighted_f1 | acc | feat_dim | sparsity_est | mem_est_mb | fit_s | pred_s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        mem = r.footprint.get("sparse_mb_est") or r.footprint.get("dense_mb_est")
        lines.append(
            f"| {r.mode} | {r.metrics.get('macro_f1', 0):.4f} | {r.metrics.get('weighted_f1', 0):.4f} | "
            f"{r.metrics.get('accuracy', 0):.4f} | {r.footprint.get('feature_dim')} | "
            f"{(r.footprint.get('sparsity_ratio_est') if r.footprint.get('sparsity_ratio_est') is not None else '')} | "
            f"{(f'{mem:.3f}' if mem is not None else '')} | {r.timing.get('fit_s', 0):.3f} | {r.timing.get('predict_s', 0):.3f} |"
        )

    # priority extras
    if target == "priority":
        lines.append("")
        lines.append("## Priority-specific (P0/P1 as high)")
        lines.append("| mode | p0p1_recall | p0p1_precision |")
        lines.append("|---|---:|---:|")
        for r in results:
            lines.append(
                f"| {r.mode} | {r.metrics.get('p0p1_recall', 0):.4f} | {r.metrics.get('p0p1_precision', 0):.4f} |"
            )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare feature baselines: one-hot vs BoW vs TF-IDF vs dense embeddings.")
    ap.add_argument("--data", type=str, default="data/tickets.csv")
    ap.add_argument("--target", choices=["category", "priority"], required=True)
    ap.add_argument("--mode", choices=["onehot", "bow", "tfidf", "svd", "embbag", "all"], default="all")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # sparse vectorizer knobs
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-features", type=int, default=200_000)
    ap.add_argument("--ngram-min", type=int, default=1)
    ap.add_argument("--ngram-max", type=int, default=2)

    # embedding knobs
    ap.add_argument("--svd-dim", type=int, default=256)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--buckets", type=int, default=50_000)

    ap.add_argument("--out-base", type=str, default="artifacts/feature_baselines")
    ap.add_argument("--report-md", type=str, default="")
    ap.add_argument("--report-json", type=str, default="")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if "text" not in df.columns:
        raise SystemExit("Expected 'text' column in CSV.")
    if args.target not in df.columns:
        raise SystemExit(f"Expected target column '{args.target}' in CSV.")

    df = df[df[args.target].notna()].copy()
    X = df["text"].astype(str).tolist()
    y = df[args.target].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(args.test_size), random_state=int(args.seed), stratify=y
    )

    cfg = SparseVectorizerConfig(
        min_df=int(args.min_df),
        max_features=int(args.max_features),
        ngram_min=int(args.ngram_min),
        ngram_max=int(args.ngram_max),
    )

    modes = ["onehot", "bow", "tfidf", "svd", "embbag"] if args.mode == "all" else [args.mode]

    out_base = Path(args.out_base)
    results: List[RunResult] = []
    for m in modes:
        res = run_one(
            m,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            target=args.target,
            seed=int(args.seed),
            cfg=cfg,
            svd_dim=int(args.svd_dim),
            emb_dim=int(args.emb_dim),
            buckets=int(args.buckets),
            out_base=out_base,
        )
        results.append(res)
        print(f"[OK] {m} -> macro_f1={res.metrics['macro_f1']:.4f} acc={res.metrics['accuracy']:.4f}")

    report_md = Path(args.report_md) if args.report_md else Path("reports") / f"feature_baselines_{args.target}.md"
    report_json = Path(args.report_json) if args.report_json else Path("reports") / f"feature_baselines_{args.target}.json"
    render_report(args.target, results, report_md, report_json)

    print(f"[OK] Wrote {report_md} and {report_json}")


if __name__ == "__main__":
    main()
