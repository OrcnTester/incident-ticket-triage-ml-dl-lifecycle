from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import pandas as pd


# ---- Optional (best-effort) process RSS measurement ----
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


def _now() -> float:
    return time.perf_counter()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except FileNotFoundError:
        return 0.0


def _rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def stream_csv_rows(
    path: str,
    *,
    encoding: str = "utf-8",
    text_col: str = "text",
    priority_col: str = "priority",
    category_col: str = "category",
    timestamp_col: str = "timestamp",
) -> Iterator[Tuple[str, Optional[str], Optional[str], Optional[str]]]:
    """
    Minimal streaming CSV reader (generator).
    Yields tuples: (text, priority, category, timestamp)

    NOTE: This is intentionally simple: no shuffling, no parallelism.
    """
    with open(path, newline="", encoding=encoding) as f:
        r = csv.DictReader(f)
        for row in r:
            yield (
                (row.get(text_col) or "").strip(),
                (row.get(priority_col) or None),
                (row.get(category_col) or None),
                (row.get(timestamp_col) or None),
            )


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str  # "memory" | "stream"
    data_path: str
    file_size_mb: float
    n_rows: int
    elapsed_s: float
    rows_per_s: float

    # memory-ish metrics (best effort)
    rss_mb_before: Optional[float]
    rss_mb_after: Optional[float]
    df_memory_mb: Optional[float]

    # environment
    python: str
    platform: str
    pandas: str


def benchmark_memory(
    data_path: str,
    *,
    usecols: Optional[list[str]] = None,
    encoding: str = "utf-8",
) -> Tuple[int, float, Optional[float]]:
    """
    Loads full CSV into RAM using pandas.
    Returns (n_rows, elapsed_s, df_memory_mb).
    """
    t0 = _now()
    df = pd.read_csv(data_path, encoding=encoding, usecols=usecols)
    # touch the dataframe to ensure parse actually happened
    n = int(len(df))
    _ = df.head(1)
    elapsed = _now() - t0

    try:
        df_mem = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    except Exception:
        df_mem = None
    return n, elapsed, df_mem


def benchmark_stream(
    data_path: str,
    *,
    encoding: str = "utf-8",
    limit: int = 0,
) -> Tuple[int, float]:
    """
    Iterates CSV row-by-row using DictReader generator.
    Returns (n_rows, elapsed_s).
    """
    t0 = _now()
    n = 0
    for _text, _prio, _cat, _ts in stream_csv_rows(data_path, encoding=encoding):
        n += 1
        if limit and n >= limit:
            break
    elapsed = _now() - t0
    return n, elapsed


def write_reports(res: BenchmarkResult, out_json: Path, out_md: Path) -> None:
    _ensure_dir(out_json.parent)
    out_json.write_text(json.dumps(asdict(res), indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# Loading Benchmark")
    md.append("")
    md.append(f"- **mode**: `{res.mode}`")
    md.append(f"- **data**: `{res.data_path}`")
    md.append(f"- **file size**: {res.file_size_mb:.2f} MB")
    md.append(f"- **rows**: {res.n_rows}")
    md.append(f"- **elapsed**: {res.elapsed_s:.4f} s")
    md.append(f"- **throughput**: {res.rows_per_s:.2f} rows/s")
    md.append("")
    md.append("## Memory (best-effort)")
    md.append(f"- RSS before: {res.rss_mb_before if res.rss_mb_before is not None else 'n/a'} MB")
    md.append(f"- RSS after: {res.rss_mb_after if res.rss_mb_after is not None else 'n/a'} MB")
    md.append(f"- pandas df.memory_usage: {res.df_memory_mb if res.df_memory_mb is not None else 'n/a'} MB")
    md.append("")
    md.append("## Environment")
    md.append(f"- Python: {res.python}")
    md.append(f"- Platform: {res.platform}")
    md.append(f"- pandas: {res.pandas}")
    md.append("")

    out_md.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark CSV loading: in-memory (pandas) vs streaming generator")
    ap.add_argument("--data", type=str, default="data/tickets.csv")
    ap.add_argument("--mode", type=str, choices=["memory", "stream"], default="memory")
    ap.add_argument("--encoding", type=str, default="utf-8")
    ap.add_argument("--limit", type=int, default=0, help="Stream mode only: stop after N rows (0 = all)")
    ap.add_argument("--usecols", nargs="*", default=None, help="Memory mode only: restrict columns for fair comparisons")
    ap.add_argument("--out-json", type=str, default="reports/loading_benchmark.json")
    ap.add_argument("--out-md", type=str, default="reports/loading_benchmark.md")
    args = ap.parse_args()

    data_path = Path(args.data)
    rss_before = _rss_mb()

    df_mem_mb: Optional[float] = None
    if args.mode == "memory":
        n, elapsed, df_mem_mb = benchmark_memory(
            str(data_path),
            usecols=list(args.usecols) if args.usecols else None,
            encoding=args.encoding,
        )
    else:
        n, elapsed = benchmark_stream(
            str(data_path),
            encoding=args.encoding,
            limit=int(args.limit),
        )

    rss_after = _rss_mb()
    rows_per_s = (n / elapsed) if elapsed > 0 else 0.0

    res = BenchmarkResult(
        mode=args.mode,
        data_path=str(data_path),
        file_size_mb=_file_size_mb(data_path),
        n_rows=int(n),
        elapsed_s=float(elapsed),
        rows_per_s=float(rows_per_s),
        rss_mb_before=rss_before,
        rss_mb_after=rss_after,
        df_memory_mb=df_mem_mb,
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}",
        pandas=pd.__version__,
    )

    write_reports(res, Path(args.out_json), Path(args.out_md))
    print(f"[OK] Wrote {args.out_json} and {args.out_md}")
    print(json.dumps(asdict(res), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
