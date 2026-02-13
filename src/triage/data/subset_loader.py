from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.triage.data.subset_manifest import SubsetManifest, load_json, parse_subset_manifest, sha256_file


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dataset_fingerprint(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    st = p.stat()
    return {
        "path": str(p),
        "size_bytes": int(st.st_size),
        "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat(),
    }


def stable_sample_id(row: pd.Series) -> str:
    """
    Create a stable-ish sample id from a row.
    Uses fields that exist in this repo's synthetic tickets.
    """
    parts = [
        str(row.get("timestamp", "")),
        str(row.get("system", "")),
        str(row.get("source", "")),
        str(row.get("error_code", "")),
        str(row.get("routing_team", "")),
        str(row.get("text", "")),
    ]
    raw = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:12]


def apply_subset_manifest(df: pd.DataFrame, manifest: SubsetManifest) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filters df according to include/exclude rules.

    include: column -> allowed values
    exclude: column -> disallowed values

    Returns: (filtered_df, meta)
    """
    before = int(len(df))
    mask = pd.Series([True] * len(df), index=df.index)

    # include filters
    for col, allowed in manifest.include.items():
        if col not in df.columns:
            raise ValueError(f"include column '{col}' not found in df columns: {list(df.columns)}")
        if allowed:
            mask &= df[col].astype(str).isin([str(x) for x in allowed])

    # exclude filters
    for col, banned in manifest.exclude.items():
        if col not in df.columns:
            raise ValueError(f"exclude column '{col}' not found in df columns: {list(df.columns)}")
        if banned:
            mask &= ~df[col].astype(str).isin([str(x) for x in banned])

    out = df[mask].copy()

    # attach sample_id for traceability (non-destructive)
    if "sample_id" not in out.columns:
        out["sample_id"] = out.apply(stable_sample_id, axis=1)

    meta = {
        "subset_id": manifest.subset_id,
        "schema_version": manifest.schema_version,
        "n_before": before,
        "n_after": int(len(out)),
        "filters": {
            "include": manifest.include,
            "exclude": manifest.exclude,
        },
        "generated_at": utc_now_iso(),
    }
    return out, meta


def load_and_apply_subset(df: pd.DataFrame, subset_manifest_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = load_json(subset_manifest_path)
    manifest = parse_subset_manifest(raw)
    df2, meta = apply_subset_manifest(df, manifest)
    meta["manifest_path"] = str(subset_manifest_path)
    meta["manifest_sha256"] = sha256_file(subset_manifest_path)
    return df2, meta


def write_subset_artifacts(meta: Dict[str, Any], out_dir: str | Path) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "subset_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
