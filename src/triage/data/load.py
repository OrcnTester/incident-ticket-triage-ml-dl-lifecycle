from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd


CATEGORY_CANDIDATES = ["category", "label", "incident_category", "type"]
PRIORITY_CANDIDATES = ["priority", "prio", "p", "sev", "severity"]

TITLE_CANDIDATES = ["title", "summary", "subject", "short_description"]
DESC_CANDIDATES = ["description", "body", "details", "long_description"]


def _first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _normalize_priority(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().upper()
    if s == "" or s == "NAN":
        return None

    # Already P0..P3
    if s in {"P0", "P1", "P2", "P3"}:
        return s

    # Numeric-like -> map to P{0..3}
    try:
        n = int(float(s))
        if n in {0, 1, 2, 3}:
            return f"P{n}"
    except Exception:
        pass

    # Sometimes "SEV0" etc.
    if s.startswith("SEV"):
        tail = s.replace("SEV", "")
        try:
            n = int(float(tail))
            if n in {0, 1, 2, 3}:
                return f"P{n}"
        except Exception:
            return s

    return s


@dataclass
class LoadedTickets:
    df: pd.DataFrame
    text_col: str
    category_col: Optional[str]
    priority_col: Optional[str]


def load_tickets(
    csv_path: str | Path = "data/tickets.csv",
    *,
    text_cols: Optional[List[str]] = None,
    category_col: Optional[str] = None,
    priority_col: Optional[str] = None,
    drop_duplicates_on_text: bool = True,
) -> LoadedTickets:
    """
    Loads tickets from CSV and builds a unified `text` column.
    Tries to auto-detect label columns if not provided.

    Returns a cleaned dataframe with:
      - text (str)
      - category (optional)
      - priority (optional normalized to P0..P3 when possible)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Detect columns if not provided
    cols = list(df.columns)

    if category_col is None:
        category_col = _first_existing(cols, CATEGORY_CANDIDATES)

    if priority_col is None:
        priority_col = _first_existing(cols, PRIORITY_CANDIDATES)

    # Determine text columns
    if text_cols is None:
        title_col = _first_existing(cols, TITLE_CANDIDATES)
        desc_col = _first_existing(cols, DESC_CANDIDATES)

        chosen: List[str] = []
        if title_col:
            chosen.append(title_col)
        if desc_col and desc_col != title_col:
            chosen.append(desc_col)

        # Fallback: use all object columns except labels
        if not chosen:
            label_set = {c for c in [category_col, priority_col] if c}
            obj_cols = [c for c in cols if c not in label_set and df[c].dtype == "object"]
            chosen = obj_cols[:2] if obj_cols else []

        if not chosen:
            raise ValueError("Could not infer text columns. Provide --text-cols explicitly.")
        text_cols = chosen

    # Build text
    def join_text(row) -> str:
        parts = []
        for c in text_cols or []:
            v = row.get(c, "")
            if v is None:
                continue
            s = str(v).strip()
            if s:
                parts.append(s)
        return "\n".join(parts).strip()

    df = df.copy()
    df["text"] = df.apply(join_text, axis=1)

    # Normalize priority if exists
    if priority_col and priority_col in df.columns:
        df["priority_norm"] = df[priority_col].apply(_normalize_priority)
    else:
        df["priority_norm"] = None

    # Rename/standardize label columns (keep originals too)
    if category_col and category_col in df.columns:
        df["category_norm"] = df[category_col].astype(str).str.strip()
        df.loc[df["category_norm"].isin(["", "nan", "None"]), "category_norm"] = None
    else:
        df["category_norm"] = None

    # Drop rows with empty text
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    # Drop duplicates to reduce leakage
    if drop_duplicates_on_text:
        df = df.drop_duplicates(subset=["text"]).copy()

    df = df.reset_index(drop=True)

    return LoadedTickets(
        df=df,
        text_col="text",
        category_col="category_norm" if df["category_norm"].notna().any() else None,
        priority_col="priority_norm" if df["priority_norm"].notna().any() else None,
    )
