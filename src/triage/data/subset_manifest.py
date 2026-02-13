from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Manifest specs (lightweight)
# -------------------------
# Subset manifest (recommended)
# {
#   "schema_version": "1.0",
#   "subset_id": "payments_and_auth_only",
#   "include": { "system": ["payments", "auth"], "source": ["monitoring"] },
#   "exclude": { "error_code": ["OOM"] },
#   "notes": "...",
#   "created_at": "2026-02-13T00:00:00Z"
# }
#
# Split manifest (optional, most audit-friendly)
# {
#   "schema_version": "1.0",
#   "split_id": "holdout_system_gateway",
#   "group_field": "system",
#   "train_groups": ["auth", "payments"],
#   "val_groups": ["inventory"],
#   "test_groups": ["gateway"],
#   "notes": "...",
#   "created_at": "..."
# }


@dataclass(frozen=True)
class SubsetManifest:
    schema_version: str
    subset_id: str
    include: Dict[str, List[str]]
    exclude: Dict[str, List[str]]
    notes: Optional[str] = None
    created_at: Optional[str] = None


@dataclass(frozen=True)
class SplitManifest:
    schema_version: str
    split_id: str
    group_field: str
    train_groups: List[str]
    val_groups: List[str]
    test_groups: List[str]
    notes: Optional[str] = None
    created_at: Optional[str] = None


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    return sha256_bytes(p.read_bytes())


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _get(d: Dict[str, Any], k: str, default: Any) -> Any:
    v = d.get(k, default)
    return v if v is not None else default


def parse_subset_manifest(raw: Dict[str, Any]) -> SubsetManifest:
    schema_version = str(raw.get("schema_version", "1.0"))
    subset_id = str(raw.get("subset_id", "")).strip()
    if not subset_id:
        raise ValueError("subset manifest missing required field: subset_id")

    include = _get(raw, "include", {})
    exclude = _get(raw, "exclude", {})

    if not isinstance(include, dict) or not isinstance(exclude, dict):
        raise ValueError("subset manifest fields include/exclude must be objects (dict)")

    # normalize values to list[str]
    def norm(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]

    include_n = {str(k): norm(v) for k, v in include.items()}
    exclude_n = {str(k): norm(v) for k, v in exclude.items()}

    return SubsetManifest(
        schema_version=schema_version,
        subset_id=subset_id,
        include=include_n,
        exclude=exclude_n,
        notes=raw.get("notes"),
        created_at=raw.get("created_at"),
    )


def parse_split_manifest(raw: Dict[str, Any]) -> SplitManifest:
    schema_version = str(raw.get("schema_version", "1.0"))
    split_id = str(raw.get("split_id", "")).strip()
    if not split_id:
        raise ValueError("split manifest missing required field: split_id")

    group_field = str(raw.get("group_field", "")).strip()
    if not group_field:
        raise ValueError("split manifest missing required field: group_field")

    def to_list(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i) for i in x]
        return [str(x)]

    train_groups = to_list(raw.get("train_groups"))
    val_groups = to_list(raw.get("val_groups"))
    test_groups = to_list(raw.get("test_groups"))

    if not train_groups and not test_groups and not val_groups:
        raise ValueError("split manifest must include at least one of train_groups/val_groups/test_groups")

    return SplitManifest(
        schema_version=schema_version,
        split_id=split_id,
        group_field=group_field,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        notes=raw.get("notes"),
        created_at=raw.get("created_at"),
    )
