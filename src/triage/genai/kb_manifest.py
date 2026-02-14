from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

SOURCE_TYPES = {"runbook", "kb_article", "postmortem", "ticket"}


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    path: str
    source_type: str
    title: Optional[str] = None
    service: Optional[str] = None
    owner: Optional[str] = None
    timestamp: Optional[str] = None  # ISO string
    tags: Optional[List[str]] = None


def _stable_doc_id(path: str) -> str:
    import hashlib
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:12]
    return f"doc_{h}"


def build_manifest_from_dir(kb_dir: str, *, source_type: str = "runbook") -> Dict[str, Any]:
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"Invalid source_type={source_type}. Choose from {sorted(SOURCE_TYPES)}")

    root = Path(kb_dir)
    if not root.exists():
        raise FileNotFoundError(kb_dir)

    docs: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        if p.suffix.lower() not in {".md", ".txt"}:
            continue

        rel = str(p.as_posix())
        docs.append(
            {
                "doc_id": _stable_doc_id(rel),
                "path": rel,
                "source_type": source_type,
                "title": p.stem,
                "service": None,
                "owner": None,
                "timestamp": None,
                "tags": [],
            }
        )

    return {"version": 1, "docs": docs}


def load_manifest(path: str) -> Dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "docs" not in raw or not isinstance(raw["docs"], list):
        raise ValueError("Manifest must contain a list field 'docs'.")
    return raw


def parse_docs(raw: Dict[str, Any]) -> List[SourceDoc]:
    docs: List[SourceDoc] = []
    for d in raw.get("docs", []):
        st = str(d.get("source_type", "")).strip()
        if st not in SOURCE_TYPES:
            raise ValueError(f"Invalid source_type={st}. Allowed={sorted(SOURCE_TYPES)}")

        doc_id = str(d.get("doc_id") or _stable_doc_id(str(d.get("path", "")))).strip()
        path = str(d.get("path", "")).strip()
        if not path:
            raise ValueError("Each doc entry must have 'path'.")

        docs.append(
            SourceDoc(
                doc_id=doc_id,
                path=path,
                source_type=st,
                title=d.get("title"),
                service=d.get("service"),
                owner=d.get("owner"),
                timestamp=d.get("timestamp"),
                tags=list(d.get("tags") or []),
            )
        )
    return docs


def save_manifest(raw: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")


def schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["version", "docs"],
        "properties": {
            "version": {"type": "integer"},
            "docs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["doc_id", "path", "source_type"],
                    "properties": {
                        "doc_id": {"type": "string"},
                        "path": {"type": "string"},
                        "source_type": {"type": "string", "enum": sorted(SOURCE_TYPES)},
                        "title": {"type": ["string", "null"]},
                        "service": {"type": ["string", "null"]},
                        "owner": {"type": ["string", "null"]},
                        "timestamp": {"type": ["string", "null"]},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }
