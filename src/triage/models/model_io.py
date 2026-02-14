from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import sklearn
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class ModelBundleMeta:
    python: str
    sklearn: str
    platform: str
    notes: str


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_default_meta(notes: str = "") -> ModelBundleMeta:
    return ModelBundleMeta(
        python=".".join(map(str, sys.version_info[:3])),
        sklearn=str(sklearn.__version__),
        platform=str(platform.platform()),
        notes=str(notes),
    )


def save_bundle(out_dir: str | Path, pipe: Pipeline, meta: Dict[str, Any]) -> Path:
    out = Path(out_dir)
    ensure_dir(out)
    joblib.dump(pipe, out / "model.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_bundle(dir_path: str | Path) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """
    Returns (pipeline, meta, compatibility_report).
    """
    d = Path(dir_path)
    pipe: Pipeline = joblib.load(d / "model.joblib")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}

    compat = {
        "python_current": ".".join(map(str, sys.version_info[:3])),
        "sklearn_current": str(sklearn.__version__),
        "python_saved": meta.get("env", {}).get("python"),
        "sklearn_saved": meta.get("env", {}).get("sklearn"),
        "warnings": [],
    }

    if compat["sklearn_saved"] and compat["sklearn_saved"] != compat["sklearn_current"]:
        compat["warnings"].append(
            f"sklearn version mismatch: saved={compat['sklearn_saved']} current={compat['sklearn_current']}"
        )
    if compat["python_saved"] and compat["python_saved"] != compat["python_current"]:
        compat["warnings"].append(
            f"python version mismatch: saved={compat['python_saved']} current={compat['python_current']}"
        )

    return pipe, meta, compat
