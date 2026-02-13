from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.triage.data.subset_loader import dataset_fingerprint, load_and_apply_subset
from src.triage.data.subset_manifest import load_json, parse_split_manifest, sha256_file
from src.triage.data.group_split import explicit_group_split, group_holdout_split


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def label_dist(y: pd.Series) -> Dict[str, Any]:
    y = y.astype(str)
    counts = y.value_counts().to_dict()
    n = int(y.shape[0])
    ratios = {k: (v / n if n else 0.0) for k, v in counts.items()}
    return {"n": n, "counts": counts, "ratios": ratios}


def write_md_report(path: Path, meta: Dict[str, Any]) -> None:
    lines = []
    lines.append("# Subset / Region Demo Report")
    lines.append("")
    lines.append(f"- subset_id: `{meta['subset']['subset_id']}`")
    lines.append(f"- manifest_sha256: `{meta['subset']['manifest_sha256']}`")
    lines.append(f"- data: `{meta['dataset']['path']}`")
    lines.append(f"- n_before: {meta['subset']['n_before']}")
    lines.append(f"- n_after: {meta['subset']['n_after']}")
    lines.append("")
    lines.append("## Split")
    s = meta["split"]["meta"]
    lines.append(f"- strategy: `{s['strategy']}`")
    lines.append(f"- group_field: `{s['group_field']}`")
    if s["strategy"] == "group_holdout":
        lines.append(f"- seed: {s['seed']}  test_size: {s['test_size']}  val_size: {s['val_size']}")
    lines.append(f"- groups: n={s.get('n_groups', 'n/a')}")
    lines.append(f"- train_groups: {s.get('train_groups')}")
    lines.append(f"- val_groups: {s.get('val_groups')}")
    lines.append(f"- test_groups: {s.get('test_groups')}")
    lines.append("")
    lines.append("## Label distributions")
    lines.append("### Category")
    lines.append(f"- train: {meta['labels']['category']['train']}")
    lines.append(f"- test:  {meta['labels']['category']['test']}")
    lines.append("")
    lines.append("### Priority")
    lines.append(f"- train: {meta['labels']['priority']['train']}")
    lines.append(f"- test:  {meta['labels']['priority']['test']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo: subset selection + group-disjoint split with traceability artifacts")
    ap.add_argument("--data", type=str, default="data/tickets.csv")
    ap.add_argument("--subset", type=str, required=True, help="Path to subset manifest JSON")
    ap.add_argument("--split", type=str, default="", help="Optional split manifest JSON (explicit group holdout)")
    ap.add_argument("--group-field", type=str, default="", help="Group field for group holdout (default: inferred)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="artifacts/subset_demo")
    ap.add_argument("--report", type=str, default="reports/subset_demo.md")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df_sub, subset_meta = load_and_apply_subset(df, args.subset)

    # choose group field
    group_field = args.group_field.strip()
    if not group_field:
        # best-effort inference: prefer system, routing_team, source
        for c in ["system", "routing_team", "source", "error_code"]:
            if c in df_sub.columns:
                group_field = c
                break
        if not group_field:
            raise SystemExit("No group field provided and none inferred. Use --group-field <column>.")

    # split
    if args.split.strip():
        split_raw = load_json(args.split)
        sm = parse_split_manifest(split_raw)
        split_res = explicit_group_split(
            df_sub,
            group_field=sm.group_field,
            train_groups=sm.train_groups,
            val_groups=sm.val_groups,
            test_groups=sm.test_groups,
        )
        split_manifest_sha = sha256_file(args.split)
        split_meta = {"split_manifest_path": args.split, "split_manifest_sha256": split_manifest_sha}
    else:
        split_res = group_holdout_split(
            df_sub,
            group_field=group_field,
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
        )
        split_meta = {"split_manifest_path": None, "split_manifest_sha256": None}

    # label distributions
    cat = "category" if "category" in df_sub.columns else None
    pr = "priority" if "priority" in df_sub.columns else None

    def dist_for(col: Optional[str]) -> Dict[str, Any]:
        if not col:
            return {"train": None, "test": None}
        y_train = df_sub.iloc[split_res.train_idx][col]
        y_test = df_sub.iloc[split_res.test_idx][col]
        return {
            "train": label_dist(y_train),
            "test": label_dist(y_test),
        }

    meta: Dict[str, Any] = {
        "dataset": dataset_fingerprint(args.data),
        "subset": subset_meta,
        "split": {
            "meta": split_res.meta,
            "train_idx_count": len(split_res.train_idx),
            "val_idx_count": len(split_res.val_idx),
            "test_idx_count": len(split_res.test_idx),
            **split_meta,
        },
        "labels": {
            "category": dist_for(cat),
            "priority": dist_for(pr),
        },
    }

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # also persist indices for downstream usage
    (out_dir / "split_indices.json").write_text(
        json.dumps(
            {"train_idx": split_res.train_idx, "val_idx": split_res.val_idx, "test_idx": split_res.test_idx},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ensure_dir(Path(args.report).parent)
    write_md_report(Path(args.report), meta)

    print(f"[OK] Wrote {out_dir / 'meta.json'} and {out_dir / 'split_indices.json'}")
    print(f"[OK] Wrote report {args.report}")


if __name__ == "__main__":
    main()
