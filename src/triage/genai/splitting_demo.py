from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.triage.genai.text_splitters import SplitterCfg, token_split, sentence_split, recursive_split, tokenize


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_docs(kb_dir: Path) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    for p in sorted(kb_dir.rglob("*")):
        if p.is_dir():
            continue
        if p.suffix.lower() not in {".md", ".txt"}:
            continue
        doc_id = p.stem
        text = p.read_text(encoding="utf-8", errors="ignore")
        docs.append((doc_id, text))
    return docs


def _p95(vals: List[int]) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    idx = int(round(0.95 * (len(s) - 1)))
    return int(s[idx])


def _dup_ratio(chunks: List[Dict[str, Any]]) -> float:
    seen = set()
    dups = 0
    for c in chunks:
        h = hash(c.get("text", ""))
        if h in seen:
            dups += 1
        else:
            seen.add(h)
    return float(dups) / float(max(1, len(chunks)))


def _sentence_boundary_ok(text: str) -> bool:
    t = (text or "").rstrip()
    if not t:
        return False
    return t[-1] in {".", "!", "?"} or t.endswith("...")


def _summarize(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    toks = [int(c.get("n_tokens") or len(tokenize(c.get("text", "")))) for c in chunks]
    chs = [int(c.get("n_chars") or len(c.get("text", ""))) for c in chunks]
    boundary = sum(1 for c in chunks if _sentence_boundary_ok(str(c.get("text", ""))))
    return {
        "n_chunks": int(len(chunks)),
        "empty_chunks": int(sum(1 for c in chunks if not str(c.get("text", "")).strip())),
        "duplicate_ratio": float(_dup_ratio(chunks)),
        "n_tokens_min": int(min(toks) if toks else 0),
        "n_tokens_median": float(statistics.median(toks) if toks else 0),
        "n_tokens_p95": int(_p95(toks)),
        "n_tokens_max": int(max(toks) if toks else 0),
        "n_chars_min": int(min(chs) if chs else 0),
        "n_chars_median": float(statistics.median(chs) if chs else 0),
        "n_chars_p95": int(_p95(chs)),
        "n_chars_max": int(max(chs) if chs else 0),
        "sentence_boundary_ratio": float(boundary) / float(max(1, len(chunks))),
    }


def _run_strategy(name: str, docs: List[Tuple[str, str]], cfg: SplitterCfg) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for doc_id, text in docs:
        if name == "token":
            all_chunks.extend(token_split(text, doc_id=doc_id, cfg=cfg))
        elif name == "sentence":
            all_chunks.extend(sentence_split(text, doc_id=doc_id, cfg=cfg))
        elif name == "recursive":
            all_chunks.extend(recursive_split(text, doc_id=doc_id, cfg=cfg))
        else:
            raise ValueError(name)
    elapsed = time.perf_counter() - t0
    stats = _summarize(all_chunks)
    stats["elapsed_s"] = float(elapsed)
    return all_chunks, stats


def _write_report(out_md: Path, meta: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Text Splitting Demo Report")
    lines.append("")
    lines.append(f"- kb_dir: `{meta['kb_dir']}`")
    lines.append(f"- docs: **{meta['n_docs']}**")
    lines.append("")
    for s in meta["strategies"]:
        lines.append(f"## {s['name']}")
        st = s["stats"]
        lines.append(f"- n_chunks: **{st['n_chunks']}** | elapsed_s: {st['elapsed_s']:.4f}")
        lines.append(f"- duplicate_ratio: {st['duplicate_ratio']:.4f} | empty_chunks: {st['empty_chunks']}")
        lines.append(f"- token len min/median/p95/max: {st['n_tokens_min']} / {st['n_tokens_median']} / {st['n_tokens_p95']} / {st['n_tokens_max']}")
        lines.append(f"- char  len min/median/p95/max: {st['n_chars_min']} / {st['n_chars_median']} / {st['n_chars_p95']} / {st['n_chars_max']}")
        lines.append(f"- sentence_boundary_ratio: {st['sentence_boundary_ratio']:.2f}")
        lines.append("")
        # preview first 1-2 chunks
        prev = s["preview"]
        if prev:
            lines.append("Preview:")
            for i, p in enumerate(prev, 1):
                text = p["text"]
                if len(text) > 300:
                    text = text[:300] + "…"
                lines.append(f"- chunk {i} ({p['doc_id']}): `{text}`")
            lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare chunking strategies: token vs sentence vs recursive.")
    ap.add_argument("--kb-dir", type=str, default="data/kb")
    ap.add_argument("--strategies", type=str, default="all", help="all | token | sentence | recursive (comma-separated ok)")
    ap.add_argument("--chunk-size", type=int, default=200, help="tokens for token/sentence; chars for recursive")
    ap.add_argument("--overlap", type=int, default=40, help="tokens for token/sentence; chars for recursive")
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--out-json", type=str, default="artifacts/text_splitting/metrics.json")
    ap.add_argument("--report", type=str, default="reports/text_splitting_demo.md")
    args = ap.parse_args()

    kb_dir = Path(args.kb_dir)
    if not kb_dir.exists():
        raise SystemExit(f"kb-dir not found: {kb_dir}")

    docs = _read_docs(kb_dir)
    if not docs:
        raise SystemExit("No .md/.txt found under kb-dir. Add files to data/kb/.")

    wanted = [x.strip() for x in args.strategies.split(",")]
    if wanted == ["all"]:
        wanted = ["token", "sentence", "recursive"]

    out_json = Path(args.out_json)
    out_md = Path(args.report)
    _ensure_dir(out_json.parent)
    _ensure_dir(out_md.parent)

    meta: Dict[str, Any] = {"kb_dir": str(kb_dir), "n_docs": len(docs), "strategies": []}

    for name in wanted:
        cfg = SplitterCfg(chunk_size=int(args.chunk_size), overlap=int(args.overlap), min_size=int(args.min_size), normalize=True)
        chunks, stats = _run_strategy(name, docs, cfg)
        meta["strategies"].append(
            {
                "name": name,
                "cfg": cfg.__dict__,
                "stats": stats,
                "preview": chunks[:2],
            }
        )

    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(out_md, meta)

    print(f"[OK] Wrote {out_json}")
    print(f"[OK] Wrote {out_md}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
