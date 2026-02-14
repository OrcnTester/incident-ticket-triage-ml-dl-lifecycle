from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.triage.text.tokenization import TokenizerConfig, tokenize
from src.triage.text.vocab import VocabConfig, build_vocab, Vocab
from src.triage.text.encode import encode_tokens, compute_encode_stats
from src.triage.text.collate import CollateConfig, collate_batch


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_label_map(series: pd.Series) -> Dict[str, int]:
    labels = sorted(series.dropna().astype(str).unique().tolist())
    return {lab: i for i, lab in enumerate(labels)}


def chunk_indices(n: int, batch_size: int) -> List[Tuple[int, int]]:
    out = []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        out.append((start, end))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Padding/Truncation/Collate demo for batch consistency")
    ap.add_argument("--data", type=str, default="data/tickets.csv")
    ap.add_argument("--text-col", type=str, default="text")
    ap.add_argument("--label-col", type=str, default="category")

    # tokenization/vocab knobs
    ap.add_argument("--mode", type=str, choices=["word", "char", "subword"], default="word")
    ap.add_argument("--lowercase", action="store_true", default=True)
    ap.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=20000)
    ap.add_argument("--ngram-min", type=int, default=3)
    ap.add_argument("--ngram-max", type=int, default=5)

    # collate knobs
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pad-to", type=str, choices=["batch", "fixed"], default="batch")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--no-truncation", action="store_true")
    ap.add_argument("--trunc-side", type=str, choices=["head", "tail"], default="head")

    ap.add_argument("--out-dir", type=str, default="artifacts/collate")
    ap.add_argument("--report", type=str, default="reports/collate_demo.md")
    ap.add_argument("--sample-batches", type=int, default=2)

    # optional: load existing vocab (recommended for reproducibility)
    ap.add_argument("--load-vocab", type=str, default="", help="Path to vocab.json to reuse (e.g., artifacts/tokenization/vocab.json)")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.text_col not in df.columns:
        raise SystemExit(f"text column '{args.text_col}' not found: {list(df.columns)}")
    if args.label_col not in df.columns:
        raise SystemExit(f"label column '{args.label_col}' not found: {list(df.columns)}")

    texts = df[args.text_col].fillna("").astype(str).tolist()
    y_raw = df[args.label_col].fillna("").astype(str)

    tok_cfg = TokenizerConfig(
        mode=args.mode,
        lowercase=bool(args.lowercase),
        ngram_min=int(args.ngram_min),
        ngram_max=int(args.ngram_max),
    )
    tokenized = [tokenize(t, tok_cfg) for t in texts]

    if args.load_vocab:
        vocab = Vocab.load(args.load_vocab)
    else:
        vcfg = VocabConfig(min_freq=int(args.min_freq), max_size=int(args.max_size))
        vocab, _counter = build_vocab(tokenized, vcfg)

    encoded = [encode_tokens(toks, vocab, max_len=0, pad_to_max=False) for toks in tokenized]
    enc_stats = compute_encode_stats(encoded, vocab)

    label_map = build_label_map(y_raw)
    y = [label_map[lab] for lab in y_raw.tolist()]

    coll_cfg = CollateConfig(
        pad_id=vocab.pad_id,
        pad_to=str(args.pad_to),
        max_len=int(args.max_len) if str(args.pad_to) == "fixed" or not args.no_truncation else 0,
        truncation=not bool(args.no_truncation),
        truncation_side=str(args.trunc_side),
    )

    idx_chunks = chunk_indices(len(encoded), int(args.batch_size))
    batches_to_show = min(int(args.sample_batches), len(idx_chunks))

    batch_summaries = []
    for bi in range(batches_to_show):
        s, e = idx_chunks[bi]
        batch = collate_batch(encoded[s:e], y[s:e], coll_cfg)
        batch_summaries.append(
            {
                "batch_index": bi,
                "range": [s, e],
                "input_ids_shape": list(batch["input_ids"].shape),
                "attention_mask_shape": list(batch["attention_mask"].shape),
                "labels_shape": list(batch["labels"].shape),
                "seq_len": int(batch["seq_len"]),
                "lengths_min": int(min(batch["lengths"])),
                "lengths_max": int(max(batch["lengths"])),
            }
        )

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "tokenizer": tok_cfg.__dict__,
                "vocab": {"vocab_size": len(vocab.id_to_token), "loaded_from": args.load_vocab or None},
                "collate": {
                    "pad_to": args.pad_to,
                    "max_len": int(args.max_len),
                    "truncation": not bool(args.no_truncation),
                    "trunc_side": args.trunc_side,
                    "batch_size": int(args.batch_size),
                },
                "label_col": args.label_col,
                "text_col": args.text_col,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rep = []
    rep.append("# Collate / Batch Consistency Demo")
    rep.append("")
    rep.append(f"- label_col: `{args.label_col}`  text_col: `{args.text_col}`")
    rep.append(f"- tokenizer mode: `{args.mode}` (lowercase={bool(args.lowercase)})")
    if args.mode == "subword":
        rep.append(f"- subword n-grams: `{args.ngram_min}..{args.ngram_max}`")
    rep.append(f"- vocab_size: `{len(vocab.id_to_token)}`  OOV rate: `{enc_stats.oov_rate:.4f}`")
    rep.append("")
    rep.append("## Collate config")
    rep.append(f"- pad_to: `{args.pad_to}`")
    rep.append(f"- max_len: `{int(args.max_len)}`")
    rep.append(f"- truncation: `{not bool(args.no_truncation)}` (side={args.trunc_side})")
    rep.append(f"- batch_size: `{int(args.batch_size)}`")
    rep.append("")
    rep.append("## Example batch shapes")
    for bs in batch_summaries:
        rep.append(
            f"- batch {bs['batch_index']} idx[{bs['range'][0]}:{bs['range'][1]}] "
            f"input_ids={bs['input_ids_shape']} mask={bs['attention_mask_shape']} labels={bs['labels_shape']} "
            f"(seq_len={bs['seq_len']}, lengths={bs['lengths_min']}..{bs['lengths_max']})"
        )

    rep.append("")
    rep.append("## Notes")
    rep.append("- `attention_mask=1` marks real tokens; `0` marks padding.")
    rep.append("- Use **the same vocab/tokenizer artifacts across splits** to avoid train/inference mismatch.")

    report_path = Path(args.report)
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(rep), encoding="utf-8")

    print(f"[OK] Wrote {out_dir / 'config.json'} and {out_dir / 'label_map.json'}")
    print(f"[OK] Wrote report {report_path}")
    print(json.dumps({'vocab_size': len(vocab.id_to_token), 'oov_rate': enc_stats.oov_rate, 'example_batches': batch_summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
