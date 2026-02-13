from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from src.triage.text.tokenization import TokenizerConfig, tokenize
from src.triage.text.vocab import VocabConfig, build_vocab
from src.triage.text.encode import encode_tokens, compute_encode_stats


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tokenization + vocab + IDs demo for incident triage text")
    ap.add_argument("--data", type=str, default="data/tickets.csv")
    ap.add_argument("--text-col", type=str, default="text")

    ap.add_argument("--mode", type=str, choices=["word", "char", "subword"], default="word")
    ap.add_argument("--lowercase", action="store_true", default=True)
    ap.add_argument("--no-lowercase", dest="lowercase", action="store_false")

    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=20000)

    ap.add_argument("--ngram-min", type=int, default=3)
    ap.add_argument("--ngram-max", type=int, default=5)

    ap.add_argument("--max-len", type=int, default=0, help="Optional truncation length for encoding (0 = no truncation)")
    ap.add_argument("--pad", action="store_true", help="If set with --max-len, pad sequences to max_len")

    ap.add_argument("--out-dir", type=str, default="artifacts/tokenization")
    ap.add_argument("--report", type=str, default="reports/tokenization_demo.md")
    ap.add_argument("--sample", type=int, default=3, help="How many sample texts to include in the report")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.text_col not in df.columns:
        raise SystemExit(f"text column '{args.text_col}' not found in dataset columns: {list(df.columns)}")

    cfg_tok = TokenizerConfig(
        mode=args.mode,
        lowercase=bool(args.lowercase),
        ngram_min=int(args.ngram_min),
        ngram_max=int(args.ngram_max),
    )

    # tokenize all texts (small dataset); for big datasets you'd stream
    texts = df[args.text_col].fillna("").astype(str).tolist()
    tokenized: List[List[str]] = [tokenize(t, cfg_tok) for t in texts]

    cfg_vocab = VocabConfig(min_freq=int(args.min_freq), max_size=int(args.max_size))
    vocab, counter = build_vocab(tokenized, cfg_vocab)

    encoded = [encode_tokens(toks, vocab, max_len=int(args.max_len), pad_to_max=bool(args.pad)) for toks in tokenized]
    stats = compute_encode_stats(encoded, vocab)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    (out_dir / "vocab.json").write_text(json.dumps(vocab.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "tokenizer": cfg_tok.__dict__,
                "vocab": cfg_vocab.__dict__,
                "max_len": int(args.max_len),
                "pad": bool(args.pad),
                "text_col": args.text_col,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Build a short report
    rep = []
    rep.append("# Tokenization Demo Report")
    rep.append("")
    rep.append(f"- mode: `{cfg_tok.mode}`  lowercase: `{cfg_tok.lowercase}`")
    if cfg_tok.mode == "subword":
        rep.append(f"- subword n-grams: `{cfg_tok.ngram_min}..{cfg_tok.ngram_max}`")
    rep.append(f"- vocab_size: `{len(vocab.id_to_token)}`  (min_freq={cfg_vocab.min_freq}, max_size={cfg_vocab.max_size})")
    rep.append(f"- OOV rate: `{stats.oov_rate:.4f}`  (unk={stats.n_unk_total} / tokens={stats.n_tokens_total})")
    rep.append("")
    rep.append("## Sequence length stats (tokens per text)")
    rep.append(f"- min: {stats.lengths['min']}")
    rep.append(f"- median: {stats.lengths['median']}")
    rep.append(f"- p95: {stats.lengths['p95']}")
    rep.append(f"- max: {stats.lengths['max']}")
    rep.append("")

    rep.append("## Samples (text → tokens → ids)")
    k = max(0, int(args.sample))
    for i in range(min(k, len(texts))):
        t = texts[i]
        toks = tokenized[i][:50]
        ids = encoded[i][:50]
        rep.append(f"### Sample {i}")
        rep.append("")
        rep.append("**Text**")
        rep.append(f"`{t[:300]}`")
        rep.append("")
        rep.append("**Tokens (first 50)**")
        rep.append("```")
        rep.append(str(toks))
        rep.append("```")
        rep.append("**IDs (first 50)**")
        rep.append("```")
        rep.append(str(ids))
        rep.append("```")

    report_path = Path(args.report)
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(rep), encoding="utf-8")

    print(f"[OK] Wrote {out_dir / 'vocab.json'} and {out_dir / 'config.json'}")
    print(f"[OK] Wrote report {report_path}")
    print(json.dumps({'vocab_size': len(vocab.id_to_token), 'oov_rate': stats.oov_rate, 'lengths': stats.lengths}, indent=2))


if __name__ == "__main__":
    main()
