from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

from .full_doc_prompting import decide_full_doc_vs_rag, decision_to_dict, read_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", type=str, required=True, help="Path to a single doc (e.g., data/kb/runbook_api_5xx.md)")
    ap.add_argument("--context-limit", type=int, default=8192)
    ap.add_argument("--overhead", type=int, default=800, help="System/prompt/tool overhead tokens")
    ap.add_argument("--out-tokens", type=int, default=500, help="Expected output token budget")
    ap.add_argument("--margin", type=float, default=0.8, help="Budget safety margin ratio")
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    ap.add_argument("--out-dir", type=str, default="artifacts/full_doc_prompting")
    ap.add_argument("--report", type=str, default="reports/full_doc_prompting_demo.md")
    args = ap.parse_args()

    text = read_text(args.doc)
    decision = decide_full_doc_vs_rag(
        text,
        context_limit=args.context_limit,
        overhead_tokens=args.overhead,
        expected_output_tokens=args.out_tokens,
        safety_margin_ratio=args.margin,
        chars_per_token=args.chars_per_token,
        doc_count=1,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decision_path = out_dir / "decision.json"
    decision_path.write_text(json.dumps(decision_to_dict(decision), indent=2), encoding="utf-8")

    # Markdown report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    md = []
    md.append(f"# Full-Document Prompting Decision Report")
    md.append("")
    md.append(f"- Generated: `{now}`")
    md.append(f"- Doc: `{args.doc}`")
    md.append("")
    md.append("## Decision")
    md.append(f"- Strategy: **{decision.strategy.upper()}**")
    md.append(f"- Fits budget: `{decision.fits_budget}`")
    md.append("")
    md.append("## Token Budget")
    md.append(f"- Context limit: `{decision.context_limit}`")
    md.append(f"- Safety margin ratio: `{decision.safety_margin_ratio}` (budget = `{int(decision.context_limit * decision.safety_margin_ratio)}`)")
    md.append(f"- Doc tokens (estimated): `{decision.doc_tokens}`")
    md.append(f"- Overhead tokens: `{decision.overhead_tokens}`")
    md.append(f"- Expected output tokens: `{decision.expected_output_tokens}`")
    md.append(f"- Total estimated: `{decision.doc_tokens + decision.overhead_tokens + decision.expected_output_tokens}`")
    md.append("")
    md.append("## Reason")
    md.append(decision.reason)
    md.append("")
    md.append("## Suggested guardrails")
    md.append("- Answer using only the document.")
    md.append("- If not found, output `NOT_FOUND`.")
    md.append("- Provide up to 3 short quotes as evidence.")
    md.append("")

    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] Wrote artifact: {decision_path}")
    print(f"[OK] Wrote report:  {report_path}")
    print(json.dumps(decision_to_dict(decision), indent=2))


if __name__ == "__main__":
    main()
