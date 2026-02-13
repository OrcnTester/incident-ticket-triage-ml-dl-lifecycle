from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.triage.data.load import load_tickets
from src.triage.genai.retrieval_stub import TfidfRetriever, RetrievedSnippet
from src.triage.genai.summarizer_contract import TicketSummary, Evidence as SumEvidence
from src.triage.genai.router_rationale_contract import (
    RoutingRationale,
    RoutingCandidate,
    Evidence as RouteEvidence,
)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _extract_key_entities(text: str) -> List[str]:
    """
    Lightweight entity extraction for demo purposes.
    """
    t = text.lower()
    entities: List[str] = []
    for kw in ["timeout", "timed out", "5xx", "500", "502", "503", "504", "latency", "db", "database", "checkout", "payment"]:
        if kw in t:
            entities.append(kw)
    # de-dup while preserving order
    seen = set()
    out = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out[:12]


def _parse_teams_from_snippets(snips: List[RetrievedSnippet]) -> List[str]:
    teams: List[str] = []
    for s in snips:
        lines = s.text.splitlines()
        for ln in lines:
            low = ln.lower().strip()
            if low.startswith("owner:"):
                team = ln.split(":", 1)[1].strip()
                if team:
                    teams.append(team)
            if low.startswith("team "):  # e.g. "Team Payments"
                team = ln.strip()
                if team:
                    teams.append(team)
    # unique
    seen = set()
    uniq = []
    for t in teams:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def build_summary_no_retrieval(ticket_text: str) -> TicketSummary:
    # Pure generation baseline (stub): no external grounding
    one_liner = _short(ticket_text.splitlines()[0] if ticket_text else "Ticket summary", 180)
    ev = [
        SumEvidence(type="ticket_text", quote=_short(ticket_text, 220)),
    ]
    s = TicketSummary(
        one_liner=one_liner,
        impact="unknown",
        suspected_service="unknown",
        severity_hint="unknown",
        key_entities=_extract_key_entities(ticket_text),
        open_questions=[
            "Which service/system is affected?",
            "What is the user/customer impact (if any)?",
        ],
        evidence=ev,
        prompt_version="rag_stub:no_retrieval:v1",
        model_id=None,
    )
    s.validate()
    return s


def build_summary_with_retrieval(ticket_text: str, snips: List[RetrievedSnippet]) -> TicketSummary:
    one_liner = _short(ticket_text.splitlines()[0] if ticket_text else "Ticket summary", 180)

    evidence = [SumEvidence(type="ticket_text", quote=_short(ticket_text, 200))]
    for sn in snips[:2]:
        evidence.append(SumEvidence(type="metadata", quote=_short(f"[{sn.doc_id}] {sn.text}", 240)))

    open_q = []
    if not snips:
        open_q.append("No relevant runbook/KB found. Which service is impacted?")
    else:
        open_q.append("Does the symptom match the retrieved runbook/ownership note?")

    s = TicketSummary(
        one_liner=one_liner,
        impact="unknown",
        suspected_service="unknown",
        severity_hint="unknown",
        key_entities=_extract_key_entities(ticket_text),
        open_questions=open_q[:6],
        evidence=evidence,
        prompt_version="rag_stub:with_retrieval:v1",
        model_id=None,
    )
    s.validate()
    return s


def build_routing_with_retrieval(snips: List[RetrievedSnippet]) -> RoutingRationale:
    teams = _parse_teams_from_snippets(snips)
    candidates: List[RoutingCandidate] = []

    # Normalize scores to 0..1 for demo confidence
    max_score = max([s.score for s in snips], default=0.0) or 1.0

    if teams:
        for i, team in enumerate(teams[:3]):
            # Use best snippet score as rough confidence
            base = snips[0].score / max_score if snips else 0.2
            conf = float(max(0.15, min(0.95, base - i * 0.1)))

            ev = []
            for sn in snips[:2]:
                ev.append(RouteEvidence(type="team_map", note=_short(f"[{sn.doc_id}] {sn.text}", 220)))

            candidates.append(
                RoutingCandidate(
                    team=team,
                    confidence=conf,
                    rationale="Suggested based on retrieved ownership/runbook hints (assistive; not auto-assign).",
                    evidence=tuple(ev),
                )
            )
    else:
        # No team info found → stay honest
        candidates.append(
            RoutingCandidate(
                team="unknown",
                confidence=0.2,
                rationale="No ownership/team evidence retrieved. Keeping routing as unknown.",
                evidence=tuple(),
            )
        )

    rr = RoutingRationale(
        recommended_teams=candidates,
        open_questions=[
            "Which service is this ticket about?",
            "Is there an internal ownership map for this service?",
        ],
        what_would_change_my_mind=[
            "A KB/runbook snippet that explicitly names the owning team.",
            "A service tag/metadata field mapping the ticket to a known component.",
        ],
        prompt_version="rag_stub:with_retrieval:v1",
        model_id=None,
    )
    rr.validate()
    return rr


def extract_runbook_hints(snips: List[RetrievedSnippet]) -> List[str]:
    hints: List[str] = []
    for sn in snips:
        for ln in sn.text.splitlines():
            line = ln.strip()
            if line.startswith("- "):
                hints.append(line[2:].strip())
    # unique + cap
    seen = set()
    uniq = []
    for h in hints:
        if h and h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq[:8]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb-dir", default="data/kb")
    ap.add_argument("--tickets", default="data/tickets.csv")
    ap.add_argument("--row", type=int, default=0, help="Which row to pick from tickets.csv after cleaning")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out-dir", default="artifacts/rag_demo")
    args = ap.parse_args()

    loaded = load_tickets(args.tickets, drop_duplicates_on_text=True)
    df = loaded.df
    if df.empty:
        raise SystemExit("No tickets found after cleaning.")

    if args.row < 0 or args.row >= len(df):
        raise SystemExit(f"--row out of range. Must be 0..{len(df)-1}")

    ticket_text = str(df.iloc[args.row][loaded.text_col])

    retriever = TfidfRetriever(args.kb_dir)
    snippets = retriever.retrieve(ticket_text, top_k=args.top_k)

    # Build both “no retrieval” and “with retrieval” outputs for comparison
    no_retrieval_summary = build_summary_no_retrieval(ticket_text)
    with_retrieval_summary = build_summary_with_retrieval(ticket_text, snippets)
    routing = build_routing_with_retrieval(snippets)
    runbook_hints = extract_runbook_hints(snippets)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(Path("reports"))

    artifact = {
        "created_at": _utc_now_iso(),
        "ticket_row": args.row,
        "ticket_text_preview": _short(ticket_text, 300),
        "retrieval": [
            {
                "doc_id": s.doc_id,
                "path": s.path,
                "score": s.score,
                "excerpt": _short(s.text, 420),
            }
            for s in snippets
        ],
        "outputs": {
            "no_retrieval": {
                "ticket_summary": no_retrieval_summary.to_dict(),
            },
            "with_retrieval": {
                "ticket_summary": with_retrieval_summary.to_dict(),
                "routing_rationale": routing.to_dict(),
                "runbook_hints": runbook_hints,
            },
        },
        "simple_metrics": {
            "citation_coverage_no_retrieval": len(no_retrieval_summary.evidence),
            "citation_coverage_with_retrieval": len(with_retrieval_summary.evidence),
            "retrieved_snippets": len(snippets),
        },
        "boundaries": {
            "assistive_only": True,
            "no_auto_assign": True,
            "unknown_allowed": True,
        },
    }

    out_path = out_dir / f"ticket_{args.row:03d}_rag.json"
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    # Human-friendly markdown report
    md_lines = []
    md_lines.append("# RAG Demo Report")
    md_lines.append(f"- created_at: {artifact['created_at']}")
    md_lines.append(f"- ticket_row: {args.row}")
    md_lines.append("")
    md_lines.append("## Retrieved snippets (top-k)")
    if snippets:
        for s in snippets:
            md_lines.append(f"### {s.doc_id} (score={s.score:.4f})")
            md_lines.append(f"- path: {s.path}")
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(_short(s.text, 900))
            md_lines.append("```")
            md_lines.append("")
    else:
        md_lines.append("_No snippets retrieved._")
        md_lines.append("")

    md_lines.append("## Output comparison")
    md_lines.append("### No retrieval (pure generation baseline)")
    md_lines.append("```json")
    md_lines.append(json.dumps(artifact["outputs"]["no_retrieval"]["ticket_summary"], indent=2, ensure_ascii=False))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("### With retrieval (grounded)")
    md_lines.append("```json")
    md_lines.append(json.dumps(artifact["outputs"]["with_retrieval"]["ticket_summary"], indent=2, ensure_ascii=False))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("### Routing rationale (assistive top-k)")
    md_lines.append("```json")
    md_lines.append(json.dumps(artifact["outputs"]["with_retrieval"]["routing_rationale"], indent=2, ensure_ascii=False))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("### Runbook hints")
    for h in runbook_hints:
        md_lines.append(f"- {h}")

    report_path = Path("reports") / "rag_demo_report.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[OK] Wrote artifact: {out_path}")
    print(f"[OK] Wrote report:  {report_path}")


if __name__ == "__main__":
    main()
