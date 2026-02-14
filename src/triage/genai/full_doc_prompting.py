from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# NOTE: Token estimation without tiktoken.
# For English-like text, a common approximation is ~4 chars per token.
# This is imperfect but good enough to make a "budget decision" deterministic.
def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    if not text:
        return 0
    return int(math.ceil(len(text) / chars_per_token))


@dataclass(frozen=True)
class FullDocDecision:
    strategy: str  # "full_doc" or "rag"
    reason: str
    context_limit: int
    doc_tokens: int
    overhead_tokens: int
    expected_output_tokens: int
    safety_margin_ratio: float
    fits_budget: bool


def decide_full_doc_vs_rag(
    doc_text: str,
    *,
    context_limit: int = 8192,
    overhead_tokens: int = 800,
    expected_output_tokens: int = 500,
    safety_margin_ratio: float = 0.8,
    chars_per_token: float = 4.0,
    doc_count: int = 1,
) -> FullDocDecision:
    doc_tokens = estimate_tokens(doc_text, chars_per_token=chars_per_token)
    total = doc_tokens + overhead_tokens + expected_output_tokens
    budget = int(context_limit * safety_margin_ratio)
    fits_budget = total <= budget

    if fits_budget and doc_count <= 1:
        return FullDocDecision(
            strategy="full_doc",
            reason="Doc fits token budget and doc_count is small → simplest grounding is full-doc prompting.",
            context_limit=context_limit,
            doc_tokens=doc_tokens,
            overhead_tokens=overhead_tokens,
            expected_output_tokens=expected_output_tokens,
            safety_margin_ratio=safety_margin_ratio,
            fits_budget=True,
        )

    # Otherwise prefer RAG.
    why = []
    if not fits_budget:
        why.append("doc does not fit budget (would be expensive / risk truncation)")
    if doc_count > 1:
        why.append("multiple docs (better handled by retrieval)")
    reason = "Prefer RAG because " + (" and ".join(why) if why else "it is the safer default for multi-source / long docs.")

    return FullDocDecision(
        strategy="rag",
        reason=reason,
        context_limit=context_limit,
        doc_tokens=doc_tokens,
        overhead_tokens=overhead_tokens,
        expected_output_tokens=expected_output_tokens,
        safety_margin_ratio=safety_margin_ratio,
        fits_budget=fits_budget,
    )


def read_text(path: str | Path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")


def decision_to_dict(d: FullDocDecision) -> dict[str, Any]:
    return {
        "strategy": d.strategy,
        "reason": d.reason,
        "context_limit": d.context_limit,
        "doc_tokens": d.doc_tokens,
        "overhead_tokens": d.overhead_tokens,
        "expected_output_tokens": d.expected_output_tokens,
        "safety_margin_ratio": d.safety_margin_ratio,
        "fits_budget": d.fits_budget,
    }
