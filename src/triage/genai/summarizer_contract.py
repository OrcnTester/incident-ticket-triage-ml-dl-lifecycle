"""
Summarization contract for incident ticket triage.

Derived from docs/07_genai_incident_ops.md:
- Output schema with evidence quotes
- Boundaries: do not guess, prefer 'unknown'
- Assistive: no authoritative actions
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional
import json
import re


Impact = Literal["customer-facing", "internal", "unknown"]
SeverityHint = Literal["P0", "P1", "P2", "P3", "unknown"]
EvidenceType = Literal["ticket_text", "log_snippet", "metadata"]


def _clamp_text(s: str, max_len: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _is_probably_secret(text: str) -> bool:
    """
    Very light heuristic. We don't "redact" automatically here (contract layer),
    but we can block obvious secrets from being stored in evidence quotes.
    """
    patterns = [
        r"AKIA[0-9A-Z]{16}",  # AWS access key id (common)
        r"(?i)secret\s*=\s*['\"][^'\"]{8,}['\"]",
        r"(?i)api[_-]?key\s*[:=]\s*['\"][^'\"]{8,}['\"]",
        r"(?i)password\s*[:=]\s*['\"][^'\"]{6,}['\"]",
        r"-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----",
    ]
    return any(re.search(p, text) for p in patterns)


@dataclass(frozen=True)
class Evidence:
    """
    Evidence quote used to ground the summary.
    Keep quotes SHORT. No secrets, no PII.
    """
    type: EvidenceType
    quote: str

    def validate(self) -> None:
        if self.type not in ("ticket_text", "log_snippet", "metadata"):
            raise ValueError(f"Invalid evidence.type: {self.type}")

        q = (self.quote or "").strip()
        if not q:
            raise ValueError("Evidence.quote must be non-empty")

        # Keep quotes short to reduce leakage risk
        if len(q) > 240:
            raise ValueError("Evidence.quote too long (max 240 chars)")

        if _is_probably_secret(q):
            raise ValueError("Evidence.quote appears to contain a secret; redact before storing.")


@dataclass
class TicketSummary:
    """
    Contract for triage summarization.

    Key design choices:
    - 'unknown' is a valid output for low-signal tickets
    - evidence[] provides auditability (short quotes only)
    - this object is a *draft*, not a system-of-record update
    """
    one_liner: str
    impact: Impact = "unknown"
    suspected_service: str = "unknown"
    severity_hint: SeverityHint = "unknown"
    key_entities: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)

    # Optional metadata (useful for traceability in artifacts/reports)
    prompt_version: Optional[str] = None
    model_id: Optional[str] = None

    def validate(self) -> None:
        ol = (self.one_liner or "").strip()
        if not ol:
            raise ValueError("one_liner must be non-empty")
        if len(ol) > 200:
            raise ValueError("one_liner too long (max 200 chars)")

        if self.impact not in ("customer-facing", "internal", "unknown"):
            raise ValueError(f"Invalid impact: {self.impact}")

        if self.severity_hint not in ("P0", "P1", "P2", "P3", "unknown"):
            raise ValueError(f"Invalid severity_hint: {self.severity_hint}")

        svc = (self.suspected_service or "").strip() or "unknown"
        self.suspected_service = _clamp_text(svc, 80)

        # Entities: keep small & clean
        if len(self.key_entities) > 20:
            raise ValueError("Too many key_entities (max 20)")
        for e in self.key_entities:
            if not isinstance(e, str) or not e.strip():
                raise ValueError("key_entities must be non-empty strings")
            if len(e) > 40:
                raise ValueError("key_entity too long (max 40 chars)")

        # Open questions: short list, short text
        if len(self.open_questions) > 8:
            raise ValueError("Too many open_questions (max 8)")
        for q in self.open_questions:
            if not isinstance(q, str) or not q.strip():
                raise ValueError("open_questions must be non-empty strings")
            if len(q) > 160:
                raise ValueError("open_question too long (max 160 chars)")

        # Evidence: 0..6 items recommended
        if len(self.evidence) > 6:
            raise ValueError("Too many evidence items (max 6)")
        for ev in self.evidence:
            if not isinstance(ev, Evidence):
                raise ValueError("evidence must contain Evidence objects")
            ev.validate()

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        d = asdict(self)
        # dataclasses -> dict already converts Evidence into dicts
        return d

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TicketSummary":
        evidence_items = []
        for raw in (data.get("evidence") or []):
            evidence_items.append(Evidence(type=raw["type"], quote=raw["quote"]))

        obj = cls(
            one_liner=data.get("one_liner", ""),
            impact=data.get("impact", "unknown"),
            suspected_service=data.get("suspected_service", "unknown"),
            severity_hint=data.get("severity_hint", "unknown"),
            key_entities=list(data.get("key_entities") or []),
            open_questions=list(data.get("open_questions") or []),
            evidence=evidence_items,
            prompt_version=data.get("prompt_version"),
            model_id=data.get("model_id"),
        )
        obj.validate()
        return obj

    @classmethod
    def from_json(cls, s: str) -> "TicketSummary":
        return cls.from_dict(json.loads(s))

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        """
        JSON Schema for tooling, docs, or future validators.
        """
        return {
            "type": "object",
            "required": ["one_liner", "impact", "suspected_service", "severity_hint", "key_entities", "open_questions", "evidence"],
            "properties": {
                "one_liner": {"type": "string", "maxLength": 200},
                "impact": {"type": "string", "enum": ["customer-facing", "internal", "unknown"]},
                "suspected_service": {"type": "string", "maxLength": 80},
                "severity_hint": {"type": "string", "enum": ["P0", "P1", "P2", "P3", "unknown"]},
                "key_entities": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 20},
                "open_questions": {"type": "array", "items": {"type": "string", "maxLength": 160}, "maxItems": 8},
                "evidence": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {
                        "type": "object",
                        "required": ["type", "quote"],
                        "properties": {
                            "type": {"type": "string", "enum": ["ticket_text", "log_snippet", "metadata"]},
                            "quote": {"type": "string", "maxLength": 240},
                        },
                    },
                },
                "prompt_version": {"type": ["string", "null"]},
                "model_id": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        }
