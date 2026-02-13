"""
Routing rationale contract for incident ticket triage.

Derived from docs/07_genai_incident_ops.md:
- Suggest top-k teams (assistive, not authoritative)
- Must include rationale + uncertainty
- Should include "what would change my mind?" questions
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple
import json


Confidence = float
EvidenceType = Literal["ticket_text", "log_snippet", "metadata", "team_map", "model_pred"]


def _check_confidence(x: float, field_name: str) -> None:
    if not isinstance(x, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    if x < 0.0 or x > 1.0:
        raise ValueError(f"{field_name} must be within [0, 1]")


@dataclass(frozen=True)
class Evidence:
    type: EvidenceType
    note: str

    def validate(self) -> None:
        if self.type not in ("ticket_text", "log_snippet", "metadata", "team_map", "model_pred"):
            raise ValueError(f"Invalid evidence.type: {self.type}")
        n = (self.note or "").strip()
        if not n:
            raise ValueError("Evidence.note must be non-empty")
        if len(n) > 220:
            raise ValueError("Evidence.note too long (max 220 chars)")


@dataclass(frozen=True)
class ModelPrediction:
    """
    Baseline model output (e.g., classical ML category/priority top-k).
    """
    label: str
    confidence: Confidence

    def validate(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("ModelPrediction.label must be a non-empty string")
        if len(self.label) > 80:
            raise ValueError("ModelPrediction.label too long (max 80 chars)")
        _check_confidence(float(self.confidence), "ModelPrediction.confidence")


@dataclass(frozen=True)
class RoutingCandidate:
    """
    A single team suggestion with a rationale (draft).
    """
    team: str
    confidence: Confidence
    rationale: str
    evidence: Tuple[Evidence, ...] = ()

    def validate(self) -> None:
        if not isinstance(self.team, str) or not self.team.strip():
            raise ValueError("RoutingCandidate.team must be non-empty")
        if len(self.team) > 80:
            raise ValueError("RoutingCandidate.team too long (max 80 chars)")

        _check_confidence(float(self.confidence), "RoutingCandidate.confidence")

        r = (self.rationale or "").strip()
        if not r:
            raise ValueError("RoutingCandidate.rationale must be non-empty")
        if len(r) > 320:
            raise ValueError("RoutingCandidate.rationale too long (max 320 chars)")

        if len(self.evidence) > 6:
            raise ValueError("Too many candidate evidence items (max 6)")
        for ev in self.evidence:
            ev.validate()


@dataclass
class RoutingRationale:
    """
    Contract for routing rationale output.

    Important boundaries:
    - This is NOT an assignment.
    - Always return top-k suggestions (1..5), prefer 3.
    """
    recommended_teams: List[RoutingCandidate]
    open_questions: List[str] = field(default_factory=list)
    what_would_change_my_mind: List[str] = field(default_factory=list)

    # Optional trace fields
    prompt_version: Optional[str] = None
    model_id: Optional[str] = None

    def validate(self) -> None:
        if not self.recommended_teams:
            raise ValueError("recommended_teams must be non-empty (top-k suggestions)")
        if len(self.recommended_teams) > 5:
            raise ValueError("recommended_teams max is 5 (top-k)")

        # Validate and (optionally) ensure sorted by confidence desc
        prev = None
        for c in self.recommended_teams:
            if not isinstance(c, RoutingCandidate):
                raise ValueError("recommended_teams must contain RoutingCandidate objects")
            c.validate()
            if prev is not None and c.confidence > prev + 1e-9:
                # Not fatal, but helps consistency.
                raise ValueError("recommended_teams should be sorted by confidence (desc)")
            prev = float(c.confidence)

        if len(self.open_questions) > 10:
            raise ValueError("Too many open_questions (max 10)")
        for q in self.open_questions:
            if not isinstance(q, str) or not q.strip():
                raise ValueError("open_questions must be non-empty strings")
            if len(q) > 180:
                raise ValueError("open_question too long (max 180 chars)")

        if len(self.what_would_change_my_mind) > 10:
            raise ValueError("Too many what_would_change_my_mind items (max 10)")
        for w in self.what_would_change_my_mind:
            if not isinstance(w, str) or not w.strip():
                raise ValueError("what_would_change_my_mind items must be non-empty strings")
            if len(w) > 180:
                raise ValueError("what_would_change_my_mind item too long (max 180 chars)")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingRationale":
        candidates: List[RoutingCandidate] = []
        for raw in (data.get("recommended_teams") or []):
            evs = tuple(Evidence(type=e["type"], note=e["note"]) for e in (raw.get("evidence") or []))
            candidates.append(
                RoutingCandidate(
                    team=raw["team"],
                    confidence=float(raw["confidence"]),
                    rationale=raw["rationale"],
                    evidence=evs,
                )
            )

        obj = cls(
            recommended_teams=candidates,
            open_questions=list(data.get("open_questions") or []),
            what_would_change_my_mind=list(data.get("what_would_change_my_mind") or []),
            prompt_version=data.get("prompt_version"),
            model_id=data.get("model_id"),
        )
        obj.validate()
        return obj

    @classmethod
    def from_json(cls, s: str) -> "RoutingRationale":
        return cls.from_dict(json.loads(s))

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["recommended_teams", "open_questions", "what_would_change_my_mind"],
            "properties": {
                "recommended_teams": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "required": ["team", "confidence", "rationale", "evidence"],
                        "properties": {
                            "team": {"type": "string", "maxLength": 80},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "rationale": {"type": "string", "maxLength": 320},
                            "evidence": {
                                "type": "array",
                                "maxItems": 6,
                                "items": {
                                    "type": "object",
                                    "required": ["type", "note"],
                                    "properties": {
                                        "type": {"type": "string", "enum": ["ticket_text", "log_snippet", "metadata", "team_map", "model_pred"]},
                                        "note": {"type": "string", "maxLength": 220},
                                    },
                                },
                            },
                        },
                    },
                },
                "open_questions": {"type": "array", "maxItems": 10, "items": {"type": "string", "maxLength": 180}},
                "what_would_change_my_mind": {"type": "array", "maxItems": 10, "items": {"type": "string", "maxLength": 180}},
                "prompt_version": {"type": ["string", "null"]},
                "model_id": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        }
