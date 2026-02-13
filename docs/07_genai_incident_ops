# Generative AI in Incident Ops: Where It Actually Helps

This doc maps high-value Generative AI (GenAI) use cases to incident workflows **without over-claiming autonomy**.
Principle: **assistive, not authoritative** (human-in-the-loop).

---

## Why GenAI is a fit (and where it isn’t)

### Good fit
- Unstructured text: ticket descriptions, summaries, stakeholder updates
- High cognitive load: extracting signal from noisy context
- Repetitive comms: status updates, templated responses
- Knowledge navigation: runbooks, SOPs, playbooks (retrieval + synthesis)

### Poor fit (avoid / guard heavily)
- Fully automated incident actions (deployments, restarts, config changes)
- “Single-shot truth”: making definitive claims without evidence
- Handling sensitive data without redaction / policy controls

---

## Incident workflow mapping (5 real-world GenAI applications)

| Incident Stage | GenAI Use Case | Output | Guardrails (must) | Success Signals |
|---|---|---|---|---|
| Intake / Triage | **Ticket summarization & normalization** | concise summary + key entities | cite sources; allow “unknown”; length cap | time-to-triage ↓, analyst satisfaction ↑ |
| Triage | **Categorization support** | top-k category suggestions + rationale | top-k not top-1; confidence; never overwrite labels | category correction rate ↓ |
| Investigation | **Runbook drafting / retrieval** | next-steps list from runbooks | retrieval-only grounding; show links/sections | time-to-mitigation ↓, fewer dead-ends |
| Communication | **Response suggestions** | stakeholder update draft | templated; no speculation; approvals required | comms latency ↓, clarity ↑ |
| Post-incident | **Timeline + RCA assistant** | event timeline + candidate causes | evidence tagging; “hypothesis” labeling | postmortem time ↓, action quality ↑ |

> Note: “Content / Code / Image synthesis” mapping:
> - Content: summaries, updates, RCA drafts
> - Code: runbook snippets, query drafts (e.g., log filters), automation suggestions (NOT execution)
> - Image: optional—diagrams of incident timeline / architecture (portfolio only)

---

## Triage-specific deep dives (pick 2)

### Use case A — Ticket Summarization (high ROI)
**Input**
- `title`, `description`, `system/service`, `source`, `timestamp`
- optional: small log/error snippet (redacted)

**Output (recommended schema)**
```json
{
  "one_liner": "string",
  "impact": "customer-facing | internal | unknown",
  "suspected_service": "string | unknown",
  "severity_hint": "P0 | P1 | P2 | P3 | unknown",
  "key_entities": ["error_code", "endpoint", "region", "host", "version"],
  "open_questions": ["string", "string"],
  "evidence": [
    {"type": "ticket_text|log_snippet|metadata", "quote": "short excerpt"}
  ]
}
```

**Boundaries**
- Must not invent facts not present in the input
- If signal is insufficient: output `unknown` + ask clarifying questions
- Never assign ownership / priority automatically

**Evaluation ideas**
- Human rating (helpfulness, factuality, completeness)
- Factual consistency checks (does evidence quote support the summary?)
- Time saved per ticket (self-reported or measured)

---

### Use case B — Routing Rationale (reduces handoffs)
**Input**
- Baseline model predictions: `category_topk`, `priority_topk` + confidences
- Ticket metadata: `service`, `source`, `region`, `time`
- Team map: `team -> owned services/domains`, on-call rotation (if available)

**Output**
- Recommended teams (top-3) + rationale for each
- Explicit uncertainty when ambiguous
- “What would change my mind?” questions

**Boundaries**
- Never auto-assign; only suggest
- Must show top-k options and confidence
- Must not claim “Team X owns this” unless grounded in the team map

**Evaluation ideas**
- Reassignment rate ↓
- Escalation accuracy ↑
- First-touch resolution ↑
- Mean time to correct owner ↓

---

## Guardrails: assistive, not authoritative

### Operational guardrails
- **Human-in-the-loop**: AI suggestions are drafts; user confirms actions
- **No system-of-record writes**: no auto-updates to Jira/ServiceNow/etc.
- **Auditability**: log inputs, prompt version, output, and user decision
- **Fallback**: if confidence low → show top-k + questions, not a decision

### Safety / security guardrails
- **Redaction**: scrub secrets/PII before model input
- **Policy filters**: block disallowed content / sensitive leakage
- **Prompt hardening**: instruct “don’t guess”; prefer “unknown”
- **Data boundaries**: only non-confidential sample data for this repo

---

## Success metrics (make it measurable)

### Primary (business/ops)
- **Time-to-triage** (median/p95) ↓
- **Escalation accuracy** ↑ (correct team on first handoff)
- **Priority accuracy** ↑ (alignment with human decisions / outcomes)
- **Analyst load** ↓ (time per ticket, context switching)

### Secondary (quality)
- Summary factuality score ↑
- Reopen rate ↓
- On-call interruptions ↓
- Stakeholder update clarity ↑ (human scoring / template adherence)

### Risk metrics (must track)
- Hallucination rate (outputs not supported by evidence)
- Sensitive data leakage incidents (should be 0)
- Overconfidence rate (high confidence but wrong)

---

## Minimal implementation footprint (portfolio-friendly)

This repo does NOT ship production automation. A “good enough” demo approach:
- Keep GenAI modules as **interfaces** + **stubs** (or local/offline model optional)
- Store outputs as artifacts under `artifacts/` and a report under `reports/`
- Demonstrate evaluation via:
  - human-labeled sample set (small)
  - measurable latency & correctness proxies

---

## Appendix — Example prompt constraints (conceptual)
- “Only use provided text; if unsure say ‘unknown’.”
- “Return JSON; include evidence quotes.”
- “Provide top-3 options; never pick a single team as final.”
