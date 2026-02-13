# 09 — RAG for Incident Triage: Why Retrieval Beats Pure Generation

This doc explains **why Retrieval-Augmented Generation (RAG)** is often the safest and most useful GenAI pattern for incident operations:
it grounds outputs in **retrieved sources** (runbooks, KB articles, past incidents), reducing hallucinations and increasing traceability.

---

## 1) RAG in 2–3 sentences (simple definition)

**RAG = retrieve → augment context → generate.**  
Before generating an answer, the assistant **retrieves** the most relevant snippets from trusted knowledge (runbooks/KB/past incident notes),
**injects** them into the context, and then **generates** a response that is explicitly grounded in those sources.

Why it matters for incident ops: incidents are high-stakes and time-sensitive, so we prefer **answers with citations** over confident guesses.

---

## 2) Why retrieval beats pure generation in incident workflows

Pure generation can:
- produce plausible-but-wrong steps (hallucinations)
- hide uncertainty (overconfidence)
- fail silently when knowledge is missing or outdated

Retrieval helps because:
- it turns “I think” into “I can point to the source”
- it makes uncertainty explicit (no relevant docs → “unknown” + questions)
- it enables auditability (what was retrieved, from where, and why)

---

## 3) 3 triage use cases (high value, low risk)

### Use case A — Ticket summarization (grounded)
**Goal:** create a short, ops-friendly summary without inventing facts.  
**RAG contribution:** retrieve similar past incidents + relevant runbook section to anchor terminology and likely root causes.

**Expected output shape:** summary + key entities + open questions + **evidence quotes**.

---

### Use case B — Suggested routing rationale (assistive top-k)
**Goal:** propose **top 3 likely teams** with reasons, not an auto-assignment.  
**RAG contribution:** retrieve a “service ownership / on-call map” snippet or KB entry linking symptoms to service/team.

**Expected output shape:** top-k teams + confidence + rationale + retrieved source notes.

---

### Use case C — Runbook hints (next-steps suggestions)
**Goal:** propose the first 3 investigation steps fast.  
**RAG contribution:** retrieve a runbook section matching error codes, service name, or symptoms.

**Expected output shape:** 3–5 suggested steps + citations + “stop conditions” (when to escalate).

---

## 4) Boundaries (assistive, cite sources, human-in-the-loop)

**Non-negotiable boundaries**
- **Assistive only:** suggestions, not authoritative actions
- **No system-of-record writes:** no auto-updates to Jira/ServiceNow; no auto-assign
- **Cite sources:** retrieved snippets must be referenced (short quotes / doc ids)
- **Unknown is allowed:** when retrieval returns weak/no evidence, output `unknown` + clarify questions
- **Human-in-the-loop:** analyst approves, edits, and decides

**Security boundaries**
- redact secrets/PII before indexing or querying
- restrict retrieval to approved knowledge sources
- log retrieval + outputs for auditability

---

## 5) Success metrics (how we know it helped)

### Primary ops metrics
- **Time-to-triage** (median/p95) ↓
- **Escalation accuracy** ↑ (right team on first handoff)
- **Resolution consistency** ↑ (fewer contradictory suggestions)

### Quality metrics
- **Citation coverage** ↑ (percent of claims supported by retrieved snippets)
- **Factuality score** ↑ (human-rated or rule-based checks)

### Risk metrics (must track)
- **Unsupported claims rate** (statements without evidence)
- **Hallucination rate** (outputs contradicting retrieved sources)
- **Sensitive data leakage** incidents (should be 0)

---

## 6) Minimal implementation plan (portfolio-friendly)

This repo does **not** ship production RAG. A minimal and safe demo path:

### Step 1 — Retrieval stub (no LLM required)
- Build a tiny knowledge base (3–10 docs) under `data/kb/` (markdown or txt)
- Use a simple retriever (TF-IDF or BM25) to return top-k snippets

### Step 2 — Contract-driven outputs
- Feed retrieved snippets into existing **GenAI output contracts**
- Always include a `sources` list + short evidence quotes

### Step 3 — Artifact generation
- Save retrieved snippets and outputs to `artifacts/rag_demo/`
- Add a report that compares:
  - “no retrieval” baseline vs “with retrieval” grounded output (qualitative + simple metrics)

---

## 7) Where this connects in the repo

- GenAI contracts: `src/triage/genai/*_contract.py`
- Future retrieval stub: `src/triage/genai/retrieval_stub.py` (planned)
- Demo artifacts: `artifacts/rag_demo/` (planned)

This keeps GenAI **operational**: predictable outputs, grounded evidence, and measurable success.
