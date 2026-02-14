# When to Use Full-Document Prompting (No Retrieval)

This card defines **when it is reasonable to paste an entire document into a prompt** (no retrieval step), and when that approach becomes **risky / inefficient**.

In incident operations, the “document” is typically a **runbook**, **service ownership sheet**, **on-call playbook**, or a **single policy page**.

---

## Mental model

You have two ways to ground an LLM:

1. **Full‑doc prompting (no retrieval)**  
   Put the **whole document** into the prompt and ask the model to answer using only that content.

2. **RAG (retrieval + prompt)**  
   Index many docs → retrieve top‑k chunks → put only relevant chunks in the prompt.

Full‑doc prompting is **simpler** but only works when the doc is **small + clean + single source of truth**.

---

## Use full‑document prompting when… (3 conditions)

### 1) The doc is *short enough* for your context window
A practical rule:

- `doc_tokens + prompt_overhead + expected_output_tokens <= context_limit * 0.8`

Why 0.8? You keep buffer for system instructions, tool metadata, and safety margin.

**Examples (good):**
- a 1–3 page runbook
- a single ownership table
- a short policy doc

### 2) The doc is the *single source of truth*
If the question can be answered from **one authoritative doc**, full‑doc prompting avoids retrieval errors.

**Examples (good):**
- “Who owns payments-api?”
- “What are the escalation steps for API 5xx spike?”

### 3) You want *maximum simplicity / minimum infra*
Early prototypes, internal demos, or constrained environments where you **don’t want to maintain indexing**.

---

## Avoid full‑document prompting when… (3 conditions)

### 1) The doc is long (or there are many docs)
Long docs increase:
- **cost** (more input tokens)
- **latency** (bigger prompt)
- **failure risk** (truncation / context overflow)

If you’re past the 0.8 rule above → do not full‑doc.

### 2) The content is noisy or mixed
If docs contain logs, tables, repeated boilerplate, unrelated sections, or multiple topics, the model can “latch” onto wrong parts.

RAG works better because you feed only **relevant chunks**.

### 3) You need multi‑source grounding
If the answer requires combining:
- multiple runbooks
- incident history
- KB + tickets + postmortems

Full‑doc prompting breaks down. RAG is the right default.

---

## Decision rule: Full‑doc vs RAG

Use **Full‑doc prompting** if all are true:
- `doc_fits_budget == true` (token budget passes)
- `doc_count <= 1` (or 2 short docs max)
- task is **extract / summarize / answer** from that doc
- you can enforce “**answer only from provided doc**” and allow “not found”

Otherwise choose **RAG**:
- many docs
- long docs
- multi‑source questions
- need citations per chunk

---

## Acceptance criteria (what “Done” means)

- ✅ A clear rule for “fits budget” (token + overhead + output).
- ✅ A clear “use vs avoid” list (3+3 conditions).
- ✅ A simple demo that prints a recommendation (Full‑doc vs RAG) for a given file.
- ✅ Predictable cost/latency: you can estimate tokens before calling the model.
- ✅ Hallucination minimization pattern:
  - require **quotes/snippets** from the doc
  - allow “**not found**” as a valid outcome

---

## Suggested prompt guardrails (copy/paste)

**System / instruction snippet:**
- “Answer using only the document below.”
- “If the answer is not in the document, reply: `NOT_FOUND`.”
- “Provide up to 3 short supporting quotes.”

This makes full‑doc prompting behave more like an auditable ops assistant.

---

## Repo linkage

- Demo code: `src/triage/genai/full_doc_demo.py`
- Helper: `src/triage/genai/full_doc_prompting.py`
- Output:
  - `reports/full_doc_prompting_demo.md`
  - `artifacts/full_doc_prompting/decision.json`

