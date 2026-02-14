# 19 — Knowledge Sources & Chunking Strategy (RAG‑Ready Data)

This card defines **what external knowledge will be indexed** for RAG (runbooks, KB articles, past incidents) and **how to chunk it** so retrieval stays high‑precision while remaining traceable and leakage‑safe.

> RAG kalitesi çoğunlukla **data + chunking + metadata** işidir.
> Chunk çok büyük olursa retrieval gürültü yapar, çok küçük olursa anlam kırılır.

---

## 1) Target sources (what we index)

Recommended sources:

1. **Runbooks / SOPs** (Markdown / text)
2. **KB articles** (Markdown / text)
3. **Postmortems / RCAs** (Markdown / text)

Optional later (policy required):
- ticket history, chat transcripts

---

## 2) Metadata to keep (minimum viable)

Per-document (applies to all its chunks):
- `source_type`: runbook | kb_article | postmortem | ticket
- `doc_id`: stable ID
- `path`: relative file path
- `title`: optional
- `service`: payments/auth/gateway…
- `owner`: team/oncall owner
- `timestamp`: last updated (ISO)
- `tags`: list

Per-chunk:
- `chunk_id`: stable hash of (doc_id + start/end)
- `start`, `end`: offsets (token veya char)
- `n_chars`, `n_tokens`
- `split`: train/test

Traceability rule: **Every retrieved chunk must point back to a doc + offset range.**

---

## 3) Chunking policy (size + overlap)

Default:
- `mode=tokens`
- `chunk_size=200`
- `overlap=40`

Alternatives:
- `mode=chars` (e.g., 1200 with 200 overlap)
- later: heading‑aware chunking for Markdown

---

## 4) Split rules (avoid leakage)

**Document-level split**:
- a document’s chunks must stay together (no chunk-level random split)
- acceptance: `doc_ids_train ∩ doc_ids_test == ∅`

---

## 5) Acceptance criteria (quality checks)

- ✅ `doc_id` present for every chunk
- ✅ `chunk_id` unique
- ✅ no empty chunks
- ✅ length stats (median / p95) sane
- ✅ duplicate chunk ratio low
- ✅ doc leakage check passes

Outputs:
- `artifacts/kb_chunks/train.jsonl`
- `artifacts/kb_chunks/test.jsonl`
- `artifacts/kb_chunks/meta.json`
- `reports/kb_chunking_report.md`

---

## 6) Repro commands

### A) Build chunks from `data/kb/` (auto-manifest)
```bash
python -m src.triage.genai.kb_chunk_demo --kb-dir data/kb --chunk-mode tokens --chunk-size 200 --overlap 40
```

### B) Use explicit manifest (recommended)
```bash
python -m src.triage.genai.kb_chunk_demo --sources-manifest data/kb/sources_manifest.json --chunk-mode tokens --chunk-size 200 --overlap 40
```

---

## Done checklist (paste)

- [ ] Defined indexable sources + minimum metadata fields
- [ ] Implemented deterministic chunking (size + overlap) with traceability offsets
- [ ] Implemented doc-level split (no leakage) and verified disjoint doc_ids between train/test
- [ ] Added acceptance checks and generated a markdown report
