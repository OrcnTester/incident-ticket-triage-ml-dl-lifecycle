# Full-Document Prompting Decision Report

- Generated: `2026-02-14 10:05:02Z`
- Doc: `data/kb/runbook_api_5xx.md`

## Decision
- Strategy: **FULL_DOC**
- Fits budget: `True`

## Token Budget
- Context limit: `8192`
- Safety margin ratio: `0.8` (budget = `6553`)
- Doc tokens (estimated): `86`
- Overhead tokens: `800`
- Expected output tokens: `500`
- Total estimated: `1386`

## Reason
Doc fits token budget and doc_count is small → simplest grounding is full-doc prompting.

## Suggested guardrails
- Answer using only the document.
- If not found, output `NOT_FOUND`.
- Provide up to 3 short quotes as evidence.
