# 20 — Text Splitting Strategies | Chunk Size, Overlap, Trade‑offs

This card explains **why splitting/chunking matters for RAG**, compares common strategies, and defines **practical defaults + acceptance checks**.

## Why splitting helps (the “why”)
- **Context limits:** LLMs and embedding models have finite context windows. Splitting lets you index large documents safely.
- **Retrieval granularity:** Smaller chunks improve “needle finding” (higher precision) but can lose meaning if too small.
- **Latency & cost:** Smaller chunks = more vectors to search; larger chunks = fewer vectors but noisier matches.
- **Traceability:** Stable chunk IDs + offsets make answers auditable (“which paragraph said this?”).

## 3 splitting strategies (and when to use them)

### 1) Token-based (fixed tokens + overlap)
**How:** tokenize → slide window of `chunk_size` tokens with `overlap` tokens.  
**Pros:** simple, deterministic, model-agnostic.  
**Cons:** can cut sentences/sections in the middle → meaning break.

**Use when:** you need **determinism** and you don’t care about perfect sentence boundaries.

### 2) Sentence/semantic-ish (sentence boundary packing)
**How:** split into sentences → pack sentences until token budget.  
**Pros:** chunks end at sentence boundaries → meaning preserved better.  
**Cons:** sentence detection is heuristic; can fail on logs or bullet lists.

**Use when:** docs are narrative (runbooks, KB articles) and you want cleaner chunks.

### 3) Recursive split (separator-aware; “LangChain style”)
**How:** recursively split by separators (`\n\n`, `\n`, `. `, ` `) until pieces fit, then merge into chunks.  
**Pros:** better structure preservation (paragraphs/sections).  
**Cons:** more code; still heuristic.

**Use when:** Markdown docs with headings/paragraphs; you want “structure-respecting” chunking.

## Chunk size + overlap defaults (practical)
**Default starting point (RAG for ops docs):**
- `chunk_size ≈ 200 tokens`
- `overlap ≈ 40 tokens` (20%)
- Choose **sentence/recursive** if you want meaning-preserving boundaries.

Rule of thumb:
- If answers feel “missing context” → increase overlap (e.g., 60).
- If retrieval returns too much irrelevant text → decrease chunk_size (e.g., 120–160).
- If vector DB grows too fast → increase chunk_size (250–350) or reduce overlap.

## Acceptance checks (quality gates)
A chunking run is acceptable if:
- ✅ `doc_id` present on every chunk
- ✅ `chunk_id` unique & deterministic
- ✅ `train/test` split is **doc-level** (no leakage)
- ✅ empty chunk count = 0
- ✅ duplicate ratio is low
- ✅ length distribution sane (median/p95 not exploding)
- ✅ boundary quality reasonable (optional): most chunks end on sentence boundary for sentence/recursive split

## Outputs (what the demo should generate)
- `reports/text_splitting_demo.md` (comparison + recommended defaults)
- `artifacts/text_splitting/metrics.json` (stats per strategy)

## Repro command
```bash
python -m src.triage.genai.splitting_demo --kb-dir data/kb --strategies all --chunk-size 200 --overlap 40
```

## Done checklist (paste)
- [ ] Documented why splitting improves retrieval quality + responsiveness
- [ ] Compared 3 strategies (token, sentence, recursive) with clear “when to use”
- [ ] Defined chunk size + overlap defaults
- [ ] Implemented acceptance checks + generated a comparison report
