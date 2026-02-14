# KB Chunking Report

- kb_dir: `data\kb`
- manifest: `data/kb/sources_manifest.json`
- chunk_mode: `tokens`
- chunk_size: `200` | overlap: `40`
- n_docs: **3** | n_chunks: **3**

## Split leakage check
- train_docs: 2 | test_docs: 1
- doc_id intersection: **0** (must be 0)

## Chunk quality stats
- empty_chunks: 0
- duplicate_ratio: 0.0000
- token_len min/median/p95/max: 34 / 54.0 / 55 / 55
- char_len  min/median/p95/max: 243 / 317.0 / 329 / 329

## Acceptance
- ✅ doc-level split enforced (no leakage)
- ✅ chunk_id unique
- ✅ offsets present for traceability