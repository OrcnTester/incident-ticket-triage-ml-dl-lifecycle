# 15 — Preprocessing for Batch Consistency (Padding, Truncation, Collate)

This card documents and demonstrates the **preprocessing step needed for deep learning** style training loops:

> token IDs (variable length) → **fixed-shape batches** (padding/truncation) + aligned labels

Even if you start with classical ML (TF‑IDF), you eventually hit this requirement once you move to:
- Embeddings + MLP
- RNN/CNN text models
- Transformers

---

## Why batch consistency exists (simple intuition)

A batch is like a stack of rectangles.
If each ticket produces a different-length sequence, you can’t stack them into one tensor unless they share the same width.

So we do one of two things:

### A) Pad to max length **in the batch** (dynamic)
- Find `max_len` among sequences in the current batch
- Pad shorter sequences with `<pad>` (usually ID 0)
- Keep an `attention_mask` so the model knows what is real vs padding

**Pros**
- less wasted compute than padding everything to a global max
- good default for many workloads

**Cons**
- batch shapes vary (can complicate caching / compilation)

### B) Pad/truncate to a **fixed** max length (static)
- Choose a global `max_len` (e.g., 128)
- Truncate longer sequences, pad shorter ones

**Pros**
- stable shapes (good for deployment + predictable latency)
- easier profiling

**Cons**
- truncation may drop important info
- padding waste if `max_len` is too big

---

## Truncation strategy

If you must truncate:
- **head** truncation keeps the beginning of the ticket (often includes system/context)
- **tail** truncation keeps the end (often includes error code/log line)

Default: keep the head.

---

## Collate (batch builder) responsibilities

A collate function should output:

- `input_ids`: shape `[batch, seq_len]`
- `attention_mask`: shape `[batch, seq_len]` (1 = real token, 0 = pad)
- `labels`: shape `[batch]` (int class indices)
- optional: `lengths` (original lengths)

---

## Label handling (alignment)

Labels must be:
- **stable integers** (e.g., `{"P0":0,"P1":1,"P2":2,"P3":3}`)
- generated once and re-used across train/val/test

Never “re-fit” label mapping separately on each split (it can reorder classes).

---

## Reproducibility note (most important ops rule)

**Tokenization + vocab must be identical across splits**.

So:
- fit vocab on TRAIN
- save `vocab.json` + `config.json`
- load them everywhere (val/test/inference)

This repo provides a demo that can either:
- **load** existing `artifacts/tokenization/vocab.json` (recommended), or
- build vocab on the fly (for quick experimentation)

---

## Demo: run + outputs

### Run (category labels)
```bash
python -m src.triage.text.collate_demo --data data/tickets.csv --label-col category --mode word --pad-to batch --batch-size 32
```

### Run (priority labels, fixed length)
```bash
python -m src.triage.text.collate_demo --data data/tickets.csv --label-col priority --mode subword --ngram-min 3 --ngram-max 5 --pad-to fixed --max-len 128
```

### Outputs
- `reports/collate_demo.md`
- `artifacts/collate/config.json`
- `artifacts/collate/label_map.json`

---

## Done checklist (paste)

- [ ] Explain padding/truncation and why batch shapes must match
- [ ] Define a collate strategy conceptually (pad to max length per batch)
- [ ] Note label handling (stable class index mapping + shape alignment)
- [ ] Add reproducibility note (same vocab/tokenization across splits)
- [ ] Provide runnable demo that writes artifacts + report
