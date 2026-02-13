# 14 — Tokenization & Vocabulary Building: From Text to IDs

This card explains **how raw incident text becomes numeric model input**:

**raw text → tokens → vocabulary → token IDs → sequences (length stats)**

It also provides a **small runnable demo** that:
- tokenizes `data/tickets.csv`
- builds a vocab with `min_freq` / `max_size`
- encodes texts to token IDs
- writes a short report with **sequence length + OOV stats**

---

## Why this matters (incident triage context)

Incident tickets contain:
- human-written sentences
- product/system names
- **error codes** (`503`, `OOM`, `AUTH_INVALID_TOKEN`)
- paths / endpoints (`/api/payments/charge`)
- timestamps / IDs

Models cannot learn from raw strings. Tokenization is the “translator” that converts text into stable symbols.

---

## 1) Tokenization options (and when to use each)

### A) Word tokenization
**What it does:** splits text into word-like chunks.

**Best for**
- classical ML baselines (TF‑IDF + Logistic Regression)
- interpretable features
- clean, stable language

**Pros**
- simple, fast
- interpretable (you can inspect top tokens per class)

**Cons**
- OOV is common for rare IDs/codes
- spelling variations create many unique tokens

---

### B) Character tokenization
**What it does:** treats each character as a token.

**Best for**
- noisy text (typos, weird formats)
- heavy use of codes/IDs where character patterns matter
- languages / domains with many rare words

**Pros**
- almost no OOV
- robust to misspellings and new IDs

**Cons**
- sequences are long (slower models)
- loses word-level semantics unless model is strong

---

### C) Subword tokenization (BPE / WordPiece / Unigram)
**What it does:** splits words into common pieces (e.g., `authoriz` + `ation`).

**Best for**
- deep learning (Transformers, LSTMs)
- mixed natural language + technical strings
- keeping vocab size bounded while reducing OOV

**Pros**
- low OOV
- better trade-off between word and char

**Cons**
- needs training a tokenizer (extra step + config)
- requires consistent preprocessing everywhere

> **In this repo’s demo code** we implement a “classical ML friendly” **subword approximation**:
**character n‑grams (fastText-style)**. It’s a practical middle ground for non-transformer pipelines.

---

## 2) Vocabulary building (vocab → indices)

Vocabulary is a mapping:
- token → integer ID

We control it with:

### `min_freq`
Drop tokens that appear fewer than `min_freq` times.
- reduces noise and accidental one-off IDs

### `max_size`
Keep only the top‑N most frequent tokens.
- bounds memory and model size

### Special tokens
Common choices:
- `<pad>` = 0 (padding)
- `<unk>` = 1 (unknown token)

**Operational rule:** fit vocab on **train only** (never peek at test tokens).

---

## 3) Outputs you should document

When you run tokenization + vocab, you should record:

- `vocab_size`
- `oov_rate` (what fraction of tokens became `<unk>`)
- sequence length stats:
  - min / median / p95 / max
- 1–2 sample encodings (text → tokens → ids)

These are “pipeline health metrics.” They predict training stability and latency.

---

## 4) Two common failure modes (and fixes)

### Failure mode #1: OOV explosion
**Symptoms**
- many tokens map to `<unk>`
- model loses useful signal (error codes, new endpoints)

**Fixes**
- switch to subword (BPE / char n‑grams)
- increase `max_size`
- normalize tokens (lowercase, unify digits)
- treat codes specially (keep `503` as token)

---

### Failure mode #2: Inconsistent preprocessing
**Symptoms**
- train and inference tokenize differently
- evaluation looks fine, production fails

**Fixes**
- centralize tokenization in one module
- save tokenizer config + vocab artifact
- add a unit test: “same input → same token IDs”

---

## 5) Demo (repo-ready)

### Files (suggested)
- `src/triage/text/tokenization.py`
- `src/triage/text/vocab.py`
- `src/triage/text/encode.py`
- `src/triage/text/tokenize_demo.py`

### Run
```bash
python -m src.triage.text.tokenize_demo --data data/tickets.csv --mode word --min-freq 2 --max-size 20000
python -m src.triage.text.tokenize_demo --data data/tickets.csv --mode char
python -m src.triage.text.tokenize_demo --data data/tickets.csv --mode subword --ngram-min 3 --ngram-max 5
```

### Outputs
- `artifacts/tokenization/vocab.json`
- `artifacts/tokenization/config.json`
- `reports/tokenization_demo.md`

---

## “Done” checklist (paste)

- [ ] Define tokenization options (word / subword / char) + when to use each
- [ ] Define vocab building knobs (`min_freq`, `max_size`) + special tokens
- [ ] Document outputs: token IDs + sequence length stats + OOV rate
- [ ] Add two failure modes (OOV explosion, inconsistent preprocessing) + mitigations
- [ ] Provide a runnable demo that writes artifacts + report
