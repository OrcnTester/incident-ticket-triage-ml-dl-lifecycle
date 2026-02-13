# 12 — Data Loading Strategies: In-Memory vs Generator/Streaming

This card documents **when to load the full dataset into memory** vs using **generator/streaming** pipelines—especially for **image-heavy** or **large-scale** ML/DL workloads.

Even though this repo starts with text-based baselines, the same principles apply to:
- large ticket corpora (millions of rows),
- deep learning tracks (Keras / PyTorch),
- any pipeline where **I/O, RAM, and reproducibility** are the hard parts.

---

## TL;DR (decision in 30 seconds)

Use **in-memory** when:
- The dataset fits comfortably in RAM (rule of thumb: **< 30–40%** of available RAM after overhead)
- Preprocessing/augmentation is light
- You want the **simplest reproducible** baseline

Use **generator/streaming** when:
- Data is too large for RAM **or** you need to scale
- You do heavy augmentation (images) and want parallelism
- You want stable throughput with prefetching and background workers

---

## 1) Comparison Table

| Strategy | Pros | Cons | Best for | Repro notes |
|---|---|---|---|---|
| **In-memory (load-all)** | simplest code, fastest random access, easy true shuffling, deterministic splits | RAM-bound, slow startup on large data, may duplicate memory during transforms | small/medium datasets, baselines, prototyping, CPU-friendly featurization | deterministic split via saved indices; shuffle with fixed seed |
| **Generator / streaming** | scales to huge datasets, low RAM, can overlap I/O+CPU+GPU, easy to add prefetch | harder reproducibility, “true shuffle” is tricky, debugging slightly harder | big datasets, image augmentation, online ingestion, production-like pipelines | use *buffered shuffle*, fixed seeds, stable sample IDs, log order windows |

---

## 2) Practical decision rules

### Rule A — “Does it fit in RAM?”
Estimate memory:
- **Text rows**: often manageable (but embeddings/tokenized tensors can explode)
- **Images**: can be huge:
  - raw float32 tensor memory ≈ `H * W * C * 4 bytes`
  - example 224×224×3 float32 ≈ 224*224*3*4 ≈ ~0.6 MB / image  
    → 100k images ≈ ~60 GB (not happening in RAM)

**If you’re near RAM limit → streaming wins**.

### Rule B — “Is augmentation expensive?”
If augmentation is heavy (image crops, color jitter, resizing):
- prefer generator pipelines with parallel workers
- use prefetch to keep GPU busy

### Rule C — “Where is the bottleneck?”
- **I/O-bound**: use caching, local SSD, compression formats (TFRecord/Parquet), prefetch
- **CPU-bound**: multiprocessing workers, vectorized transforms
- **GPU under-utilized**: increase prefetch/batch size, move decode/augment to CPU pipeline

### Rule D — “Do you need audit-grade reproducibility?”
If yes, **you must**:
- pin dataset version + checksums
- persist split indices
- log environment versions + seeds
- make shuffling deterministic (or at least bounded + traceable)

---

## 3) Success criteria (what to measure)

Minimum metrics to call a loader “good”:

### Throughput
- samples/sec (or images/sec)
- epoch wall time

### Resource caps
- peak RAM usage (target a cap, e.g. < 2–4 GB for dev runs)
- CPU utilization (avoid single-core bottleneck)
- GPU utilization (if DL)

### Determinism checks
- same split sizes and label distributions across runs
- same *first K sample IDs* per epoch (when deterministic mode is enabled)
- same metrics (within tolerance) with fixed seeds

---

## 4) Defense-grade note: reproducibility + auditability

To make a data pipeline “defensible” in real ops:

### A) Dataset identity
Record:
- dataset source path(s)
- file size + SHA256 checksums (or a manifest)
- schema version

### B) Split identity
Persist:
- train/test indices (like `split.json` in this repo)
- cutoff timestamp when using time-aware splits
- gap window used (if any)

### C) Run identity
Log:
- git commit hash
- Python + library versions
- random seeds (Python / NumPy / framework)

### D) Output identity
Store artifacts:
- metrics.json
- report.txt / report.md
- sample-level trace for a small subset (IDs + predictions + evidence)

---

## 5) Minimal code templates (copy-paste friendly)

### 5.1 PyTorch — In-memory Dataset (simple)
```python
from torch.utils.data import Dataset

class InMemoryDataset(Dataset):
    def __init__(self, items):
        self.items = items  # list of dicts or tuples

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
```

### 5.2 PyTorch — Streaming via IterableDataset (scales)
```python
import csv
from torch.utils.data import IterableDataset

class CSVTicketsStream(IterableDataset):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                yield row["text"], row.get("priority"), row.get("category")
```

**Note:** true shuffling is hard in pure streaming. Use a shuffle buffer if needed.

### 5.3 Keras / TF — tf.data (fast + prefetch)
```python
import tensorflow as tf

ds = tf.data.TextLineDataset("data/tickets.csv").skip(1)  # skip header
# parse CSV line here (or use tf.data.experimental.CsvDataset)
ds = ds.shuffle(10_000, seed=42, reshuffle_each_iteration=True)
ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
```

---

## 6) How this maps to this repository

Current baselines (scikit-learn) use small CSVs:
- in-memory loading is fine and simplest

For the DL track (future):
- start with in-memory (fast iteration, deterministic)
- move to generator/streaming once:
  - dataset grows,
  - augmentation is heavy,
  - or you want production-like behavior

---

## 7) “Done” checklist for the GitHub card

- [ ] Add this doc: `docs/12_data_loading_strategies.md`
- [ ] Decision table included (memory vs generator)
- [ ] Decision rules included (size, augmentation, IO bottleneck)
- [ ] Success criteria defined (throughput, RAM cap, determinism)
- [ ] Defense-grade note added (auditability + reproducibility)
