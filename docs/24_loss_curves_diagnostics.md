# Loss Curves & Diagnostics for NLP Training (Text Classification)

This card documents how to **read training/validation curves** (loss + accuracy) to quickly diagnose:
- **underfitting** (model too weak / not learning)
- **overfitting** (memorizing train, failing to generalize)
- **data/label issues** (bad labels, leakage, inconsistent preprocessing)

## 1) What a “healthy” curve looks like

**Healthy training** (typical):
- `train_loss` decreases steadily.
- `val_loss` also decreases (maybe slower) and then **plateaus**.
- `train_acc` increases; `val_acc` increases and stays within a **reasonable gap** of train.

Interpretation: the model learns signal and generalizes.

## 2) Three red flags (fast triage)

### A) Loss not decreasing (underfit / training broken)
Symptoms:
- `train_loss` flat (or barely moves)
- `train_acc` ~ random baseline

Likely causes:
- wrong preprocessing / tokenization mismatch
- learning rate too low/high
- model capacity too small
- labels mis-encoded

### B) Validation gap blow-up (overfit)
Symptoms:
- `train_loss` keeps improving
- `val_loss` worsens after some epoch
- `train_acc` >> `val_acc`

Likely causes:
- too many epochs
- insufficient regularization
- too small / too clean train set (vs messy real world)

### C) Random accuracy / noisy signal (data/label issues)
Symptoms:
- `train_acc` improves slightly but `val_acc` stays near chance
- `val_loss` stays high
- lots of instability across runs

Likely causes:
- label noise, inconsistent labeling
- train/test leakage rules wrong (or “time split” not applied correctly)
- train and val distributions differ drastically

## 3) Label sanity checks (must-have)

Before blaming the model:
- **Class distribution**: do you have extreme imbalance? (P0/P1 rare in real life)
- **Label encoding**: stable mapping (same `label→index`) across splits/runs
- **Text pipeline stability**: same tokenization + vocab for train/val/test
- **Leakage**: do not split by random if “future tickets” contain patterns not allowed in training

## 4) Practical acceptance criteria (this card is “Done” when…)
- We can run a script that reads a training history (JSON/CSV) and outputs a **diagnostic report**:
  - curve trend stats
  - top red flags (if any)
  - suggested next actions
- Report includes a **small, human-readable summary**: “healthy / underfit / overfit / suspicious labels”

## 5) Default decision rules (simple + effective)

Recommended heuristics (rule of thumb):
- **Not learning**: `train_loss` drop < ~10% from start to end.
- **Overfit**: `val_loss` increases by > ~15% after its minimum *and* train keeps improving.
- **Random**: `val_acc` close to `1 / n_classes` (within a small margin) for most epochs.

These are heuristics—not proofs—meant for fast iteration speed.

## 6) Implementation
See:
- `src/triage/eval/loss_diagnostics.py` (CLI)
Outputs:
- `reports/loss_diagnostics.md`
- `artifacts/loss_diagnostics/summary.json`
