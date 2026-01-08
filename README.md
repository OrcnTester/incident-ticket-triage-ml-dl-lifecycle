# Incident Ticket Triage — ML/DL Lifecycle Board

Board-first repository that documents an end-to-end ML/DL lifecycle design for incident ticket triage.
Focus: **engineering decision-making** (data strategy, baselines, deep learning tracks, evaluation, ops, and risk).

## What this is
- A structured lifecycle board + docs that capture **why** decisions are made
- A portfolio-style artifact for ML/DL system design discussions (enterprise/ops context)
- Track coverage: **Classical ML (scikit-learn)** + **Deep Learning (Keras / PyTorch)** + **GenAI risk-aware notes**

## What this is not
- Not a production incident system
- Not a tutorial that ships one final model
- Not using any confidential or proprietary data

## Problem context
Incident tickets often include:
- Unstructured text descriptions
- Limited structured signals (system, source, timestamp, error codes)

Goal:
- Classify incident categories
- Predict priority levels (P0–P3)
- Support routing decisions to the right teams

## Repo structure (initial)
- `docs/` — board spec, card templates, design notes
- `assets/` — diagrams / visuals used in docs
