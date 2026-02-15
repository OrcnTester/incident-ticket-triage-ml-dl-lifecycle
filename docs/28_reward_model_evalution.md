# 27 — Reward Model Evaluation

## Objective

Evaluate how well the reward model separates high-quality vs low-quality responses.

Focus:
- Pairwise ranking accuracy
- Agreement metrics
- Failure mode analysis
- Qualitative disagreement audit

---

## 1. Core Metric — Pairwise Accuracy

Given response pairs (A, B):

If human prefers A over B,
reward_model_score(A) > reward_model_score(B)

Metric:

Pairwise Accuracy = correct_rankings / total_pairs

---

## 2. Ranking Agreement Metrics

- Pairwise Accuracy
- Spearman correlation
- Kendall tau (optional)

---

## 3. Failure Mode Analysis

Inspect top disagreements:

Common failure types:
- Verbosity bias (longer answers win)
- Style over substance
- Confident but incorrect
- Refusal when answer required
- Over-safe behavior

Log top 20 largest score disagreements.

---

## 4. Acceptance Criteria

- Pairwise accuracy ≥ 0.70
- Clear separation between preferred/non-preferred score distributions
- Top disagreements manually audited
- Failure modes documented

---

## 5. Outputs

Artifacts:
artifacts/reward_eval/results.json

Report:
reports/reward_model_evaluation.md
