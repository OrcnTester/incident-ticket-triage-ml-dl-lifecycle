# Error Analysis — category

- Model: `artifacts/baseline_category/model.joblib`
- Split: `time` | test_size=0.2 | seed=42 | gap_days=1
- Train window: 2026-01-26T07:25:49+00:00 → 2026-02-06T07:34:49+00:00
- Test window:  2026-02-07T07:38:49+00:00 → 2026-02-10T07:05:49+00:00

## Metrics (holdout)
- Macro F1: **0.9955**
- Weighted F1: **0.9950**
- Accuracy: **0.9950**

## Confusion matrix
|true\pred|auth_issue|data_issue|deployment_issue|latency|outage|payment_issue|
|---|---|---|---|---|---|---|
|auth_issue|51|0|0|0|0|0|
|data_issue|0|53|0|0|0|0|
|deployment_issue|0|0|65|0|0|0|
|latency|0|0|0|74|0|0|
|outage|0|0|0|0|82|0|
|payment_issue|0|0|0|2|0|73|


## Top confusions (true → predicted)
- **payment_issue → latency** (count=2)

## Examples — payment_issue → latency
- **true=payment_issue pred=latency** | ts=2026-02-07T08:56:49Z | system=inventory | source=partner | err=E429
  - text: `[inventory] paymen tdeclined (E429). urgent`
- **true=payment_issue pred=latency** | ts=2026-02-10T06:26:49Z | system=gateway | source=batch_job | err=E500
  - text: `[gateway] webhoo kfailed (E500). intermittent`

## Notes / Next steps
- If *Critical FN* exists, treat as a **blocking** issue for autonomous escalation decisions.
- Use these examples to refine synthetic generation rules or add new features (error_code/system/source).
- When real data exists: add human review tags and build a labeled “hard cases” set.
