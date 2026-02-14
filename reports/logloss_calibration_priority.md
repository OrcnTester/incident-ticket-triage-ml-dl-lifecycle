# Log Loss & Calibration Report — priority

- Generated: `2026-02-14T12:02:06Z`
- Model: `artifacts/baseline_priority/model.joblib`
- Split: `time` (test_size=0.2)
  - time_col=timestamp, gap_days=1
  - train_max_ts=2026-02-07T07:38:49+00:00
  - test_min_ts=2026-02-07T07:38:49+00:00
- Calibration: `sigmoid`

## Why log loss (cross entropy)?
- Accuracy: **right/wrong**
- Log loss: **probability quality** (penalizes **confident but wrong**)

## Metrics
- Accuracy: **0.6450**
- Macro F1: **0.5459**
- Log loss: **0.8327**
- ECE (10 bins): **0.0426**

### Priority safety view (P0/P1 vs others)
- P0/P1 binary log loss: **0.3980**
- P0/P1 binary ECE: **0.1775**

## Top confident-but-wrong examples
- **P2 → P1** (conf=0.796) — `[payments] version mismatch (E429). occurred right after deployment.`
- **P2 → P1** (conf=0.781) — `[search] config error (TIMEOUT). occurred right after deployment.`
- **P2 → P1** (conf=0.781) — `[billing] version mismatch (OOM). occurred right after deployment.`
- **P2 → P1** (conf=0.779) — `[inventory] version mismatch (E503). occurred right after deployment.`
- **P2 → P1** (conf=0.778) — `[auth] after deploy (E503). user says app is broken. occurred right after deployment.`
- **P2 → P1** (conf=0.778) — `[payments] config error (TIMEOUT). occurred right after deployment. urgent`
- **P2 → P1** (conf=0.778) — `[auth] config error (OOM). occurred right after deployment. intermittent`
- **P2 → P1** (conf=0.777) — `[billing] token invalid (E503). intermittent`
- **P2 → P1** (conf=0.776) — `[billing] config error (E500). occurred right after deployment.`
- **P2 → P1** (conf=0.769) — `[payments] config error (DB_CONN). occurred right after deployment.`

## Interpretation
- If log loss is high while accuracy is decent, the model is likely **miscalibrated** (over/under-confident).
- For triage, focus on reducing **high-confidence mistakes** on P0/P1 to avoid expensive escalations.

## Files
- JSON: `artifacts/calibration/priority.json`
- This report: `reports/logloss_calibration_priority.md`