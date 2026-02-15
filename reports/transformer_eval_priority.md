# Transformer Classification Evaluation — priority

## Summary

- Accuracy: **0.8025**
- Macro F1: **0.7636**
- Weighted F1: **0.8077**
- Log loss (cross-entropy): **0.7714**
- ECE (calibration error): **0.0401**

## Confusion Matrix

| true \\ pred | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| P0 | 85 | 4 | 7 | 5 |
| P1 | 19 | 153 | 15 | 9 |
| P2 | 7 | 5 | 63 | 3 |
| P3 | 3 | 0 | 2 | 20 |

## Top Confusions

- **P1 → P0**: 19
- **P1 → P2**: 15
- **P1 → P3**: 9
- **P2 → P0**: 7
- **P0 → P2**: 7

## Example Error Buckets

### P1 → P0
- [inventory] lag (OOM). (conf=0.85)
- [observability] version mismatch (E500). alert triggered from monitoring. occurred right after deployment. pls fix asap (conf=0.83)
- [billing] version mismatch (E503). occurred right after deployment. (conf=0.81)

### P1 → P2
- [auth] degraded (DB_CONN). (conf=0.81)
- [search] p95 high (E503). user says app is broken. repro steps unknown (conf=0.86)
- [billing] 500 spike (E408). (conf=0.80)

### P1 → P3
- [gateway] config error (E408). occurred right after deployment. (conf=0.80)
- [mobile] 500 spike (OOM). (conf=0.82)
- [billing] unauthorized (E429). alert triggered from monitoring. (conf=0.82)

### P2 → P0
- [billing] lag (E504). seen by customers (conf=0.87)
- [observability] lag (TIMEOUT). (conf=0.84)
- [billing] incorrect data (E403). urgent (conf=0.86)

### P0 → P2
- [observability] cannot reach (E401). user says app is broken. repro steps unknown (conf=0.81)
- [inventory] charge failed (E403). (conf=0.86)
- [gateway] webhook failed (E408). (conf=0.84)
