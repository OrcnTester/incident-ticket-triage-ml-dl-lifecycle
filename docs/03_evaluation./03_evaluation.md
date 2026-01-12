# Evaluation & Metrics

## Metric Strategy

### Description
This project optimizes for operational impact, not just overall accuracy. We evaluate two primary tasks:

1) **Priority prediction (P0–P3)**
2) **Category classification**

### Priority prediction (P0–P3)
**Primary goal:** Protect high-impact incidents (P0/P1).
- **Why:** False negatives on P0/P1 are expensive (missed escalation, extended downtime).
- **Primary metric:** Recall for P0/P1 (or class-weighted recall).
- **Secondary metrics:** Precision for P0/P1 (to control alert fatigue), confusion matrix, and PR curves.

**Operational view**
- A “confident but wrong” P0/P1 miss is worse than a false alarm.
- Thresholding may be used for escalation decisions (separate from classification quality).

### Category classification
**Primary goal:** Avoid majority-class bias and preserve coverage across categories.
- **Primary metric:** Macro F1 (treats each class equally)
- **Secondary metrics:** Per-class F1, support-weighted F1, error buckets by system/source.

### Validation protocol (initial)
- Stratified split for class imbalance (baseline)
- Time-aware split considered when timestamps simulate real incident streams (separate card)

### Deliverable
- This doc section + linked project card

### Done checklist
- [ ] Priority task: metric rationale documented (P0/P1 recall focus)
- [ ] Category task: macro F1 rationale documented
- [ ] Secondary metrics listed (confusion matrix + per-class breakdown)
- [ ] Note added: thresholding/escalation is an ops decision layer

### Risks / Notes
- **Risk:** Over-optimizing recall can increase false positives (alert fatigue).
- **Trade-off:** Conservative escalation vs operational noise.
- **Assumption:** Synthetic data imbalance approximates real incident distributions sufficiently for metric reasoning.
