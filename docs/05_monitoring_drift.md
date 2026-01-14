# Monitoring & Drift

## Monitoring and Feedback Loops

### Description
After deployment, model quality can degrade due to changes in ticket language, systems, incident patterns, and labeling behavior. Monitoring ensures we detect regressions early—especially for high-impact classes (P0/P1)—and trigger safe response actions.

### What we monitor (signals)
**Data drift**
- Changes in token/embedding distributions, top n-grams, or metadata frequencies (system/source)
- New error codes or service names appearing

**Label drift**
- Priority/categorization policies change over time (human process drift)
- Inconsistent labeling across teams

**Performance drift**
- Rolling-window Macro F1 for category
- Rolling-window P0/P1 recall and precision for priority
- Confusion matrix shifts and false negative rates for P0/P1

### Operational actions
- Alert thresholds and review workflow (human-in-the-loop)
- Retraining triggers (time-based + drift-based)
- Rollback plan if P0/P1 safety metrics regress

### Deliverable
- Monitoring checklist + drift signals list (concept-level)

### Done checklist
- [ ] Data/label/performance drift differentiated clearly
- [ ] P0/P1 safety metric monitoring explicitly listed
- [ ] Trigger conditions documented (retrain vs rollback vs review)
- [ ] At least 1 observability note included (logs/metrics dashboards)

### Risks / Notes
- **Risk:** Optimizing offline metrics may hide real-world regressions without monitoring.
- **Trade-off:** Monitoring depth vs cost/complexity.
- **Assumption:** Synthetic data can still support a concept-level monitoring plan.
