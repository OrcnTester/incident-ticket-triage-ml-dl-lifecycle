# Roles & Tooling

## AI Engineer vs Data Scientist

### Summary
Both roles can overlap, but they optimize for different outcomes.

### Data Scientist (DS)
Focus: discovery and decision support.
- Exploratory analysis, hypothesis testing, and experimentation
- Feature understanding and statistical validation
- Communicating insights and business impact
- Often owns: analysis notebooks, experiments, stakeholder narratives

### AI/ML Engineer (AIE / MLE)
Focus: building reliable systems that run in production.
- Training/inference pipelines, reproducibility, and CI/CD
- Deployment constraints (latency, cost, reliability, scaling)
- Monitoring, drift detection, and rollback strategies
- Data contracts, model packaging, and operational playbooks
- Often owns: productionization, system design, operational safety

### In incident triage context
- DS: explores patterns in ticket text/metadata, evaluates label quality, proposes features/targets.
- AIE/MLE: builds deployable triage pipeline, enforces traceability, and monitors P0/P1 safety metrics.

### Deliverable
- This doc section + linked project card

### Done checklist
- [ ] Clear role distinction stated without “either/or” oversimplification
- [ ] Incident triage example included
- [ ] At least 1 collaboration touchpoint described (handoff boundary)
- [ ] Risk/trade-off note captured

### Risks / Notes
- **Risk:** Role definitions vary by company; avoid rigid labeling.
- **Trade-off:** DS speed vs production guardrails; both are needed for real systems.
