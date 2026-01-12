# Data Design (Synthetic)

## Synthetic Data Strategy

### Description
All data in this portfolio project is synthetically generated. No real users, companies, systems, or confidential artifacts are used. The goal is to create realistic-enough incident tickets to discuss ML/DL lifecycle decisions (data strategy, baselines, evaluation, ops, and risks).

### Synthetic generation approach
**Ticket text**
- Template-driven incident narratives (service, symptom, environment, error hints)
- Controlled variability to mimic real phrasing:
  - typos, synonyms, paraphrases
  - optional “noise” tokens and inconsistent formatting
- Intentional ambiguity for harder cases (to test triage robustness)

**Structured signals (optional fields)**
- `source` (e.g., monitoring, user report, oncall)
- `system/service` (component name)
- `timestamp` (for time-aware splits)
- `error_code` (when applicable)

### Deliverable
- This doc section + linked project card

### Done checklist
- [ ] Explicit statement: no proprietary/confidential data is used
- [ ] Template + noise strategy described (text variability)
- [ ] Structured fields listed and justified
- [ ] Note added: leakage prevention handled via split strategy (separate card)

### Risks / Notes
- **Risk:** Synthetic data may not match true operational distributions and language patterns.
- **Trade-off:** Public/safe dataset vs realism and coverage.
- **Assumption:** Template + controlled noise is sufficient to demonstrate lifecycle reasoning and evaluation design.
