# Problem & Scope

## Define the ML Task

### Description
We design an incident ticket triage system that supports operations teams by structuring and prioritizing incoming incidents.

**Input**
- Unstructured ticket text
- Limited structured signals (e.g., source, system, timestamp, error_code)

**Outputs**
- Incident category (multi-class)
- Priority level (P0–P3)
- Routing team suggestion

**Why ML**
Rule-based systems degrade as ticket phrasing, systems, and failure modes evolve. ML enables better generalization and consistent triage decisions.

### Deliverable
- This doc section + linked project card

### Done checklist
- [ ] Inputs/outputs clearly defined
- [ ] Scope boundaries stated (assistive system, not auto-close)
- [ ] Success criteria stub added (metrics defined in Metric Strategy card)
- [ ] At least 1 key risk/trade-off captured below

### Risks / Notes
- **Risk:** Misclassifying P0/P1 incidents has high operational cost.
- **Trade-off:** Automation vs human-in-the-loop; system must support traceability.
- **Assumption:** No confidential/proprietary data is used in this public portfolio.
