# Board Spec — Incident Ticket Triage (ML/DL Lifecycle)

This repository is organized as a **lifecycle board**. Cards represent engineering decisions, not just tasks.
Each card should produce a small artifact (doc/diagram/snippet) and record rationale + risks.

## Board Columns (Purpose)

### No Lifecycle
Parking area for early notes or incomplete items that must be moved into a lifecycle stage.

### Problem & Scope
Define the ML problem, constraints, stakeholders, and the “shape” of the solution.
Outputs: problem statement, labels, success criteria, failure costs, scope boundaries.

### Data Design (Synthetic)
Design data schema and generation assumptions.
Outputs: feature/label schema, split strategy, imbalance plan, data loader strategy, dataset manifests.

### Modeling (scikit-learn)
Classical baselines and explainable models used to set reference performance.
Outputs: baseline model choices, feature pipeline notes, tuning plan, baseline evaluation summary.

### Evaluation & Metrics
Define what “good” means and how failures are measured.
Outputs: metric definitions, error analysis approach, thresholding, calibration notes, validation protocol.

### Lifecycle & Ops
Production constraints: monitoring, logging, drift, retraining triggers, rollout, incident response.
Outputs: lifecycle diagrams, operational checklists, runbook notes, SLO/SLI ideas.

### Roles & Tooling
Who uses the system and what tools support it (HF, PyTorch, Keras, CI, docs).
Outputs: tooling rationale, stack choices, reproducibility guidance.

### Risks & Ethics
Risk identification and mitigation (bias, privacy, hallucinations, escalation harms).
Outputs: risk register entries, mitigations, evaluation prompts, safety constraints.

### Deliverables
Concrete artifacts that make the project reviewable and shareable.
Outputs: README sections, diagrams, notebooks, demo snippets, evaluation summaries.

### Deep Learning (Keras)
Keras/TF track fundamentals and architecture reasoning.
Outputs: pipeline notes, augmentation plan, model architecture choices, training strategy summary.

### Deep Learning (PyTorch)
PyTorch track fundamentals (tensors, autograd, Dataset/DataLoader) and deep model patterns.
Outputs: PyTorch foundations notes, dataset patterns, training flow outline, architecture patterns.

## Card Quality Standard (Definition of Done)

A card is “Done” only if it contains:
- **Description**: why it exists + what it proves
- **Deliverable**: a tangible artifact (doc/diagram/snippet) or a linked section in README
- **Done checklist**: 4–8 measurable checkboxes
- **Risks/Notes**: at least 1 risk or trade-off

## Naming Convention (Recommended)

Use a structured title:
`Domain | Topic | Outcome`

Examples:
- `PyTorch | DataLoader | Memory vs Streaming Strategy`
- `Evaluation | Log Loss | Confidence vs Accuracy`
- `GenAI | Hallucinations | Detection & Mitigation`

## WIP Rule (Professional Focus)

Keep Work-In-Progress small:
- Max **3** cards actively “In Progress”
- Everything else stays “Todo” until picked up

## Optional Fields (Recommended)

If you use custom fields in GitHub Projects:
- Track: Classical ML / Keras / PyTorch / GenAI / Ops
- Artifact: Doc / Diagram / Code / Workflow / Eval
- Priority: P0 / P1 / P2
- Risk: Low / Med / High
