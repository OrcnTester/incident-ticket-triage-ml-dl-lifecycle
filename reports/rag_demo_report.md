# RAG Demo Report
- created_at: 2026-02-13T09:11:51.113264+00:00
- ticket_row: 0

## Retrieved snippets (top-k)
### runbook_db_latency (score=0.0768)
- path: data/kb/runbook_db_latency.md

```
# Runbook: DB Latency / Timeouts

Signals:
- p95 latency increased
- connection pool saturation
- timeout exceptions in app logs

First steps:
- Check DB CPU / IOPS / connections
- Identify top slow queries (if available)
- Check if traffic spike correlates with latency
- Consider read replica health

Owner: Team Data Platform
```

### runbook_api_5xx (score=0.0674)
- path: data/kb/runbook_api_5xx.md

```
# Runbook: API 5xx Spike

When you see elevated 5xx:
- Check recent deployments (last 30-60 mins)
- Check upstream dependency latency/timeouts
- Verify error rate by endpoint and region
- If DB latency is high, consult DB latency runbook

Stop conditions / escalate:
- If P0 customer impact confirmed and error rate > threshold, page on-call.
```

### ownership_payments (score=0.0000)
- path: data/kb/ownership_payments.md

```
# Service Ownership: Payments

Service: payments-api
Owner: Team Payments
On-call: payments-oncall

Common symptoms:
- Elevated 5xx on /checkout
- Spike in timeout errors

Routing note:
If errors mention "checkout" or "payment_intent", route to Team Payments.
```

## Output comparison
### No retrieval (pure generation baseline)
```json
{
  "one_liner": "[auth] p95 high (E500). alert triggered from montioring.",
  "impact": "unknown",
  "suspected_service": "unknown",
  "severity_hint": "unknown",
  "key_entities": [
    "500"
  ],
  "open_questions": [
    "Which service/system is affected?",
    "What is the user/customer impact (if any)?"
  ],
  "evidence": [
    {
      "type": "ticket_text",
      "quote": "[auth] p95 high (E500). alert triggered from montioring.\nauth"
    }
  ],
  "prompt_version": "rag_stub:no_retrieval:v1",
  "model_id": null
}
```

### With retrieval (grounded)
```json
{
  "one_liner": "[auth] p95 high (E500). alert triggered from montioring.",
  "impact": "unknown",
  "suspected_service": "unknown",
  "severity_hint": "unknown",
  "key_entities": [
    "500"
  ],
  "open_questions": [
    "Does the symptom match the retrieved runbook/ownership note?"
  ],
  "evidence": [
    {
      "type": "ticket_text",
      "quote": "[auth] p95 high (E500). alert triggered from montioring.\nauth"
    },
    {
      "type": "metadata",
      "quote": "[runbook_db_latency] # Runbook: DB Latency / Timeouts\n\nSignals:\n- p95 latency increased\n- connection pool saturation\n- timeout exceptions in app logs\n\nFirst steps:\n- Check DB CPU / IOPS / connections\n- Identify top slow queries (if availab…"
    },
    {
      "type": "metadata",
      "quote": "[runbook_api_5xx] # Runbook: API 5xx Spike\n\nWhen you see elevated 5xx:\n- Check recent deployments (last 30-60 mins)\n- Check upstream dependency latency/timeouts\n- Verify error rate by endpoint and region\n- If DB latency is high, consult DB…"
    }
  ],
  "prompt_version": "rag_stub:with_retrieval:v1",
  "model_id": null
}
```

### Routing rationale (assistive top-k)
```json
{
  "recommended_teams": [
    {
      "team": "Team Data Platform",
      "confidence": 0.95,
      "rationale": "Suggested based on retrieved ownership/runbook hints (assistive; not auto-assign).",
      "evidence": [
        {
          "type": "team_map",
          "note": "[runbook_db_latency] # Runbook: DB Latency / Timeouts\n\nSignals:\n- p95 latency increased\n- connection pool saturation\n- timeout exceptions in app logs\n\nFirst steps:\n- Check DB CPU / IOPS / connections\n- Identify top slow…"
        },
        {
          "type": "team_map",
          "note": "[runbook_api_5xx] # Runbook: API 5xx Spike\n\nWhen you see elevated 5xx:\n- Check recent deployments (last 30-60 mins)\n- Check upstream dependency latency/timeouts\n- Verify error rate by endpoint and region\n- If DB latency…"
        }
      ]
    },
    {
      "team": "Team Payments",
      "confidence": 0.9,
      "rationale": "Suggested based on retrieved ownership/runbook hints (assistive; not auto-assign).",
      "evidence": [
        {
          "type": "team_map",
          "note": "[runbook_db_latency] # Runbook: DB Latency / Timeouts\n\nSignals:\n- p95 latency increased\n- connection pool saturation\n- timeout exceptions in app logs\n\nFirst steps:\n- Check DB CPU / IOPS / connections\n- Identify top slow…"
        },
        {
          "type": "team_map",
          "note": "[runbook_api_5xx] # Runbook: API 5xx Spike\n\nWhen you see elevated 5xx:\n- Check recent deployments (last 30-60 mins)\n- Check upstream dependency latency/timeouts\n- Verify error rate by endpoint and region\n- If DB latency…"
        }
      ]
    }
  ],
  "open_questions": [
    "Which service is this ticket about?",
    "Is there an internal ownership map for this service?"
  ],
  "what_would_change_my_mind": [
    "A KB/runbook snippet that explicitly names the owning team.",
    "A service tag/metadata field mapping the ticket to a known component."
  ],
  "prompt_version": "rag_stub:with_retrieval:v1",
  "model_id": null
}
```

### Runbook hints
- p95 latency increased
- connection pool saturation
- timeout exceptions in app logs
- Check DB CPU / IOPS / connections
- Identify top slow queries (if available)
- Check if traffic spike correlates with latency
- Consider read replica health
- Check recent deployments (last 30-60 mins)