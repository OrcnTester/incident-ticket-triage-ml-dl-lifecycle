# Text Splitting Demo Report

- kb_dir: `data\kb`
- docs: **3**

## token
- n_chunks: **3** | elapsed_s: 0.0001
- duplicate_ratio: 0.0000 | empty_chunks: 0
- token len min/median/p95/max: 34 / 54.0 / 55 / 55
- char  len min/median/p95/max: 243 / 317.0 / 329 / 329
- sentence_boundary_ratio: 0.67

Preview:
- chunk 1 (ownership_payments): `service ownership payments service payments-api owner team payments on-call payments-oncall common symptoms - elevated 5xx on /checkout - spike in timeout errors routing note if errors mention checkout or payment_intent route to team payments.`
- chunk 2 (runbook_api_5xx): `runbook api 5xx spike when you see elevated 5xx - check recent deployments last 30-60 mins - check upstream dependency latency/timeouts - verify error rate by endpoint and region - if db latency is high consult db latency runbook stop conditions / escalate - if p0 customer impact confirmed and error…`

## sentence
- n_chunks: **3** | elapsed_s: 0.0001
- duplicate_ratio: 0.0000 | empty_chunks: 0
- token len min/median/p95/max: 34 / 54.0 / 55 / 55
- char  len min/median/p95/max: 259 / 328.0 / 342 / 342
- sentence_boundary_ratio: 0.67

Preview:
- chunk 1 (ownership_payments): `# Service Ownership: Payments

Service: payments-api
Owner: Team Payments
On-call: payments-oncall

Common symptoms:
- Elevated 5xx on /checkout
- Spike in timeout errors

Routing note:
If errors mention "checkout" or "payment_intent", route to Team Payments.`
- chunk 2 (runbook_api_5xx): `# Runbook: API 5xx Spike

When you see elevated 5xx:
- Check recent deployments (last 30-60 mins)
- Check upstream dependency latency/timeouts
- Verify error rate by endpoint and region
- If DB latency is high, consult DB latency runbook

Stop conditions / escalate:
- If P0 customer impact confirmed…`

## recursive
- n_chunks: **6** | elapsed_s: 0.0001
- duplicate_ratio: 0.0000 | empty_chunks: 0
- token len min/median/p95/max: 20 / 25.5 / 34 / 34
- char  len min/median/p95/max: 128 / 177.0 / 198 / 198
- sentence_boundary_ratio: 0.33

Preview:
- chunk 1 (ownership_payments): `# Service Ownership: Payments

Service: payments-api
Owner: Team Payments
On-call: payments-oncall

Common symptoms:
- Elevated 5xx on /checkout
- Spike in timeout errors`
- chunk 2 (ownership_payments): `x on /checkout
- Spike in timeout errors Routing note:
If errors mention "checkout" or "payment_intent", route to Team Payments.`
