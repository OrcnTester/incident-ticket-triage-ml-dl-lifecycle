# Runbook: API 5xx Spike

When you see elevated 5xx:
- Check recent deployments (last 30-60 mins)
- Check upstream dependency latency/timeouts
- Verify error rate by endpoint and region
- If DB latency is high, consult DB latency runbook

Stop conditions / escalate:
- If P0 customer impact confirmed and error rate > threshold, page on-call.
