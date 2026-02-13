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
