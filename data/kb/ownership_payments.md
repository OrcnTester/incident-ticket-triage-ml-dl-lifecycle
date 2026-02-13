# Service Ownership: Payments

Service: payments-api
Owner: Team Payments
On-call: payments-oncall

Common symptoms:
- Elevated 5xx on /checkout
- Spike in timeout errors

Routing note:
If errors mention "checkout" or "payment_intent", route to Team Payments.
