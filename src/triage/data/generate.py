from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
now = datetime.now(timezone.utc)
import pandas as pd


@dataclass(frozen=True)
class Ticket:
    text: str
    system: str
    source: str
    error_code: str
    timestamp: str
    category: str
    priority: str
    routing_team: str


SYSTEMS = ["payments", "auth", "search", "inventory", "gateway", "billing", "mobile", "observability"]
SOURCES = ["monitoring", "user_report", "oncall", "batch_job", "partner", "qa"]
ERROR_CODES = ["E401", "E403", "E408", "E429", "E500", "E502", "E503", "E504", "TIMEOUT", "OOM", "DB_CONN"]

CATEGORIES = [
    ("outage", ["service down", "500 spike", "unavailable", "cannot reach", "timeout"], "SRE"),
    ("latency", ["slow response", "p95 high", "timeouts", "degraded", "lag"], "SRE"),
    ("auth_issue", ["login fails", "token invalid", "unauthorized", "permission denied"], "Security"),
    ("payment_issue", ["charge failed", "payment declined", "checkout broken", "webhook failed"], "Payments"),
    ("data_issue", ["incorrect data", "missing records", "stale cache", "inconsistent state"], "Data"),
    ("deployment_issue", ["after deploy", "rollback", "version mismatch", "config error"], "Platform"),
]

PRIORITIES = ["P0", "P1", "P2", "P3"]

# Simple, opinionated mapping (you can refine later)
CATEGORY_TO_PRIORITY = {
    "outage": ["P0", "P1"],
    "latency": ["P1", "P2"],
    "auth_issue": ["P1", "P2"],
    "payment_issue": ["P0", "P1", "P2"],
    "data_issue": ["P2", "P3"],
    "deployment_issue": ["P1", "P2"],
}

NOISE = [
    "pls fix asap", "urgent", "seen by customers", "intermittent", "repro steps unknown",
    "happens since morning", "after last change", "randomly", "needs investigation",
]


def _rand_timestamp(days_back: int = 14) -> str:
    now = datetime.utcnow()
    dt = now - timedelta(days=random.randint(0, days_back), hours=random.randint(0, 23), minutes=random.randint(0, 59))
    return dt.isoformat(timespec="seconds") + "Z"


def _maybe_noise(p: float = 0.35) -> str:
    return (" " + random.choice(NOISE)) if random.random() < p else ""


def _typo(text: str, p: float = 0.15) -> str:
    if random.random() > p or len(text) < 8:
        return text
    i = random.randint(1, len(text) - 2)
    return text[:i] + text[i + 1] + text[i] + text[i + 2:]


def make_ticket() -> Ticket:
    system = random.choice(SYSTEMS)
    source = random.choice(SOURCES)
    error_code = random.choice(ERROR_CODES)

    category, phrases, team = random.choice(CATEGORIES)
    phrase = random.choice(phrases)

    # Pick priority from category bucket, with skew
    pr_choices = CATEGORY_TO_PRIORITY[category]
    if "P0" in pr_choices and random.random() < 0.55:
        priority = "P0"
    elif "P1" in pr_choices and random.random() < 0.55:
        priority = "P1"
    else:
        priority = random.choice(pr_choices)

    # Build text
    text = f"[{system}] {phrase} ({error_code})."
    if source == "user_report":
        text += " user says app is broken."
    if source == "monitoring":
        text += " alert triggered from monitoring."
    if "deploy" in phrase or category == "deployment_issue":
        text += " occurred right after deployment."

    text = _typo(text)
    text += _maybe_noise()

    return Ticket(
        text=text,
        system=system,
        source=source,
        error_code=error_code,
        timestamp=_rand_timestamp(),
        category=category,
        priority=priority,
        routing_team=team,
    )


def generate(n: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    rows = [make_ticket().__dict__ for _ in range(n)]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/tickets.csv")
    args = ap.parse_args()

    df = generate(args.n, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"✅ Wrote {len(df)} synthetic tickets to {out_path}")


if __name__ == "__main__":
    main()
