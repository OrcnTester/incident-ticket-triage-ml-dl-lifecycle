import json
import random
from pathlib import Path
from typing import List, Dict

# --------------------------
# Prompt Suite (static)
# --------------------------

PROMPTS = [
    {
        "id": "p1",
        "prompt": "Classify this ticket as P0-P3: Database completely down.",
        "expected_keywords": ["P0"]
    },
    {
        "id": "p2",
        "prompt": "Return JSON with keys: category and priority. Ticket: Payment API timeout.",
        "expected_schema": ["category", "priority"]
    }
]

# --------------------------
# Mock model runner
# Replace with real LLM call
# --------------------------

def mock_model(prompt: str, seed: int = 42) -> str:
    random.seed(seed)
    if "Database completely down" in prompt:
        return "P0"
    if "JSON" in prompt:
        return json.dumps({"category": "payment_issue", "priority": "P1"})
    return "Unknown"

# --------------------------
# Rubric Scoring
# --------------------------

def score_response(prompt_data: Dict, response: str) -> Dict:
    score = 0
    format_ok = True

    # Correctness
    if "expected_keywords" in prompt_data:
        if any(k in response for k in prompt_data["expected_keywords"]):
            score += 2
        else:
            score += 0

    # Format adherence
    if "expected_schema" in prompt_data:
        try:
            parsed = json.loads(response)
            if all(k in parsed for k in prompt_data["expected_schema"]):
                score += 2
            else:
                score += 0
        except:
            format_ok = False

    return {
        "score": score,
        "format_ok": format_ok
    }

# --------------------------
# Evaluation Runner
# --------------------------

def run_evaluation(seed: int = 42):
    results = []
    total_score = 0
    format_fail = 0

    for p in PROMPTS:
        response = mock_model(p["prompt"], seed=seed)
        scored = score_response(p, response)
        total_score += scored["score"]

        if not scored["format_ok"]:
            format_fail += 1

        results.append({
            "id": p["id"],
            "response": response,
            "score": scored["score"]
        })

    avg_score = total_score / len(PROMPTS)

    summary = {
        "avg_score": avg_score,
        "format_failures": format_fail,
        "total_prompts": len(PROMPTS)
    }

    return results, summary

# --------------------------
# CLI
# --------------------------

def main():
    results, summary = run_evaluation()

    Path("artifacts/instruction_eval").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    with open("artifacts/instruction_eval/results.json", "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    with open("reports/instruction_eval.md", "w") as f:
        f.write("# Instruction Tuning Evaluation Report\n\n")
        f.write(f"Average Score: {summary['avg_score']}\n")
        f.write(f"Format Failures: {summary['format_failures']}\n")

    print("[OK] Instruction evaluation complete.")
    print(summary)


if __name__ == "__main__":
    main()
