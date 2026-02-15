import json
import random
from pathlib import Path
from typing import List, Dict
from scipy.stats import spearmanr

# ----------------------------
# Synthetic Pair Dataset
# ----------------------------

PAIR_DATA = [
    {
        "id": "1",
        "response_a": "Database is completely down. This is critical. P0.",
        "response_b": "Maybe there is some issue.",
        "preferred": "A"
    },
    {
        "id": "2",
        "response_a": "The issue seems minor.",
        "response_b": "System-wide outage detected. Immediate action required.",
        "preferred": "B"
    }
]

# ----------------------------
# Mock Reward Model
# Replace with real RM inference
# ----------------------------

def reward_model_score(text: str, seed: int = 42) -> float:
    random.seed(seed + len(text))
    return random.random()

# ----------------------------
# Evaluation
# ----------------------------

def evaluate_reward_model(seed: int = 42):
    correct = 0
    total = 0

    scores_a = []
    scores_b = []
    human_labels = []

    disagreements = []

    for pair in PAIR_DATA:
        score_a = reward_model_score(pair["response_a"], seed)
        score_b = reward_model_score(pair["response_b"], seed)

        predicted = "A" if score_a > score_b else "B"
        human = pair["preferred"]

        if predicted == human:
            correct += 1
        else:
            disagreements.append({
                "id": pair["id"],
                "score_a": score_a,
                "score_b": score_b,
                "preferred": human
            })

        scores_a.append(score_a)
        scores_b.append(score_b)
        human_labels.append(1 if human == "A" else 0)

        total += 1

    pairwise_accuracy = correct / total

    # Spearman (flatten ranking comparison)
    flat_scores = scores_a + scores_b
    flat_labels = human_labels + [1 - x for x in human_labels]

    corr, _ = spearmanr(flat_scores, flat_labels)

    return {
        "pairwise_accuracy": pairwise_accuracy,
        "spearman_correlation": float(corr),
        "total_pairs": total,
        "disagreements": disagreements
    }

# ----------------------------
# CLI
# ----------------------------

def main():
    results = evaluate_reward_model()

    Path("artifacts/reward_eval").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    with open("artifacts/reward_eval/results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("reports/reward_model_evaluation.md", "w") as f:
        f.write("# Reward Model Evaluation Report\n\n")
        f.write(f"Pairwise Accuracy: {results['pairwise_accuracy']}\n")
        f.write(f"Spearman Correlation: {results['spearman_correlation']}\n")
        f.write(f"Total Pairs: {results['total_pairs']}\n")
        f.write("\n## Disagreements\n")
        for d in results["disagreements"]:
            f.write(f"- Pair {d['id']} disagreement\n")

    print("[OK] Reward model evaluation complete.")
    print(results)


if __name__ == "__main__":
    main()
