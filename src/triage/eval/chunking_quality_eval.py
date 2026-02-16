import json
from typing import List, Dict


def recall_at_k(results: List[Dict]) -> float:
    correct = 0
    for r in results:
        if r["gold_doc"] in r["retrieved_docs"]:
            correct += 1
    return correct / len(results)


def citation_rate(results: List[Dict]) -> float:
    count = 0
    for r in results:
        if "[" in r["answer"] and "]" in r["answer"]:
            count += 1
    return count / len(results)


def groundedness_rate(results: List[Dict]) -> float:
    grounded = 0
    for r in results:
        context = " ".join(r["retrieved_text"]).lower()
        answer = r["answer"].lower()

        if any(token in context for token in answer.split()):
            grounded += 1

    return grounded / len(results)


def avg_chunk_tokens(results: List[Dict]) -> float:
    total = 0
    for r in results:
        total += sum(len(t.split()) for t in r["retrieved_text"])
    return total / len(results)


def evaluate_chunking(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)


    rec = recall_at_k(data)
    cit = citation_rate(data)
    grd = groundedness_rate(data)
    tokens = avg_chunk_tokens(data)

    quality_score = 0.5 * rec + 0.5 * grd
    winner_score = quality_score - 0.001 * tokens

    result = {
        "recall@k": rec,
        "groundedness_rate": grd,
        "citation_rate": cit,
        "avg_tokens_per_prompt": tokens,
        "quality_score": quality_score,
        "winner_score": winner_score
    }

    print("[OK] Chunking evaluation complete.")
    print(result)

    return result


if __name__ == "__main__":
    evaluate_chunking("artifacts/chunking_eval_input.json")

import sys

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/chunking_eval_input.json"
    evaluate_chunking(path)
