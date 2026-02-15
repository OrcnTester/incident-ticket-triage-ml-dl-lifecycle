import json
from typing import List, Dict


def recall_at_k(results: List[Dict], k: int = 5) -> float:
    hits = 0
    for r in results:
        if r["gold_doc"] in r["retrieved_docs"][:k]:
            hits += 1
    return hits / len(results)


def citation_rate(results: List[Dict]) -> float:
    count = 0
    for r in results:
        if "[" in r["answer"] and "]" in r["answer"]:
            count += 1
    return count / len(results)

import re

def _strip_citations(text: str) -> str:
    # [doc_id] gibi bracket içlerini temizle
    return re.sub(r"\[[^\]]+\]", "", text).strip()

def _tokenize(text: str) -> set:
    # çok basit tokenization
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

def groundedness_rate(results: List[Dict]) -> float:
    grounded = 0
    for r in results:
        context = " ".join(r["retrieved_text"]).lower()
        answer = _strip_citations(r["answer"]).lower()

        # kısa cevaplarda "overlap" kontrollü
        ans_tokens = _tokenize(answer)
        ctx_tokens = _tokenize(context)

        if not ans_tokens:
            continue

        overlap = len(ans_tokens & ctx_tokens) / len(ans_tokens)

        # %30 token örtüşmesi varsa "supported" kabul et
        if overlap >= 0.30:
            grounded += 1

    return grounded / len(results)



def hallucination_rate(results: List[Dict]) -> float:
    return 1 - groundedness_rate(results)


def evaluate(results: List[Dict]) -> Dict:
    return {
        "recall@5": recall_at_k(results, 5),
        "citation_rate": citation_rate(results),
        "groundedness_rate": groundedness_rate(results),
        "hallucination_rate": hallucination_rate(results),
    }


if __name__ == "__main__":
    with open("artifacts/rag_eval_input.json", "r") as f:
        data = json.load(f)

    metrics = evaluate(data)

    print("[OK] RAG evaluation complete.")
    print(metrics)
