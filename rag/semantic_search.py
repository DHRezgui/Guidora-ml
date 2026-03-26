"""Semantic FAQ search using cosine similarity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from .semantic_embedder import SemanticEmbedder
except ImportError:
    from semantic_embedder import SemanticEmbedder

# Make ml/ importable no matter where the script is launched from.
ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from config import FAQ_EMBEDDING_PATHS, SEMANTIC_SEARCH_CONFIG


class FaqSemanticSearcher:
    """Retrieve most relevant FAQ entries for a natural-language query."""

    def __init__(
        self,
        embeddings_path: str | None = None,
        embedder: SemanticEmbedder | None = None,
    ) -> None:
        selected_path = Path(embeddings_path or FAQ_EMBEDDING_PATHS["output_embeddings"])
        if not selected_path.is_absolute():
            selected_path = ML_ROOT / selected_path

        self.embeddings_path = str(selected_path)
        self.embedder = embedder or SemanticEmbedder()
        self.faq_items = self._load_embeddings(Path(self.embeddings_path))

    @staticmethod
    def _load_embeddings(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path}")

        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError("Embeddings payload must contain an 'items' array")

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Embedding item at index {idx} must be an object")
            if not isinstance(item.get("question"), str):
                raise ValueError(f"Embedding item at index {idx} missing 'question'")
            if not isinstance(item.get("embedding"), list):
                raise ValueError(f"Embedding item at index {idx} missing 'embedding' list")

        return items

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        if len(vec_a) != len(vec_b):
            raise ValueError("Vector dimensions do not match for cosine similarity")

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int | None = None, min_similarity: float | None = None) -> List[Dict[str, Any]]:
        query_text = query.strip()
        if not query_text:
            return []

        effective_top_k = top_k or SEMANTIC_SEARCH_CONFIG["top_k"]
        effective_min_similarity = (
            SEMANTIC_SEARCH_CONFIG["min_similarity"] if min_similarity is None else min_similarity
        )

        query_vec = self.embedder.embed([query_text])[0]
        scored_results: List[Dict[str, Any]] = []

        for item in self.faq_items:
            score = self._cosine_similarity(query_vec, item["embedding"])
            if score < effective_min_similarity:
                continue

            scored_results.append(
                {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "answer": item.get("answer", ""),
                    "category": item.get("category", "general"),
                    "priority": item.get("priority", "medium"),
                    "score": score,
                }
            )

        scored_results.sort(key=lambda row: row["score"], reverse=True)
        return scored_results[:effective_top_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic FAQ search with cosine similarity")
    parser.add_argument("query", help="User natural-language question")
    parser.add_argument(
        "--embeddings",
        default=FAQ_EMBEDDING_PATHS["output_embeddings"],
        help="Path to FAQ embeddings JSON",
    )
    parser.add_argument("--top-k", type=int, default=SEMANTIC_SEARCH_CONFIG["top_k"], help="Top K results")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=SEMANTIC_SEARCH_CONFIG["min_similarity"],
        help="Minimum cosine similarity threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    searcher = FaqSemanticSearcher(embeddings_path=args.embeddings)
    results = searcher.search(query=args.query, top_k=args.top_k, min_similarity=args.min_similarity)

    if not results:
        print("No relevant FAQ results found.")
        return

    print(f"Top {len(results)} results for: {args.query}")
    for idx, item in enumerate(results, start=1):
        print(f"{idx}. [{item['score']:.4f}] ({item['category']}) {item['question']}")
        print(f"   {item['answer']}")


if __name__ == "__main__":
    main()
