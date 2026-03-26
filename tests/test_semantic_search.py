"""Unit tests for FAQ semantic search using cosine similarity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.rag.semantic_search import FaqSemanticSearcher


class FakeEmbedder:
    """Simple deterministic embedder for tests."""

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self.vectors_by_text = vectors_by_text

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors_by_text[text] for text in texts]


def _write_embeddings(path: Path) -> None:
    payload = {
        "model": "fake-model",
        "count": 3,
        "dimension": 3,
        "items": [
            {
                "id": "faq-a",
                "question": "reset password",
                "answer": "use forgot password flow",
                "category": "auth",
                "priority": "high",
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "id": "faq-b",
                "question": "change email",
                "answer": "go to account settings",
                "category": "profile",
                "priority": "medium",
                "embedding": [0.0, 1.0, 0.0],
            },
            {
                "id": "faq-c",
                "question": "billing invoice",
                "answer": "open billing tab",
                "category": "billing",
                "priority": "low",
                "embedding": [0.0, 0.0, 1.0],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_semantic_search_returns_most_similar_result(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "faq_embeddings.json"
    _write_embeddings(embeddings_path)

    embedder = FakeEmbedder(
        vectors_by_text={
            "forgot my password": [0.99, 0.1, 0.0],
        }
    )
    searcher = FaqSemanticSearcher(embeddings_path=str(embeddings_path), embedder=embedder)

    results = searcher.search(query="forgot my password", top_k=2, min_similarity=0.0)

    assert len(results) == 2
    assert results[0]["id"] == "faq-a"
    assert results[0]["score"] >= results[1]["score"]


def test_semantic_search_applies_min_similarity(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "faq_embeddings.json"
    _write_embeddings(embeddings_path)

    embedder = FakeEmbedder(vectors_by_text={"unknown topic": [0.1, 0.1, 0.0]})
    searcher = FaqSemanticSearcher(embeddings_path=str(embeddings_path), embedder=embedder)

    results = searcher.search(query="unknown topic", top_k=3, min_similarity=0.9)

    assert results == []


def test_semantic_search_rejects_invalid_embeddings_payload(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "invalid_embeddings.json"
    embeddings_path.write_text(json.dumps({"model": "x", "items": "bad"}), encoding="utf-8")

    embedder = FakeEmbedder(vectors_by_text={"q": [1.0]})

    with pytest.raises(ValueError):
        FaqSemanticSearcher(embeddings_path=str(embeddings_path), embedder=embedder)
