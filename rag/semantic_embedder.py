"""Semantic embedding utilities for FAQ retrieval."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

# Make ml/ importable no matter where the script is launched from.
ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from config import SEMANTIC_CONFIG


class SemanticEmbedder:
    """Wrapper around sentence-transformers for FAQ embedding generation."""

    def __init__(
        self,
        model_name: str | None = None,
        normalize_embeddings: bool | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or SEMANTIC_CONFIG["model_name"]
        self.normalize_embeddings = (
            SEMANTIC_CONFIG["normalize_embeddings"]
            if normalize_embeddings is None
            else normalize_embeddings
        )
        self.batch_size = batch_size or SEMANTIC_CONFIG["batch_size"]
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return dense vector embeddings for input texts."""
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


if __name__ == "__main__":
    sample_questions = [
        "Comment réinitialiser mon mot de passe ?",
        "Pourquoi mon onboarding est bloqué à l'étape KYC ?",
    ]

    embedder = SemanticEmbedder()
    embeddings = embedder.embed(sample_questions)

    print(f"Model: {embedder.model_name}")
    print(f"Questions embedded: {len(embeddings)}")
    if embeddings:
        print(f"Vector dimension: {len(embeddings[0])}")
