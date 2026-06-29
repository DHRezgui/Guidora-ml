"""Generate vector embeddings for FAQ questions.

Usage:
    python ml/rag/generate_faq_embeddings.py
    python ml/rag/generate_faq_embeddings.py --input ml/data/raw/faq_questions.json --output ml/results/faq_embeddings.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from semantic_embedder import SemanticEmbedder

# Make ml/ importable no matter where the script is launched from.
ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from config import FAQ_EMBEDDING_PATHS


def load_faq_items(input_path: Path) -> List[Dict[str, Any]]:
    """Load and validate FAQ entries from JSON.

    Supported format: list of objects containing at least `question`.
    Optional fields: `id`, `answer`, `category`.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"FAQ input file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("FAQ input must be a JSON array of objects")

    valid_items: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"FAQ entry at index {idx} must be an object")

        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"FAQ entry at index {idx} requires a non-empty 'question'")

        valid_items.append(item)

    return valid_items


def build_embedding_records(faq_items: List[Dict[str, Any]], vectors: List[List[float]]) -> List[Dict[str, Any]]:
    """Combine FAQ metadata and embedding vectors into output records."""
    records: List[Dict[str, Any]] = []
    for idx, (faq_item, embedding) in enumerate(zip(faq_items, vectors)):
        records.append(
            {
                "id": faq_item.get("id", idx),
                "question": faq_item["question"],
                "answer": faq_item.get("answer", ""),
                "category": faq_item.get("category", "general"),
                "priority": faq_item.get("priority", "medium"),
                "embedding": embedding,
            }
        )
    return records


def generate_embeddings(input_file: str, output_file: str) -> None:
    """Generate and persist FAQ embeddings to JSON."""
    input_path = Path(input_file)
    output_path = Path(output_file)

    faq_items = load_faq_items(input_path)
    questions = [item["question"] for item in faq_items]

    embedder = SemanticEmbedder()
    vectors = embedder.embed(questions)

    records = build_embedding_records(faq_items, vectors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "model": embedder.model_name,
        "count": len(records),
        "dimension": len(records[0]["embedding"]) if records else 0,
        "items": records,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("FAQ embeddings generated successfully")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Questions embedded: {output_payload['count']}")
    print(f"Embedding dimension: {output_payload['dimension']}")
    print(f"Model: {output_payload['model']}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for input/output file paths."""
    parser = argparse.ArgumentParser(description="Generate semantic embeddings for FAQ questions")
    parser.add_argument(
        "--input",
        default=FAQ_EMBEDDING_PATHS["input_questions"],
        help="Path to FAQ questions JSON file",
    )
    parser.add_argument(
        "--output",
        default=FAQ_EMBEDDING_PATHS["output_embeddings"],
        help="Path to output embeddings JSON file",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for FAQ embedding generation."""
    args = parse_args()
    generate_embeddings(input_file=args.input, output_file=args.output)


if __name__ == "__main__":
    main()
