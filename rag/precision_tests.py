"""Precision testing suite for semantic FAQ search."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from .semantic_embedder import SemanticEmbedder
    from .semantic_search import FaqSemanticSearcher
except ImportError:
    from semantic_embedder import SemanticEmbedder
    from semantic_search import FaqSemanticSearcher

# Make ml/ importable
ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from config import FAQ_EMBEDDING_PATHS


class PrecisionValidator:
    """Evaluate semantic search precision and recall metrics."""

    def __init__(self, embeddings_path: str | None = None) -> None:
        """Initialize precision validator with FAQ embeddings."""
        self.searcher = FaqSemanticSearcher(embeddings_path=embeddings_path)
        self.test_cases: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []

    def add_test_case(
        self,
        query: str,
        expected_ids: List[str],
        category: str = "general",
        description: str = "",
    ) -> None:
        """Add a test case with expected relevant FAQ IDs.

        Args:
            query: User question to search for
            expected_ids: List of FAQ IDs that should rank highly
            category: Category of the question
            description: Human-readable description
        """
        self.test_cases.append(
            {
                "query": query,
                "expected_ids": set(expected_ids),
                "category": category,
                "description": description,
            }
        )

    def _compute_metrics(
        self,
        retrieved_ids: List[str],
        expected_ids: set[str],
        top_k: int = 5,
    ) -> Dict[str, float]:
        """Compute precision, recall, and NDCG metrics.

        Args:
            retrieved_ids: Ranked list of FAQ IDs returned by search
            expected_ids: Set of relevant FAQ IDs
            top_k: Number of results to consider

        Returns:
            Dict with precision, recall, NDCG metrics
        """
        # Precision@K: relevant items in top-k / k
        top_k_retrieved = set(retrieved_ids[:top_k])
        relevant_in_top_k = len(top_k_retrieved & expected_ids)
        precision_at_k = relevant_in_top_k / top_k if top_k > 0 else 0.0

        # Recall@K: relevant items in top-k / total relevant
        recall_at_k = relevant_in_top_k / len(expected_ids) if expected_ids else 0.0

        # NDCG (Normalized Discounted Cumulative Gain)
        dcg = 0.0
        for idx, faq_id in enumerate(retrieved_ids[:top_k], 1):
            if faq_id in expected_ids:
                # discount = 1 / log2(idx + 1)
                dcg += 1.0 / (1.0 + (idx - 1) ** 0.5)

        # Ideal DCG: all relevant items ranked first
        idcg = 0.0
        for idx in range(1, min(len(expected_ids), top_k) + 1):
            idcg += 1.0 / (1.0 + (idx - 1) ** 0.5)

        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            "precision@5": precision_at_k,
            "recall@5": recall_at_k,
            "ndcg@5": ndcg,
        }

    def run_tests(self, top_k: int = 5, min_similarity: float = 0.2) -> None:
        """Execute all test cases and compute metrics."""
        self.results = []

        for test_case in self.test_cases:
            result = self.searcher.search(
                query=test_case["query"],
                top_k=top_k,
                min_similarity=min_similarity,
            )
            retrieved_ids = [item["id"] for item in result]

            metrics = self._compute_metrics(
                retrieved_ids,
                test_case["expected_ids"],
                top_k,
            )

            found_count = len([id_ for id_ in retrieved_ids if id_ in test_case["expected_ids"]])

            self.results.append(
                {
                    "query": test_case["query"],
                    "description": test_case["description"],
                    "category": test_case["category"],
                    "expected_ids": list(test_case["expected_ids"]),
                    "retrieved_ids": retrieved_ids,
                    "found_count": found_count,
                    "metrics": metrics,
                }
            )

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive precision test report."""
        if not self.results:
            return {"error": "No test results available. Run run_tests() first."}

        # Aggregate metrics
        avg_precision = sum(r["metrics"]["precision@5"] for r in self.results) / len(self.results)
        avg_recall = sum(r["metrics"]["recall@5"] for r in self.results) / len(self.results)
        avg_ndcg = sum(r["metrics"]["ndcg@5"] for r in self.results) / len(self.results)

        # Category breakdown
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"count": 0, "precision": 0, "recall": 0}
            categories[cat]["count"] += 1
            categories[cat]["precision"] += result["metrics"]["precision@5"]
            categories[cat]["recall"] += result["metrics"]["recall@5"]

        for cat in categories:
            categories[cat]["precision"] /= categories[cat]["count"]
            categories[cat]["recall"] /= categories[cat]["count"]

        return {
            "summary": {
                "total_tests": len(self.results),
                "avg_precision@5": avg_precision,
                "avg_recall@5": avg_recall,
                "avg_ndcg@5": avg_ndcg,
            },
            "by_category": categories,
            "detailed_results": self.results,
        }

    def print_report(self) -> None:
        """Print formatted precision test report."""
        report = self.generate_report()

        if "error" in report:
            print(f" {report['error']}")
            return

        summary = report["summary"]
        print("\n" + "=" * 70)
        print("SEMANTIC SEARCH PRECISION TEST REPORT")
        print("=" * 70)

        print(f"\n OVERALL METRICS:")
        print(f"   Tests run: {summary['total_tests']}")
        print(f"   Precision@5: {summary['avg_precision@5']:.3f}")
        print(f"   Recall@5: {summary['avg_recall@5']:.3f}")
        print(f"   NDCG@5: {summary['avg_ndcg@5']:.3f}")

        print(f"\n BY CATEGORY:")
        for cat, metrics in report["by_category"].items():
            print(f"   {cat} (n={metrics['count']}):")
            print(f"      Precision@5: {metrics['precision']:.3f}")
            print(f"      Recall@5: {metrics['recall']:.3f}")

        print(f"\n DETAILED RESULTS:")
        for i, result in enumerate(report["detailed_results"], 1):
            status = "" if result["found_count"] > 0 else ""
            print(f"\n   Test {i}: {status} {result['description']}")
            print(f"      Query: {result['query']}")
            print(f"      Found: {result['found_count']}/{len(result['expected_ids'])} expected")
            print(f"      Precision: {result['metrics']['precision@5']:.3f}")
            print(f"      NDCG: {result['metrics']['ndcg@5']:.3f}")

        print("\n" + "=" * 70 + "\n")


def create_benchmark_suite() -> PrecisionValidator:
    """Create comprehensive benchmark test suite with real FAQ data."""
    validator = PrecisionValidator()

    # Load real FAQ data from embeddings
    embeddings_path = ML_ROOT / "results" / "faq_embeddings.json"
    if embeddings_path.exists():
        try:
            data = json.loads(embeddings_path.read_text(encoding="utf-8-sig"))
            faq_items = data.get("items", [])

            if len(faq_items) >= 10:
                # Test 1: Account creation
                q0 = faq_items[0]
                validator.add_test_case(
                    "create account",
                    [q0["id"]],
                    "account",
                    "Semantic match: Create account",
                )

                # Test 2: Password reset
                q2 = faq_items[2]
                validator.add_test_case(
                    "reset password",
                    [q2["id"]],
                    "auth",
                    "Semantic match: Reset password",
                )

                # Test 3: Activation email
                q1 = faq_items[1]
                validator.add_test_case(
                    "missing activation email",
                    [q1["id"]],
                    "account",
                    "Semantic match: Email not received",
                )

                # Test 4: Organization creation
                q7 = faq_items[7]
                validator.add_test_case(
                    "create organization",
                    [q7["id"]],
                    "organization",
                    "Semantic match: Create organization",
                )

                # Test 5: User roles
                q9 = faq_items[9]
                validator.add_test_case(
                    "modify user role",
                    [q9["id"]],
                    "organization",
                    "Semantic match: Change user role",
                )

                # Test 6: Multilingual - French exact
                validator.add_test_case(
                    "comment reinitialiser mon mot de passe",
                    [q2["id"]],
                    "auth",
                    "Exact French match: Reset password",
                )

                # Test 7: Partial match
                validator.add_test_case(
                    "password change security",
                    [q2["id"]],
                    "auth",
                    "Partial semantic match: Password-related",
                )

                # Test 8: Edge case
                validator.add_test_case(
                    "xyz abc 123",
                    [],
                    "edge",
                    "Nonsensical query (expect no match)",
                )

        except Exception as e:
            print(f"  Could not load real FAQ data: {e}")

    # Fallback: ensure at least some tests exist
    if not validator.test_cases:
        print("  No real FAQ data found, using example test cases...")
        validator.add_test_case(
            "reset password",
            ["faq-003"],
            "auth",
            "Example test",
        )

    return validator


if __name__ == "__main__":
    print(" Loading benchmark suite...")
    suite = create_benchmark_suite()

    print(" Running precision tests...")
    suite.run_tests(top_k=5, min_similarity=0.2)

    print("\n Generating report...")
    suite.print_report()

    # Save report to JSON
    report = suite.generate_report()
    output_path = ML_ROOT / "results" / "precision_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f" Report saved to {output_path}")
