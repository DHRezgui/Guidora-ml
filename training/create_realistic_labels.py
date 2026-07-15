#!/usr/bin/env python3
"""
Create abandonment labels from backend-exported behavior analysis data.

Supports two label sources:
- real: uses `isAbandoned` / `is_abandoned` from user_progress when available
- synthetic: behavioral proxy with realistic noise (legacy fallback)

Use --prefer-real to keep ground-truth labels and only synthesize missing rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def assign_realistic_abandonment_label(record: dict[str, Any], seed_value: int | None = None):
    """Assign abandonment label based on behavioral patterns + realistic noise."""

    time_on_page = record.get("timeOnPage", 0)
    scroll_depth = record.get("scrollDepth", 0)
    click_misses = record.get("clickMisses", 0)
    hesitations = record.get("hesitations", 0)
    help_triggered = record.get("helpTriggered", False)

    frustration_score = 0
    if time_on_page > 120:
        frustration_score += 25
    elif time_on_page > 60:
        frustration_score += 10

    if scroll_depth < 30:
        frustration_score += 20
    elif scroll_depth < 50:
        frustration_score += 10

    frustration_score += min(click_misses * 5, 20)
    frustration_score += min(hesitations * 10, 20)

    if help_triggered:
        frustration_score += 10

    base_abandon_prob = min(frustration_score / 100.0, 1.0)
    noise = np.random.RandomState(seed_value).normal(0, 0.15)
    final_prob = float(np.clip(base_abandon_prob + noise, 0, 1))
    label = 1 if final_prob > 0.5 else 0

    return label, frustration_score, final_prob


def extract_real_label(record: dict[str, Any]) -> int | None:
    """Read ground-truth abandonment from backend export fields."""

    if record.get("labelSource") == "real" and record.get("label") in (0, 1):
        return int(record["label"])

    for key in ("isAbandoned", "is_abandoned"):
        if key not in record:
            continue
        value = record[key]
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return 1 if int(value) == 1 else 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return 1
            if normalized in {"false", "0", "no"}:
                return 0
    return None


def resolve_label(
    record: dict[str, Any],
    *,
    prefer_real: bool,
    seed_value: int,
) -> tuple[int, str, float | None, float | None]:
    if prefer_real:
        real_label = extract_real_label(record)
        if real_label is not None:
            return real_label, "real", None, None

    label, frustration, prob = assign_realistic_abandonment_label(record, seed_value=seed_value)
    return label, "synthetic", float(frustration), float(prob)


def load_dataset(input_file: str) -> list[dict[str, Any]]:
    logger.info("Loading dataset from %s...", input_file)
    with open(input_file, "r", encoding="utf-8") as f:
        content = json.load(f)

    if isinstance(content, dict) and "features" in content:
        data = content["features"]
    elif isinstance(content, list):
        data = content
    else:
        raise ValueError("Unsupported dataset format. Expected list or {features: [...]}.")

    logger.info("Loaded %s records", len(data))
    return data


def synthesize_page_time(
    time_on_page: float,
    multiple_pages: int,
    seed_value: int,
) -> float:
    """
    Synthesize page dwell when exports lack pageTime (behavior_analysis is session-level).

    - Single-page rows: page ≈ session (85–100%).
    - Multi-page rows: page is a fraction of session (20–70%).
    """
    top = max(0.0, float(time_on_page or 0))
    rng = np.random.RandomState(seed_value + 101)
    if int(multiple_pages or 0) == 1:
        ratio = float(rng.uniform(0.2, 0.7))
    else:
        ratio = float(rng.uniform(0.85, 1.0))
    return round(top * ratio, 3)


def ensure_page_time(record: dict[str, Any], seed_value: int) -> None:
    time_on_page = float(record.get("timeOnPage", 0) or 0)
    multiple_pages = int(record.get("multiplePages", 0) or 0)
    existing = record.get("pageTime")
    if existing is not None and existing != "":
        page_time = max(0.0, float(existing))
        if time_on_page > 0:
            page_time = min(page_time, time_on_page)
        record["pageTime"] = page_time
        return
    record["pageTime"] = synthesize_page_time(time_on_page, multiple_pages, seed_value)


def add_labels(data: list[dict[str, Any]], *, prefer_real: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    logger.info("Assigning labels (prefer_real=%s)...", prefer_real)

    labeled_data: list[dict[str, Any]] = []
    stats = {
        "real": 0,
        "synthetic": 0,
        "abandoned": 0,
        "pageTimeSynthesized": 0,
    }

    for idx, record in enumerate(data):
        had_page_time = record.get("pageTime") not in (None, "")
        ensure_page_time(record, seed_value=idx)
        if not had_page_time:
            stats["pageTimeSynthesized"] += 1

        label, label_source, frustration, prob = resolve_label(
            record,
            prefer_real=prefer_real,
            seed_value=idx,
        )
        record["label"] = label
        record["label_source"] = label_source
        record["labelSource"] = label_source

        if frustration is not None:
            record["frustration_score"] = frustration
        if prob is not None:
            record["abandon_probability"] = prob

        labeled_data.append(record)
        stats[label_source] += 1
        if label == 1:
            stats["abandoned"] += 1

    return labeled_data, stats


def save_labeled_dataset(data: list[dict[str, Any]], output_file: str) -> None:
    logger.info("Saving labeled dataset to %s...", output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info("Dataset saved: %s", output_file)


def print_statistics(data: list[dict[str, Any]], stats: dict[str, int]) -> None:
    total = len(data)
    completing_count = total - stats["abandoned"]

    logger.info("\n" + "=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info("Total records:        %s", total)
    logger.info("Real labels:          %s (%.1f%%)", stats["real"], 100 * stats["real"] / total if total else 0)
    logger.info(
        "Synthetic labels:     %s (%.1f%%)",
        stats["synthetic"],
        100 * stats["synthetic"] / total if total else 0,
    )
    logger.info(
        "Completed onboarding: %s (%.1f%%)",
        completing_count,
        100 * completing_count / total if total else 0,
    )
    logger.info(
        "Abandoned:            %s (%.1f%%)",
        stats["abandoned"],
        100 * stats["abandoned"] / total if total else 0,
    )
    logger.info(
        "pageTime synthesized: %s (%.1f%%)",
        stats.get("pageTimeSynthesized", 0),
        100 * stats.get("pageTimeSynthesized", 0) / total if total else 0,
    )
    logger.info(
        "Imbalance ratio:      %.2f",
        stats["abandoned"] / completing_count if completing_count > 0 else 0,
    )
    logger.info("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create abandonment labels for ML training")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "raw", "dataset_export_real.json"),
        help="Backend-exported dataset JSON",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "raw", "dataset_realistic_labels.json"),
        help="Output JSON with labels",
    )
    parser.add_argument(
        "--prefer-real",
        action="store_true",
        help="Keep ground-truth labels from user_progress when exported by the backend",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.input)
    labeled_data, stats = add_labels(data, prefer_real=args.prefer_real)
    save_labeled_dataset(labeled_data, args.output)
    print_statistics(labeled_data, stats)
    logger.info("Label generation complete")


if __name__ == "__main__":
    main()
