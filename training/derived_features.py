"""Canonical derived-feature formulas — shared by training, inference, and tests.

Must stay aligned with:
  backend/src/ml/abandonment-derived-features.ts
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from ml.config import PREPROCESSING_PARAMS

if TYPE_CHECKING:
    import pandas as pd

FRICTION_WEIGHTS = PREPROCESSING_PARAMS["friction_weights"]
HIGH_FRICTION_THRESHOLD = 0.7
PAGES_SINGLE = int(PREPROCESSING_PARAMS["default_pages_single"])
PAGES_MULTI = int(PREPROCESSING_PARAMS["default_pages_multi"])


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def compute_derived_features_row(
    time_on_page: float,
    scroll_depth: float,
    click_misses: float,
    hesitations: float,
    multiple_pages: int,
    idle_seconds: float = 0.0,
    page_time: float | None = None,
) -> dict[str, float | int]:
    """Compute derived features for a single observation (train/serve contract)."""
    time_on_page = max(0.0, float(time_on_page))
    page_time = max(0.0, float(time_on_page if page_time is None else page_time))
    scroll_depth = max(0.0, min(100.0, float(scroll_depth)))
    click_misses = max(0.0, float(click_misses))
    hesitations = max(0.0, float(hesitations))
    pages = PAGES_MULTI if int(multiple_pages) == 1 else PAGES_SINGLE

    # pageTime drives per-page dwell; session timeOnPage stays a separate raw feature.
    time_per_page = page_time / max(pages, 1)
    click_miss_rate = click_misses / max(time_on_page, 1.0)
    hesitation_rate = hesitations / max(time_on_page, 1.0)
    idle_seconds = max(0.0, float(idle_seconds))
    idle_ratio = idle_seconds / max(page_time, 1.0)

    fw = FRICTION_WEIGHTS
    friction_score = _clip01(
        click_misses * float(fw["clickMisses"])
        + hesitations * float(fw["hesitations"])
        + min(page_time / 120.0, 1.0) * float(fw["timeOnPage"])
        + (1.0 - scroll_depth / 100.0) * float(fw["scrollDepth"])
    )

    return {
        "pageTime": page_time,
        "timePerPage": time_per_page,
        "clickMissRate": click_miss_rate,
        "hesitationRate": hesitation_rate,
        "idleSeconds": idle_seconds,
        "idleRatio": idle_ratio,
        "frictionScore": friction_score,
        "highFriction": 1 if friction_score > HIGH_FRICTION_THRESHOLD else 0,
        "multipleIssues": 1 if click_misses > 1 and hesitations > 0 else 0,
    }


def enrich_features_dict(features: dict[str, Any]) -> dict[str, Any]:
    """Merge raw SDK features with canonical derived fields (overwrites stale derived)."""
    result = dict(features)
    raw_page_time = result.get("pageTime", None)

    derived = compute_derived_features_row(
        time_on_page=result.get("timeOnPage", 0),
        scroll_depth=result.get("scrollDepth", 0),
        click_misses=result.get("clickMisses", 0),
        hesitations=result.get("hesitations", 0),
        multiple_pages=int(result.get("multiplePages", 0) or 0),
        idle_seconds=result.get("idleSeconds", 0),
        page_time=None if raw_page_time is None else raw_page_time,
    )
    result.update(derived)
    return result


def build_derived_features_frame(frame: "pd.DataFrame") -> None:
    """In-place derived columns for training preprocessing."""
    import pandas as pd

    if "pageTime" not in frame.columns:
        frame["pageTime"] = frame["timeOnPage"]
    else:
        frame["pageTime"] = pd.to_numeric(frame["pageTime"], errors="coerce")
        frame["pageTime"] = frame["pageTime"].fillna(frame["timeOnPage"]).clip(lower=0)

    estimated_pages = np.where(frame["multiplePages"] == 1, PAGES_MULTI, PAGES_SINGLE)
    frame["timePerPage"] = frame["pageTime"] / np.maximum(estimated_pages, 1)

    frame["clickMissRate"] = frame["clickMisses"] / np.maximum(frame["timeOnPage"], 1)
    frame["hesitationRate"] = frame["hesitations"] / np.maximum(frame["timeOnPage"], 1)

    fw = FRICTION_WEIGHTS
    frame["frictionScore"] = (
        frame["clickMisses"] * float(fw["clickMisses"])
        + frame["hesitations"] * float(fw["hesitations"])
        + np.minimum(frame["pageTime"] / 120.0, 1.0) * float(fw["timeOnPage"])
        + (1.0 - frame["scrollDepth"] / 100.0) * float(fw["scrollDepth"])
    ).clip(lower=0, upper=1)

    frame["highFriction"] = (frame["frictionScore"] > HIGH_FRICTION_THRESHOLD).astype(int)
    frame["multipleIssues"] = (
        (frame["clickMisses"] > 1) & (frame["hesitations"] > 0)
    ).astype(int)
