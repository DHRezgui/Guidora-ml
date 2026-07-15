from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"
MODEL_PATH = ML_DIR / "models" / "lightgbm_model.pkl"
FEATURES_PATH = ML_DIR / "models" / "feature_names.json"
PREDICT_SCRIPT = ML_DIR / "predict.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.predict import PredictionEngine, compute_abandonment_confidence


def _sample_features() -> dict[str, float]:
    return {
        "timeOnPage": 83,
        "scrollDepth": 41.4,
        "clickMisses": 0,
        "hesitations": 2,
        "abandonmentRisk": 0.2,
        "helpTriggered": 0,
        "hasError": 0,
        "multiplePages": 1,
        "timePerPage": 41.5,
        "clickMissRate": 0.0,
        "hesitationRate": 0.4,
        "frictionScore": 0.25,
        "highFriction": 0,
        "multipleIssues": 0,
    }


def _require_artifacts() -> None:
    assert MODEL_PATH.exists(), f"Missing model artifact: {MODEL_PATH}"
    assert FEATURES_PATH.exists(), f"Missing features artifact: {FEATURES_PATH}"


@pytest.mark.model
def test_prediction_engine_loads_artifacts() -> None:
    _require_artifacts()

    engine = PredictionEngine()

    assert engine.model is not None
    assert isinstance(engine.feature_names, list)
    assert len(engine.feature_names) > 0


@pytest.mark.model
def test_prediction_returns_expected_schema() -> None:
    _require_artifacts()
    engine = PredictionEngine()

    prediction = engine.predict(_sample_features(), threshold=0.5)

    assert set(prediction.keys()) == {
        "abandonmentRisk",
        "willAbandon",
        "confidence",
        "threshold",
    }
    assert isinstance(prediction["willAbandon"], bool)
    assert isinstance(prediction["abandonmentRisk"], float)
    assert 0.0 <= prediction["abandonmentRisk"] <= 1.0
    assert 0.0 <= prediction["confidence"] <= 1.0
    expected_confidence = compute_abandonment_confidence(
        prediction["abandonmentRisk"],
        prediction["threshold"],
    )
    assert prediction["confidence"] == pytest.approx(round(expected_confidence, 4))
    assert prediction["threshold"] == 0.5


def test_confidence_is_distance_from_threshold() -> None:
    assert compute_abandonment_confidence(0.2642, 0.5) == pytest.approx(0.4716, rel=1e-3)
    assert compute_abandonment_confidence(0.78, 0.4) == pytest.approx(0.6333, rel=1e-3)
    assert compute_abandonment_confidence(0.5, 0.5) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.model
def test_prediction_can_attach_shap_explanation() -> None:
    pytest.importorskip("shap")
    _require_artifacts()
    engine = PredictionEngine()

    prediction = engine.predict(_sample_features(), threshold=0.5, explain=True)

    assert "explanation" in prediction
    explanation = prediction["explanation"]
    assert isinstance(explanation.get("topFeatures"), list)
    assert len(explanation["topFeatures"]) > 0
    assert "feature" in explanation["topFeatures"][0]
    assert "contribution" in explanation["topFeatures"][0]


@pytest.mark.model
def test_prediction_supports_missing_features_with_imputation() -> None:
    _require_artifacts()
    engine = PredictionEngine()

    partial_features = {
        "timeOnPage": 120,
        "scrollDepth": 60,
        "clickMisses": 1,
        "hesitations": 1,
        "abandonmentRisk": 0.35,
        "helpTriggered": 0,
        "hasError": 0,
        "multiplePages": 1,
    }

    prediction = engine.predict(partial_features, threshold=0.5)

    assert isinstance(prediction["willAbandon"], bool)
    assert 0.0 <= prediction["abandonmentRisk"] <= 1.0


@pytest.mark.model
def test_cli_prediction_from_stdin_returns_valid_json() -> None:
    _require_artifacts()

    payload = json.dumps({"features": _sample_features()})
    proc = subprocess.run(
        [sys.executable, str(PREDICT_SCRIPT)],
        input=payload,
        text=True,
        capture_output=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout)

    assert body.get("success") is True
    assert "prediction" in body
    assert 0.0 <= body["prediction"]["abandonmentRisk"] <= 1.0


@pytest.mark.model
def test_threshold_affects_binary_decision() -> None:
    _require_artifacts()
    engine = PredictionEngine()

    features = _sample_features()
    p_low = engine.predict(features, threshold=0.0)
    p_high = engine.predict(features, threshold=1.0)

    assert p_low["willAbandon"] is True
    assert p_high["willAbandon"] is False
