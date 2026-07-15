"""Persistent stdin/stdout worker for LightGBM abandonment inference.

Protocol (newline-delimited JSON on stdout, one response per request line):

  Request:  {"id":"<uuid>","action":"warmup"}
            {"id":"<uuid>","features":{...},"threshold":0.4,"explain":true,"debug":true}

  Response: {"id":"<uuid>","ok":true,"result":{...}}
            {"id":"<uuid>","ok":false,"error":"...","errorType":"..."}

The LightGBM model is loaded once on worker startup and reused for all
subsequent requests in this process.

stderr carries human logs; stdout is JSON-only.
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict

try:
    from predict import PredictionEngine  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    from ml.predict import PredictionEngine  # type: ignore[import-not-found]

READY_MARKER = "ABANDONMENT_WORKER_READY"

_engine: PredictionEngine | None = None


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _get_engine() -> PredictionEngine:
    global _engine
    if _engine is None:
        _engine = PredictionEngine()
    return _engine


def _handle_request(body: Dict[str, Any]) -> Dict[str, Any]:
    action = body.get("action")
    if action == "warmup":
        engine = _get_engine()
        return {
            "ready": True,
            "modelPath": str(engine.model_path),
            "featureCount": len(engine.feature_names or []),
        }

    features = body.get("features")
    if not isinstance(features, dict):
        raise ValueError("features must be a JSON object")

    threshold_raw = body.get("threshold", 0.5)
    threshold = float(threshold_raw)
    explain = bool(body.get("explain")) and bool(body.get("debug"))
    return _get_engine().predict(features, threshold=threshold, explain=explain)


def main() -> None:
    _log("Abandonment prediction worker starting…")
    try:
        engine = _get_engine()
        _log(f"Model loaded: {engine.model_path}")
    except Exception as exc:  # noqa: BLE001
        _log(f"Warmup failed during startup: {exc}")
        print(READY_MARKER, file=sys.stderr, flush=True)
        sys.exit(1)

    print(READY_MARKER, file=sys.stderr, flush=True)
    _log("Worker ready (model loaded).")

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        req_id = None
        try:
            body = json.loads(line)
            if not isinstance(body, dict):
                raise ValueError("request must be a JSON object")
            req_id = body.get("id")
            result = _handle_request(body)
            response = {"id": req_id, "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            response = {
                "id": req_id,
                "ok": False,
                "error": str(exc),
                "errorType": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
