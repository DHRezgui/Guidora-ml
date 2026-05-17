"""Persistent stdin/stdout worker for tour semantic role inference.

Protocol (newline-delimited JSON on stdout, one response per request line):

  Request:  {"id":"<uuid>","action":"warmup"}
            {"id":"<uuid>","payload":{...}}   # same shape as tour_semantic_role_inference.infer

  Response: {"id":"<uuid>","ok":true,"result":{...}}
            {"id":"<uuid>","ok":false,"error":"...","errorType":"..."}

The SentenceTransformer model is loaded once on the first inference call
and reused for all subsequent requests in this process.

stderr carries human logs; stdout is JSON-only.
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict

try:
    from .tour_semantic_role_inference import infer, warmup_model  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    from tour_semantic_role_inference import infer, warmup_model  # type: ignore[import-not-found]

READY_MARKER = "TOUR_SEMANTIC_WORKER_READY"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _handle_request(body: Dict[str, Any]) -> Dict[str, Any]:
    action = body.get("action")
    if action == "warmup":
        return {
            "success": True,
            "roles": [],
            "preferredOrder": [],
            "implementation": {
                "kind": "sentence-transformers",
                "model": warmup_model(),
                "phase": "phase-2-embeddings",
            },
        }
    payload = body.get("payload")
    if isinstance(payload, dict):
        return infer(payload)
    return infer(body)


def main() -> None:
    _log("Tour semantic role worker starting…")
    try:
        model_name = warmup_model()
        _log(f"Model loaded: {model_name}")
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
