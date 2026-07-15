#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

PROFILES = [
    (
        "2 click-miss, 5s page=session",
        {
            "timeOnPage": 5,
            "pageTime": 5,
            "scrollDepth": 0,
            "clickMisses": 2,
            "hesitations": 0,
            "helpTriggered": 0,
            "hasError": 0,
            "multiplePages": 0,
        },
    ),
    (
        "2 click-miss, session 95s / page 20s",
        {
            "timeOnPage": 95,
            "pageTime": 20,
            "scrollDepth": 5,
            "clickMisses": 2,
            "hesitations": 0,
            "helpTriggered": 0,
            "hasError": 0,
            "multiplePages": 1,
        },
    ),
    (
        "2 click-miss, 72s page=session",
        {
            "timeOnPage": 72,
            "pageTime": 72,
            "scrollDepth": 5,
            "clickMisses": 2,
            "hesitations": 0,
            "helpTriggered": 0,
            "hasError": 0,
            "multiplePages": 0,
        },
    ),
    (
        "2 click-miss, 83s scroll 30%",
        {
            "timeOnPage": 83,
            "pageTime": 83,
            "scrollDepth": 30,
            "clickMisses": 2,
            "hesitations": 0,
            "helpTriggered": 0,
            "hasError": 0,
            "multiplePages": 0,
        },
    ),
]

root = Path(__file__).resolve().parents[2]
predict = root / "ml" / "predict.py"

for name, features in PROFILES:
    payload = json.dumps({"features": features})
    proc = subprocess.run(
        [sys.executable, str(predict), "--threshold", "0.35"],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout)
    risk = result["prediction"]["abandonmentRisk"]
    print(f"{name}: {risk * 100:.1f}%")
