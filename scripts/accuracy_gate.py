"""
Milestone 3 – Automated accuracy gate for CI
Fails (exit 1) if best model accuracy is below threshold.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")
DEFAULT_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "90.0"))


def main():
    if not os.path.exists(METRICS_PATH):
        print("ERROR: metrics.json not found at", METRICS_PATH)
        sys.exit(1)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    primary = metrics.get("primary") or metrics.get("models", {}).get(
        metrics.get("best_model", "logistic_regression"), {}
    )
    acc = float(primary.get("accuracy", 0))
    print(f"Best model accuracy: {acc}")
    print(f"Threshold: {DEFAULT_THRESHOLD}")

    if acc < DEFAULT_THRESHOLD:
        print("ACCURACY GATE FAILED")
        sys.exit(1)

    print("ACCURACY GATE PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
