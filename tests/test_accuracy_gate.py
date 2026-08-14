import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "models", "metrics.json")


def test_metrics_file_exists():
    assert os.path.exists(METRICS)


def test_accuracy_above_gate():
    with open(METRICS, "r", encoding="utf-8") as f:
        m = json.load(f)
    acc = float(m["primary"]["accuracy"])
    assert acc >= 90.0
