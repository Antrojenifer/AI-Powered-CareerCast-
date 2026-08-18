import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")
THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "90.0"))

def main():
    if not os.path.exists(METRICS_PATH):
        print("ERROR: metrics.json missing"); sys.exit(1)
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    primary = metrics.get("primary") or metrics.get("models", {}).get(metrics.get("best_model", "logistic_regression"), {})
    acc = float(primary.get("accuracy", 0))
    print(f"Best model accuracy: {acc}")
    print(f"Threshold: {THRESHOLD}")
    if acc < THRESHOLD:
        print("ACCURACY GATE FAILED"); sys.exit(1)
    print("ACCURACY GATE PASSED"); sys.exit(0)

if __name__ == "__main__":
    main()
