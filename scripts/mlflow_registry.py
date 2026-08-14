"""
Milestone 3 – MLflow Model Registry helper
Logs model metrics and registers the best model.
Usage:
  python scripts/mlflow_registry.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mlflow
from mlflow.tracking import MlflowClient

METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")
MODEL_PATH = os.path.join(ROOT, "models", "career_model.pkl")
EXPERIMENT = "CareerCast-Milestone3"


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file://" + os.path.join(ROOT, "mlruns")))
    mlflow.set_experiment(EXPERIMENT)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    models = metrics.get("models", {})
    best = metrics.get("best_model", "logistic_regression")

    with mlflow.start_run(run_name=f"register-{best}") as run:
        mlflow.log_param("best_model", best)
        mlflow.log_param("feature_type", metrics.get("feature_type", "tfidf"))
        for name, m in models.items():
            prefix = name
            for k in ("accuracy", "precision", "recall", "f1_score"):
                if k in m:
                    mlflow.log_metric(f"{prefix}_{k}", float(m[k]))

        if os.path.exists(MODEL_PATH):
            mlflow.log_artifact(MODEL_PATH, artifact_path="models")
        mlflow.log_artifact(METRICS_PATH, artifact_path="metrics")

        print("MLflow run id:", run.info.run_id)
        print("Best model:", best)
        print("Tracking URI:", mlflow.get_tracking_uri())

    # Register model name in registry (local file store)
    client = MlflowClient()
    model_name = "CareerCastCareerPredictor"
    try:
        client.create_registered_model(model_name)
        print(f"Created registered model: {model_name}")
    except Exception:
        print(f"Registered model already exists: {model_name}")

    print("Done. View with: mlflow ui --backend-store-uri", mlflow.get_tracking_uri())


if __name__ == "__main__":
    main()
