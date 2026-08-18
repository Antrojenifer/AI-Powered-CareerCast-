import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mlflow
from mlflow.tracking import MlflowClient

METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")
MODEL_PATH = os.path.join(ROOT, "models", "career_model.pkl")

def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file://" + os.path.join(ROOT, "mlruns")))
    mlflow.set_experiment("CareerCast-Milestone3")
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    models = metrics.get("models", {})
    best = metrics.get("best_model", "logistic_regression")
    with mlflow.start_run(run_name=f"register-{best}") as run:
        mlflow.log_param("best_model", best)
        mlflow.log_param("feature_type", metrics.get("feature_type", "tfidf"))
        for name, m in models.items():
            for k in ("accuracy", "precision", "recall", "f1_score"):
                if k in m:
                    mlflow.log_metric(f"{name}_{k}", float(m[k]))
        if os.path.exists(MODEL_PATH):
            mlflow.log_artifact(MODEL_PATH, artifact_path="models")
        mlflow.log_artifact(METRICS_PATH, artifact_path="metrics")
        print("run_id", run.info.run_id)
    client = MlflowClient()
    try:
        client.create_registered_model("CareerCastCareerPredictor")
    except Exception:
        pass
    print("Done")

if __name__ == "__main__":
    main()
