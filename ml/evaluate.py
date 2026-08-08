"""
Milestone 2 – Validation & Benchmark Evaluation
- Stratified cross-validation
- Model comparison (LR vs RF vs XGBoost)
- LinkedIn-style career transition holdout evaluation
- SemEval-style ranking metrics (Precision@K, MRR)
"""

from __future__ import annotations

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "career_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_PATH = os.path.join(MODEL_DIR, "evaluation_report.json")


def load_xy():
    df = pd.read_csv(DATASET_PATH)
    df["features"] = (
        df["Skills"].astype(str) + " " +
        df["Degree"].astype(str) + " " +
        df["Experience"].astype(str) + " years experience"
    )
    return df["features"], df["Career"], df


def precision_at_k(y_true, proba, classes, k=3):
    """SemEval-style Precision@K for multi-class ranking."""
    hits = 0
    for i, true_label in enumerate(y_true):
        top_idx = np.argsort(proba[i])[::-1][:k]
        top_labels = [classes[j] for j in top_idx]
        if true_label in top_labels:
            hits += 1
    return round(hits / len(y_true) * 100, 1)


def mean_reciprocal_rank(y_true, proba, classes):
    """MRR – common ranking metric used in SemEval-style evaluations."""
    rr = []
    for i, true_label in enumerate(y_true):
        order = np.argsort(proba[i])[::-1]
        ranked = [classes[j] for j in order]
        if true_label in ranked:
            rank = ranked.index(true_label) + 1
            rr.append(1.0 / rank)
        else:
            rr.append(0.0)
    return round(float(np.mean(rr)), 3)


def linkedin_transition_eval(df: pd.DataFrame):
    """
    Curated LinkedIn-style transition evaluation:
    Simulate career transitions by pairing junior profiles with target careers
    and measuring whether the model ranks the target career in Top-3.
    """
    # Build simple transition pairs from same-domain progressions
    transitions = [
        ("python sql excel power bi", "Data Analyst"),
        ("python machine learning pandas scikit-learn", "Data Scientist"),
        ("python deep learning tensorflow pytorch", "ML Engineer"),
        ("html css javascript react", "Frontend Developer"),
        ("python flask django sql", "Backend Developer"),
        ("docker kubernetes aws linux", "DevOps Engineer"),
        ("python spark sql etl", "Data Engineer"),
        ("python nlp transformers spacy", "NLP Engineer"),
    ]

    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    lr_path = os.path.join(MODEL_DIR, "career_model.pkl")

    if not os.path.exists(vectorizer_path):
        return {"note": "Models not trained yet", "top3_hit_rate": None}

    vectorizer = joblib.load(vectorizer_path)
    if os.path.exists(best_path):
        bundle = joblib.load(best_path)
    else:
        bundle = {"name": "logistic_regression", "model": joblib.load(lr_path), "feature": "tfidf"}

    hits = 0
    details = []
    for skills, target in transitions:
        text = f"{skills} B.Tech Computer Science 2 years experience"
        X = vectorizer.transform([text])
        name = bundle.get("name", "logistic_regression")
        model = bundle["model"]

        if name == "xgboost":
            xgb, le = model["model"], model["label_encoder"]
            proba = xgb.predict_proba(X.toarray())[0]
            classes = list(le.classes_)
        elif name == "random_forest":
            proba = model.predict_proba(X.toarray())[0]
            classes = list(model.classes_)
        else:
            proba = model.predict_proba(X)[0]
            classes = list(model.classes_)

        top3 = [classes[i] for i in np.argsort(proba)[::-1][:3]]
        hit = target in top3
        hits += int(hit)
        details.append({"skills": skills, "target": target, "top3": top3, "hit": hit})

    return {
        "top3_hit_rate": round(hits / len(transitions) * 100, 1),
        "n_transitions": len(transitions),
        "details": details,
        "note": "Curated LinkedIn-style skill→career transition pairs",
    }


def run_evaluation():
    X_text, y, df = load_xy()
    report = {"dataset_size": len(df), "n_classes": int(y.nunique())}

    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic Regression baseline
    lr = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    proba = lr.predict_proba(X_test)
    report["logistic_regression"] = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 1),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 1),
        "precision_at_3": precision_at_k(list(y_test), proba, lr.classes_, k=3),
        "mrr": mean_reciprocal_rank(list(y_test), proba, lr.classes_),
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train.toarray(), y_train)
    y_pred_rf = rf.predict(X_test.toarray())
    proba_rf = rf.predict_proba(X_test.toarray())
    report["random_forest"] = {
        "accuracy": round(accuracy_score(y_test, y_pred_rf) * 100, 1),
        "f1_weighted": round(f1_score(y_test, y_pred_rf, average="weighted", zero_division=0) * 100, 1),
        "precision_at_3": precision_at_k(list(y_test), proba_rf, rf.classes_, k=3),
        "mrr": mean_reciprocal_rank(list(y_test), proba_rf, rf.classes_),
    }

    # Cross-validation
    min_class = int(y.value_counts().min())
    n_splits = max(2, min(3, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_lr = cross_val_score(lr, X, y, cv=cv, scoring="f1_weighted")
    cv_rf = cross_val_score(rf, X.toarray(), y, cv=cv, scoring="f1_weighted")
    report["cross_validation"] = {
        "n_splits": n_splits,
        "logistic_regression_f1_mean": round(float(cv_lr.mean()) * 100, 1),
        "logistic_regression_f1_std": round(float(cv_lr.std()) * 100, 1),
        "random_forest_f1_mean": round(float(cv_rf.mean()) * 100, 1),
        "random_forest_f1_std": round(float(cv_rf.std()) * 100, 1),
    }

    # LinkedIn-style transitions
    report["linkedin_transition_benchmark"] = linkedin_transition_eval(df)

    # SemEval-style summary
    report["semeval_style_ranking"] = {
        "protocol": "Precision@K and MRR on held-out test set (SemEval-style ranking evaluation)",
        "precision_at_3_lr": report["logistic_regression"]["precision_at_3"],
        "precision_at_3_rf": report["random_forest"]["precision_at_3"],
        "mrr_lr": report["logistic_regression"]["mrr"],
        "mrr_rf": report["random_forest"]["mrr"],
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(json.dumps(report, indent=2))
    logger.info(f"Report saved → {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_evaluation()
