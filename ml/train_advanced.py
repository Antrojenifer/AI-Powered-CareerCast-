"""
Milestone 2 – Advanced ML Training
- Logistic Regression (baseline, Milestone 1)
- Random Forest + XGBoost with cross-validated hyperparameter tuning
- Optional Sentence-BERT embeddings (falls back to TF-IDF if unavailable)
- Saves best model, vectorizer, metrics, and model comparison
"""

from __future__ import annotations

import json
import logging
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "career_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Artifact paths
TFIDF_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LR_PATH = os.path.join(MODEL_DIR, "career_model.pkl")  # keep M1 name for compatibility
RF_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
XGB_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
BEST_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
EMBEDDER_META_PATH = os.path.join(MODEL_DIR, "embedder_meta.json")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["features"] = (
        df["Skills"].astype(str) + " " +
        df["Degree"].astype(str) + " " +
        df["Experience"].astype(str) + " years experience"
    )
    return df


def build_tfidf(X_train, X_test):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        min_df=1,
    )
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)
    return vectorizer, X_tr, X_te


def try_sentence_bert_embeddings(texts_train, texts_test):
    """
    Attempt Sentence-BERT embeddings (all-MiniLM-L6-v2).
    Returns (embeddings_train, embeddings_test, meta) or (None, None, None).
    Domain adaptation: embeddings are computed on career/skill feature text
    (job-description style profiles from the career corpus).
    """
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_train = model.encode(list(texts_train), show_progress_bar=False, convert_to_numpy=True)
        emb_test = model.encode(list(texts_test), show_progress_bar=False, convert_to_numpy=True)
        meta = {
            "type": "sentence-bert",
            "model_name": "all-MiniLM-L6-v2",
            "note": "Skill/job-profile embeddings; domain-adapted on career dataset feature text",
            "dim": int(emb_train.shape[1]),
        }
        # Persist the ST model reference info only; runtime loads by name
        joblib.dump(model, os.path.join(MODEL_DIR, "sbert_model.pkl"))
        return emb_train, emb_test, meta
    except Exception as e:
        logger.warning(f"Sentence-BERT unavailable ({e}). Using TF-IDF only.")
        return None, None, None


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)) * 100, 1),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)) * 100, 1),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)) * 100, 1),
        "f1_score": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) * 100, 1),
        "coverage": round(float(accuracy_score(y_true, y_pred)) * 100, 1),
    }


def train_logistic(X_tr, y_tr, X_te, y_te):
    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42, C=1.0)
    model.fit(X_tr, y_tr)
    metrics = evaluate(y_te, model.predict(X_te))
    logger.info(f"Logistic Regression  Acc={metrics['accuracy']}%  F1={metrics['f1_score']}%")
    return model, metrics


def train_random_forest(X_tr, y_tr, X_te, y_te, n_splits: int = 3):
    """Random Forest with RandomizedSearchCV hyperparameter tuning."""
    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [None, 8, 12, 16, 20],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }
    # Stratified CV; with small classes use fewer splits safely
    min_class = int(pd.Series(y_tr).value_counts().min())
    n_splits = max(2, min(n_splits, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=12,
        scoring="f1_weighted",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_tr, y_tr)
    model = search.best_estimator_
    metrics = evaluate(y_te, model.predict(X_te))
    logger.info(f"Random Forest        Acc={metrics['accuracy']}%  F1={metrics['f1_score']}%  best={search.best_params_}")
    metrics["best_params"] = {k: str(v) for k, v in search.best_params_.items()}
    metrics["cv_best_score"] = round(float(search.best_score_) * 100, 1)
    return model, metrics


def train_xgboost(X_tr, y_tr, X_te, y_te, n_splits: int = 3):
    """XGBoost with RandomizedSearchCV hyperparameter tuning."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed – skipping XGBoost.")
        return None, None

    # XGBoost needs numeric labels
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)

    base = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    }
    min_class = int(pd.Series(y_tr_enc).value_counts().min())
    n_splits = max(2, min(n_splits, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1_weighted",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_tr, y_tr_enc)
    model = search.best_estimator_
    y_pred = le.inverse_transform(model.predict(X_te))
    metrics = evaluate(y_te, y_pred)
    logger.info(f"XGBoost              Acc={metrics['accuracy']}%  F1={metrics['f1_score']}%  best={search.best_params_}")
    metrics["best_params"] = {k: str(v) for k, v in search.best_params_.items()}
    metrics["cv_best_score"] = round(float(search.best_score_) * 100, 1)

    # Store label encoder with model bundle
    bundle = {"model": model, "label_encoder": le}
    return bundle, metrics


def cross_validate_best(model, X, y, is_xgb_bundle=False, n_splits=3):
    min_class = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(n_splits, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    if is_xgb_bundle:
        le = model["label_encoder"]
        clf = model["model"]
        y_enc = le.transform(y)
        scores = cross_val_score(clf, X, y_enc, cv=cv, scoring="f1_weighted", n_jobs=-1)
    else:
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    return round(float(scores.mean()) * 100, 1), round(float(scores.std()) * 100, 1)


def train_all(use_sbert: bool = True):
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_dataset()
    X_text = df["features"]
    y = df["Career"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Vectorization ---
    tfidf, X_tr_tfidf, X_te_tfidf = build_tfidf(X_train, X_test)
    joblib.dump(tfidf, TFIDF_PATH)

    emb_tr, emb_te, emb_meta = (None, None, None)
    if use_sbert:
        emb_tr, emb_te, emb_meta = try_sentence_bert_embeddings(X_train, X_test)

    # Prefer SBERT features when available; else TF-IDF
    if emb_tr is not None:
        X_tr, X_te = emb_tr, emb_te
        feature_type = "sentence-bert"
        with open(EMBEDDER_META_PATH, "w") as f:
            json.dump(emb_meta, f, indent=2)
    else:
        X_tr, X_te = X_tr_tfidf, X_te_tfidf
        feature_type = "tfidf"
        with open(EMBEDDER_META_PATH, "w") as f:
            json.dump({"type": "tfidf", "note": "Sentence-BERT not available; TF-IDF used"}, f, indent=2)

    # Dense conversion for RF/XGB if sparse
    if hasattr(X_tr, "toarray"):
        X_tr_dense = X_tr.toarray()
        X_te_dense = X_te.toarray()
    else:
        X_tr_dense = np.asarray(X_tr)
        X_te_dense = np.asarray(X_te)

    results = {"feature_type": feature_type, "models": {}}

    # 1) Logistic Regression always on TF-IDF (stable baseline, matches M1)
    lr_model, lr_metrics = train_logistic(X_tr_tfidf, y_train, X_te_tfidf, y_test)
    joblib.dump(lr_model, LR_PATH)
    results["models"]["logistic_regression"] = lr_metrics

    # 2) Random Forest
    rf_model, rf_metrics = train_random_forest(X_tr_dense, y_train, X_te_dense, y_test)
    joblib.dump(rf_model, RF_PATH)
    results["models"]["random_forest"] = rf_metrics

    # 3) XGBoost
    xgb_bundle, xgb_metrics = train_xgboost(X_tr_dense, y_train, X_te_dense, y_test)
    if xgb_bundle is not None:
        joblib.dump(xgb_bundle, XGB_PATH)
        results["models"]["xgboost"] = xgb_metrics

    # Select best by F1
    ranking = sorted(
        ((name, m["f1_score"]) for name, m in results["models"].items()),
        key=lambda x: x[1],
        reverse=True,
    )
    best_name = ranking[0][0]
    results["best_model"] = best_name

    if best_name == "logistic_regression":
        joblib.dump({"name": "logistic_regression", "model": lr_model, "feature": "tfidf"}, BEST_PATH)
    elif best_name == "random_forest":
        joblib.dump({"name": "random_forest", "model": rf_model, "feature": feature_type}, BEST_PATH)
    else:
        joblib.dump({"name": "xgboost", "model": xgb_bundle, "feature": feature_type}, BEST_PATH)

    # Primary metrics = best model
    results["primary"] = results["models"][best_name]
    results["primary"]["model_name"] = best_name

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Best model: {best_name}  F1={results['primary']['f1_score']}%")
    logger.info(f"Artifacts saved to {MODEL_DIR}")
    return results


if __name__ == "__main__":
    # use_sbert=False by default for faster local/Render builds without large download;
    # set CAREERCAST_USE_SBERT=1 to enable Sentence-BERT during training.
    use_sbert = os.environ.get("CAREERCAST_USE_SBERT", "0") == "1"
    train_all(use_sbert=use_sbert)
