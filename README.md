# AI-Powered Career Intelligence Platform

## Milestone 1 – Parsing & Baseline Prediction
- Resume parsing (PDF/DOCX) + skill extraction
- TF-IDF + Logistic Regression baseline
- Career ranking + skill-gap recommendations
- Professional dashboard UI

## Milestone 2 – Advanced ML & Recommendation Engine
- **Random Forest** and **XGBoost** with cross-validated hyperparameter tuning (`RandomizedSearchCV`)
- **Top-K career ranking** with confidence scores and skill-alignment metrics
- Optional **Sentence-BERT** skill embeddings (`CAREERCAST_USE_SBERT=1`)
- Validation: stratified CV, Precision@K / MRR (SemEval-style), LinkedIn-style transition benchmark

### Train models
```bash
pip install -r requirements.txt
python -m ml.train_advanced          # TF-IDF + LR + RF + XGBoost
# Optional embeddings:
# pip install sentence-transformers torch
# CAREERCAST_USE_SBERT=1 python -m ml.train_advanced
python -m ml.evaluate                # benchmarks → models/evaluation_report.json
```

### Run locally
```bash
python app.py
# open http://127.0.0.1:5000
```

### Deploy (Render)
- Build: `pip install -r requirements.txt && python -m ml.train_advanced`
- Start: `gunicorn app:app`
- Optional env: `SESSION_SECRET`, `CAREERCAST_USE_SBERT=0`

### API
`POST /api/predict`
```json
{"skills": "python sql power bi", "degree": "B.Tech", "experience": "2", "top_k": 5}
```
