# CareerCast – Milestone 3
## Skill Gap Analysis, API & CI Integration

Extends Milestone 1 + 2 with:
- FastAPI REST service (`/predict`, `/recommend`, `/gap-report`)
- Skill gap analysis module with actionable suggestions
- MLflow model registry helper
- GitHub Actions CI with accuracy gate (>= 90%)
- Streamlit review UI with charts and report export

## Quick Start

```bash
pip install -r requirements.txt

# FastAPI
uvicorn api.main:app --reload --port 8000
# Docs: http://127.0.0.1:8000/docs

# Streamlit UI
streamlit run streamlit_app/app.py

# Accuracy gate
python scripts/accuracy_gate.py

# MLflow registry log
python scripts/mlflow_registry.py

# Tests
pytest -q
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Career prediction |
| POST | `/recommend` | Top-K recommendation |
| POST | `/gap-report` | Skill gap report |

### Example body
```json
{
  "skills": "Python, SQL, Machine Learning, Pandas",
  "degree": "B.Tech",
  "experience": 1.0
}
```

## Model Metrics (from Milestone 2)
- Logistic Regression: 96%
- Random Forest: 94%
- XGBoost: 91%

## Project layout
```
api/               FastAPI service
ml/                ML + skill_gap
streamlit_app/     Review UI
scripts/           MLflow + accuracy gate
.github/workflows/ CI pipeline
models/            Trained artifacts
```
