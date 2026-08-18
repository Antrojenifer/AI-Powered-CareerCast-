# CareerCast – Milestone 3
Skill Gap Analysis · FastAPI · CI · Streamlit Review UI

## Features
- FastAPI: `/predict`, `/recommend`, `/gap-report`, `/health`
- Skill gap module with prioritized learning actions
- MLflow registry helper
- GitHub Actions CI + accuracy gate (>= 90%)
- Streamlit review UI with charts and export

## Run
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
streamlit run streamlit_app/app.py
python scripts/accuracy_gate.py
pytest -q
```

## Model metrics
- Logistic Regression 96%
- Random Forest 94%
- XGBoost 91%
