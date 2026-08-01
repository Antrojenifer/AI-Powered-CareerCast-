# CareerCast Pro – AI Career Path Prediction

Professional AI-powered career recommendation system.

## Run Locally
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python ml/train.py
python app.py
```

## Deploy on Render
- Build Command: `pip install -r requirements.txt && python -m spacy download en_core_web_sm && python ml/train.py`
- Start Command: `gunicorn app:app`
- Add environment variable (optional): `SECRET_KEY` or `SESSION_SECRET`
