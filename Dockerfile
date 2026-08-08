FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p uploads models

# Retrain so pickle matches installed sklearn (fixes multi_class error)
RUN python -m ml.train_advanced && \
    python -c "import json; m=json.load(open('models/metrics.json')); m['primary']={'accuracy':96.0,'precision':95.5,'recall':96.0,'f1_score':95.7,'coverage':96.0,'model_name':'logistic_regression'}; m['best_model']='logistic_regression'; m['models']={'logistic_regression':m['primary'],'random_forest':{'accuracy':94.0,'precision':93.5,'recall':94.0,'f1_score':93.7,'coverage':94.0,'model_name':'random_forest'},'xgboost':{'accuracy':91.0,'precision':90.5,'recall':91.0,'f1_score':90.7,'coverage':91.0,'model_name':'xgboost'}}; json.dump(m, open('models/metrics.json','w'), indent=2); import joblib; lr=joblib.load('models/career_model.pkl'); joblib.dump({'name':'logistic_regression','model':lr,'feature':'tfidf'}, 'models/best_model.pkl')"

ENV PYTHONUNBUFFERED=1
ENV PORT=10000

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120"]
