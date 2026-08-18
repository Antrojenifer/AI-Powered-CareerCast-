"""Milestone 3 FastAPI – /predict /recommend /gap-report /health"""
from __future__ import annotations
import os, sys
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.schemas import ProfileRequest, HealthResponse
from ml.predict import predict_career
from ml.skill_gap import build_gap_report

app = FastAPI(title="CareerCast API", description="Milestone 3 REST API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def _parse_skills(skills: str) -> List[str]:
    if "," in (skills or ""):
        return [p.strip() for p in skills.replace(";", ",").split(",") if p.strip()]
    return [p.strip() for p in (skills or "").split() if p.strip()]

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "service": "CareerCast FastAPI", "milestone": "3"}

@app.post("/predict")
def predict(req: ProfileRequest):
    try:
        r = predict_career(skills=req.skills, degree=req.degree or "Not Specified", experience=float(req.experience or 0), top_k=5)
        return {"predicted_career": r["predicted_career"], "confidence": r["confidence"], "top_careers": r["top_careers"], "model_used": r.get("model_used", "logistic_regression"), "user_skills": r.get("user_skills", _parse_skills(req.skills))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
def recommend(req: ProfileRequest):
    try:
        r = predict_career(skills=req.skills, degree=req.degree or "Not Specified", experience=float(req.experience or 0), top_k=5)
        return {"predicted_career": r["predicted_career"], "confidence": r["confidence"], "top_careers": r["top_careers"], "skill_alignment": r.get("skill_alignment", r["confidence"]), "model_used": r.get("model_used", "logistic_regression"), "user_skills": r.get("user_skills", _parse_skills(req.skills))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gap-report")
def gap_report(req: ProfileRequest):
    try:
        r = predict_career(skills=req.skills, degree=req.degree or "Not Specified", experience=float(req.experience or 0), top_k=5)
        user_skills = r.get("user_skills") or _parse_skills(req.skills)
        return build_gap_report(user_skills, r.get("top_careers", []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "CareerCast Milestone 3 API", "docs": "/docs", "endpoints": ["/health", "/predict", "/recommend", "/gap-report"]}
