"""
Milestone 3 – FastAPI REST Service
Endpoints:
  GET  /health
  POST /predict
  POST /recommend
  POST /gap-report
"""
from __future__ import annotations

import os
import sys
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow importing project root modules
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.schemas import (
    ProfileRequest,
    PredictionResponse,
    RecommendationResponse,
    GapReportResponse,
    HealthResponse,
)
from ml.predict import predict_career
from ml.skill_gap import build_gap_report, analyze_skill_gap

app = FastAPI(
    title="CareerCast API",
    description="Milestone 3 – Prediction, Recommendation, and Skill Gap REST API",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_skills(skills: str) -> List[str]:
    raw = (skills or "").replace(";", ",").replace("\n", ",")
    parts = []
    for chunk in raw.split(","):
        for token in chunk.split():
            t = token.strip()
            if t:
                parts.append(t)
    # also keep comma-separated multi-word skills
    if "," in (skills or ""):
        parts = [p.strip() for p in skills.replace(";", ",").split(",") if p.strip()]
    return parts


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "service": "CareerCast FastAPI",
        "milestone": "3",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: ProfileRequest):
    try:
        result = predict_career(
            skills=req.skills,
            degree=req.degree or "Not Specified",
            experience=float(req.experience or 0),
            top_k=5,
        )
        return {
            "predicted_career": result["predicted_career"],
            "confidence": result["confidence"],
            "top_careers": result["top_careers"],
            "model_used": result.get("model_used", "logistic_regression"),
            "user_skills": result.get("user_skills", _parse_skills(req.skills)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: ProfileRequest):
    try:
        result = predict_career(
            skills=req.skills,
            degree=req.degree or "Not Specified",
            experience=float(req.experience or 0),
            top_k=5,
        )
        return {
            "predicted_career": result["predicted_career"],
            "confidence": result["confidence"],
            "top_careers": result["top_careers"],
            "skill_alignment": result.get("skill_alignment", result["confidence"]),
            "model_used": result.get("model_used", "logistic_regression"),
            "user_skills": result.get("user_skills", _parse_skills(req.skills)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gap-report", response_model=GapReportResponse)
def gap_report(req: ProfileRequest):
    try:
        result = predict_career(
            skills=req.skills,
            degree=req.degree or "Not Specified",
            experience=float(req.experience or 0),
            top_k=5,
        )
        user_skills = result.get("user_skills") or _parse_skills(req.skills)
        report = build_gap_report(user_skills, result.get("top_careers", []))
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "CareerCast Milestone 3 API",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/recommend", "/gap-report"],
    }
