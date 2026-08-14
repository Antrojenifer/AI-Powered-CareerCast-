"""Pydantic request/response models for Milestone 3 FastAPI service."""
from __future__ import annotations
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class ProfileRequest(BaseModel):
    skills: str = Field(..., description="Comma or space separated skills")
    name: Optional[str] = None
    email: Optional[str] = None
    degree: Optional[str] = "Not Specified"
    experience: Optional[float] = 0.0


class PredictionResponse(BaseModel):
    predicted_career: str
    confidence: float
    top_careers: List[Dict[str, Any]]
    model_used: str
    user_skills: List[str]


class RecommendationResponse(BaseModel):
    predicted_career: str
    confidence: float
    top_careers: List[Dict[str, Any]]
    skill_alignment: float
    model_used: str
    user_skills: List[str]


class GapReportResponse(BaseModel):
    primary_career: Optional[str]
    primary_gap: Optional[Dict[str, Any]]
    career_gaps: List[Dict[str, Any]]
    user_skills: List[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    milestone: str
