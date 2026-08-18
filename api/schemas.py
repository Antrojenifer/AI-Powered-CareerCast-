from __future__ import annotations
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class ProfileRequest(BaseModel):
    skills: str = Field(..., description="Comma-separated skills")
    name: Optional[str] = None
    email: Optional[str] = None
    degree: Optional[str] = "Not Specified"
    experience: Optional[float] = 0.0

class HealthResponse(BaseModel):
    status: str
    service: str
    milestone: str
