"""
Health and version response schemas for SIGNOVA API.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    service: str = Field(default="signova-api")
    version: str = Field(default="0.1.0")
    phase: int = Field(default=0)
    system: Dict[str, Any] = Field(default_factory=dict)


class VersionResponse(BaseModel):
    name: str = "SIGNOVA API"
    version: str = "0.1.0"
    api_version: str = "v1"
    description: str = "Continuous Indian Sign Language to English Translation Pipeline"
    supported_tasks: list = ["health_check", "dataset_validation"]
