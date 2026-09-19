"""
Health and version API routes for SIGNOVA.
"""

from fastapi import APIRouter
from apps.api.schemas.health import HealthResponse, VersionResponse
from apps.api.services.system_service import get_system_health

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint confirming API operational status."""
    return HealthResponse(
        status="ok",
        service="signova-api",
        version="0.1.0",
        phase=0,
        system=get_system_health(),
    )


@router.get("/version", response_model=VersionResponse)
async def version_info():
    """Version and capability descriptor endpoint."""
    return VersionResponse()
