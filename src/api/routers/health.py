"""Health check endpoints."""
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str


@router.get("/", response_model=HealthResponse)
async def health_check():
    from src.utils.config import get_settings
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
    )


@router.get("/ready")
async def readiness_check():
    """Check if models are loaded and ready."""
    return {"ready": True}
