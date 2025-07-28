from fastapi import APIRouter
from datetime import datetime
import os

from ..models import HealthResponse, RootResponse
from config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        openai_configured=bool(settings.OPENAI_API_KEY),
        service="Furniture Category Classification API"
    )

@router.get("/", response_model=RootResponse)
async def root():
    """Root endpoint"""
    return RootResponse(
        message="Furniture Category Classification API",
        version=settings.API_VERSION,
        endpoints={
            "classify-category": "/classify-category",
            "health": "/health"
        },
        supported_categories=settings.FURNITURE_CATEGORIES
    )