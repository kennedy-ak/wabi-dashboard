from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class FurnitureItem(BaseModel):
    """Pydantic model for furniture item data."""
    product_name: str
    type: Optional[str] = None
    style: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    price_range_usd: Optional[str] = None
    product_url: Optional[str] = None
    image_url: Optional[str] = None
    suggested_placement: Optional[List[str]] = None

class CategoryClassificationRequest(BaseModel):
    """Request model for category classification."""
    data: List[FurnitureItem]
    toggle: int  # 1 for text, 0 for image

class CategoryResult(BaseModel):
    name: str
    predicted_category: str
    confidence: float
    reasoning: str
    description: Optional[str] = None
    embedding: Optional[List[float]] = None

class BatchResult(BaseModel):
    batch_id: int
    results: List[CategoryResult]
    timestamp: str
    processing_type: str
    error: Optional[str] = None
    total_processed: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    openai_configured: bool
    service: str

class RootResponse(BaseModel):
    message: str
    version: str
    endpoints: Dict[str, str]
    supported_categories: List[str]