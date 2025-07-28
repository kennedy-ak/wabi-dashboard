from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class CategoryClassificationRequest(BaseModel):
    data: List[Dict[str, Any]]
    toggle: int  # 1 for text, 0 for URL/image
    product_column: str = "Product URL"

class CategoryResult(BaseModel):
    product_name: str
    category: str
    primary_style: Optional[str] = None
    secondary_style: Optional[str] = None
    style_tags: List[str] = []
    placement_tags: List[str] = []
    confidence: float
    reasoning: str

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