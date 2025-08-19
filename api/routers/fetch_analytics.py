from fastapi import APIRouter, HTTPException
from api.services.langsmith_integration import fetch_langsmith_metrics_async
import json

langsmith_router = APIRouter(
    prefix="/langsmith",
    tags=["LangSmith Metrics"]
)

@langsmith_router.get("/metrics")
async def get_langsmith_metrics():
    """
    Fetch and return LangSmith metrics (latency, error rates, trends).
    """
    metrics = await fetch_langsmith_metrics_async()
    if not metrics:
        raise HTTPException(status_code=404, detail="No LangSmith metrics available")
    return json.dumps(metrics, indent=2)
