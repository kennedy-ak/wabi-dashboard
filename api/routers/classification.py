from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import asyncio
import json
import logging
from datetime import datetime

from api.models.schemas import CategoryClassificationRequest, BatchResult
from api.services.furniture_classifier import FurnitureCategoryClassifier

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize classifier outside the function
furniture_classifier = FurnitureCategoryClassifier()

async def generate_classification_stream(data: List[Dict[str, Any]], toggle: int, request_id: str):
    """Generate streaming response for furniture category classification"""
    batch_size = 5  # Process in batches of 5
    total_items = len(data)

    logger.info(f"Request ID: {request_id} - Processing {total_items} furniture items with toggle={toggle}, batch_size={batch_size}")

    # Process in batches
    for i in range(0, total_items, batch_size):
        batch_data = data[i:i + batch_size]
        batch_id = i // batch_size + 1

        logger.info(f"Request ID: {request_id} - Processing furniture batch {batch_id} with {len(batch_data)} items")

        try:
            if toggle == 1:
                # Text classification
                result = await furniture_classifier.classify_text_batch(batch_data, batch_id)
            else:
                # Image classification using direct image URLs
                result = await furniture_classifier.classify_image_batch(batch_data, batch_id)

            # Yield result as JSON
            yield f"data: {json.dumps(result.model_dump())}\n\n"

            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Request ID: {request_id} - Error processing furniture batch {batch_id}: {str(e)}")
            error_result = BatchResult(
                batch_id=batch_id,
                results=[],
                timestamp=datetime.now().isoformat(),
                processing_type="error",
                error=str(e)
            )
            yield f"data: {json.dumps(error_result.model_dump())}\n\n"

    # Send completion signal
    completion_result = BatchResult(
        batch_id=-1,
        results=[],
        timestamp=datetime.now().isoformat(),
        processing_type="complete",
        total_processed=total_items
    )
    yield f"data: {json.dumps(completion_result.model_dump())}\n\n"

@router.post("/classify-category")
async def classify_furniture_category(request: CategoryClassificationRequest):
    """
    Classify furniture categories based on text or image data
    """
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    try:
        # Validate request
        if not request.data:
            raise HTTPException(status_code=400, detail="No data provided")

        if request.toggle not in [0, 1]:
            raise HTTPException(status_code=400, detail="Toggle must be 0 or 1")

        # Check if product_url or image_url exists when using image mode
        if request.toggle == 0:
            for item in request.data:
                if not item.product_url and not item.image_url:
                    raise HTTPException(status_code=400, detail="product_url or image_url is required for image classification")

        logger.info(f"Request ID: {request_id} - Starting furniture category classification for {len(request.data)} items, toggle={request.toggle}")

        return StreamingResponse(
            generate_classification_stream([item.model_dump() for item in request.data], request.toggle, request_id),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Unexpected error in classify_furniture_category: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
