from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
import asyncio
import logging
import re
from datetime import datetime

from api.models import CategoryResult, BatchResult

load_dotenv()

logger = logging.getLogger(__name__)

class ImageClassifier:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _create_category_result(self, result_data: Dict[str, Any], original_item: Dict[str, Any], index: int) -> CategoryResult:
        """Create category classification result from parsed data"""
        # Extract product name
        product_name = (
            result_data.get('product_name') or
            original_item.get('Product Name') or
            original_item.get('product_name') or
            f"Furniture Item {index + 1}"
        )

        return CategoryResult(
            product_name=product_name,
            category=result_data.get('category', 'OTHER'),
            confidence=float(result_data.get('confidence', 0.5)),
            reasoning=result_data.get('reasoning', 'Category classification completed')
        )

    def _create_error_result(self, item: Dict[str, Any], index: int, error_message: str) -> CategoryResult:
        """Create error result"""
        product_name = (
            item.get('Product Name') or
            item.get('product_name') or
            f"Furniture Item {index + 1}"
        )

        return CategoryResult(
            product_name=product_name,
            category="ERROR",
            confidence=0.0,
            reasoning=error_message
        )

    def classify_image(self, image_url):
        """Classify a single image using GPT-4 vision with robust error handling"""
        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": """Analyze this furniture image and respond ONLY with valid JSON:
                    {
                        "product_name": "string",
                        "category": "SOFA|CHAIR|BED|TABLE|NIGHTSTAND|STOOL|OTHER",
                        "confidence": 0.95,
                        "reasoning": "string"
                    }
                    """},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            )

            response = self.model.invoke([message])

            # First try parsing as pure JSON
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                # If failed, try extracting JSON from markdown or other formatting
                json_str = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
                if json_str:
                    return json.loads(json_str.group(1))
                # If still no JSON, try parsing the raw content
                return {"raw_response": response.content, "error": "Could not parse as JSON"}

        except Exception as e:
            logger.error(f"Classification error for {image_url}: {str(e)}")
            return {"error": str(e), "image_url": image_url}

    def process_product(self, image_url, index):
        """Process a single image URL with improved error handling"""
        try:
            if not image_url or not image_url.startswith('http'):
                return {
                    "index": index,
                    "image_url": image_url,
                    "error": "Invalid image URL provided",
                    "success": False
                }

            logger.info(f"Processing image {index+1}: {image_url}")

            result = self.classify_image(image_url)

            if result and not result.get("error"):
                return {
                    "index": index,
                    "image_url": image_url,
                    "result": result,
                    "success": True
                }
            else:
                error_msg = result.get("error", "Classification failed") if result else "Empty response"
                return {
                    "index": index,
                    "image_url": image_url,
                    "error": f"Classification failed: {error_msg}",
                    "success": False
                }

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return {
                "index": index,
                "image_url": image_url,
                "error": f"Processing error: {str(e)}",
                "success": False
            }

    async def classify_image_batch(self, batch_data: List[Dict[str, Any]], batch_id: int) -> BatchResult:
        """Process a batch of image URLs"""
        results = []
        for i, item in enumerate(batch_data):
            image_url = item.get("Image URL", item.get("image_url", ""))
            processed_result = self.process_product(image_url, i)

            if processed_result.get("success"):
                try:
                    category_result = self._create_category_result(processed_result["result"], item, i)
                    results.append(category_result)
                except Exception as e:
                    logger.error(f"Error creating category result: {str(e)}")
                    category_result = self._create_error_result(item, i, f"Error creating category result: {str(e)}")
                    results.append(category_result)
            else:
                # Create an error CategoryResult
                error_msg = processed_result.get("error", "Unknown error")
                category_result = self._create_error_result(item, i, error_msg)
                results.append(category_result)

        successful_count = sum(1 for r in results if r.category != "ERROR")
        failed_count = len(results) - successful_count

        return BatchResult(
            batch_id=batch_id,
            results=results,
            timestamp=datetime.now().isoformat(),
            processing_type="image",
            total_processed=len(batch_data),
            successful=successful_count,
            failed=failed_count
        )

# Example usage
if __name__ == "__main__":
    classifier = ImageClassifier()

    test_batch = [
        {"Image URL": "https://example.com/image1.jpg"},
        {"image_url": "https://example.com/image2.jpg"}
    ]

    batch_result = asyncio.run(classifier.classify_image_batch(test_batch, 1))
    print(json.dumps(batch_result.model_dump(), indent=2))