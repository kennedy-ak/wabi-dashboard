from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
import asyncio
import logging
import re
from datetime import datetime
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from api.models import CategoryResult, BatchResult

load_dotenv()

logger = logging.getLogger(__name__)

class ImageClassifier:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # Add thread pool for concurrent processing
        self.max_workers = 3  # Conservative to avoid rate limits

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
        """Process a batch of image URLs with concurrent processing"""
        
        # Create tasks for concurrent processing
        tasks = []
        for i, item in enumerate(batch_data):
            image_url = item.get("Image URL", item.get("image_url", ""))
            task = asyncio.create_task(self._process_product_async(image_url, i, item))
            tasks.append(task)
        
        # Process all items concurrently with limited concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        results = await asyncio.gather(*[self._process_with_semaphore(semaphore, task) for task in tasks])
        
        # Convert results to CategoryResult objects
        category_results = []
        for result in results:
            if result.get("success"):
                try:
                    category_result = self._create_category_result(result["result"], result["original_item"], result["index"])
                    category_results.append(category_result)
                except Exception as e:
                    category_result = self._create_error_result(result["original_item"], result["index"], f"Error creating result: {str(e)}")
                    category_results.append(category_result)
            else:
                category_result = self._create_error_result(result["original_item"], result["index"], result.get("error", "Unknown error"))
                category_results.append(category_result)
        
        successful_count = sum(1 for r in category_results if r.category != "ERROR")
        failed_count = len(category_results) - successful_count
        
        return BatchResult(
            batch_id=batch_id,
            results=category_results,
            timestamp=datetime.now().isoformat(),
            processing_type="image",
            total_processed=len(batch_data),
            successful=successful_count,
            failed=failed_count
        )
    
    async def _process_with_semaphore(self, semaphore: asyncio.Semaphore, task):
        """Process task with semaphore to limit concurrency"""
        async with semaphore:
            return await task
    
    async def _process_product_async(self, image_url: str, index: int, original_item: Dict[str, Any]):
        """Async wrapper for process_product"""
        loop = asyncio.get_event_loop()
        
        # Run the synchronous classify_image in a thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                result = await loop.run_in_executor(executor, self.process_product, image_url, index)
                result["original_item"] = original_item
                return result
            except Exception as e:
                return {
                    "index": index,
                    "image_url": image_url,
                    "error": f"Async processing error: {str(e)}",
                    "success": False,
                    "original_item": original_item
                }

# Example usage
if __name__ == "__main__":
    classifier = ImageClassifier()

    test_batch = [
        {"Image URL": "https://example.com/image1.jpg"},
        {"image_url": "https://example.com/image2.jpg"}
    ]

    batch_result = asyncio.run(classifier.classify_image_batch(test_batch, 1))
    print(json.dumps(batch_result.model_dump(), indent=2))
