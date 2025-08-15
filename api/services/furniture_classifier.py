"""Furniture Category Classifier with multithreading support."""

import asyncio
import base64
import concurrent.futures
import json
import logging
import os
from datetime import datetime
from functools import partial
from typing import List, Dict, Any, Optional

import openai
import requests
from dotenv import load_dotenv

from api.models.schemas import CategoryResult, BatchResult
from api.utils.constants import FURNITURE_CATEGORY_CLASSIFICATION_PROMPT
from api.services.embedding_service import EmbeddingService
from api.services.description_service import DescriptionService

load_dotenv()

logger = logging.getLogger(__name__)


class FurnitureCategoryClassifier:
    """Furniture category classifier with multithreading support."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.max_workers = min(32, (os.cpu_count() or 1) + 4)  # Limit concurrent threads
        self.embedding_service = EmbeddingService()
        self.description_service = DescriptionService()

    def _classify_single_text_item(self, item: Dict[str, Any], index: int) -> CategoryResult:
        """Classify a single item using text data - synchronous function for threading"""
        try:
            # Extract relevant text fields for classification
            text_parts = []

            # Add product name if available
            if "product_name" in item:
                text_parts.append(f"Product Name: {item['product_name']}")

            # Add type if available
            if "type" in item:
                text_parts.append(f"Type: {item['type']}")

            # Add style if available
            if "style" in item and item["style"]:
                if isinstance(item["style"], list):
                    style_str = ", ".join(item["style"])
                else:
                    style_str = str(item["style"])
                text_parts.append(f"Style: {style_str}")

            # Add tags if available
            if "tags" in item and item["tags"]:
                if isinstance(item["tags"], list):
                    tags_str = ", ".join(item["tags"])
                else:
                    tags_str = str(item["tags"])
                text_parts.append(f"Tags: {tags_str}")

            # Add price range if available
            if "price_range_usd" in item:
                text_parts.append(f"Price Range: {item['price_range_usd']}")

            furniture_description = " | ".join(text_parts)

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": FURNITURE_CATEGORY_CLASSIFICATION_PROMPT},
                    {"role": "user", "content": f"Classify this furniture item: {furniture_description}"}
                ],
                temperature=0.3,
                max_tokens=500
            )

            classification_text = response.choices[0].message.content

            # Parse JSON response
            try:
                # Extract JSON from response
                if '{' in classification_text:
                    json_start = classification_text.find('{')
                    json_end = classification_text.rfind('}') + 1
                    json_str = classification_text[json_start:json_end]
                    result_data = json.loads(json_str)

                    return self._create_category_result(result_data, item, index, None)
                else:
                    return self._create_error_result(item, index, "No JSON found in response")

            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON for item {index}: {e}")
                return self._create_error_result(item, index, "Could not parse classification response")

        except Exception as api_error:
            logger.error(f"OpenAI API error for item {index}: {str(api_error)}")
            return self._create_error_result(item, index, f"API error: {str(api_error)}")

    def _classify_single_image_item(self, item: Dict[str, Any], index: int) -> CategoryResult:
        """Classify a single item using image data - synchronous function for threading"""
        try:
            # Prioritize direct image_url over product_url for better success rate
            image_url = item.get("image_url", "") or item.get("product_url", "")

            if not image_url:
                return self._create_error_result(item, index, "No image_url or product_url provided")

            # Download and encode the image (synchronous version)
            logger.info(f"Processing image for item {index+1}: {image_url}")
            image_data = self._download_and_encode_image_sync(image_url)

            if not image_data:
                # Fallback to text classification if image download fails
                logger.warning(f"Image download failed for item {index+1}, falling back to text classification")
                return self._classify_single_text_item(item, index)

            # Use OpenAI Vision API for classification
            context_text = self._build_context_text(item)
            prompt_text = f"{FURNITURE_CATEGORY_CLASSIFICATION_PROMPT}\n\nAnalyze this furniture image and classify its category."
            if context_text:
                prompt_text += f"\n\nAdditional context: {context_text}"

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            classification_text = response.choices[0].message.content

            # Parse JSON response
            try:
                if '{' in classification_text:
                    json_start = classification_text.find('{')
                    json_end = classification_text.rfind('}') + 1
                    json_str = classification_text[json_start:json_end]
                    result_data = json.loads(json_str)
                    return self._create_category_result(result_data, item, index, image_data)
                else:
                    return self._create_error_result(item, index, "Image analyzed but no JSON response")
            except json.JSONDecodeError:
                return self._create_error_result(item, index, "Could not parse vision response")

        except Exception as vision_error:
            logger.error(f"Vision API error for item {index}: {str(vision_error)}")
            return self._create_error_result(item, index, f"Vision processing error: {str(vision_error)}")

    def _normalize_field_name(self, item: Dict[str, Any], field_variations: List[str]) -> Any:
        """Get field value using multiple possible field names"""
        for field_name in field_variations:
            if field_name in item and item[field_name] is not None:
                return item[field_name]
        return None

    def _extract_product_info(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize product information from item"""
        return {
            'product_name': self._normalize_field_name(item, ['product_name', 'Product_Name', 'Product Name']),
            'type': self._normalize_field_name(item, ['type', 'Type']),
            'style': self._normalize_field_name(item, ['style', 'Style']),
            'tags': self._normalize_field_name(item, ['tags', 'Tags']),
            'price_range': self._normalize_field_name(item, ['price_range_usd', 'Price_Range_USD', 'price_range']),
            'product_url': self._normalize_field_name(item, ['product_url', 'Product_URL', 'Product URL']),
            'image_url': self._normalize_field_name(item, ['image_url', 'Image_URL', 'ImageUrl'])
        }

    async def classify_text_batch(self, batch_data: List[Dict[str, Any]], batch_id: int) -> BatchResult:
        """Classify furniture categories using text data with multithreading"""
        try:
            logger.info(f"Processing text batch {batch_id} with {len(batch_data)} items using multithreading")

            # Use ThreadPoolExecutor for concurrent processing
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create partial function with the method bound
                classify_func = partial(self._classify_single_text_item)

                # Submit all tasks to thread pool
                futures = [
                    loop.run_in_executor(executor, classify_func, item, i)
                    for i, item in enumerate(batch_data)
                ]

                # Wait for all results
                results = await asyncio.gather(*futures, return_exceptions=True)

                # Handle any exceptions
                final_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error in thread for item {i}: {str(result)}")
                        final_results.append(self._create_error_result(batch_data[i], i, f"Threading error: {str(result)}"))
                    else:
                        final_results.append(result)

            return BatchResult(
                batch_id=batch_id,
                results=final_results,
                timestamp=datetime.now().isoformat(),
                processing_type="text_multithreaded"
            )

        except Exception as e:
            logger.error(f"Error in text classification: {str(e)}")
            return BatchResult(
                batch_id=batch_id,
                results=[
                    self._create_error_result(item, i, str(e))
                    for i, item in enumerate(batch_data)
                ],
                timestamp=datetime.now().isoformat(),
                processing_type="text_multithreaded"
            )

    async def classify_image_batch(self, batch_data: List[Dict[str, Any]], batch_id: int) -> BatchResult:
        """Classify furniture categories using provided image URLs with multithreading"""
        try:
            logger.info(f"Processing image batch {batch_id} with {len(batch_data)} items using multithreading")

            # Use ThreadPoolExecutor for concurrent processing
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create partial function with the method bound
                classify_func = partial(self._classify_single_image_item)

                # Submit all tasks to thread pool
                futures = [
                    loop.run_in_executor(executor, classify_func, item, i)
                    for i, item in enumerate(batch_data)
                ]

                # Wait for all results
                results = await asyncio.gather(*futures, return_exceptions=True)

                # Handle any exceptions
                final_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error in thread for item {i}: {str(result)}")
                        final_results.append(self._create_error_result(batch_data[i], i, f"Threading error: {str(result)}"))
                    else:
                        final_results.append(result)

            return BatchResult(
                batch_id=batch_id,
                results=final_results,
                timestamp=datetime.now().isoformat(),
                processing_type="image_multithreaded"
            )

        except Exception as e:
            logger.error(f"Error in image classification: {str(e)}")
            return BatchResult(
                batch_id=batch_id,
                results=[
                    self._create_error_result(item, i, f"Batch processing error: {str(e)}")
                    for i, item in enumerate(batch_data)
                ],
                timestamp=datetime.now().isoformat(),
                processing_type="image_multithreaded"
            )

    def _build_context_text(self, item: Dict[str, Any]) -> str:
        """Build context text from item data for image classification"""
        product_info = self._extract_product_info(item)
        context_parts = []

        if product_info['product_name']:
            context_parts.append(f"Name: {product_info['product_name']}")
        if product_info['type']:
            context_parts.append(f"Type: {product_info['type']}")
        if product_info['style']:
            if isinstance(product_info['style'], list):
                style_str = ", ".join(product_info['style'])
            else:
                style_str = str(product_info['style'])
            context_parts.append(f"Style: {style_str}")
        if product_info['tags']:
            if isinstance(product_info['tags'], list):
                tags_str = ", ".join(product_info['tags'])
            else:
                tags_str = str(product_info['tags'])
            context_parts.append(f"Tags: {tags_str}")

        return " | ".join(context_parts)

    def _download_and_encode_image_sync(self, image_url: str) -> Optional[str]:
        """Download image and encode as base64 - synchronous version for threading"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(image_url, headers=headers, timeout=15)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                logger.warning(f"Invalid content type for image: {content_type}")
                return None

            # Check file size (limit to 10MB)
            content_length = len(response.content)
            if content_length > 10 * 1024 * 1024:
                logger.warning(f"Image too large: {content_length} bytes")
                return None

            # Encode as base64
            img_data = base64.b64encode(response.content).decode()
            logger.info(f"Successfully downloaded and encoded image: {len(img_data)} chars")
            return img_data

        except Exception as e:
            logger.error(f"Error downloading image {image_url}: {str(e)}")
            return None

    def _create_category_result(self, result_data: Dict[str, Any], original_item: Dict[str, Any], index: int, image_data: str = None) -> CategoryResult:
        """Create category classification result from parsed data"""
        product_info = self._extract_product_info(original_item)

        # Extract product name
        product_name = (
            result_data.get('product_name') or
            product_info['product_name'] or
            f"Furniture Item {index + 1}"
        )

        # Use predicted_category if available, fallback to category
        predicted_category = (
            result_data.get('predicted_category') or
            result_data.get('category') or
            'OTHER'
        )

        # Generate description
        description = None
        if image_data:
            # For image-based classification, generate description from image
            description = self.description_service.generate_image_description_from_base64(image_data)
        else:
            # For text-based classification, generate description from product info
            description = self.description_service.generate_text_description(product_info)

        # Generate embedding from description
        embedding = None
        if description:
            embedding = self.embedding_service.generate_embedding(description)

        return CategoryResult(
            name=product_name,
            predicted_category=predicted_category,
            confidence=float(result_data.get('confidence', 0.5)),
            reasoning=result_data.get('reasoning', 'Category classification completed'),
            description=description,
            embedding=embedding
        )

    def _create_error_result(self, item: Dict[str, Any], index: int, error_msg: str) -> CategoryResult:
        """Create error result"""
        product_info = self._extract_product_info(item)

        product_name = (
            product_info['product_name'] or
            f"Furniture Item {index + 1}"
        )

        return CategoryResult(
            name=product_name,
            predicted_category="ERROR",
            confidence=0.0,
            reasoning=f"Error: {error_msg}",
            description=None,
            embedding=None
        )