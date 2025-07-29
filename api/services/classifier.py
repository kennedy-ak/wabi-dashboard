import openai
import asyncio
import json
import logging
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import settings
from api.models import CategoryResult, BatchResult
from api.utils.constants import CATEGORY_CLASSIFICATION_PROMPT, CATEGORY_CLASSIFICATION_PROMPT_TEXT_BASED
from api.utils.scraper import PoliteScraper

logger = logging.getLogger(__name__)

class FurnitureCategoryClassifier:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.scraper = PoliteScraper()

    async def classify_text_batch(self, batch_data: List[Dict[str, Any]], batch_id: int) -> BatchResult:
        """Classify furniture categories using text data"""
        try:
            logger.info(f"Processing text batch {batch_id} with {len(batch_data)} items")

            # Prepare furniture data for each item
            furniture_descriptions = []
            for item in batch_data:
                # Extract relevant text fields
                text_fields = []
                for key, value in item.items():
                    if key not in ["Product Name", "Tags"] and value:
                        text_fields.append(f"{key}: {value}")
                furniture_descriptions.append(" | ".join(text_fields))

            # Create prompt for batch
            batch_prompt = f"{CATEGORY_CLASSIFICATION_PROMPT}\n\nFurniture items to classify:\n"
            for i, desc in enumerate(furniture_descriptions):
                batch_prompt += f"{i+1}. {desc}\n"

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CATEGORY_CLASSIFICATION_PROMPT},
                    {"role": "user", "content": f"Classify these furniture items into categories:\n{batch_prompt}"}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            # Parse response
            classification_text = response.choices[0].message.content

            # Try to extract JSON from response
            results = []
            try:
                # Attempt to parse as JSON array
                if classification_text.strip().startswith('['):
                    parsed_results = json.loads(classification_text)
                else:
                    # Extract individual JSON objects
                    import re
                    json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', classification_text)
                    parsed_results = [json.loads(match) for match in json_matches]

                # Convert to structured results
                for i, result_data in enumerate(parsed_results):
                    if i < len(batch_data):
                        result = self._create_category_result(result_data, batch_data[i], i)
                        results.append(result)

                # Fill missing results if needed
                while len(results) < len(batch_data):
                    i = len(results)
                    fallback_result = self._create_fallback_result(batch_data[i], i)
                    results.append(fallback_result)

            except json.JSONDecodeError:
                # Fallback: create basic results
                results = [
                    self._create_fallback_result(item, i)
                    for i, item in enumerate(batch_data)
                ]

            return BatchResult(
                batch_id=batch_id,
                results=results,
                timestamp=datetime.now().isoformat(),
                processing_type="text"
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
                processing_type="text"
            )

    async def classify_image_batch(self, batch_data: List[Dict[str, Any]], batch_id: int) -> BatchResult:
        """Classify furniture categories using scraped images"""
        try:
            results = []

            for i, item in enumerate(batch_data):
                product_url = item.get("Product URL", item.get("product", ""))

                if not product_url:
                    results.append(self._create_error_result(item, i, "No product URL provided"))
                    continue

                # Scrape image using polite scraper
                logger.info(f"Scraping image for item {i+1}: {product_url}")
                image_url = await self.scraper.scrape_images_safely(product_url)


                if not image_url:
                    logger.warning(f"Image scraping failed for item {i+1}, attempting text-based fallback")
                    # Fallback to text-based classification if available
                    fallback_result = await self._try_text_fallback(item, i)
                    results.append(fallback_result)
                    continue

                # Download and encode the image
                image_data = await self._download_and_encode_image(image_url)

                if not image_data:
                    logger.warning(f"Image download failed for item {i+1}, attempting text-based fallback")
                    # Fallback to text-based classification if available
                    fallback_result = await self._try_text_fallback(item, i)
                    results.append(fallback_result)
                    continue

                # Use OpenAI Vision API for classification
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4.o-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"{CATEGORY_CLASSIFICATION_PROMPT}\n\nAnalyze this furniture image and classify its category:"
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

                    # Try to parse JSON response
                    try:
                        if '{' in classification_text:
                            json_start = classification_text.find('{')
                            json_end = classification_text.rfind('}') + 1
                            json_str = classification_text[json_start:json_end]
                            result_data = json.loads(json_str)
                            result = self._create_category_result(result_data, item, i)
                        else:
                            raise json.JSONDecodeError("No JSON found", "", 0)
                    except json.JSONDecodeError:
                        # Fallback parsing
                        result = self._create_fallback_result(item, i)
                        result.reasoning = "Image analyzed but could not parse structured response"

                    results.append(result)

                except Exception as vision_error:
                    logger.error(f"Vision API error: {str(vision_error)}")
                    results.append(self._create_error_result(item, i, f"Vision processing error: {str(vision_error)}"))

            return BatchResult(
                batch_id=batch_id,
                results=results,
                timestamp=datetime.now().isoformat(),
                processing_type="image"
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
                processing_type="image"
            )

    async def _download_and_encode_image(self, image_url: str) -> Optional[str]:
        """Download image and encode as base64"""
        try:
            headers = self.scraper.get_headers()
            response = self.scraper.session.get(image_url, headers=headers, timeout=10)
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

    def _create_category_result(self, result_data: Dict[str, Any], original_item: Dict[str, Any], index: int) -> CategoryResult:
        """Create category classification result from parsed data"""

        # Extract product name
        product_name = (
            result_data.get('product_name') or
            original_item.get('Product Name') or
            original_item.get('product_name') or
            f"Furniture Item {index + 1}"
        )

        # Extract style tags and placement tags, ensuring they're lists
        style_tags = result_data.get('style_tags', [])
        if isinstance(style_tags, str):
            style_tags = [style_tags]

        placement_tags = result_data.get('placement_tags', [])
        if isinstance(placement_tags, str):
            placement_tags = [placement_tags]

        return CategoryResult(
            product_name=product_name,
            category=result_data.get('category', 'OTHER'),
            primary_style=result_data.get('primary_style'),
            secondary_style=result_data.get('secondary_style'),
            style_tags=style_tags,
            placement_tags=placement_tags,
            confidence=float(result_data.get('confidence', 0.5)),
            reasoning=result_data.get('reasoning', 'Category classification completed')
        )

    def _create_fallback_result(self, item: Dict[str, Any], index: int) -> CategoryResult:
        """Create fallback result when parsing fails"""
        product_name = (
            item.get('Product Name') or
            item.get('product_name') or
            f"Furniture Item {index + 1}"
        )

        # Try to guess category from existing data
        category = item.get('Category', item.get('Type', 'OTHER')).upper()

        return CategoryResult(
            product_name=product_name,
            category=category,
            primary_style=None,
            secondary_style=None,
            style_tags=[],
            placement_tags=[],
            confidence=0.3,
            reasoning="Could not parse classification response, used existing data"
        )

    def _create_error_result(self, item: Dict[str, Any], index: int, error_msg: str) -> CategoryResult:
        """Create error result"""
        product_name = (
            item.get('Product Name') or
            item.get('product_name') or
            f"Furniture Item {index + 1}"
        )

        return CategoryResult(
            product_name=product_name,
            category="ERROR",
            primary_style=None,
            secondary_style=None,
            style_tags=[],
            placement_tags=[],
            confidence=0.0,
            reasoning=error_msg
        )

    async def _try_text_fallback(self, item: Dict[str, Any], index: int) -> CategoryResult:
        """Attempt text-based classification when image scraping fails"""
        try:
            # Extract text data from the item
            text_fields = []
            for key, value in item.items():
                if key not in ["Product URL", "product"] and value and isinstance(value, str):
                    text_fields.append(f"{key}: {value}")

            if not text_fields:
                return self._create_error_result(item, index, "No text data available for fallback classification")

            description = " | ".join(text_fields)
            logger.info(f"Attempting text-based fallback classification for item {index + 1}")

            # Use simplified prompt for single item
            fallback_prompt = f"{CATEGORY_CLASSIFICATION_PROMPT}\n\nClassify this furniture item:\n{description}"

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CATEGORY_CLASSIFICATION_PROMPT},
                    {"role": "user", "content": fallback_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            classification_text = response.choices[0].message.content

            # Try to parse JSON response
            try:
                if '{' in classification_text:
                    json_start = classification_text.find('{')
                    json_end = classification_text.rfind('}') + 1
                    json_str = classification_text[json_start:json_end]
                    result_data = json.loads(json_str)
                    result = self._create_category_result(result_data, item, index)
                    result.reasoning = f"Text-based fallback: {result.reasoning}"
                    return result
                else:
                    raise json.JSONDecodeError("No JSON found", "", 0)
            except json.JSONDecodeError:
                # Create basic fallback result
                result = self._create_fallback_result(item, index)
                result.reasoning = "Text-based fallback: Could not parse AI response"
                return result

        except Exception as e:
            logger.error(f"Text fallback failed for item {index + 1}: {str(e)}")
            return self._create_error_result(item, index, f"Both image and text classification failed: {str(e)}")
