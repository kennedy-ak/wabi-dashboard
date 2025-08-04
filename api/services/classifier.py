from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import asyncio
import json
import logging
import base64
import os
import concurrent.futures
from functools import partial
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils import CATEGORY_CLASSIFICATION_PROMPT
from ..utils.scraper import PoliteScraper
from ..models import CategoryResult, BatchResult
from config import settings

logger = logging.getLogger(__name__)

class FurnitureCategoryClassifier:
    def __init__(self):
        self.client = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.scraper = PoliteScraper()
        self.max_workers = min(32, (os.cpu_count() or 1) + 4)  # Limit concurrent threads

    def _classify_single_text_item(self, item: Dict[str, Any], index: int) -> CategoryResult:
        """Classify a single item using text data - synchronous function for threading"""
        try:
            # Extract relevant text fields for classification
            text_parts = []
            
            # Add product name if available
            product_name_key = next((k for k in item.keys() if k.lower() in ['product_name', 'product name']), None)
            if product_name_key and item[product_name_key]:
                text_parts.append(f"Product Name: {item[product_name_key]}")
            
            # Add type if available
            type_key = next((k for k in item.keys() if k.lower() == 'type'), None)
            if type_key and item[type_key]:
                text_parts.append(f"Type: {item[type_key]}")
            
            # Add category if available
            category_key = next((k for k in item.keys() if k.lower() == 'category'), None)
            if category_key and item[category_key]:
                text_parts.append(f"Current Category: {item[category_key]}")
            
            # Add style if available
            style_key = next((k for k in item.keys() if k.lower() == 'style'), None)
            if style_key and item[style_key]:
                text_parts.append(f"Style: {item[style_key]}")
            
            # Add tags if available
            tags_key = next((k for k in item.keys() if k.lower() == 'tags'), None)
            if tags_key and item[tags_key]:
                if isinstance(item[tags_key], list):
                    tags_str = ", ".join(item[tags_key])
                else:
                    tags_str = str(item[tags_key])
                text_parts.append(f"Tags: {tags_str}")
            
            # Add price range if available
            price_key = next((k for k in item.keys() if 'price' in k.lower()), None)
            if price_key and item[price_key]:
                text_parts.append(f"Price Range: {item[price_key]}")
            
            furniture_description = " | ".join(text_parts)
            
            # Call OpenAI API via langchain (synchronous for threading)
            message = HumanMessage(content=f"{CATEGORY_CLASSIFICATION_PROMPT}\n\nClassify this furniture item: {furniture_description}")
            response = self.client.invoke([message])
            
            classification_text = response.content
            
            # Parse JSON response
            try:
                # Extract JSON from response
                if '{' in classification_text:
                    json_start = classification_text.find('{')
                    json_end = classification_text.rfind('}') + 1
                    json_str = classification_text[json_start:json_end]
                    result_data = json.loads(json_str)
                    return self._create_category_result(result_data, item, index)
                else:
                    return self._create_fallback_result(item, index)
            
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON for item {index}: {e}")
                return self._create_fallback_result(item, index)
                
        except Exception as api_error:
            logger.error(f"OpenAI API error for item {index}: {str(api_error)}")
            return self._create_error_result(item, index, f"API error: {str(api_error)}")

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

    def _classify_single_image_item(self, item: Dict[str, Any], index: int) -> CategoryResult:
        """Classify a single item using image data - synchronous function for threading"""
        try:
            product_url = item.get("Product URL", item.get("product", ""))
            
            if not product_url:
                return self._create_error_result(item, index, "No Product_URL provided")
            
            # Scrape image using polite scraper (synchronous version)
            logger.info(f"Processing image for item {index+1}: {product_url}")
            # Note: This would need a synchronous version of scrape_images_safely
            # For now, we'll simulate the scraping
            image_url = product_url  # Assuming the URL itself is the image
            
            if not image_url:
                return self._create_error_result(item, index, "Could not scrape image from URL")
            
            # Download and encode the image
            image_data = self._download_and_encode_image_sync(image_url)
            
            if not image_data:
                return self._create_error_result(item, index, "Could not download and encode image")
            
            # Use OpenAI Vision API for classification
            context_text = self._build_context_text(item)
            prompt_text = f"{CATEGORY_CLASSIFICATION_PROMPT}\n\nAnalyze this furniture image and classify its category."
            if context_text:
                prompt_text += f"\n\nAdditional context: {context_text}"
            
            message = HumanMessage(
                content=[
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
            )
            
            response = self.client.invoke([message])
            classification_text = response.content
            
            # Parse JSON response
            try:
                if '{' in classification_text:
                    json_start = classification_text.find('{')
                    json_end = classification_text.rfind('}') + 1
                    json_str = classification_text[json_start:json_end]
                    result_data = json.loads(json_str)
                    return self._create_category_result(result_data, item, index)
                else:
                    return self._create_error_result(item, index, "Image analyzed but no JSON response")
            except json.JSONDecodeError:
                return self._create_error_result(item, index, "Could not parse vision response")
                
        except Exception as vision_error:
            logger.error(f"Vision API error for item {index}: {str(vision_error)}")
            return self._create_error_result(item, index, f"Vision processing error: {str(vision_error)}")

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

    def _download_and_encode_image_sync(self, image_url: str) -> Optional[str]:
        """Download image and encode as base64 - synchronous version for threading"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            import requests
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

    async def _download_and_encode_image(self, image_url: str) -> Optional[str]:
        """Download image and encode as base64 - async wrapper for backwards compatibility"""
        return self._download_and_encode_image_sync(image_url)

    def _build_context_text(self, item: Dict[str, Any]) -> str:
        """Build context text from item data for image classification"""
        context_parts = []
        
        # Extract product name
        product_name_key = next((k for k in item.keys() if k.lower() in ['product_name', 'product name']), None)
        if product_name_key and item[product_name_key]:
            context_parts.append(f"Name: {item[product_name_key]}")
        
        # Extract type
        type_key = next((k for k in item.keys() if k.lower() == 'type'), None)
        if type_key and item[type_key]:
            context_parts.append(f"Type: {item[type_key]}")
        
        # Extract style
        style_key = next((k for k in item.keys() if k.lower() == 'style'), None)
        if style_key and item[style_key]:
            context_parts.append(f"Style: {item[style_key]}")
        
        # Extract tags
        tags_key = next((k for k in item.keys() if k.lower() == 'tags'), None)
        if tags_key and item[tags_key]:
            if isinstance(item[tags_key], list):
                tags_str = ", ".join(item[tags_key])
            else:
                tags_str = str(item[tags_key])
            context_parts.append(f"Tags: {tags_str}")
        
        return " | ".join(context_parts)

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

