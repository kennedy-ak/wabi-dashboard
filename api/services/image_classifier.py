from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import time
import random
from fake_useragent import UserAgent
from typing import List, Dict, Any
import json
import asyncio
import logging
from datetime import datetime
import concurrent.futures
import os
from functools import partial

from api.models import CategoryResult, BatchResult

load_dotenv()

logger = logging.getLogger(__name__)

class PoliteScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.last_request_time = 0
        self.request_count = 0
        self.session = requests.Session()

    def get_random_delay(self):
        """Return a random delay between 2-10 seconds with occasional longer pauses"""
        if random.random() < 0.1:  # 10% chance of a longer delay
            return random.uniform(15, 30)
        return random.uniform(2, 10)

    def get_headers(self):
        """Generate random headers for each request"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': random.choice(['en-US,en;q=0.5', 'en-GB,en;q=0.5', 'fr,fr-FR;q=0.8']),
            'Referer': random.choice(['https://www.google.com/', 'https://www.bing.com/', 'https://duckduckgo.com/']),
            'DNT': str(random.randint(0, 1)),
        }

    def make_request(self, url):
        """Make a request with proper rate limiting and headers"""
        elapsed = time.time() - self.last_request_time
        delay = self.get_random_delay()

        if elapsed < delay:
            wait_time = delay - elapsed
            logger.info(f"Waiting {wait_time:.1f} seconds to avoid rate limiting...")
            time.sleep(wait_time)

        try:
            headers = self.get_headers()
            logger.info(f"Making request #{self.request_count + 1} with User-Agent: {headers['User-Agent']}")

            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            self.last_request_time = time.time()
            self.request_count += 1

            if random.random() < 0.3:
                self.session.cookies.clear()

            return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                logger.warning("Hit rate limit! Waiting longer...")
                time.sleep(random.uniform(30, 60))
                return self.make_request(url)
            return None

class ImageClassifier:
    def __init__(self):
        self.scraper = PoliteScraper()
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.max_workers = min(32, (os.cpu_count() or 1) + 4)  # Limit concurrent threads

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
            confidence=float(result_data.get('confidence', 0.5)) if result_data.get('confidence') not in [None, float('inf'), float('-inf')] and str(result_data.get('confidence')).lower() != 'nan' else 0.5,
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
            primary_style=None,
            secondary_style=None,
            style_tags=[],
            placement_tags=[],
            confidence=0.0,
            reasoning=error_message
        )

    def scrape_images(self, url):
        """Scrape all image URLs from a webpage"""
        html = self.scraper.make_request(url)
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        image_urls = []

        # Regular image tags
        for img in soup.find_all('img'):
            if hasattr(img, 'attrs') and 'src' in img.attrs:
                src = img['src']
                if not src.startswith('data:'):
                    absolute_url = urljoin(url, src)
                    image_urls.append(absolute_url)

        # Background images
        for tag in soup.find_all(style=re.compile(r'background-image:\s*url\((.*?)\)')):
            if hasattr(tag, 'attrs') and 'style' in tag.attrs:
                style = tag['style']
                match = re.search(r'url\([\"\']?(.*?)[\"\']?\)', style)
                if match and not match.group(1).startswith('data:'):
                    absolute_url = urljoin(url, match.group(1))
                    image_urls.append(absolute_url)

        # JavaScript embedded images
        script_pattern = re.compile(r'https?://[^\s\"\']+?\.(jpg|jpeg|png|webp)(?:\?\w+)?', re.IGNORECASE)
        image_urls.extend(urljoin(url, u) for u in script_pattern.findall(html))

        # Filter and prioritize likely product images
        filtered_urls = [
            url for url in list(set(image_urls))
            if not any(ignore in url.lower() for ignore in ['logo', 'icon', 'svg', 'placeholder'])
        ]

        # Return only the longest URL (most likely to be main product image)
        return max(filtered_urls, key=len)

    def classify_image(self, image_url):
        """Classify a single image using GPT-4 vision with robust error handling"""
        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": """Analyze this furniture image and respond ONLY with valid JSON:
                    {
                        "product_name": "string",
                        "category": "SOFA|CHAIR|BED|TABLE|NIGHTSTAND|STOOL|STORAGE|DESK|BENCH|OTTOMAN|LIGHTING|DECOR|VASE|TV_STAND|PAINTINGS|OTHER",
                        "primary_style": "string",
                        "secondary_style": "string",
                        "style_tags": ["string"],
                        "placement_tags": ["string"],
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

    def _process_single_item_sync(self, item: Dict[str, Any], index: int) -> CategoryResult:
        """Process a single item synchronously for multithreading"""
        try:
            # Try multiple possible URL column names
            product_url = (
                item.get("Product URL") or 
                item.get("product") or 
                item.get("url") or 
                item.get("URL") or 
                item.get("product_url") or
                ""
            )
            
            # More comprehensive URL validation
            if not product_url or not isinstance(product_url, str) or not (product_url.startswith('http://') or product_url.startswith('https://')):
                return self._create_error_result(item, index, f"Invalid URL provided: '{product_url}' - must start with http:// or https://")

            logger.info(f"Processing product {index+1}: {product_url}")

            image_url = self.scrape_images(product_url)
            if not image_url:
                return self._create_error_result(item, index, "No valid images found")

            logger.info(f"Using image URL: {image_url}")

            result = self.classify_image(image_url)

            if result and not result.get("error"):
                return self._create_category_result(result, item, index)
            else:
                error_msg = result.get("error", "Classification failed") if result else "Empty response"
                return self._create_error_result(item, index, f"Classification failed: {error_msg}")

        except Exception as e:
            logger.error(f"Processing error for item {index}: {str(e)}")
            return self._create_error_result(item, index, f"Processing error: {str(e)}")

    def process_product(self, product_url, index):
        """Process a single product URL with improved error handling (legacy method)"""
        try:
            # More comprehensive URL validation
            if not product_url or not isinstance(product_url, str) or not (product_url.startswith('http://') or product_url.startswith('https://')):
                return {
                    "index": index,
                    "product_url": product_url,
                    "error": f"Invalid URL provided: '{product_url}' - must start with http:// or https://",
                    "success": False
                }

            logger.info(f"Processing product {index+1}: {product_url}")

            image_url = self.scrape_images(product_url)
            if not image_url:
                return {
                    "index": index,
                    "product_url": product_url,
                    "error": "No valid images found",
                    "success": False
                }

            logger.info(f"Using image URL: {image_url}")

            result = self.classify_image(image_url)

            if result and not result.get("error"):
                return {
                    "index": index,
                    "product_url": product_url,
                    "image_url": image_url,
                    "result": result,
                    "success": True
                }
            else:
                error_msg = result.get("error", "Classification failed") if result else "Empty response"
                return {
                    "index": index,
                    "product_url": product_url,
                    "error": f"Classification failed: {error_msg}",
                    "success": False
                }

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return {
                "index": index,
                "product_url": product_url,
                "error": f"Processing error: {str(e)}",
                "success": False
            }

    async def classify_image_batch(self, batch_data: List[Dict[str, Any]], batch_id: int) -> BatchResult:
        """Process a batch of product URLs with multithreading"""
        try:
            logger.info(f"Processing image batch {batch_id} with {len(batch_data)} items using multithreading (max_workers={self.max_workers})")

            # Use ThreadPoolExecutor for concurrent processing
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create partial function with the method bound
                classify_func = partial(self._process_single_item_sync)

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

            successful_count = sum(1 for r in final_results if r.category != "ERROR")
            failed_count = len(final_results) - successful_count

            return BatchResult(
                batch_id=batch_id,
                results=final_results,
                timestamp=datetime.now().isoformat(),
                processing_type="image_multithreaded",
                total_processed=len(batch_data),
                successful=successful_count,
                failed=failed_count
            )

        except Exception as e:
            logger.error(f"Error in image classification batch {batch_id}: {str(e)}")
            # Create error results for all items
            error_results = [
                self._create_error_result(item, i, f"Batch processing error: {str(e)}")
                for i, item in enumerate(batch_data)
            ]
            
            return BatchResult(
                batch_id=batch_id,
                results=error_results,
                timestamp=datetime.now().isoformat(),
                processing_type="image_multithreaded",
                total_processed=len(batch_data),
                successful=0,
                failed=len(batch_data)
            )

# Example usage
if __name__ == "__main__":
    classifier = ImageClassifier()

    test_batch = [
        {"Product URL": "https://www.wayfair.com/furniture/pdp/george-oliver-winfree-2-piece-787-upholstered-sofa-w100055074.html"},
        {"product": "https://www.ikea.com/us/en/p/product-page"}
    ]

    batch_result = asyncio.run(classifier.classify_image_batch(test_batch, 1))
    print(json.dumps(batch_result, indent=2))
