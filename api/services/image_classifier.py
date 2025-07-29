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

load_dotenv()

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
            print(f"Waiting {wait_time:.1f} seconds to avoid rate limiting...")
            time.sleep(wait_time)

        try:
            headers = self.get_headers()
            print(f"Making request #{self.request_count + 1} with User-Agent: {headers['User-Agent']}")

            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            self.last_request_time = time.time()
            self.request_count += 1

            if random.random() < 0.3:
                self.session.cookies.clear()

            return response.text

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                print("Hit rate limit! Waiting longer...")
                time.sleep(random.uniform(30, 60))
                return self.make_request(url)
            return None

class ImageClassifier:
    def __init__(self):
        self.scraper = PoliteScraper()
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @staticmethod
    def _create_error_result(item, index, error_message):
        return {
            "index": index,
            "product_url": item.get("Product URL", item.get("product", "")),
            "error": error_message,
            "success": False
        }

    def scrape_images(self, url):
        """Scrape all image URLs from a webpage"""
        html = self.scraper.make_request(url)
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        image_urls = []

        # Regular image tags
        for img in soup.find_all('img'):
            if 'src' in img.attrs:
                src = img['src']
                if not src.startswith('data:'):
                    absolute_url = urljoin(url, src)
                    image_urls.append(absolute_url)

        # Background images
        for tag in soup.find_all(style=re.compile(r'background-image:\s*url\(.*\)')):
            style = tag['style']
            match = re.search(r'url\(["\']?(.*?)["\']?\)', style)
            if match and not match.group(1).startswith('data:'):
                absolute_url = urljoin(url, match.group(1))
                image_urls.append(absolute_url)

        # JavaScript embedded images
        script_pattern = re.compile(r'https?://[^\s"\']+?\.(jpg|jpeg|png|webp)(?:\?\w+)?', re.IGNORECASE)
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
                        "category": "SOFA|CHAIR|BED|TABLE|NIGHTSTAND|STOOL|OTHER",
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
            print(f"Classification error for {image_url}: {str(e)}")
            return {"error": str(e), "image_url": image_url}

    def process_product(self, product_url, index):
        """Process a single product URL with improved error handling"""
        try:
            if not product_url or not product_url.startswith('http'):
                return self._create_error_result({"product_url": product_url}, index, "Invalid URL provided")

            print(f"\nProcessing product {index+1}: {product_url}")

            image_url = self.scrape_images(product_url)
            if not image_url:
                return self._create_error_result({"product_url": product_url}, index, "No valid images found")

            print(f"Using image URL: {image_url}")

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
                return self._create_error_result(
                    {"product_url": product_url},
                    index,
                    f"Classification failed: {error_msg}"
                )

        except Exception as e:
            return self._create_error_result(
                {"product_url": product_url},
                index,
                f"Processing error: {str(e)}"
            )

    async def classify_image_batch(self, batch_data: List[Dict[str, Any]], batch_id: int) -> Dict:
        """Process a batch of product URLs"""
        results = []
        for i, item in enumerate(batch_data):
            product_url = item.get("Product URL", item.get("product", ""))
            result = self.process_product(product_url, i)
            results.append(result)

            # Be extra polite between items
            time.sleep(random.uniform(1, 3))

        return {
            "batch_id": batch_id,
            "total_items": len(batch_data),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "results": results
        }

# Example usage
if __name__ == "__main__":
    classifier = ImageClassifier()

    test_batch = [
        {"Product URL": "https://www.wayfair.com/furniture/pdp/george-oliver-winfree-2-piece-787-upholstered-sofa-w100055074.html"},
        {"product": "https://www.ikea.com/us/en/p/product-page"}
    ]

    batch_result = asyncio.run(classifier.classify_image_batch(test_batch, 1))
    print(json.dumps(batch_result, indent=2))
