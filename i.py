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
        try:
            self.ua = UserAgent()
        except Exception as e:
            print(f"Warning: UserAgent initialization failed: {e}")
            self.ua = None
        self.last_request_time = 0
        self.request_count = 0
        self.session = requests.Session()
        self.fallback_user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]

    def get_random_delay(self):
        """Return a random delay between 5-20 seconds with occasional longer pauses"""
        if random.random() < 0.2:  # 20% chance of a longer delay
            return random.uniform(30, 60)
        return random.uniform(5, 20)

    def get_user_agent(self):
        """Get a random user agent with fallback"""
        if self.ua:
            try:
                return self.ua.random
            except Exception as e:
                print(f"Warning: fake_useragent failed: {e}")
        
        # Fallback to predefined user agents
        return random.choice(self.fallback_user_agents)

    def get_headers(self):
        """Generate random headers for each request with enhanced anti-detection"""
        user_agent = self.get_user_agent()
        
        # Extract browser info from user agent for consistent headers
        is_chrome = 'Chrome' in user_agent
        is_firefox = 'Firefox' in user_agent
        is_safari = 'Safari' in user_agent and 'Chrome' not in user_agent
        
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': random.choice([
                'en-US,en;q=0.9',
                'en-GB,en;q=0.9',
                'en-US,en;q=0.8,es;q=0.6',
                'en-GB,en-US;q=0.9,en;q=0.8'
            ]),
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': random.choice([
                'https://www.google.com/',
                'https://www.google.com/search?q=furniture+sofa',
                'https://www.google.com/search?q=home+decor',
                'https://www.bing.com/search?q=furniture',
                'https://duckduckgo.com/?q=furniture',
                'https://www.pinterest.com/search/pins/?q=furniture'
            ]),
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': random.choice(['none', 'cross-site']),
            'Sec-Fetch-User': '?1',
        }
        
        # Add browser-specific headers
        if is_chrome:
            headers['sec-ch-ua'] = '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"'
            headers['sec-ch-ua-mobile'] = '?0'
            headers['sec-ch-ua-platform'] = f'"{random.choice(["Windows", "macOS", "Linux"])}"'
        
        # Randomly omit some headers to look more natural
        if random.random() < 0.3:
            headers.pop('DNT', None)
        else:
            headers['DNT'] = '1'
            
        if random.random() < 0.2:
            headers.pop('Sec-Fetch-User', None)
            
        return headers

    def create_fresh_session(self):
        """Create a fresh session with new settings"""
        if hasattr(self, 'session'):
            self.session.close()
        
        self.session = requests.Session()
        
        # Set session-level configuration
        self.session.max_redirects = 3
        self.session.verify = True
        
        # Random session configuration
        if random.random() < 0.5:
            self.session.trust_env = False

    def make_request(self, url, max_retries=5):
        """Make a request with exponential backoff and proper rate limiting"""
        for attempt in range(max_retries):
            # Create fresh session on first attempt or after rate limits
            if attempt == 0 or (hasattr(self, '_rate_limited') and self._rate_limited):
                self.create_fresh_session()
                self._rate_limited = False
            
            # Ensure minimum delay between requests (longer after rate limits)
            elapsed = time.time() - self.last_request_time
            base_delay = self.get_random_delay()
            
            # Add extra delay if we've been rate limited recently
            if hasattr(self, '_last_rate_limit') and (time.time() - self._last_rate_limit) < 300:  # 5 minutes
                base_delay += random.uniform(10, 30)
                print("🐌 Adding extra delay due to recent rate limiting...")
            
            if elapsed < base_delay:
                wait_time = base_delay - elapsed
                print(f"⏳ Waiting {wait_time:.1f} seconds to avoid rate limiting...")
                time.sleep(wait_time)

            try:
                headers = self.get_headers()
                print(f"🌐 Making request #{self.request_count + 1} (attempt {attempt + 1}/{max_retries})")
                print(f"   User-Agent: {headers['User-Agent'][:60]}...")
                print(f"   Referer: {headers.get('Referer', 'None')}")

                # Add random jitter before request
                time.sleep(random.uniform(0.5, 2.0))
                
                response = self.session.get(url, headers=headers, timeout=20, allow_redirects=True)
                response.raise_for_status()

                self.last_request_time = time.time()
                self.request_count += 1

                # Clear cookies occasionally to avoid tracking
                if random.random() < 0.4:
                    self.session.cookies.clear()
                    print("🔄 Cleared session cookies")

                print(f"✅ Request successful! Response size: {len(response.text)} chars")
                return response.text

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    self._rate_limited = True
                    self._last_rate_limit = time.time()
                    
                    # Much more aggressive exponential backoff for rate limiting
                    backoff_time = (3 ** attempt) * 20 + random.uniform(30, 60)
                    print(f"❌ Rate limited (429)! Attempt {attempt + 1}/{max_retries}")
                    print(f"🕐 Waiting {backoff_time:.1f} seconds before retry...")
                    
                    if attempt < max_retries - 1:
                        time.sleep(backoff_time)
                        continue
                    else:
                        print(f"❌ Max retries reached for rate limiting. URL: {url}")
                        return None
                        
                elif e.response.status_code == 403:
                    print(f"❌ Forbidden (403) - likely blocked by anti-bot measures")
                    # Try with longer delay and fresh session
                    if attempt < max_retries - 1:
                        wait_time = random.uniform(60, 120)
                        print(f"🕐 Waiting {wait_time:.1f} seconds and creating fresh session...")
                        time.sleep(wait_time)
                        continue
                    return None
                    
                else:
                    print(f"❌ HTTP Error {e.response.status_code}: {e}")
                    if attempt < max_retries - 1 and e.response.status_code >= 500:
                        # Server errors - retry with delay
                        wait_time = (attempt + 1) * 10 + random.uniform(5, 15)
                        print(f"🕐 Server error, retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                        continue
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # Progressive delay for other errors
                    wait_time = (attempt + 1) * 8 + random.uniform(2, 8)
                    print(f"🕐 Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Max retries reached. Giving up on {url}")
                    return None

        return None

class ImageClassifier:
    def __init__(self):
        self.scraper = PoliteScraper()
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)

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
        if filtered_urls:
            return max(filtered_urls, key=len)
        return None

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
        """Process a batch of product URLs with enhanced rate limiting"""
        results = []
        consecutive_failures = 0
        total_processing_time = time.time()
        
        print(f"🚀 Starting batch {batch_id} with {len(batch_data)} items")
        
        for i, item in enumerate(batch_data):
            product_url = item.get("Product URL", item.get("product", ""))
            
            # If we've had multiple consecutive failures, add extra delay and reset scraper
            if consecutive_failures >= 2:
                extra_delay = min(consecutive_failures * 15, 120) + random.uniform(10, 30)
                print(f"⚠️ {consecutive_failures} consecutive failures detected.")
                print(f"🛡️ Taking a break for {extra_delay:.1f} seconds and resetting scraper...")
                time.sleep(extra_delay)
                
                # Reset the scraper's rate limit memory
                if hasattr(self.scraper, '_last_rate_limit'):
                    delattr(self.scraper, '_last_rate_limit')
            
            item_start_time = time.time()
            result = self.process_product(product_url, i)
            item_duration = time.time() - item_start_time
            
            results.append(result)
            
            # Track consecutive failures
            if not result.get("success", False):
                consecutive_failures += 1
                print(f"❌ Item {i+1} failed (consecutive failures: {consecutive_failures})")
            else:
                if consecutive_failures > 0:
                    print(f"✅ Item {i+1} succeeded! Breaking failure streak of {consecutive_failures}")
                consecutive_failures = 0
            
            # Be extra polite between items - adaptive delays
            if i < len(batch_data) - 1:  # Don't delay after the last item
                # Base delay increases with consecutive failures
                base_delay = random.uniform(5, 12)
                
                # Add failure penalty
                if consecutive_failures > 0:
                    failure_penalty = min(consecutive_failures * 10, 60)
                    base_delay += failure_penalty
                    print(f"🐌 Adding {failure_penalty:.1f}s penalty for {consecutive_failures} failures")
                
                # Add processing time consideration
                if item_duration > 30:  # If last item took a long time
                    base_delay += random.uniform(5, 10)
                    print(f"⏱️ Last item took {item_duration:.1f}s, adding extra delay")
                
                print(f"💤 Waiting {base_delay:.1f} seconds before next item...")
                time.sleep(base_delay)

        total_duration = time.time() - total_processing_time
        success_rate = (sum(1 for r in results if r.get("success", False)) / len(batch_data)) * 100
        
        print(f"🏁 Batch {batch_id} completed in {total_duration:.1f}s")
        print(f"📊 Success rate: {success_rate:.1f}% ({sum(1 for r in results if r.get('success', False))}/{len(batch_data)})")

        return {
            "batch_id": batch_id,
            "total_items": len(batch_data),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "results": results,
            "processing_time": total_duration,
            "success_rate": success_rate
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