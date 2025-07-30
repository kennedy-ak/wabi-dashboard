import asyncio
import time
import random
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from typing import Optional, List

logger = logging.getLogger(__name__)

class PoliteScraper:
    """Enhanced web scraper with anti-detection and rate limiting"""
    
    def __init__(self):
        self.last_request_time = 0
        self.request_count = 0
        self.failed_attempts = {}  # Track failed attempts per domain
        self.session = requests.Session()
        self.max_concurrent_requests = 3  # Limit concurrent requests per domain
        self.domain_semaphores = {}  # Per-domain rate limiting
        
        # More diverse and recent user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0'
        ]

    def get_random_delay(self, domain=None):
        """Return a random delay with exponential backoff for problematic domains"""
        base_delay = random.uniform(3, 8)  # Increased base delay
        
        # Check if this domain has had recent failures
        if domain and domain in self.failed_attempts:
            failure_count = self.failed_attempts[domain]
            # Exponential backoff: 2^failures * base_delay
            backoff_multiplier = min(2 ** failure_count, 16)  # Cap at 16x
            base_delay *= backoff_multiplier
            logger.info(f"Applying exponential backoff for {domain}: {failure_count} failures, delay: {base_delay:.1f}s")
        
        # Occasional longer pauses for natural behavior
        if random.random() < 0.15:  # 15% chance of longer delay
            base_delay += random.uniform(20, 45)
            
        return base_delay

    def get_headers(self, url=None):
        """Generate realistic headers for each request"""
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': random.choice([
                'en-US,en;q=0.9', 'en-GB,en;q=0.8', 'en-US,en;q=0.5',
                'en-CA,en;q=0.7', 'en-AU,en;q=0.6'
            ]),
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': random.choice(['none', 'same-origin', 'cross-site']),
            'Cache-Control': 'max-age=0',
        }
        
        # Add referer based on the URL being requested
        if url and 'wayfair.com' in url:
            headers['Referer'] = 'https://www.wayfair.com/'
        else:
            headers['Referer'] = random.choice([
                'https://www.google.com/', 'https://www.bing.com/', 
                'https://duckduckgo.com/', 'https://www.pinterest.com/'
            ])
        
        # Randomly include or exclude DNT
        if random.random() < 0.7:
            headers['DNT'] = str(random.randint(0, 1))
            
        return headers

    async def make_request(self, url, max_retries=3):
        """Make a request with enhanced anti-detection and exponential backoff"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        for attempt in range(max_retries):
            try:
                # Apply rate limiting with domain-specific backoff
                elapsed = time.time() - self.last_request_time
                delay = self.get_random_delay(domain)

                if elapsed < delay:
                    wait_time = delay - elapsed
                    logger.info(f"Waiting {wait_time:.1f} seconds before request to {domain}...")
                    await asyncio.sleep(wait_time)

                # Generate headers with URL context
                headers = self.get_headers(url)
                logger.info(f"Making request #{self.request_count + 1} to {url[:50]}... (attempt {attempt + 1}/{max_retries})")

                # Make request with longer timeout
                response = self.session.get(url, headers=headers, timeout=30, verify=True)
                
                # Check response status
                if response.status_code == 200:
                    # Success! Reset failure count for this domain
                    if domain in self.failed_attempts:
                        del self.failed_attempts[domain]
                    
                    self.last_request_time = time.time()
                    self.request_count += 1

                    # Occasionally rotate session to appear more natural
                    if random.random() < 0.2:  # 20% chance
                        self.session.cookies.clear()
                        logger.info("Cleared session cookies for natural behavior")

                    return response.text
                
                elif response.status_code == 429:
                    # Rate limited - implement exponential backoff
                    self._handle_rate_limit(domain, attempt)
                    if attempt < max_retries - 1:
                        backoff_time = min(60 * (2 ** attempt), 300)  # Cap at 5 minutes
                        logger.warning(f"Rate limited. Backing off for {backoff_time:.1f} seconds...")
                        await asyncio.sleep(backoff_time)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for rate limited URL: {url}")
                        return None
                        
                elif response.status_code == 403:
                    logger.error(f"Access forbidden (403) for {url} - likely blocked by anti-bot")
                    self._handle_rate_limit(domain, attempt)
                    return None
                    
                else:
                    logger.warning(f"Unexpected status code {response.status_code} for {url}")
                    response.raise_for_status()

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1} for {url}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(random.uniform(5, 15))
                    continue
                    
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on attempt {attempt + 1} for {url}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(random.uniform(5, 15))
                    continue
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(random.uniform(2, 8))
                    continue

        # All attempts failed
        logger.error(f"All {max_retries} attempts failed for {url}")
        return None

    def _handle_rate_limit(self, domain, attempt):
        """Track and handle rate limiting for a domain"""
        if domain not in self.failed_attempts:
            self.failed_attempts[domain] = 0
        self.failed_attempts[domain] += 1
        logger.info(f"Recorded failure #{self.failed_attempts[domain]} for domain {domain}")

    async def scrape_images_safely(self, url):
        """Extract images from URL using polite scraping"""
        html = await self.make_request(url)
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        image_urls = []

        # Find all image tags
        for img in soup.find_all('img'):
            if 'src' in img.attrs:
                src = img['src']
                if not src.startswith('data:'):
                    absolute_url = urljoin(url, src)
                    image_urls.append(absolute_url)

        # Find background images
        for tag in soup.find_all(style=re.compile(r'background-image:\s*url\(.*\)')):
            style = tag['style']
            match = re.search(r'url\(["\']?(.*?)["\']?\)', style)
            if match and not match.group(1).startswith('data:'):
                absolute_url = urljoin(url, match.group(1))
                image_urls.append(absolute_url)

        # Find in JavaScript data (specifically for Wayfair)
        script_pattern = re.compile(r'https?://assets\.wfcdn\.com/[^\s"\']+?\.(jpg|jpeg)(?:\?\w+)?', re.IGNORECASE)
        js_images = script_pattern.findall(html)
        image_urls.extend(js_images)

        # Remove duplicates and filter for likely product images
        unique_urls = list(set(image_urls))
        
        # Try to find the best product image (longest URL often has more detail)
        if unique_urls:
            # Filter for likely product images
            product_images = [url for url in unique_urls if self._is_likely_product_image(url)]
            if product_images:
                # Return the URL with most detail (often the longest)
                return max(product_images, key=len)
            else:
                # Fallback to longest URL
                return max(unique_urls, key=len)
        
        return None

    def get_domain_semaphore(self, domain: str) -> asyncio.Semaphore:
        """Get or create semaphore for domain-specific rate limiting"""
        if domain not in self.domain_semaphores:
            self.domain_semaphores[domain] = asyncio.Semaphore(self.max_concurrent_requests)
        return self.domain_semaphores[domain]
    
    async def scrape_multiple_urls(self, urls: List[str]) -> List[Optional[str]]:
        """Scrape multiple URLs concurrently with per-domain rate limiting"""
        from urllib.parse import urlparse
        
        tasks = []
        for url in urls:
            domain = urlparse(url).netloc
            semaphore = self.get_domain_semaphore(domain)
            task = asyncio.create_task(self._scrape_with_domain_limit(semaphore, url))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        return [result if not isinstance(result, Exception) else None for result in results]
    
    async def _scrape_with_domain_limit(self, semaphore: asyncio.Semaphore, url: str):
        """Scrape URL with domain-specific rate limiting"""
        async with semaphore:
            return await self.scrape_images_safely(url)

    def _is_likely_product_image(self, url):
        """Check if URL is likely a product image"""
        url_lower = url.lower()
        
        # Good indicators
        good_indicators = ['product', 'furniture', 'item', 'catalog', 'hero', 'main', 'primary']
        if any(indicator in url_lower for indicator in good_indicators):
            return True
        
        # Bad indicators
        bad_indicators = ['icon', 'logo', 'banner', 'nav', 'thumb', 'small', 'mini']
        if any(indicator in url_lower for indicator in bad_indicators):
            return False
        
        # Check for reasonable image dimensions in URL
        if re.search(r'\d{3,4}x\d{3,4}', url) or re.search(r'w=\d{3,4}', url):
            return True
            
        return True  # Default to True if no clear indicators
