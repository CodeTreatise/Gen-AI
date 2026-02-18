---
title: "Web Scraping with AI"
---

# Web Scraping with AI

## Introduction

Traditional web scraping relies on rigid selectors and hand-crafted parsing rules that break whenever a site changes its HTML structure. AI-powered scraping flips this model: instead of writing brittle extraction rules, we let language models **understand** the content and extract structured data from messy, unpredictable pages.

In this lesson, we combine Playwright's browser automation with LLMs to build scrapers that adapt to layout changes, extract meaning from unstructured text, and handle the messy reality of the modern web.

### What We'll Cover
- Extracting raw content with Playwright
- Using LLMs to structure unstructured page content
- Pattern recognition across multiple pages
- Handling anti-bot measures responsibly
- Building an adaptive scraping pipeline

### Prerequisites
- Playwright navigation and page interaction (Lesson 01)
- Visual element identification (Lesson 04)
- LLM API integration (Unit 4)
- Understanding of HTML structure (Unit 1)

---

## Extracting Raw Content

Before an LLM can structure data, we need to extract raw content from the page. Playwright gives us multiple ways to get at page content.

### Text Extraction

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://example.com", wait_until="domcontentloaded")
    
    # Method 1: Visible text only (what a user sees)
    visible_text = page.inner_text("body")
    print(f"Visible text ({len(visible_text)} chars):")
    print(visible_text[:200])
    print()
    
    # Method 2: Full HTML source
    html = page.content()
    print(f"HTML source: {len(html)} chars")
    print()
    
    # Method 3: Text from specific elements
    heading = page.locator("h1").inner_text()
    print(f"Heading: {heading}")
    
    # Method 4: All links with their text and URLs
    links = page.evaluate("""
        () => Array.from(document.querySelectorAll('a[href]'))
            .map(a => ({
                text: a.textContent.trim(),
                href: a.href
            }))
    """)
    print(f"Links: {links}")
    
    browser.close()
```

**Output:**
```
Visible text (233 chars):
Example Domain
This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.
More information...

HTML source: 1256 chars

Heading: Example Domain
Links: [{'text': 'More information...', 'href': 'https://www.iana.org/help/example-domains'}]
```

### Structured Data Extraction

Many pages have structured data that can be extracted without an LLM:

```python
from playwright.sync_api import sync_playwright
import json

def extract_structured_data(page) -> dict:
    """Extract common structured data from a page."""
    data = {}
    
    # Meta tags
    data["meta"] = page.evaluate("""
        () => {
            const meta = {};
            document.querySelectorAll('meta').forEach(m => {
                const name = m.getAttribute('name') || 
                             m.getAttribute('property');
                if (name) meta[name] = m.getAttribute('content');
            });
            return meta;
        }
    """)
    
    # JSON-LD structured data (schema.org)
    data["json_ld"] = page.evaluate("""
        () => {
            const scripts = document.querySelectorAll(
                'script[type="application/ld+json"]'
            );
            return Array.from(scripts).map(s => {
                try { return JSON.parse(s.textContent); }
                catch { return null; }
            }).filter(Boolean);
        }
    """)
    
    # Tables
    data["tables"] = page.evaluate("""
        () => {
            return Array.from(document.querySelectorAll('table')).map(table => {
                const headers = Array.from(table.querySelectorAll('th'))
                    .map(th => th.textContent.trim());
                const rows = Array.from(table.querySelectorAll('tbody tr'))
                    .map(tr => Array.from(tr.querySelectorAll('td'))
                        .map(td => td.textContent.trim()));
                return { headers, rows };
            });
        }
    """)
    
    return data

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <html>
        <head>
            <meta name="description" content="A test product page">
            <meta property="og:title" content="Laptop Pro">
            <script type="application/ld+json">
            {
                "@type": "Product",
                "name": "Laptop Pro",
                "price": "$999",
                "brand": "TechCo"
            }
            </script>
        </head>
        <body>
            <table>
                <thead><tr><th>Spec</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td>CPU</td><td>M3 Pro</td></tr>
                    <tr><td>RAM</td><td>16GB</td></tr>
                    <tr><td>Storage</td><td>512GB</td></tr>
                </tbody>
            </table>
        </body>
        </html>
    """)
    
    data = extract_structured_data(page)
    print(json.dumps(data, indent=2))
    browser.close()
```

**Output:**
```json
{
  "meta": {
    "description": "A test product page",
    "og:title": "Laptop Pro"
  },
  "json_ld": [
    {
      "@type": "Product",
      "name": "Laptop Pro",
      "price": "$999",
      "brand": "TechCo"
    }
  ],
  "tables": [
    {
      "headers": ["Spec", "Value"],
      "rows": [
        ["CPU", "M3 Pro"],
        ["RAM", "16GB"],
        ["Storage", "512GB"]
      ]
    }
  ]
}
```

---

## AI-Powered Content Structuring

The real power comes when we send raw page content to an LLM and ask it to extract structured data. This handles messy HTML, inconsistent formatting, and layout changes.

### Basic Content Extraction with LLMs

```python
import anthropic
import json
from playwright.sync_api import sync_playwright

def extract_with_llm(page_text: str, schema: dict) -> dict:
    """Use an LLM to extract structured data from page text."""
    client = anthropic.Anthropic()
    
    schema_str = json.dumps(schema, indent=2)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Extract structured data from this web page text.

Return data matching this JSON schema:
{schema_str}

Web page text:
---
{page_text[:4000]}
---

Return ONLY valid JSON matching the schema. If a field is not found, use null."""
        }]
    )
    
    return json.loads(response.content[0].text)

# Define what we want to extract
product_schema = {
    "name": "string - product name",
    "price": "number - price in dollars",
    "description": "string - product description",
    "specs": {
        "cpu": "string",
        "ram": "string",
        "storage": "string"
    },
    "availability": "string - in stock, out of stock, or unknown"
}

# Simulate extracting from a product page
sample_text = """
Laptop Pro - Premium Performance
$999.99 - Free Shipping
The Laptop Pro delivers blazing-fast performance with the latest M3 Pro chip.
Perfect for developers, designers, and AI engineers.

Specifications:
- Processor: Apple M3 Pro
- Memory: 16GB Unified
- Storage: 512GB SSD
- Display: 14.2" Liquid Retina XDR

In Stock - Ships within 24 hours
"""

# result = extract_with_llm(sample_text, product_schema)
# print(json.dumps(result, indent=2))

# Expected output:
expected = {
    "name": "Laptop Pro",
    "price": 999.99,
    "description": "The Laptop Pro delivers blazing-fast performance with the latest M3 Pro chip. Perfect for developers, designers, and AI engineers.",
    "specs": {
        "cpu": "Apple M3 Pro",
        "ram": "16GB Unified",
        "storage": "512GB SSD"
    },
    "availability": "in stock"
}
print(json.dumps(expected, indent=2))
```

**Output:**
```json
{
  "name": "Laptop Pro",
  "price": 999.99,
  "description": "The Laptop Pro delivers blazing-fast performance with the latest M3 Pro chip. Perfect for developers, designers, and AI engineers.",
  "specs": {
    "cpu": "Apple M3 Pro",
    "ram": "16GB Unified",
    "storage": "512GB SSD"
  },
  "availability": "in stock"
}
```

> **🤖 AI Context:** The key advantage over traditional scraping: if the website changes its layout, class names, or HTML structure, the LLM still extracts the right data. It understands *meaning*, not structure.

### Extracting Lists of Items

```python
import anthropic
import json

def extract_list_from_page(page_text: str, item_type: str) -> list:
    """Extract a list of items from page text using an LLM."""
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Extract all {item_type} from this web page text.

Return a JSON array where each item has these fields:
- name: string
- price: number or null
- description: string (brief)
- url: string or null

Web page text:
---
{page_text[:6000]}
---

Return ONLY the JSON array."""
        }]
    )
    
    return json.loads(response.content[0].text)

# Simulated search results page
search_results_text = """
Search results for "laptop"

Laptop Pro - $999
Premium performance laptop with M3 Pro chip. 16GB RAM, 512GB SSD.
★★★★★ (245 reviews)
/products/laptop-pro

Budget Book 15 - $449
Great value laptop for everyday tasks. 8GB RAM, 256GB SSD.
★★★★☆ (1,023 reviews)
/products/budget-book-15

Gaming Beast X - $1,599
Ultimate gaming laptop with RTX 4080. 32GB RAM, 1TB SSD.
★★★★★ (89 reviews)
/products/gaming-beast-x

Page 1 of 12
"""

# results = extract_list_from_page(search_results_text, "laptops")
# print(json.dumps(results, indent=2))

# Expected:
expected_results = [
    {
        "name": "Laptop Pro",
        "price": 999,
        "description": "Premium performance laptop with M3 Pro chip. 16GB RAM, 512GB SSD.",
        "url": "/products/laptop-pro"
    },
    {
        "name": "Budget Book 15",
        "price": 449,
        "description": "Great value laptop for everyday tasks. 8GB RAM, 256GB SSD.",
        "url": "/products/budget-book-15"
    },
    {
        "name": "Gaming Beast X",
        "price": 1599,
        "description": "Ultimate gaming laptop with RTX 4080. 32GB RAM, 1TB SSD.",
        "url": "/products/gaming-beast-x"
    }
]
print(json.dumps(expected_results, indent=2))
```

**Output:**
```json
[
  {
    "name": "Laptop Pro",
    "price": 999,
    "description": "Premium performance laptop with M3 Pro chip. 16GB RAM, 512GB SSD.",
    "url": "/products/laptop-pro"
  },
  {
    "name": "Budget Book 15",
    "price": 449,
    "description": "Great value laptop for everyday tasks. 8GB RAM, 256GB SSD.",
    "url": "/products/budget-book-15"
  },
  {
    "name": "Gaming Beast X",
    "price": 1599,
    "description": "Ultimate gaming laptop with RTX 4080. 32GB RAM, 1TB SSD.",
    "url": "/products/gaming-beast-x"
  }
]
```

---

## Building an Adaptive Scraping Pipeline

Here's a complete scraping pipeline that combines Playwright for content extraction with an LLM for data structuring:

```python
from playwright.sync_api import sync_playwright
import json
import time

class AIScraper:
    """An AI-powered web scraper that adapts to page structure."""
    
    def __init__(self, ai_client=None):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        self._page = self._context.new_page()
        self._client = ai_client
    
    def scrape(self, url: str, schema: dict) -> dict:
        """
        Scrape a URL and extract data matching the schema.
        
        Args:
            url: The page URL to scrape
            schema: Dict describing the data to extract
            
        Returns:
            Dict with extracted data and metadata
        """
        start = time.time()
        
        # Step 1: Navigate and wait for content
        response = self._page.goto(url, wait_until="domcontentloaded")
        self._page.wait_for_load_state("networkidle")
        
        # Step 2: Extract raw content
        raw = self._extract_raw_content()
        
        # Step 3: Try structured extraction first (no LLM needed)
        structured = self._extract_structured(raw)
        
        # Step 4: Use LLM for unstructured content
        if self._client:
            extracted = self._extract_with_ai(raw["text"], schema)
        else:
            extracted = structured
        
        elapsed = time.time() - start
        
        return {
            "url": url,
            "status": response.status if response else None,
            "extracted": extracted,
            "structured_data": structured,
            "metadata": {
                "title": raw["title"],
                "text_length": len(raw["text"]),
                "html_length": len(raw["html"]),
                "elapsed_seconds": round(elapsed, 2)
            }
        }
    
    def _extract_raw_content(self) -> dict:
        """Extract all raw content from the current page."""
        return {
            "title": self._page.title(),
            "text": self._page.inner_text("body"),
            "html": self._page.content(),
            "links": self._page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({text: a.textContent.trim(), href: a.href}))
                    .filter(a => a.text.length > 0)
            """),
            "images": self._page.evaluate("""
                () => Array.from(document.querySelectorAll('img'))
                    .map(img => ({src: img.src, alt: img.alt}))
            """),
            "json_ld": self._page.evaluate("""
                () => Array.from(
                    document.querySelectorAll('script[type="application/ld+json"]')
                ).map(s => {
                    try { return JSON.parse(s.textContent); }
                    catch { return null; }
                }).filter(Boolean)
            """)
        }
    
    def _extract_structured(self, raw: dict) -> dict:
        """Extract any structured data that doesn't need AI."""
        return {
            "json_ld": raw.get("json_ld", []),
            "link_count": len(raw.get("links", [])),
            "image_count": len(raw.get("images", []))
        }
    
    def _extract_with_ai(self, text: str, schema: dict) -> dict:
        """Use LLM to extract data matching the schema."""
        schema_str = json.dumps(schema, indent=2)
        
        response = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": f"""Extract data from this webpage text matching the schema.

Schema:
{schema_str}

Page text:
---
{text[:5000]}
---

Return ONLY valid JSON matching the schema."""
            }]
        )
        
        return json.loads(response.content[0].text)
    
    def scrape_multiple(self, urls: list, schema: dict, 
                        delay: float = 1.0) -> list:
        """Scrape multiple URLs with a delay between requests."""
        results = []
        for i, url in enumerate(urls):
            if i > 0:
                time.sleep(delay)  # Respectful delay
            try:
                result = self.scrape(url, schema)
                results.append(result)
                print(f"  [{i+1}/{len(urls)}] {url} — OK")
            except Exception as e:
                results.append({
                    "url": url,
                    "error": str(e)
                })
                print(f"  [{i+1}/{len(urls)}] {url} — ERROR: {e}")
        return results
    
    def close(self):
        """Release all resources."""
        self._context.close()
        self._browser.close()
        self._pw.stop()

# Usage (without AI client for demo)
scraper = AIScraper()
try:
    result = scraper.scrape(
        "https://example.com",
        schema={"title": "string", "description": "string"}
    )
    print(json.dumps(result, indent=2))
finally:
    scraper.close()
```

**Output:**
```json
{
  "url": "https://example.com/",
  "status": 200,
  "extracted": {
    "json_ld": [],
    "link_count": 1,
    "image_count": 0
  },
  "structured_data": {
    "json_ld": [],
    "link_count": 1,
    "image_count": 0
  },
  "metadata": {
    "title": "Example Domain",
    "text_length": 233,
    "html_length": 1256,
    "elapsed_seconds": 0.45
  }
}
```

---

## Pattern Recognition Across Pages

When scraping similar pages (product listings, articles, profiles), the LLM can recognize patterns and extract consistent data:

```python
import json

def create_adaptive_extractor(sample_pages: list, schema: dict):
    """
    Show the LLM a few sample pages so it learns the pattern,
    then extract from new pages more accurately.
    """
    # Build few-shot examples from sample pages
    examples = []
    for page in sample_pages:
        examples.append({
            "input": page["text"][:1000],
            "output": page["expected_data"]
        })
    
    def extract(page_text: str, client) -> dict:
        examples_str = "\n\n".join([
            f"Example input:\n{ex['input']}\n\nExample output:\n{json.dumps(ex['output'], indent=2)}"
            for ex in examples[:3]  # Use up to 3 examples
        ])
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": f"""You are extracting structured data from web pages.
Here are examples of correct extractions:

{examples_str}

Now extract from this new page:
---
{page_text[:4000]}
---

Return ONLY JSON matching the same format as the examples."""
            }]
        )
        return json.loads(response.content[0].text)
    
    return extract

# Example usage:
sample_pages = [
    {
        "text": "Laptop Pro - $999 - Premium laptop with M3 chip, 16GB RAM",
        "expected_data": {
            "name": "Laptop Pro",
            "price": 999,
            "specs": ["M3 chip", "16GB RAM"]
        }
    },
    {
        "text": "Budget Book - $449 - Affordable laptop, 8GB RAM, 256GB storage",
        "expected_data": {
            "name": "Budget Book",
            "price": 449,
            "specs": ["8GB RAM", "256GB storage"]
        }
    }
]

extractor = create_adaptive_extractor(sample_pages, {})
# new_data = extractor("Gaming Beast - $1599 - RTX 4080, 32GB RAM, 1TB SSD", client)
print("Adaptive extractor created with 2 examples")
```

**Output:**
```
Adaptive extractor created with 2 examples
```

---

## Handling Anti-Bot Measures

Modern websites use various techniques to block automated scraping. Here's how to handle them responsibly:

### Realistic Browser Behavior

```python
from playwright.sync_api import sync_playwright
import random
import time

def create_stealth_context(browser):
    """Create a browser context that behaves more like a real user."""
    context = browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        timezone_id="America/New_York",
        color_scheme="light"
    )
    
    # Add common headers
    context.set_extra_http_headers({
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/webp,*/*;q=0.8"
        )
    })
    
    return context

def human_like_delay(min_seconds=0.5, max_seconds=2.0):
    """Add a random delay to simulate human behavior."""
    time.sleep(random.uniform(min_seconds, max_seconds))

# Usage
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = create_stealth_context(browser)
    page = context.new_page()
    
    page.goto("https://example.com")
    human_like_delay()
    
    print(f"Page loaded: {page.title()}")
    
    context.close()
    browser.close()
```

**Output:**
```
Page loaded: Example Domain
```

### Checking robots.txt

Always check `robots.txt` before scraping:

```python
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse

def check_robots_txt(page, url: str) -> dict:
    """Check if a URL is allowed by robots.txt."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    
    try:
        response = page.goto(robots_url)
        if response and response.status == 200:
            robots_text = page.inner_text("body")
            
            # Simple parser — check for Disallow rules
            disallowed = []
            current_agent = None
            for line in robots_text.split("\n"):
                line = line.strip()
                if line.startswith("User-agent:"):
                    current_agent = line.split(":", 1)[1].strip()
                elif line.startswith("Disallow:"):
                    path = line.split(":", 1)[1].strip()
                    if current_agent in ("*", None) and path:
                        disallowed.append(path)
            
            # Check if our target path is disallowed
            target_path = parsed.path or "/"
            is_allowed = not any(
                target_path.startswith(d) for d in disallowed
            )
            
            return {
                "robots_found": True,
                "allowed": is_allowed,
                "disallowed_paths": disallowed
            }
        else:
            return {"robots_found": False, "allowed": True}
    except Exception:
        return {"robots_found": False, "allowed": True}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    result = check_robots_txt(page, "https://example.com/")
    print(f"robots.txt check: {result}")
    
    browser.close()
```

**Output:**
```
robots.txt check: {'robots_found': True, 'allowed': True, 'disallowed_paths': []}
```

### Rate Limiting

```python
import time
from collections import deque

class RateLimiter:
    """Enforce rate limits for responsible scraping."""
    
    def __init__(self, requests_per_minute: int = 10):
        self.max_rpm = requests_per_minute
        self.requests = deque()
    
    def wait(self):
        """Wait if necessary to stay within rate limits."""
        now = time.time()
        
        # Remove requests older than 60 seconds
        while self.requests and self.requests[0] < now - 60:
            self.requests.popleft()
        
        # If at limit, wait for the oldest request to expire
        if len(self.requests) >= self.max_rpm:
            wait_time = 60 - (now - self.requests[0])
            if wait_time > 0:
                print(f"Rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        
        self.requests.append(time.time())

# Usage
limiter = RateLimiter(requests_per_minute=10)

for i in range(3):
    limiter.wait()
    print(f"Request {i+1} at {time.strftime('%H:%M:%S')}")

print("Rate limiter working correctly")
```

**Output:**
```
Request 1 at 14:30:01
Request 2 at 14:30:01
Request 3 at 14:30:01
Rate limiter working correctly
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Check `robots.txt` before scraping | Respects site owner's wishes and avoids legal issues |
| Use rate limiting (10-30 req/min) | Prevents server overload and bans |
| Extract structured data (JSON-LD) first | Faster, cheaper, and more accurate than LLM extraction |
| Send text, not HTML, to LLMs | Reduces token usage — strip tags and navigation elements |
| Use few-shot examples for consistent extraction | The LLM learns your exact output format |
| Cache scraped content | Avoid re-scraping unchanged pages |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Sending entire HTML to the LLM | Extract visible text first with `inner_text("body")` |
| No delay between requests | Add 1-2 second delays and use `RateLimiter` |
| Ignoring `robots.txt` | Always check and respect disallow rules |
| Using default Playwright user agent | Set a realistic browser user agent |
| Scraping without caching | Store results and check cache before re-scraping |
| Not handling page load failures | Catch exceptions, retry with backoff, log failures |

---

## Hands-on Exercise

### Your Task

Build a `SmartScraper` class that extracts structured data from web pages using both DOM-based extraction and LLM-powered extraction.

### Requirements
1. Create a `SmartScraper` class with Playwright
2. Implement `extract_metadata(url)` — return title, description, Open Graph tags, and JSON-LD data
3. Implement `extract_links(url)` — return all links grouped by internal vs external
4. Implement `extract_text(url, selector="body")` — return cleaned visible text from a specific section
5. Implement `check_allowed(url)` — check robots.txt before scraping
6. Test with `https://example.com`

### Expected Result
A clean extraction of metadata, categorized links, and page text from any URL, with robots.txt compliance checking.

<details>
<summary>💡 Hints (click to expand)</summary>

- Meta tags: `document.querySelectorAll('meta[name], meta[property]')`
- JSON-LD: `document.querySelectorAll('script[type="application/ld+json"]')`
- Internal vs external links: compare `new URL(href).hostname` to current page hostname
- `page.inner_text(selector)` gets visible text from a specific element
- Parse `robots.txt` by splitting on newlines and checking `Disallow:` rules

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse
import json

class SmartScraper:
    """DOM-based scraper with robots.txt compliance."""
    
    def __init__(self):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
        )
        self._page = self._context.new_page()
    
    def extract_metadata(self, url: str) -> dict:
        """Extract page metadata, OG tags, and JSON-LD."""
        self._page.goto(url, wait_until="domcontentloaded")
        
        return self._page.evaluate("""
            () => {
                const meta = {};
                const og = {};
                
                document.querySelectorAll('meta').forEach(m => {
                    const name = m.getAttribute('name');
                    const prop = m.getAttribute('property');
                    const content = m.getAttribute('content');
                    
                    if (name) meta[name] = content;
                    if (prop && prop.startsWith('og:'))
                        og[prop.replace('og:', '')] = content;
                });
                
                const jsonLd = Array.from(
                    document.querySelectorAll('script[type="application/ld+json"]')
                ).map(s => {
                    try { return JSON.parse(s.textContent); }
                    catch { return null; }
                }).filter(Boolean);
                
                return {
                    title: document.title,
                    meta,
                    open_graph: og,
                    json_ld: jsonLd
                };
            }
        """)
    
    def extract_links(self, url: str) -> dict:
        """Extract all links, grouped by internal vs external."""
        self._page.goto(url, wait_until="domcontentloaded")
        hostname = urlparse(url).hostname
        
        all_links = self._page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]'))
                .map(a => ({
                    text: a.textContent.trim(),
                    href: a.href
                }))
                .filter(a => a.text && a.href.startsWith('http'))
        """)
        
        internal = [l for l in all_links 
                     if urlparse(l["href"]).hostname == hostname]
        external = [l for l in all_links 
                     if urlparse(l["href"]).hostname != hostname]
        
        return {
            "internal": internal,
            "external": external,
            "total": len(all_links)
        }
    
    def extract_text(self, url: str, selector: str = "body") -> str:
        """Extract visible text from a page section."""
        self._page.goto(url, wait_until="domcontentloaded")
        return self._page.inner_text(selector).strip()
    
    def check_allowed(self, url: str) -> bool:
        """Check if scraping this URL is allowed by robots.txt."""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        try:
            resp = self._page.goto(robots_url)
            if not resp or resp.status != 200:
                return True
            
            text = self._page.inner_text("body")
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("Disallow:"):
                    path = line.split(":", 1)[1].strip()
                    if path and (parsed.path or "/").startswith(path):
                        return False
            return True
        except Exception:
            return True
    
    def close(self):
        self._context.close()
        self._browser.close()
        self._pw.stop()

# Test
scraper = SmartScraper()
try:
    # Check robots.txt
    allowed = scraper.check_allowed("https://example.com/")
    print(f"Allowed: {allowed}")
    
    # Extract metadata
    metadata = scraper.extract_metadata("https://example.com")
    print(f"Title: {metadata['title']}")
    print(f"Meta tags: {len(metadata['meta'])}")
    
    # Extract links
    links = scraper.extract_links("https://example.com")
    print(f"Links: {links['total']} (internal: {len(links['internal'])}, external: {len(links['external'])})")
    
    # Extract text
    text = scraper.extract_text("https://example.com")
    print(f"Text: {text[:80]}...")
finally:
    scraper.close()
```

**Output:**
```
Allowed: True
Title: Example Domain
Meta tags: 1
Links: 1 (internal: 0, external: 1)
Text: Example Domain
This domain is for use in illustrative examples in documents. Yo...
```

</details>

### Bonus Challenges
- [ ] Add an LLM extraction mode: `extract_structured(url, schema)` that sends text to an LLM
- [ ] Implement pagination: scrape multiple pages of results automatically
- [ ] Add a caching layer to avoid re-scraping URLs within a time window

---

## Summary

✅ Extract structured data (JSON-LD, meta tags, tables) before using an LLM — it's faster and cheaper

✅ LLMs excel at extracting structured data from unstructured text — they handle layout changes gracefully

✅ Send visible text (`inner_text("body")`), not raw HTML, to reduce token usage

✅ Always check `robots.txt` and implement rate limiting for responsible scraping

✅ Few-shot examples help LLMs produce consistent, well-formatted extraction results

**Next:** [Form Filling Automation](./06-form-filling-automation.md)

**Previous:** [Visual Element Identification](./04-visual-element-identification.md)

---

## Further Reading

- [Playwright Network Events](https://playwright.dev/python/docs/network) - Intercepting and mocking network requests
- [Schema.org](https://schema.org/) - Structured data vocabulary
- [robots.txt Standard](https://www.robotstxt.org/) - Robots exclusion protocol

<!-- 
Sources Consulted:
- Playwright Input/Actions: https://playwright.dev/python/docs/input
- Playwright Locators: https://playwright.dev/python/docs/locators
- Playwright Network: https://playwright.dev/python/docs/network
- robots.txt: https://www.robotstxt.org/
-->
