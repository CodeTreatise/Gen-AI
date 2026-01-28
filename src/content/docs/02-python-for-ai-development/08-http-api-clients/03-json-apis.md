---
title: "Working with JSON APIs"
---

# Working with JSON APIs

## Introduction

Most modern APIs communicate using JSON. This lesson covers parsing responses, handling errors, and dealing with common API patterns like pagination and rate limiting.

### What We'll Cover

- Sending JSON data
- Parsing JSON responses
- Error handling strategies
- Rate limiting
- Pagination

### Prerequisites

- Requests library basics

---

## Sending JSON Data

### POST with JSON

```python
import requests

url = 'https://httpbin.org/post'

# Method 1: json parameter (recommended)
payload = {
    'name': 'John Doe',
    'email': 'john@example.com',
    'preferences': {
        'newsletter': True,
        'notifications': ['email', 'sms']
    }
}

response = requests.post(url, json=payload)
print(response.json())
```

### Manual JSON Encoding

```python
import requests
import json

payload = {'key': 'value'}

# Manual approach (not recommended, but sometimes needed)
response = requests.post(
    url,
    data=json.dumps(payload),
    headers={'Content-Type': 'application/json'}
)
```

---

## Parsing JSON Responses

### Basic Parsing

```python
import requests

response = requests.get('https://api.github.com/users/python')

# Parse JSON response
data = response.json()

# Access nested data
print(data['login'])        # 'python'
print(data['public_repos']) # number of repos
print(data.get('bio', 'No bio'))  # Safe access with default
```

### Handling Invalid JSON

```python
import requests
from json import JSONDecodeError

response = requests.get('https://example.com/might-not-be-json')

try:
    data = response.json()
except JSONDecodeError:
    print(f"Response is not valid JSON: {response.text[:100]}")
    data = None
```

### Checking Content-Type

```python
import requests

response = requests.get('https://api.github.com')

content_type = response.headers.get('content-type', '')

if 'application/json' in content_type:
    data = response.json()
else:
    print(f"Unexpected content type: {content_type}")
    data = response.text
```

---

## Error Handling

### HTTP Status Code Handling

```python
import requests

def fetch_user(username: str):
    """Fetch user with proper error handling."""
    
    response = requests.get(
        f'https://api.github.com/users/{username}',
        timeout=10
    )
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        print(f"User '{username}' not found")
        return None
    elif response.status_code == 403:
        print("Rate limit exceeded or forbidden")
        return None
    elif response.status_code >= 500:
        print(f"Server error: {response.status_code}")
        return None
    else:
        print(f"Unexpected status: {response.status_code}")
        return None

user = fetch_user('python')
```

### API Error Responses

```python
import requests

def api_request(url: str, data: dict):
    """Handle API error responses."""
    
    response = requests.post(url, json=data, timeout=10)
    
    # Many APIs return error details in JSON
    result = response.json()
    
    if not response.ok:
        error_message = result.get('error', result.get('message', 'Unknown error'))
        error_code = result.get('code', response.status_code)
        print(f"API Error [{error_code}]: {error_message}")
        return None
    
    return result

# Example API error response:
# {"error": "Invalid email format", "code": "VALIDATION_ERROR"}
```

### Comprehensive Error Handler

```python
import requests
from requests.exceptions import (
    Timeout, 
    ConnectionError, 
    HTTPError,
    RequestException
)

def safe_api_call(url: str, **kwargs):
    """Make API call with comprehensive error handling."""
    
    try:
        response = requests.get(url, timeout=10, **kwargs)
        response.raise_for_status()
        return response.json()
    
    except Timeout:
        print("Request timed out")
    except ConnectionError:
        print("Failed to connect to server")
    except HTTPError as e:
        print(f"HTTP error: {e.response.status_code}")
        # Try to get error details from response
        try:
            error_data = e.response.json()
            print(f"Details: {error_data}")
        except:
            print(f"Response: {e.response.text[:200]}")
    except RequestException as e:
        print(f"Request failed: {e}")
    
    return None
```

---

## Rate Limiting

### Detecting Rate Limits

```python
import requests
import time

def handle_rate_limit(response):
    """Handle rate limit response."""
    
    if response.status_code == 429:
        # Check for Retry-After header
        retry_after = response.headers.get('Retry-After')
        
        if retry_after:
            wait_time = int(retry_after)
        else:
            wait_time = 60  # Default wait
        
        print(f"Rate limited. Waiting {wait_time} seconds...")
        return wait_time
    
    return 0

# Usage in loop
def fetch_with_rate_limit(url: str, max_retries: int = 3):
    for attempt in range(max_retries):
        response = requests.get(url)
        
        wait_time = handle_rate_limit(response)
        if wait_time > 0:
            time.sleep(wait_time)
            continue
        
        if response.ok:
            return response.json()
    
    return None
```

### Rate Limit Headers

```python
import requests

response = requests.get('https://api.github.com/users/python')

# GitHub rate limit headers
remaining = response.headers.get('X-RateLimit-Remaining')
limit = response.headers.get('X-RateLimit-Limit')
reset = response.headers.get('X-RateLimit-Reset')

print(f"Rate limit: {remaining}/{limit}")
if reset:
    import datetime
    reset_time = datetime.datetime.fromtimestamp(int(reset))
    print(f"Resets at: {reset_time}")
```

### Simple Rate Limiter

```python
import time
from functools import wraps

def rate_limit(calls_per_second: float):
    """Decorator to rate limit function calls."""
    min_interval = 1.0 / calls_per_second
    last_call = [0.0]  # Mutable container
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call[0]
            wait_time = min_interval - elapsed
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            result = func(*args, **kwargs)
            last_call[0] = time.time()
            return result
        
        return wrapper
    return decorator

@rate_limit(calls_per_second=2)  # Max 2 requests per second
def fetch_data(url):
    return requests.get(url).json()
```

---

## Pagination

### Offset-Based Pagination

```python
import requests

def fetch_all_items(base_url: str, per_page: int = 100):
    """Fetch all items using offset pagination."""
    
    all_items = []
    page = 1
    
    while True:
        response = requests.get(
            base_url,
            params={'page': page, 'per_page': per_page},
            timeout=10
        )
        response.raise_for_status()
        
        items = response.json()
        
        if not items:
            break
        
        all_items.extend(items)
        page += 1
        
        print(f"Fetched page {page - 1}, total items: {len(all_items)}")
    
    return all_items

# Example
# repos = fetch_all_items('https://api.github.com/users/python/repos')
```

### Link Header Pagination

```python
import requests
import re

def get_next_url(response):
    """Extract next page URL from Link header."""
    
    link_header = response.headers.get('Link', '')
    
    # Parse Link header: <url>; rel="next", <url>; rel="prev"
    links = {}
    for link in link_header.split(','):
        match = re.match(r'<(.+)>;\s*rel="(\w+)"', link.strip())
        if match:
            links[match.group(2)] = match.group(1)
    
    return links.get('next')

def fetch_paginated(start_url: str):
    """Fetch all pages using Link header."""
    
    all_items = []
    url = start_url
    
    while url:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        all_items.extend(response.json())
        url = get_next_url(response)
        
        print(f"Fetched {len(all_items)} items...")
    
    return all_items
```

### Cursor-Based Pagination

```python
import requests

def fetch_with_cursor(base_url: str, initial_cursor: str = None):
    """Fetch all items using cursor pagination."""
    
    all_items = []
    cursor = initial_cursor
    
    while True:
        params = {'limit': 100}
        if cursor:
            params['cursor'] = cursor
        
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        items = data.get('items', [])
        
        if not items:
            break
        
        all_items.extend(items)
        cursor = data.get('next_cursor')
        
        if not cursor:
            break
    
    return all_items
```

---

## Hands-on Exercise

### Your Task

```python
# Build a GitHub API client that:
# 1. Fetches all repositories for a user (with pagination)
# 2. Respects rate limits
# 3. Handles errors gracefully
# 4. Returns a list of repo names and star counts
```

<details>
<summary>✅ Solution</summary>

```python
import requests
import time

class GitHubClient:
    BASE_URL = 'https://api.github.com'
    
    def __init__(self, token: str = None):
        self.session = requests.Session()
        self.session.headers['Accept'] = 'application/vnd.github.v3+json'
        if token:
            self.session.headers['Authorization'] = f'Bearer {token}'
    
    def _check_rate_limit(self, response):
        """Handle rate limiting."""
        remaining = int(response.headers.get('X-RateLimit-Remaining', 1))
        
        if remaining == 0:
            reset = int(response.headers.get('X-RateLimit-Reset', 0))
            wait_time = max(reset - time.time(), 0) + 1
            print(f"Rate limited. Waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
            return True
        return False
    
    def _get_next_url(self, response):
        """Extract next URL from Link header."""
        link = response.headers.get('Link', '')
        for part in link.split(','):
            if 'rel="next"' in part:
                return part.split(';')[0].strip(' <>')
        return None
    
    def get_user_repos(self, username: str):
        """Fetch all repos for a user."""
        
        url = f'{self.BASE_URL}/users/{username}/repos'
        all_repos = []
        
        while url:
            try:
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 404:
                    print(f"User '{username}' not found")
                    return []
                
                if response.status_code == 403:
                    if self._check_rate_limit(response):
                        continue
                    print("Access forbidden")
                    return all_repos
                
                response.raise_for_status()
                
                repos = response.json()
                all_repos.extend(repos)
                
                url = self._get_next_url(response)
                print(f"Fetched {len(all_repos)} repos...")
                
            except requests.RequestException as e:
                print(f"Error: {e}")
                break
        
        # Return simplified data
        return [
            {'name': r['name'], 'stars': r['stargazers_count']}
            for r in all_repos
        ]
    
    def close(self):
        self.session.close()

# Usage
client = GitHubClient()
repos = client.get_user_repos('python')

print(f"\nTotal repos: {len(repos)}")
for repo in sorted(repos, key=lambda x: x['stars'], reverse=True)[:5]:
    print(f"  {repo['name']}: ⭐ {repo['stars']}")

client.close()
```
</details>

---

## Summary

✅ Use **`json=data`** to send JSON (auto-sets headers)
✅ **`response.json()`** parses JSON responses
✅ Check **status codes** and handle API error responses
✅ Handle **rate limits** with retry logic
✅ Support **pagination** (offset, Link header, cursor)

**Next:** [HTTPX Modern Client](./04-httpx.md)

---

## Further Reading

- [JSON API Best Practices](https://jsonapi.org/)
- [REST API Design](https://restfulapi.net/)

<!-- 
Sources Consulted:
- Requests Docs: https://requests.readthedocs.io/
-->
