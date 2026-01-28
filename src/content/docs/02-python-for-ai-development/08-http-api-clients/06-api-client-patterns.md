---
title: "API Client Patterns"
---

# API Client Patterns

## Introduction

Well-designed API clients improve reliability, maintainability, and developer experience. This lesson covers production-ready patterns for building robust API clients.

### What We'll Cover

- API client class design
- Configuration management
- Error handling strategies
- Retry with exponential backoff
- Request logging
- Response caching

### Prerequisites

- HTTP client libraries
- Python OOP basics

---

## Basic Client Class

### Simple API Client

```python
import requests
from typing import Any, Optional

class APIClient:
    """Basic API client with session management."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"
    
    def get(self, endpoint: str, **kwargs) -> dict:
        response = self.session.get(self._url(endpoint), **kwargs)
        response.raise_for_status()
        return response.json()
    
    def post(self, endpoint: str, data: dict, **kwargs) -> dict:
        response = self.session.post(self._url(endpoint), json=data, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def close(self):
        self.session.close()

# Usage
client = APIClient('https://api.example.com', 'my-api-key')
users = client.get('/users')
client.close()
```

### Context Manager Support

```python
import requests
from typing import Any

class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self._session = None
    
    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
        return self._session
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        if self._session:
            self._session.close()
            self._session = None
    
    def get(self, endpoint: str, **kwargs):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, **kwargs)
        response.raise_for_status()
        return response.json()

# Usage as context manager
with APIClient('https://api.example.com', 'key') as client:
    data = client.get('/users')
```

---

## Configuration Management

### Using dataclass for Config

```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class APIConfig:
    base_url: str
    api_key: str
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Load configuration from environment variables."""
        return cls(
            base_url=os.environ['API_BASE_URL'],
            api_key=os.environ['API_KEY'],
            timeout=float(os.environ.get('API_TIMEOUT', 30)),
            max_retries=int(os.environ.get('API_MAX_RETRIES', 3))
        )

# Usage
config = APIConfig(
    base_url='https://api.example.com',
    api_key='secret-key'
)
# Or from environment:
# config = APIConfig.from_env()
```

### Client with Config

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class ConfiguredClient:
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        
        # Retry strategy
        retry = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers
        session.headers.update({
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        })
        
        return session
    
    def request(self, method: str, endpoint: str, **kwargs):
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault('timeout', self.config.timeout)
        
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
```

---

## Error Handling

### Custom Exceptions

```python
class APIError(Exception):
    """Base exception for API errors."""
    pass

class AuthenticationError(APIError):
    """Authentication failed."""
    pass

class RateLimitError(APIError):
    """Rate limit exceeded."""
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Rate limited. Retry after {retry_after}s")

class NotFoundError(APIError):
    """Resource not found."""
    pass

class ServerError(APIError):
    """Server-side error."""
    pass
```

### Error Handler Method

```python
import requests
from typing import Any

class APIClient:
    def _handle_response(self, response: requests.Response) -> Any:
        """Handle response and raise appropriate exceptions."""
        
        if response.ok:
            return response.json()
        
        # Try to get error message from response
        try:
            error_data = response.json()
            message = error_data.get('error', error_data.get('message', 'Unknown error'))
        except:
            message = response.text or response.reason
        
        # Raise specific exceptions
        if response.status_code == 401:
            raise AuthenticationError(message)
        elif response.status_code == 403:
            raise AuthenticationError(f"Forbidden: {message}")
        elif response.status_code == 404:
            raise NotFoundError(message)
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            raise RateLimitError(retry_after)
        elif 500 <= response.status_code < 600:
            raise ServerError(f"Server error {response.status_code}: {message}")
        else:
            raise APIError(f"HTTP {response.status_code}: {message}")
```

---

## Retry with Exponential Backoff

### Manual Implementation

```python
import time
import random
from typing import Callable, TypeVar

T = TypeVar('T')

def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
) -> T:
    """Retry function with exponential backoff."""
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            if attempt == max_retries:
                raise
            
            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            wait_time = delay + jitter
            
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

# Usage
def make_request():
    return requests.get('https://api.example.com/data').json()

result = retry_with_backoff(
    make_request,
    max_retries=3,
    exceptions=(requests.RequestException, RateLimitError)
)
```

### Retry Decorator

```python
import functools
import time
import random

def retry(max_retries=3, base_delay=1.0, exceptions=(Exception,)):
    """Decorator for retry with exponential backoff."""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    jitter = random.uniform(0, delay * 0.1)
                    time.sleep(delay + jitter)
        
        return wrapper
    return decorator

# Usage
class APIClient:
    @retry(max_retries=3, base_delay=1.0, exceptions=(requests.RequestException,))
    def get(self, endpoint: str):
        response = self.session.get(f"{self.base_url}/{endpoint}")
        response.raise_for_status()
        return response.json()
```

---

## Request Logging

### Simple Logging

```python
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggedClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def request(self, method: str, endpoint: str, **kwargs):
        url = f"{self.base_url}/{endpoint}"
        
        logger.info(f"→ {method} {url}")
        
        try:
            response = self.session.request(method, url, **kwargs)
            
            logger.info(
                f"← {response.status_code} {response.reason} "
                f"({response.elapsed.total_seconds():.2f}s)"
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            logger.error(f"✗ Request failed: {e}")
            raise
```

### Debug Logging with Request/Response

```python
import logging
import json

class DebugClient:
    def __init__(self, base_url: str, debug: bool = False):
        self.base_url = base_url
        self.debug = debug
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
    
    def _log_request(self, method: str, url: str, **kwargs):
        if not self.debug:
            return
        
        self.logger.debug(f"Request: {method} {url}")
        if 'headers' in kwargs:
            self.logger.debug(f"Headers: {kwargs['headers']}")
        if 'json' in kwargs:
            self.logger.debug(f"Body: {json.dumps(kwargs['json'], indent=2)}")
    
    def _log_response(self, response: requests.Response):
        if not self.debug:
            return
        
        self.logger.debug(f"Response: {response.status_code}")
        self.logger.debug(f"Headers: {dict(response.headers)}")
        try:
            self.logger.debug(f"Body: {json.dumps(response.json(), indent=2)}")
        except:
            self.logger.debug(f"Body: {response.text[:500]}")
```

---

## Response Caching

### Simple In-Memory Cache

```python
import time
from functools import lru_cache
from typing import Optional

class CachedClient:
    def __init__(self, base_url: str, cache_ttl: int = 300):
        self.base_url = base_url
        self.cache_ttl = cache_ttl
        self.session = requests.Session()
        self._cache: dict = {}
    
    def _get_cache_key(self, method: str, endpoint: str, params: dict = None) -> str:
        params_str = str(sorted(params.items())) if params else ''
        return f"{method}:{endpoint}:{params_str}"
    
    def get(self, endpoint: str, params: dict = None, use_cache: bool = True) -> dict:
        cache_key = self._get_cache_key('GET', endpoint, params)
        
        # Check cache
        if use_cache and cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        
        # Make request
        response = self.session.get(
            f"{self.base_url}/{endpoint}",
            params=params
        )
        response.raise_for_status()
        data = response.json()
        
        # Cache response
        self._cache[cache_key] = (data, time.time())
        
        return data
    
    def clear_cache(self):
        self._cache.clear()
```

### Using functools.lru_cache

```python
from functools import lru_cache
import hashlib
import json

class LRUCachedClient:
    def __init__(self, base_url: str, max_cache_size: int = 128):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Create cached method
        self._cached_get = lru_cache(maxsize=max_cache_size)(self._make_get)
    
    def _make_get(self, endpoint: str, params_hash: str) -> str:
        """Make GET request and return JSON string (for caching)."""
        response = self.session.get(f"{self.base_url}/{endpoint}")
        response.raise_for_status()
        return response.text
    
    def get(self, endpoint: str, params: dict = None, use_cache: bool = True) -> dict:
        params_hash = hashlib.md5(
            json.dumps(params or {}, sort_keys=True).encode()
        ).hexdigest()
        
        if use_cache:
            response_text = self._cached_get(endpoint, params_hash)
        else:
            response_text = self._make_get(endpoint, params_hash)
        
        return json.loads(response_text)
    
    def clear_cache(self):
        self._cached_get.cache_clear()
```

---

## Complete Production Client

```python
import logging
import time
import random
from dataclasses import dataclass
from typing import Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

@dataclass
class ClientConfig:
    base_url: str
    api_key: str
    timeout: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 1.0
    cache_ttl: int = 300
    debug: bool = False

class ProductionAPIClient:
    """Production-ready API client with all best practices."""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = self._create_session()
        self._cache: dict = {}
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        
        retry = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'ProductionClient/1.0'
        })
        
        return session
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        self.session.close()
    
    def get(self, endpoint: str, params: dict = None, use_cache: bool = True) -> Any:
        return self._request('GET', endpoint, params=params, use_cache=use_cache)
    
    def post(self, endpoint: str, data: dict) -> Any:
        return self._request('POST', endpoint, json=data)
    
    def _request(self, method: str, endpoint: str, use_cache: bool = False, **kwargs) -> Any:
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault('timeout', self.config.timeout)
        
        # Check cache for GET
        cache_key = None
        if use_cache and method == 'GET':
            cache_key = f"{endpoint}:{kwargs.get('params', {})}"
            if cache_key in self._cache:
                data, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self.config.cache_ttl:
                    self.logger.debug(f"Cache hit: {endpoint}")
                    return data
        
        # Log request
        self.logger.info(f"→ {method} {url}")
        
        try:
            response = self.session.request(method, url, **kwargs)
            
            self.logger.info(
                f"← {response.status_code} ({response.elapsed.total_seconds():.2f}s)"
            )
            
            data = self._handle_response(response)
            
            # Cache response
            if cache_key:
                self._cache[cache_key] = (data, time.time())
            
            return data
        
        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    def _handle_response(self, response: requests.Response) -> Any:
        if response.ok:
            return response.json() if response.text else None
        
        try:
            error = response.json()
        except:
            error = {'message': response.text}
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            raise RateLimitError(retry_after)
        
        raise APIError(f"HTTP {response.status_code}: {error}")

# Usage
config = ClientConfig(
    base_url='https://api.example.com',
    api_key='my-secret-key',
    debug=True
)

with ProductionAPIClient(config) as client:
    users = client.get('/users')
    new_user = client.post('/users', {'name': 'John'})
```

---

## Summary

✅ Design clients as **classes** with session management
✅ Use **configuration objects** for flexibility
✅ Create **custom exceptions** for clear error handling
✅ Implement **retry with exponential backoff**
✅ Add **logging** for debugging and monitoring
✅ Use **caching** for read-heavy workloads

**Back to:** [HTTP & API Clients Overview](./00-http-api-clients.md)

---

## Further Reading

- [Python HTTP Client Best Practices](https://realpython.com/python-requests/)
- [API Design Patterns](https://cloud.google.com/apis/design)

<!-- 
Sources Consulted:
- Requests Docs: https://requests.readthedocs.io/
-->
