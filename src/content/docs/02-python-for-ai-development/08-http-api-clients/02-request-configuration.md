---
title: "Request Configuration"
---

# Request Configuration

## Introduction

Production HTTP clients need proper configuration for reliability—timeouts prevent hangs, retries handle transient failures, and authentication secures access.

### What We'll Cover

- Timeout settings
- Retry strategies
- SSL verification
- Proxy configuration
- Authentication methods

### Prerequisites

- Requests library basics

---

## Timeouts

### Why Timeouts Matter

Without timeouts, requests can hang indefinitely if a server doesn't respond.

```python
import requests

# ❌ Bad: No timeout - can hang forever
response = requests.get('https://api.example.com')

# ✅ Good: Always set timeout
response = requests.get('https://api.example.com', timeout=10)
```

### Connection vs Read Timeout

```python
import requests

# Single timeout (applies to both)
response = requests.get('https://httpbin.org/get', timeout=5)

# Tuple: (connect_timeout, read_timeout)
response = requests.get(
    'https://httpbin.org/delay/2',
    timeout=(3.05, 10)  # 3.05s to connect, 10s to read
)
```

### Handling Timeout Errors

```python
import requests
from requests.exceptions import Timeout, ConnectionError

try:
    response = requests.get('https://httpbin.org/delay/5', timeout=2)
except Timeout:
    print("Request timed out")
except ConnectionError:
    print("Connection failed")
```

---

## Retry Strategies

### Using urllib3 Retry

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure retry strategy
retry_strategy = Retry(
    total=3,                    # Total retries
    backoff_factor=1,           # Wait 1, 2, 4 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]  # Methods to retry
)

# Create adapter with retry
adapter = HTTPAdapter(max_retries=retry_strategy)

# Mount to session
session = requests.Session()
session.mount("http://", adapter)
session.mount("https://", adapter)

# Now requests will auto-retry
response = session.get('https://httpbin.org/status/503', timeout=5)
```

### Retry Parameters

| Parameter | Description |
|-----------|-------------|
| `total` | Maximum retry attempts |
| `backoff_factor` | Delay multiplier between retries |
| `status_forcelist` | HTTP status codes to retry |
| `allowed_methods` | HTTP methods to retry |
| `raise_on_status` | Raise exception on bad status |

### Exponential Backoff

```python
from urllib3.util.retry import Retry

# backoff_factor=2 means:
# Retry 1: wait 2 * (2^0) = 2 seconds
# Retry 2: wait 2 * (2^1) = 4 seconds
# Retry 3: wait 2 * (2^2) = 8 seconds

retry = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504]
)
```

---

## SSL Verification

### Default Behavior

```python
import requests

# SSL verification is ON by default
response = requests.get('https://api.github.com')  # Verifies SSL cert
```

### Disabling Verification (Development Only)

```python
import requests
import urllib3

# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ⚠️ Only for development/testing!
response = requests.get('https://self-signed.example.com', verify=False)
```

### Custom CA Bundle

```python
import requests

# Use custom CA certificate
response = requests.get(
    'https://api.example.com',
    verify='/path/to/ca-bundle.crt'
)

# Or set environment variable
# export REQUESTS_CA_BUNDLE=/path/to/ca-bundle.crt
```

### Client Certificates

```python
import requests

# Client certificate authentication
response = requests.get(
    'https://api.example.com',
    cert=('/path/to/client.cert', '/path/to/client.key')
)
```

---

## Proxy Configuration

### HTTP Proxy

```python
import requests

proxies = {
    'http': 'http://proxy.example.com:8080',
    'https': 'http://proxy.example.com:8080'
}

response = requests.get('https://api.github.com', proxies=proxies)
```

### Authenticated Proxy

```python
import requests

proxies = {
    'http': 'http://user:password@proxy.example.com:8080',
    'https': 'http://user:password@proxy.example.com:8080'
}

response = requests.get('https://api.github.com', proxies=proxies)
```

### SOCKS Proxy

```bash
pip install requests[socks]
```

```python
import requests

proxies = {
    'http': 'socks5://localhost:9050',
    'https': 'socks5://localhost:9050'
}

response = requests.get('https://api.github.com', proxies=proxies)
```

---

## Authentication

### Basic Authentication

```python
import requests
from requests.auth import HTTPBasicAuth

# Method 1: HTTPBasicAuth
response = requests.get(
    'https://api.example.com/protected',
    auth=HTTPBasicAuth('username', 'password')
)

# Method 2: Tuple shorthand
response = requests.get(
    'https://api.example.com/protected',
    auth=('username', 'password')
)
```

### Bearer Token (Most Common for APIs)

```python
import requests

token = 'your-api-token-here'

headers = {
    'Authorization': f'Bearer {token}'
}

response = requests.get('https://api.example.com/data', headers=headers)
```

### API Key Authentication

```python
import requests

# In header
headers = {'X-API-Key': 'your-api-key'}
response = requests.get('https://api.example.com/data', headers=headers)

# In query parameter
params = {'api_key': 'your-api-key'}
response = requests.get('https://api.example.com/data', params=params)
```

### Digest Authentication

```python
import requests
from requests.auth import HTTPDigestAuth

response = requests.get(
    'https://api.example.com/digest-auth',
    auth=HTTPDigestAuth('user', 'pass')
)
```

### OAuth2 with Session

```python
import requests

class OAuth2Session(requests.Session):
    def __init__(self, token):
        super().__init__()
        self.headers['Authorization'] = f'Bearer {token}'

# Usage
session = OAuth2Session('your-oauth-token')
response = session.get('https://api.example.com/me')
```

---

## Complete Configuration Example

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_configured_session(token: str = None):
    """Create a production-ready requests session."""
    
    session = requests.Session()
    
    # Retry strategy
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Default headers
    session.headers.update({
        'User-Agent': 'MyApp/1.0',
        'Accept': 'application/json'
    })
    
    # Auth if provided
    if token:
        session.headers['Authorization'] = f'Bearer {token}'
    
    return session

# Usage
session = create_configured_session(token='my-api-token')

try:
    response = session.get('https://api.example.com/data', timeout=10)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
finally:
    session.close()
```

---

## Hands-on Exercise

### Your Task

```python
# Create a robust API client that:
# 1. Has 3 retries with exponential backoff
# 2. Uses a 10-second timeout
# 3. Includes API key authentication
# 4. Handles common errors gracefully
```

<details>
<summary>✅ Solution</summary>

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class RobustAPIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.session = self._create_session(api_key)
    
    def _create_session(self, api_key: str) -> requests.Session:
        session = requests.Session()
        
        # Retry with exponential backoff
        retry = Retry(
            total=3,
            backoff_factor=2,  # 2, 4, 8 seconds
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers
        session.headers.update({
            'X-API-Key': api_key,
            'Accept': 'application/json',
            'User-Agent': 'RobustClient/1.0'
        })
        
        return session
    
    def get(self, endpoint: str, **kwargs):
        return self._request('GET', endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs):
        return self._request('POST', endpoint, **kwargs)
    
    def _request(self, method: str, endpoint: str, **kwargs):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault('timeout', 10)
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Timeout: {url}")
            raise
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error {response.status_code}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise
    
    def close(self):
        self.session.close()

# Usage
client = RobustAPIClient('https://api.example.com', 'my-api-key')
try:
    data = client.get('/users')
    print(data)
finally:
    client.close()
```
</details>

---

## Summary

✅ **Always set timeouts**: `timeout=(3.05, 10)`
✅ Use **`Retry`** with exponential backoff for reliability
✅ **Bearer tokens** are most common for API auth
✅ Configure **sessions** for production use
✅ Handle exceptions: `Timeout`, `HTTPError`, `RequestException`

**Next:** [Working with JSON APIs](./03-json-apis.md)

---

## Further Reading

- [Requests Advanced Usage](https://requests.readthedocs.io/en/latest/user/advanced/)
- [urllib3 Retry](https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html#urllib3.util.Retry)

<!-- 
Sources Consulted:
- Requests Docs: https://requests.readthedocs.io/en/latest/user/advanced/
-->
