---
title: "Requests Library"
---

# Requests Library

## Introduction

The `requests` library is Python's most popular HTTP client. It provides a simple, elegant API for making HTTP requests and handling responses.

### What We'll Cover

- Installing requests
- GET and POST requests
- Headers and parameters
- Response handling
- Session objects

### Prerequisites

- Python basics
- Understanding of HTTP

---

## Installation

```bash
pip install requests
```

```python
import requests
print(requests.__version__)
```

---

## GET Requests

### Basic GET

```python
import requests

response = requests.get('https://api.github.com')

print(response.status_code)     # 200
print(response.headers)         # Response headers
print(response.text)            # Raw text
print(response.json())          # Parse as JSON
```

### With Query Parameters

```python
import requests

# Method 1: params dict (recommended)
params = {
    'q': 'python',
    'sort': 'stars',
    'order': 'desc'
}
response = requests.get('https://api.github.com/search/repositories', params=params)

# Method 2: In URL
# response = requests.get('https://api.github.com/search/repositories?q=python&sort=stars')

print(response.url)  # Shows full URL with params
print(response.json()['total_count'])
```

---

## POST Requests

### Form Data

```python
import requests

# Form-encoded data (default)
data = {
    'username': 'user123',
    'password': 'secret'
}

response = requests.post('https://httpbin.org/post', data=data)
print(response.json())
```

### JSON Data

```python
import requests

# JSON payload
payload = {
    'name': 'John Doe',
    'email': 'john@example.com'
}

response = requests.post(
    'https://httpbin.org/post',
    json=payload  # Automatically sets Content-Type: application/json
)

print(response.json())
```

### Difference: data vs json

| Parameter | Content-Type | Use Case |
|-----------|--------------|----------|
| `data={}` | application/x-www-form-urlencoded | Form submissions |
| `json={}` | application/json | API calls |

---

## Request Headers

### Custom Headers

```python
import requests

headers = {
    'Authorization': 'Bearer your-token-here',
    'Accept': 'application/json',
    'User-Agent': 'MyApp/1.0'
}

response = requests.get(
    'https://api.github.com/user',
    headers=headers
)

print(response.status_code)
```

### Common Headers

| Header | Purpose |
|--------|---------|
| `Authorization` | Authentication token |
| `Content-Type` | Request body format |
| `Accept` | Desired response format |
| `User-Agent` | Client identification |

---

## Response Handling

### Response Object

```python
import requests

response = requests.get('https://api.github.com')

# Status
print(response.status_code)      # 200
print(response.ok)               # True if 200-299
print(response.reason)           # "OK"

# Headers
print(response.headers['content-type'])

# Content
print(response.text)             # Text content
print(response.json())           # JSON parsed
print(response.content)          # Raw bytes

# Request info
print(response.url)              # Final URL
print(response.elapsed)          # Time taken
```

### Checking Status

```python
import requests

response = requests.get('https://api.github.com/users/invalid-user-12345')

if response.ok:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code} - {response.reason}")

# Or raise exception on error
try:
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
```

---

## Other HTTP Methods

```python
import requests

# PUT - Update resource
response = requests.put('https://httpbin.org/put', json={'key': 'value'})

# PATCH - Partial update
response = requests.patch('https://httpbin.org/patch', json={'key': 'updated'})

# DELETE - Remove resource
response = requests.delete('https://httpbin.org/delete')

# HEAD - Get headers only
response = requests.head('https://httpbin.org/get')

# OPTIONS - Get allowed methods
response = requests.options('https://httpbin.org/get')
```

---

## Session Objects

### Why Sessions?

- Persist cookies across requests
- Reuse connection for performance
- Share headers/auth across requests

### Using Sessions

```python
import requests

# Create session
session = requests.Session()

# Set default headers
session.headers.update({
    'Authorization': 'Bearer token123',
    'User-Agent': 'MyApp/1.0'
})

# All requests use session settings
response1 = session.get('https://api.example.com/users')
response2 = session.get('https://api.example.com/posts')

# Close when done
session.close()
```

### Session as Context Manager

```python
import requests

with requests.Session() as session:
    session.headers['Authorization'] = 'Bearer token123'
    
    # Login (cookies saved automatically)
    session.post('https://example.com/login', data={'user': 'admin'})
    
    # Subsequent requests include cookies
    profile = session.get('https://example.com/profile')
    print(profile.json())
```

---

## Downloading Files

### Download to Memory

```python
import requests

response = requests.get('https://example.com/file.pdf')

with open('downloaded.pdf', 'wb') as f:
    f.write(response.content)
```

### Streaming Large Files

```python
import requests

url = 'https://example.com/large-file.zip'

with requests.get(url, stream=True) as response:
    response.raise_for_status()
    with open('large-file.zip', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

print("Download complete")
```

---

## Hands-on Exercise

### Your Task

```python
# Create a GitHub API client that:
# 1. Fetches user info for a given username
# 2. Lists their public repositories
# 3. Handles errors gracefully
# 4. Uses a session for efficiency
```

<details>
<summary>✅ Solution</summary>

```python
import requests

def get_github_user(username: str):
    """Fetch GitHub user info and repos."""
    
    with requests.Session() as session:
        session.headers['Accept'] = 'application/vnd.github.v3+json'
        session.headers['User-Agent'] = 'Python-GitHub-Client'
        
        # Get user info
        user_url = f'https://api.github.com/users/{username}'
        user_response = session.get(user_url)
        
        if not user_response.ok:
            print(f"User not found: {user_response.status_code}")
            return None
        
        user = user_response.json()
        print(f"User: {user['login']}")
        print(f"Name: {user.get('name', 'N/A')}")
        print(f"Public repos: {user['public_repos']}")
        
        # Get repositories
        repos_url = f'https://api.github.com/users/{username}/repos'
        repos_response = session.get(repos_url, params={'sort': 'updated', 'per_page': 5})
        
        if repos_response.ok:
            repos = repos_response.json()
            print("\nRecent repositories:")
            for repo in repos:
                print(f"  - {repo['name']} ⭐ {repo['stargazers_count']}")
        
        return user

# Test
get_github_user('python')
```
</details>

---

## Summary

✅ Install with **`pip install requests`**
✅ **`requests.get()`** for fetching data
✅ **`requests.post(url, json=data)`** for APIs
✅ Use **`params={}`** for query parameters
✅ Check **`response.ok`** or **`raise_for_status()`**
✅ Use **`Session()`** for multiple requests

**Next:** [Request Configuration](./02-request-configuration.md)

---

## Further Reading

- [Requests Quickstart](https://requests.readthedocs.io/en/latest/user/quickstart/)
- [Advanced Usage](https://requests.readthedocs.io/en/latest/user/advanced/)

<!-- 
Sources Consulted:
- Requests Docs: https://requests.readthedocs.io/
-->
