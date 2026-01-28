---
title: "Mocking"
---

# Mocking

## Introduction

Mocking replaces real objects with fake ones during testing. This isolates the code under test and makes tests faster and more reliable by avoiding external dependencies.

### What We'll Cover

- Mock and MagicMock
- patch decorator and context manager
- Mocking API calls
- Mocking file operations
- Side effects

### Prerequisites

- pytest fundamentals
- Fixtures

---

## Mock Basics

### Creating Mocks

```python
from unittest.mock import Mock

# Create a basic mock
mock = Mock()

# Call it like a function
result = mock(1, 2, key="value")

# Access attributes
mock.some_attribute

# Chain calls
mock.method().another_method()
```

### Configuring Return Values

```python
from unittest.mock import Mock

mock = Mock()

# Set return value
mock.return_value = 42
assert mock() == 42

# Set attribute return value
mock.get_user.return_value = {"name": "Alice"}
assert mock.get_user(1) == {"name": "Alice"}

# Chain return values
mock.database.query.return_value = [1, 2, 3]
assert mock.database.query("SELECT *") == [1, 2, 3]
```

### Checking Calls

```python
from unittest.mock import Mock

mock = Mock()
mock(1, 2, key="value")

# Assert called
mock.assert_called()
mock.assert_called_once()
mock.assert_called_with(1, 2, key="value")
mock.assert_called_once_with(1, 2, key="value")

# Check call count
assert mock.call_count == 1

# Get call arguments
print(mock.call_args)  # call(1, 2, key='value')
print(mock.call_args_list)  # [call(1, 2, key='value')]
```

---

## MagicMock

### Magic Methods

```python
from unittest.mock import MagicMock

mock = MagicMock()

# Magic methods work automatically
len(mock)        # Works
mock[0]          # Works
mock["key"]      # Works
str(mock)        # Works

# Configure magic methods
mock.__len__.return_value = 5
assert len(mock) == 5

mock.__getitem__.return_value = "value"
assert mock["key"] == "value"
```

### When to Use MagicMock

```python
from unittest.mock import Mock, MagicMock

# Mock: Basic mocking
mock = Mock()
# len(mock)  # TypeError!

# MagicMock: When you need magic methods
magic = MagicMock()
len(magic)  # Works

# Most common: Use MagicMock by default
```

---

## patch Decorator

### Patching Functions

```python
from unittest.mock import patch

# Module: myapp/service.py
# import requests
# def fetch_user(user_id):
#     response = requests.get(f"https://api.example.com/users/{user_id}")
#     return response.json()

@patch("myapp.service.requests.get")
def test_fetch_user(mock_get):
    # Configure mock response
    mock_get.return_value.json.return_value = {"id": 1, "name": "Alice"}
    
    # Call the function
    from myapp.service import fetch_user
    result = fetch_user(1)
    
    # Assertions
    assert result["name"] == "Alice"
    mock_get.assert_called_once_with("https://api.example.com/users/1")
```

### Patching Multiple Objects

```python
@patch("myapp.service.requests.post")
@patch("myapp.service.requests.get")
def test_multiple_requests(mock_get, mock_post):
    # Note: Order is reversed! Innermost patch = first parameter
    mock_get.return_value.json.return_value = {"id": 1}
    mock_post.return_value.status_code = 201
```

### patch as Context Manager

```python
from unittest.mock import patch

def test_with_context_manager():
    with patch("myapp.service.requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"name": "Alice"}
        
        from myapp.service import fetch_user
        result = fetch_user(1)
        
        assert result["name"] == "Alice"
```

---

## patch.object

### Patching Object Methods

```python
from unittest.mock import patch

class UserService:
    def get_user(self, user_id):
        # Real implementation
        pass

def test_user_service():
    service = UserService()
    
    with patch.object(service, "get_user") as mock_get:
        mock_get.return_value = {"id": 1, "name": "Alice"}
        
        result = service.get_user(1)
        assert result["name"] == "Alice"
```

---

## Mocking API Calls

### requests Library

```python
from unittest.mock import patch, Mock

def fetch_data(url):
    import requests
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@patch("requests.get")
def test_fetch_data_success(mock_get):
    # Setup mock response
    mock_response = Mock()
    mock_response.json.return_value = {"data": "test"}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    result = fetch_data("https://api.example.com/data")
    
    assert result == {"data": "test"}
    mock_get.assert_called_once_with("https://api.example.com/data")

@patch("requests.get")
def test_fetch_data_error(mock_get):
    from requests.exceptions import HTTPError
    
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = HTTPError("404")
    mock_get.return_value = mock_response
    
    import pytest
    with pytest.raises(HTTPError):
        fetch_data("https://api.example.com/data")
```

### Using responses Library

```python
import responses
import requests

@responses.activate
def test_with_responses():
    responses.add(
        responses.GET,
        "https://api.example.com/users/1",
        json={"id": 1, "name": "Alice"},
        status=200
    )
    
    response = requests.get("https://api.example.com/users/1")
    assert response.json()["name"] == "Alice"
```

---

## Mocking File Operations

### Basic File Mocking

```python
from unittest.mock import patch, mock_open

def read_config(path):
    with open(path) as f:
        return f.read()

def test_read_config():
    mock_content = "key=value\nother=data"
    
    with patch("builtins.open", mock_open(read_data=mock_content)):
        result = read_config("config.txt")
        assert result == mock_content
```

### JSON Files

```python
from unittest.mock import patch, mock_open
import json

def load_settings(path):
    with open(path) as f:
        return json.load(f)

def test_load_settings():
    mock_data = json.dumps({"debug": True, "port": 8080})
    
    with patch("builtins.open", mock_open(read_data=mock_data)):
        settings = load_settings("settings.json")
        assert settings["debug"] is True
        assert settings["port"] == 8080
```

---

## Side Effects

### Return Different Values

```python
from unittest.mock import Mock

mock = Mock()
mock.side_effect = [1, 2, 3]

assert mock() == 1
assert mock() == 2
assert mock() == 3
```

### Raise Exceptions

```python
from unittest.mock import Mock

mock = Mock()
mock.side_effect = ValueError("Invalid input")

import pytest
with pytest.raises(ValueError, match="Invalid input"):
    mock()
```

### Custom Function

```python
from unittest.mock import Mock

def custom_function(x):
    return x * 2

mock = Mock(side_effect=custom_function)
assert mock(5) == 10
assert mock(10) == 20
```

### Conditional Behavior

```python
from unittest.mock import Mock

def conditional_side_effect(*args, **kwargs):
    if args[0] == 1:
        return {"id": 1, "name": "Alice"}
    elif args[0] == 2:
        return {"id": 2, "name": "Bob"}
    else:
        raise ValueError("User not found")

mock = Mock(side_effect=conditional_side_effect)
assert mock(1)["name"] == "Alice"
assert mock(2)["name"] == "Bob"
```

---

## pytest-mock

### Simpler Syntax

```bash
pip install pytest-mock
```

```python
def test_with_mocker(mocker):
    # mocker is a fixture from pytest-mock
    mock_get = mocker.patch("requests.get")
    mock_get.return_value.json.return_value = {"data": "test"}
    
    import requests
    response = requests.get("https://api.example.com")
    assert response.json() == {"data": "test"}
```

### Spy on Real Objects

```python
def test_spy(mocker):
    # Spy calls the real function but tracks calls
    spy = mocker.spy(some_module, "some_function")
    
    result = some_module.some_function(1, 2)
    
    spy.assert_called_once_with(1, 2)
    # result is the real return value
```

---

## Hands-on Exercise

### Your Task

Mock the external API in this code:

```python
# weather.py
import requests

def get_weather(city: str) -> dict:
    response = requests.get(
        f"https://api.weather.com/v1/{city}",
        headers={"API-Key": "secret"}
    )
    response.raise_for_status()
    data = response.json()
    return {
        "city": city,
        "temperature": data["temp"],
        "conditions": data["conditions"]
    }
```

<details>
<summary>✅ Solution</summary>

```python
# tests/test_weather.py
from unittest.mock import patch, Mock
import pytest
from weather import get_weather

class TestGetWeather:
    @patch("weather.requests.get")
    def test_successful_weather_fetch(self, mock_get):
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {
            "temp": 72,
            "conditions": "sunny"
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Act
        result = get_weather("london")
        
        # Assert
        assert result == {
            "city": "london",
            "temperature": 72,
            "conditions": "sunny"
        }
        mock_get.assert_called_once_with(
            "https://api.weather.com/v1/london",
            headers={"API-Key": "secret"}
        )
    
    @patch("weather.requests.get")
    def test_api_error_raises_exception(self, mock_get):
        from requests.exceptions import HTTPError
        
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404")
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            get_weather("unknown_city")
    
    @patch("weather.requests.get")
    def test_different_cities(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        
        # Different responses for different cities
        def city_weather(*args, **kwargs):
            if "paris" in args[0]:
                mock_response.json.return_value = {"temp": 65, "conditions": "cloudy"}
            else:
                mock_response.json.return_value = {"temp": 50, "conditions": "rainy"}
            return mock_response
        
        mock_get.side_effect = city_weather
        
        paris = get_weather("paris")
        london = get_weather("london")
        
        assert paris["temperature"] == 65
        assert london["temperature"] == 50
```
</details>

---

## Summary

✅ **Mock** replaces real objects with test doubles
✅ **MagicMock** supports magic methods
✅ **patch** temporarily replaces during tests
✅ **side_effect** controls mock behavior
✅ **mock_open** mocks file operations
✅ **pytest-mock** provides simpler syntax

**Next:** [Test Organization](./05-test-organization.md)

---

## Further Reading

- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-mock](https://pytest-mock.readthedocs.io/)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/unittest.mock.html
-->
