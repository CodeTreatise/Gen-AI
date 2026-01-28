---
title: "Test Organization"
---

# Test Organization

## Introduction

Well-organized tests are easy to find, maintain, and run selectively. Good structure separates different test types and makes CI/CD pipelines efficient.

### What We'll Cover

- Directory structure
- Separating test types
- Test markers
- Skipping tests
- Expected failures

### Prerequisites

- pytest fundamentals
- Fixtures

---

## Directory Structure

### Recommended Layout

```
project/
├── src/
│   └── myapp/
│       ├── __init__.py
│       ├── models.py
│       ├── services.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── conftest.py          # Unit test fixtures
│   │   ├── test_models.py
│   │   ├── test_services.py
│   │   └── test_utils.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   └── test_api.py
│   └── e2e/
│       ├── __init__.py
│       ├── conftest.py
│       └── test_workflow.py
├── pyproject.toml
└── pytest.ini
```

### Alternative: Tests Alongside Code

```
project/
├── myapp/
│   ├── __init__.py
│   ├── models.py
│   ├── test_models.py    # Tests next to source
│   ├── services.py
│   └── test_services.py
└── tests/
    └── integration/      # Integration tests separate
```

---

## Separating Test Types

### Unit Tests

```python
# tests/unit/test_calculator.py
"""Fast, isolated tests for individual functions."""

def test_add():
    assert add(1, 2) == 3

def test_subtract():
    assert subtract(5, 3) == 2
```

### Integration Tests

```python
# tests/integration/test_database.py
"""Tests that verify components work together."""

import pytest

@pytest.fixture
def database():
    db = create_test_database()
    yield db
    db.drop()

def test_user_crud(database):
    # Create
    user = database.create_user("Alice")
    assert user.id is not None
    
    # Read
    fetched = database.get_user(user.id)
    assert fetched.name == "Alice"
    
    # Delete
    database.delete_user(user.id)
    assert database.get_user(user.id) is None
```

### E2E Tests

```python
# tests/e2e/test_checkout.py
"""Full workflow tests."""

def test_complete_checkout_flow(browser, api_client):
    # Login
    browser.login("user@example.com", "password")
    
    # Add to cart
    browser.add_to_cart("Widget")
    
    # Checkout
    browser.checkout(payment_method="card")
    
    # Verify order created
    orders = api_client.get_orders()
    assert len(orders) == 1
```

---

## Test Markers

### Built-in Markers

```python
import pytest

@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Not supported on Windows"
)
def test_unix_only():
    pass

@pytest.mark.xfail(reason="Known bug #123")
def test_known_bug():
    assert broken_function() == expected

@pytest.mark.parametrize("input,expected", [(1, 2), (2, 4)])
def test_double(input, expected):
    assert double(input) == expected
```

### Custom Markers

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: integration tests",
    "e2e: end-to-end tests",
    "api: API tests",
]
```

```python
# tests/test_example.py
import pytest

@pytest.mark.slow
def test_slow_operation():
    # Takes several seconds
    pass

@pytest.mark.integration
def test_database_connection():
    pass

@pytest.mark.api
def test_external_api():
    pass
```

### Running by Marker

```bash
# Run only slow tests
pytest -m slow

# Run everything except slow tests
pytest -m "not slow"

# Run integration OR api tests
pytest -m "integration or api"

# Run integration AND api tests
pytest -m "integration and api"
```

---

## Skipping Tests

### Unconditional Skip

```python
import pytest

@pytest.mark.skip(reason="Feature not implemented")
def test_unimplemented():
    pass

def test_skip_inside():
    if some_condition:
        pytest.skip("Skipping because...")
    # Rest of test
```

### Conditional Skip

```python
import pytest
import sys

@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Requires Python 3.10+"
)
def test_match_statement():
    pass

@pytest.mark.skipif(
    not shutil.which("ffmpeg"),
    reason="Requires ffmpeg"
)
def test_video_processing():
    pass
```

### Skip Module

```python
# tests/test_cuda.py
import pytest

pytest.importorskip("torch.cuda")

# All tests in this file skipped if torch.cuda unavailable
def test_gpu_computation():
    pass
```

---

## Expected Failures

### xfail Marker

```python
import pytest

@pytest.mark.xfail(reason="Bug #456 not fixed")
def test_known_bug():
    assert buggy_function() == expected  # Expected to fail

@pytest.mark.xfail(strict=True)
def test_must_fail():
    # If this passes, it's an error (test should be removed/updated)
    pass

@pytest.mark.xfail(
    sys.platform == "darwin",
    reason="Flaky on macOS"
)
def test_flaky_on_mac():
    pass
```

### xfail at Runtime

```python
def test_dynamic_xfail():
    if condition:
        pytest.xfail("Known issue under this condition")
    # Rest of test
```

---

## Running Subsets

### By Path

```bash
# Run all tests in directory
pytest tests/unit/

# Run specific file
pytest tests/unit/test_models.py

# Run specific test
pytest tests/unit/test_models.py::test_user_creation

# Run specific class
pytest tests/unit/test_models.py::TestUser

# Run specific method
pytest tests/unit/test_models.py::TestUser::test_create
```

### By Keyword

```bash
# Tests containing "user"
pytest -k user

# Tests containing "user" but not "delete"
pytest -k "user and not delete"

# Multiple patterns
pytest -k "user or order"
```

---

## Configuration

### pyproject.toml

```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "-ra",  # Show extra summary for all except passed
]
markers = [
    "slow: slow tests",
    "integration: integration tests",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --strict-markers
markers =
    slow: marks tests as slow
    integration: integration tests
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -e ".[test]"
      
      - name: Run unit tests
        run: pytest tests/unit/ -v
  
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -e ".[test]"
      
      - name: Run integration tests
        run: pytest tests/integration/ -v
```

---

## Hands-on Exercise

### Your Task

Organize tests for a user management system:

```python
# Create appropriate test structure and markers for:
# 1. Unit tests for User model
# 2. Integration tests for UserRepository (database)
# 3. API tests for user endpoints
# 4. A slow test that should be skippable
```

<details>
<summary>✅ Solution</summary>

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: slow running tests",
    "integration: requires database",
    "api: API endpoint tests",
]
```

```python
# tests/conftest.py
import pytest

@pytest.fixture(scope="session")
def database():
    from myapp.db import create_test_db
    db = create_test_db()
    yield db
    db.drop()

@pytest.fixture
def api_client():
    from myapp import create_app
    app = create_app(testing=True)
    return app.test_client()
```

```python
# tests/unit/test_user.py
from myapp.models import User

class TestUser:
    def test_create_user(self):
        user = User(name="Alice", email="alice@example.com")
        assert user.name == "Alice"
    
    def test_user_email_validation(self):
        import pytest
        with pytest.raises(ValueError):
            User(name="Alice", email="invalid")
    
    def test_user_full_name(self):
        user = User(first_name="Alice", last_name="Smith")
        assert user.full_name == "Alice Smith"
```

```python
# tests/integration/test_user_repository.py
import pytest

@pytest.mark.integration
class TestUserRepository:
    def test_create_and_fetch_user(self, database):
        from myapp.repositories import UserRepository
        repo = UserRepository(database)
        
        user = repo.create(name="Alice", email="alice@example.com")
        fetched = repo.get_by_id(user.id)
        
        assert fetched.name == "Alice"
    
    @pytest.mark.slow
    def test_bulk_user_creation(self, database):
        from myapp.repositories import UserRepository
        repo = UserRepository(database)
        
        users = [{"name": f"User{i}"} for i in range(1000)]
        repo.bulk_create(users)
        
        assert repo.count() == 1000
```

```python
# tests/api/test_user_endpoints.py
import pytest

@pytest.mark.api
class TestUserAPI:
    def test_create_user_endpoint(self, api_client):
        response = api_client.post("/users", json={
            "name": "Alice",
            "email": "alice@example.com"
        })
        assert response.status_code == 201
    
    def test_get_user_endpoint(self, api_client):
        response = api_client.get("/users/1")
        assert response.status_code == 200
    
    @pytest.mark.skipif(
        not os.getenv("API_KEY"),
        reason="Requires API_KEY"
    )
    def test_external_api_integration(self, api_client):
        response = api_client.post("/users/verify")
        assert response.status_code == 200
```

```bash
# Run specific test types
pytest tests/unit/                  # Unit only
pytest -m integration              # Integration only
pytest -m "not slow"               # Skip slow tests
pytest -m "api and not slow"       # API tests, not slow
```
</details>

---

## Summary

✅ **Directory structure** separates test types
✅ **Markers** categorize and select tests
✅ **skip/skipif** conditionally skip tests
✅ **xfail** marks known failures
✅ **`-k` and `-m`** run test subsets
✅ **Configuration** in pyproject.toml/pytest.ini

**Next:** [Code Coverage](./06-code-coverage.md)

---

## Further Reading

- [pytest Markers](https://docs.pytest.org/en/stable/how-to/mark.html)
- [pytest Configuration](https://docs.pytest.org/en/stable/reference/customize.html)

<!-- 
Sources Consulted:
- pytest Docs: https://docs.pytest.org/
-->
