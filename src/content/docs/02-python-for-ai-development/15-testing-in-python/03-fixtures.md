---
title: "Fixtures"
---

# Fixtures

## Introduction

Fixtures provide a way to set up test preconditions and clean up after tests. They make tests cleaner by separating setup logic from test logic.

### What We'll Cover

- Creating fixtures
- Fixture scope
- Fixture dependencies
- conftest.py for sharing
- Built-in fixtures

### Prerequisites

- pytest fundamentals

---

## Creating Fixtures

### Basic Fixture

```python
import pytest

@pytest.fixture
def sample_user():
    return {"name": "Alice", "email": "alice@example.com", "age": 30}

def test_user_name(sample_user):
    assert sample_user["name"] == "Alice"

def test_user_email(sample_user):
    assert "@" in sample_user["email"]
```

### Fixture with Setup and Teardown

```python
import pytest

@pytest.fixture
def database_connection():
    # Setup
    connection = create_connection("test_db")
    connection.begin_transaction()
    
    yield connection  # Provide the fixture value
    
    # Teardown (runs after test)
    connection.rollback()
    connection.close()

def test_insert_user(database_connection):
    database_connection.execute("INSERT INTO users VALUES (1, 'Alice')")
    result = database_connection.execute("SELECT * FROM users")
    assert len(result) == 1
```

### Fixture Returning Objects

```python
import pytest
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str
    active: bool = True

@pytest.fixture
def active_user():
    return User(name="Alice", email="alice@example.com")

@pytest.fixture
def inactive_user():
    return User(name="Bob", email="bob@example.com", active=False)

def test_active_user(active_user):
    assert active_user.active is True

def test_inactive_user(inactive_user):
    assert inactive_user.active is False
```

---

## Fixture Scope

### Scope Options

```python
import pytest

@pytest.fixture(scope="function")  # Default: New fixture per test
def per_test_fixture():
    return create_resource()

@pytest.fixture(scope="class")  # Once per test class
def per_class_fixture():
    return create_resource()

@pytest.fixture(scope="module")  # Once per test file
def per_module_fixture():
    return create_resource()

@pytest.fixture(scope="session")  # Once per entire test run
def per_session_fixture():
    return create_resource()
```

### Scope Comparison

| Scope | Created | Use Case |
|-------|---------|----------|
| `function` | Each test | Default, isolated tests |
| `class` | Each test class | Class-level setup |
| `module` | Each test file | Expensive module setup |
| `session` | Once per run | Database, external services |

### Session Scope Example

```python
import pytest

@pytest.fixture(scope="session")
def database():
    """Create database once for all tests."""
    print("Creating database...")
    db = Database("test_db")
    db.create_tables()
    
    yield db
    
    print("Dropping database...")
    db.drop_tables()
    db.close()

def test_create_user(database):
    database.insert("users", {"name": "Alice"})
    assert database.count("users") >= 1

def test_create_order(database):
    database.insert("orders", {"user_id": 1})
    assert database.count("orders") >= 1
```

---

## Fixture Dependencies

### Fixtures Using Other Fixtures

```python
import pytest

@pytest.fixture
def user():
    return {"id": 1, "name": "Alice"}

@pytest.fixture
def authenticated_client(user):
    """Depends on user fixture."""
    client = TestClient()
    client.login(user["id"])
    return client

@pytest.fixture
def order(user):
    """Depends on user fixture."""
    return {"id": 100, "user_id": user["id"], "items": []}

def test_create_order(authenticated_client, order):
    response = authenticated_client.post("/orders", json=order)
    assert response.status_code == 201
```

### Chain of Dependencies

```python
@pytest.fixture
def config():
    return {"db_url": "sqlite:///test.db"}

@pytest.fixture
def database(config):
    return Database(config["db_url"])

@pytest.fixture
def user_repository(database):
    return UserRepository(database)

@pytest.fixture
def user_service(user_repository):
    return UserService(user_repository)

def test_create_user(user_service):
    user = user_service.create("Alice", "alice@example.com")
    assert user.id is not None
```

---

## conftest.py

### Shared Fixtures

```python
# tests/conftest.py - automatically discovered by pytest

import pytest

@pytest.fixture
def api_client():
    """Available to all tests in tests/ directory."""
    return APIClient(base_url="http://localhost:8000")

@pytest.fixture
def sample_data():
    return {"items": [1, 2, 3], "total": 6}
```

### Directory Structure

```
tests/
├── conftest.py              # Fixtures for all tests
├── unit/
│   ├── conftest.py          # Fixtures for unit tests only
│   └── test_models.py
├── integration/
│   ├── conftest.py          # Fixtures for integration tests
│   └── test_api.py
└── e2e/
    ├── conftest.py          # Fixtures for e2e tests
    └── test_workflow.py
```

### Example conftest.py

```python
# tests/conftest.py
import pytest
from myapp import create_app, db

@pytest.fixture(scope="session")
def app():
    """Create application for testing."""
    app = create_app(testing=True)
    return app

@pytest.fixture(scope="session")
def database(app):
    """Set up test database."""
    with app.app_context():
        db.create_all()
        yield db
        db.drop_all()

@pytest.fixture
def client(app):
    """Test client for making requests."""
    return app.test_client()

@pytest.fixture
def auth_headers():
    """Authentication headers."""
    return {"Authorization": "Bearer test-token"}
```

---

## Built-in Fixtures

### tmp_path

```python
def test_create_file(tmp_path):
    """tmp_path provides a temporary directory."""
    file = tmp_path / "test.txt"
    file.write_text("Hello, World!")
    
    assert file.exists()
    assert file.read_text() == "Hello, World!"
    # Directory is automatically cleaned up
```

### tmp_path_factory

```python
@pytest.fixture(scope="session")
def shared_temp_dir(tmp_path_factory):
    """Create temp directory shared across session."""
    return tmp_path_factory.mktemp("data")
```

### capsys

```python
def test_print_output(capsys):
    """Capture stdout/stderr."""
    print("Hello, World!")
    
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"
    assert captured.err == ""
```

### caplog

```python
import logging

def test_logging(caplog):
    """Capture log messages."""
    logger = logging.getLogger(__name__)
    
    with caplog.at_level(logging.INFO):
        logger.info("Test message")
        logger.warning("Warning message")
    
    assert "Test message" in caplog.text
    assert len(caplog.records) == 2
```

### monkeypatch

```python
def test_environment_variable(monkeypatch):
    """Temporarily modify environment."""
    monkeypatch.setenv("API_KEY", "test-key")
    
    import os
    assert os.environ["API_KEY"] == "test-key"
    # Original value restored after test
```

---

## Fixture Factories

### Factory Pattern

```python
import pytest

@pytest.fixture
def make_user():
    """Factory for creating test users."""
    def _make_user(name="Test", email=None, active=True):
        email = email or f"{name.lower()}@example.com"
        return User(name=name, email=email, active=active)
    return _make_user

def test_create_multiple_users(make_user):
    alice = make_user("Alice")
    bob = make_user("Bob", active=False)
    
    assert alice.email == "alice@example.com"
    assert bob.active is False
```

---

## Hands-on Exercise

### Your Task

Create fixtures for testing a simple blog application:

```python
# blog.py
class Post:
    def __init__(self, title: str, content: str, author: str):
        self.title = title
        self.content = content
        self.author = author
        self.published = False
    
    def publish(self):
        self.published = True

class Blog:
    def __init__(self):
        self.posts = []
    
    def add_post(self, post: Post):
        self.posts.append(post)
    
    def get_published(self):
        return [p for p in self.posts if p.published]
```

<details>
<summary>✅ Solution</summary>

```python
# tests/conftest.py
import pytest
from blog import Post, Blog

@pytest.fixture
def make_post():
    """Factory for creating posts."""
    def _make_post(title="Test Post", content="Content", author="Author"):
        return Post(title=title, content=content, author=author)
    return _make_post

@pytest.fixture
def sample_post(make_post):
    """Single sample post."""
    return make_post()

@pytest.fixture
def published_post(make_post):
    """Published post."""
    post = make_post(title="Published Post")
    post.publish()
    return post

@pytest.fixture
def blog():
    """Empty blog instance."""
    return Blog()

@pytest.fixture
def blog_with_posts(blog, make_post):
    """Blog with mix of published and draft posts."""
    draft = make_post(title="Draft")
    published = make_post(title="Published")
    published.publish()
    
    blog.add_post(draft)
    blog.add_post(published)
    return blog


# tests/test_blog.py
def test_new_post_is_unpublished(sample_post):
    assert sample_post.published is False

def test_publish_post(sample_post):
    sample_post.publish()
    assert sample_post.published is True

def test_blog_starts_empty(blog):
    assert len(blog.posts) == 0

def test_add_post_to_blog(blog, sample_post):
    blog.add_post(sample_post)
    assert len(blog.posts) == 1

def test_get_published_posts(blog_with_posts):
    published = blog_with_posts.get_published()
    assert len(published) == 1
    assert published[0].title == "Published"

def test_multiple_posts_with_factory(blog, make_post):
    for i in range(5):
        blog.add_post(make_post(title=f"Post {i}"))
    assert len(blog.posts) == 5
```
</details>

---

## Summary

✅ **Fixtures** separate setup from test logic
✅ **`yield`** enables teardown after tests
✅ **Scope** controls fixture lifetime
✅ **Fixtures can depend** on other fixtures
✅ **conftest.py** shares fixtures across tests
✅ **Built-in fixtures** provide common utilities

**Next:** [Mocking](./04-mocking.md)

---

## Further Reading

- [pytest Fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [Built-in Fixtures](https://docs.pytest.org/en/stable/reference/fixtures.html)

<!-- 
Sources Consulted:
- pytest Docs: https://docs.pytest.org/
-->
