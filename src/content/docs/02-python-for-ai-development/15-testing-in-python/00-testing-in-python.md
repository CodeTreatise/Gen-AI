---
title: "Testing in Python"
---

# Testing in Python

## Overview

Testing ensures your code works correctly and continues working as you make changes. pytest is Python's most popular testing framework, offering simple syntax and powerful features.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-pytest-fundamentals.md) | pytest Fundamentals | Test discovery, assertions, running tests |
| [02](./02-writing-effective-tests.md) | Writing Effective Tests | AAA pattern, parameterized tests |
| [03](./03-fixtures.md) | Fixtures | Setup/teardown, scope, conftest.py |
| [04](./04-mocking.md) | Mocking | Mock, patch, API mocking |
| [05](./05-test-organization.md) | Test Organization | Structure, markers, skipping |
| [06](./06-code-coverage.md) | Code Coverage | pytest-cov, reports, CI/CD |

---

## Quick Start

```bash
# Install pytest
pip install pytest

# Run tests
pytest

# Verbose output
pytest -v

# Run specific file
pytest tests/test_example.py
```

```python
# tests/test_example.py
def test_addition():
    assert 1 + 1 == 2

def test_string():
    assert "hello".upper() == "HELLO"
```

---

## Why pytest?

| Feature | Benefit |
|---------|---------|
| **Simple assertions** | Just use `assert` |
| **Auto-discovery** | Finds tests automatically |
| **Fixtures** | Reusable setup/teardown |
| **Plugins** | Extensive ecosystem |
| **Detailed output** | Clear failure messages |

---

## Testing Pyramid

```
        /\
       /  \      E2E Tests (few)
      /----\
     /      \    Integration Tests
    /--------\
   /          \  Unit Tests (many)
  /------------\
```

| Level | Tests | Speed | Coverage |
|-------|-------|-------|----------|
| Unit | Many | Fast | Functions, classes |
| Integration | Some | Medium | Components together |
| E2E | Few | Slow | Full system |

---

## Prerequisites

Before starting this lesson:
- Python functions and classes
- Basic command line usage
- Understanding of modules

---

## Start Learning

Begin with [pytest Fundamentals](./01-pytest-fundamentals.md) to understand test basics.

---

## Further Reading

- [pytest Documentation](https://docs.pytest.org/)
- [Python Testing with pytest](https://pragprog.com/titles/bopytest2/python-testing-with-pytest-second-edition/)
- [Real Python Testing Guide](https://realpython.com/pytest-python-testing/)
