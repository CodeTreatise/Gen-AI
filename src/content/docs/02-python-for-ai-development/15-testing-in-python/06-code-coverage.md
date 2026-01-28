---
title: "Code Coverage"
---

# Code Coverage

## Introduction

Code coverage measures which lines of code are executed during testing. It helps identify untested code but shouldn't be the only metric for test quality.

### What We'll Cover

- pytest-cov plugin
- Coverage reports
- Coverage thresholds
- Excluding code
- CI/CD integration

### Prerequisites

- pytest fundamentals
- Test organization

---

## Installation

```bash
pip install pytest-cov
```

---

## Basic Usage

### Running with Coverage

```bash
# Basic coverage report
pytest --cov=myapp

# Coverage for specific directory
pytest --cov=src/myapp tests/

# With terminal report
pytest --cov=myapp --cov-report=term
```

### Terminal Output

```
---------- coverage: platform linux, python 3.11.0 ----------
Name                      Stmts   Miss  Cover
---------------------------------------------
myapp/__init__.py             5      0   100%
myapp/models.py              42      8    81%
myapp/services.py            65     15    77%
myapp/utils.py               28      0   100%
---------------------------------------------
TOTAL                       140     23    84%
```

---

## Report Formats

### Terminal Reports

```bash
# Missing lines shown
pytest --cov=myapp --cov-report=term-missing
```

**Output:**
```
Name                 Stmts   Miss  Cover   Missing
--------------------------------------------------
myapp/models.py         42      8    81%   45-52, 78
myapp/services.py       65     15    77%   23-30, 55-62
```

### HTML Report

```bash
pytest --cov=myapp --cov-report=html
# Opens htmlcov/index.html
```

### XML Report (CI/CD)

```bash
pytest --cov=myapp --cov-report=xml
# Creates coverage.xml
```

### Multiple Reports

```bash
pytest --cov=myapp \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml
```

---

## Coverage Thresholds

### Fail Below Threshold

```bash
# Fail if coverage < 80%
pytest --cov=myapp --cov-fail-under=80
```

### Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = ["myapp"]
branch = true

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
]
```

```ini
# .coveragerc (alternative)
[run]
source = myapp
branch = True

[report]
fail_under = 80
show_missing = True
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
```

---

## Branch Coverage

### Enable Branch Coverage

```bash
pytest --cov=myapp --cov-branch
```

### Understanding Branch Coverage

```python
def process(value):
    if value > 0:
        return "positive"
    return "non-positive"

# Line coverage: 100% if both lines executed
# Branch coverage: Requires testing both if branches
```

```
Name            Stmts   Miss Branch BrPart  Cover
-------------------------------------------------
myapp/utils.py     10      0      4      1    92%
```

- **Branch**: Total branches
- **BrPart**: Partially covered branches

---

## Excluding Code

### Inline Exclusion

```python
def debug_info():  # pragma: no cover
    """Only used for debugging."""
    return get_debug_data()

if TYPE_CHECKING:  # pragma: no cover
    from typing import Protocol
```

### Block Exclusion

```python
if sys.platform == "win32":  # pragma: no cover
    def windows_specific():
        pass
else:
    def unix_specific():
        pass
```

### Configuration Exclusions

```toml
# pyproject.toml
[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "def __str__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "class .*\\bProtocol\\):",
    "@abstractmethod",
]

exclude_also = [
    "if self.debug:",
    "if settings.DEBUG",
]

omit = [
    "*/tests/*",
    "*/__init__.py",
    "*/migrations/*",
]
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/coverage.yml
name: Coverage

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -e ".[test]" pytest-cov
      
      - name: Run tests with coverage
        run: |
          pytest --cov=myapp \
                 --cov-report=xml \
                 --cov-fail-under=80
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
          fail_ci_if_error: true
```

### Codecov Configuration

```yaml
# codecov.yml
coverage:
  precision: 2
  round: down
  range: "70...100"
  
  status:
    project:
      default:
        target: 80%
        threshold: 2%
    patch:
      default:
        target: 80%

ignore:
  - "tests/*"
  - "setup.py"
```

---

## pytest-cov Options

### Common Options

| Option | Description |
|--------|-------------|
| `--cov=PATH` | Measure coverage for path |
| `--cov-report=TYPE` | Report type (term, html, xml) |
| `--cov-fail-under=N` | Fail if coverage < N% |
| `--cov-branch` | Measure branch coverage |
| `--cov-append` | Append to existing coverage |
| `--no-cov` | Disable coverage |

### pyproject.toml Configuration

```toml
[tool.pytest.ini_options]
addopts = [
    "--cov=myapp",
    "--cov-report=term-missing",
    "--cov-fail-under=80",
]
```

---

## Coverage Best Practices

### Good Coverage Targets

| Code Type | Target |
|-----------|--------|
| Core business logic | 90%+ |
| Utility functions | 85%+ |
| API endpoints | 80%+ |
| Overall project | 80%+ |

### What Coverage Doesn't Tell You

```python
# 100% coverage but bad test!
def divide(a, b):
    return a / b

def test_divide():
    result = divide(10, 2)
    # Missing: assert result == 5
    # Missing: test for b=0
```

### Focus on Meaningful Tests

```python
# Good: Tests behavior, not just lines
def test_divide_returns_correct_result():
    assert divide(10, 2) == 5.0

def test_divide_by_zero_raises_error():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_divide_handles_floats():
    assert divide(7, 2) == 3.5
```

---

## Hands-on Exercise

### Your Task

Set up coverage for this module:

```python
# myapp/calculator.py
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def calculate(a, b, operation):
    if operation == "add":
        return add(a, b)
    elif operation == "subtract":
        return subtract(a, b)
    elif operation == "divide":
        return divide(a, b)
    else:
        raise ValueError(f"Unknown operation: {operation}")
```

<details>
<summary>✅ Solution</summary>

```toml
# pyproject.toml
[tool.coverage.run]
source = ["myapp"]
branch = true

[tool.coverage.report]
fail_under = 90
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
]

[tool.pytest.ini_options]
addopts = [
    "--cov=myapp",
    "--cov-report=term-missing",
    "--cov-branch",
]
```

```python
# tests/test_calculator.py
import pytest
from myapp.calculator import add, subtract, divide, calculate

class TestAdd:
    def test_positive_numbers(self):
        assert add(2, 3) == 5
    
    def test_negative_numbers(self):
        assert add(-1, -1) == -2

class TestSubtract:
    def test_positive_result(self):
        assert subtract(5, 3) == 2
    
    def test_negative_result(self):
        assert subtract(3, 5) == -2

class TestDivide:
    def test_integer_division(self):
        assert divide(10, 2) == 5.0
    
    def test_float_division(self):
        assert divide(7, 2) == 3.5
    
    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10, 0)

class TestCalculate:
    def test_add_operation(self):
        assert calculate(2, 3, "add") == 5
    
    def test_subtract_operation(self):
        assert calculate(5, 3, "subtract") == 2
    
    def test_divide_operation(self):
        assert calculate(10, 2, "divide") == 5.0
    
    def test_unknown_operation(self):
        with pytest.raises(ValueError, match="Unknown operation"):
            calculate(1, 2, "multiply")
```

```bash
# Run with coverage
pytest

# Output:
# Name                     Stmts   Miss Branch BrPart  Cover   Missing
# --------------------------------------------------------------------
# myapp/calculator.py         16      0      8      0   100%
# --------------------------------------------------------------------
# TOTAL                       16      0      8      0   100%
```
</details>

---

## Summary

✅ **pytest-cov** measures code coverage
✅ **Multiple report formats** for different uses
✅ **`--cov-fail-under`** enforces minimums
✅ **Branch coverage** tests conditional paths
✅ **`# pragma: no cover`** excludes code
✅ **CI integration** catches coverage drops

**Back to:** [Testing in Python Overview](./00-testing-in-python.md)

---

## Further Reading

- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Codecov](https://about.codecov.io/)

<!-- 
Sources Consulted:
- pytest-cov Docs: https://pytest-cov.readthedocs.io/
-->
