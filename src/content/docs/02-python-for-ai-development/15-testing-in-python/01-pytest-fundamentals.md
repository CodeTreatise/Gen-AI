---
title: "pytest Fundamentals"
---

# pytest Fundamentals

## Introduction

pytest makes testing simple with its intuitive syntax and powerful features. Write tests with plain `assert` statements and let pytest handle the rest.

### What We'll Cover

- Installing and running pytest
- Test file and function naming
- Assertions and failure messages
- Test discovery
- Command-line options

### Prerequisites

- Python functions
- Basic command line

---

## Installation

```bash
pip install pytest

# Verify installation
pytest --version
```

---

## Your First Test

### Simple Test File

```python
# tests/test_math.py

def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 5 - 3 == 2

def test_multiplication():
    result = 3 * 4
    assert result == 12
```

### Running Tests

```bash
pytest
```

**Output:**
```
==================== test session starts ====================
collected 3 items

tests/test_math.py ...                                 [100%]

==================== 3 passed in 0.01s ====================
```

---

## Naming Conventions

### Test Files

```
tests/
├── test_math.py        ✅ Starts with test_
├── test_strings.py     ✅ Starts with test_
├── math_test.py        ✅ Ends with _test
├── helpers.py          ❌ Won't be discovered
└── conftest.py         ✅ Special file for fixtures
```

### Test Functions

```python
# test_example.py

def test_something():           # ✅ Discovered
    pass

def test_another_thing():       # ✅ Discovered
    pass

def helper_function():          # ❌ Not discovered
    pass

def validate_data():            # ❌ Not discovered
    pass
```

### Test Classes

```python
class TestMathOperations:       # ✅ Class starts with Test
    def test_add(self):         # ✅ Method starts with test_
        assert 1 + 1 == 2
    
    def test_subtract(self):    # ✅ Method starts with test_
        assert 5 - 3 == 2
    
    def helper(self):           # ❌ Not discovered
        pass
```

---

## Assertions

### Basic Assertions

```python
def test_assertions():
    # Equality
    assert 1 + 1 == 2
    
    # Inequality
    assert 1 + 1 != 3
    
    # Truthiness
    assert True
    assert [1, 2, 3]  # Non-empty list is truthy
    assert not []     # Empty list is falsy
    
    # Comparison
    assert 5 > 3
    assert 2 <= 2
    
    # Membership
    assert 3 in [1, 2, 3]
    assert "hello" in "hello world"
    
    # Identity
    assert None is None
```

### Assertion Messages

```python
def test_with_message():
    value = 42
    assert value > 50, f"Expected value > 50, got {value}"
```

**Output on failure:**
```
AssertionError: Expected value > 50, got 42
```

### pytest's Smart Assertions

```python
def test_list_equality():
    expected = [1, 2, 3, 4]
    actual = [1, 2, 5, 4]
    assert expected == actual
```

**Output:**
```
    def test_list_equality():
        expected = [1, 2, 3, 4]
        actual = [1, 2, 5, 4]
>       assert expected == actual
E       AssertionError: assert [1, 2, 3, 4] == [1, 2, 5, 4]
E         At index 2 diff: 3 != 5
```

---

## Testing Exceptions

### pytest.raises

```python
import pytest

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(10, 0)

def test_divide_by_zero_message():
    with pytest.raises(ValueError) as exc_info:
        divide(10, 0)
    assert "Cannot divide by zero" in str(exc_info.value)

def test_divide_by_zero_match():
    with pytest.raises(ValueError, match="divide by zero"):
        divide(10, 0)
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Very verbose
pytest -vv

# Quiet (minimal output)
pytest -q
```

### Selecting Tests

```bash
# Run specific file
pytest tests/test_math.py

# Run specific test
pytest tests/test_math.py::test_addition

# Run tests matching pattern
pytest -k "add"          # Tests containing "add"
pytest -k "add or sub"   # Tests containing "add" or "sub"
pytest -k "not slow"     # Tests NOT containing "slow"

# Run tests in directory
pytest tests/unit/
```

### Useful Options

```bash
# Stop on first failure
pytest -x

# Stop after N failures
pytest --maxfail=3

# Show local variables in tracebacks
pytest -l

# Run last failed tests
pytest --lf

# Run failed tests first
pytest --ff

# Show slowest N tests
pytest --durations=10
```

---

## Test Discovery

### Default Discovery Rules

pytest finds tests by:

1. **Directories**: Starting from current or specified path
2. **Files**: Named `test_*.py` or `*_test.py`
3. **Classes**: Named `Test*` (no `__init__`)
4. **Functions**: Named `test_*`

### Project Structure

```
my_project/
├── src/
│   └── calculator.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_add.py
│   │   └── test_subtract.py
│   └── integration/
│       └── test_calculator.py
└── pytest.ini
```

### Configuration

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers
```

Or in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v"
```

---

## Test Output

### Understanding Output

```
tests/test_example.py::test_pass PASSED     [ 33%]
tests/test_example.py::test_fail FAILED     [ 66%]
tests/test_example.py::test_skip SKIPPED    [100%]
```

| Symbol | Meaning |
|--------|---------|
| `.` or `PASSED` | Test passed |
| `F` or `FAILED` | Test failed |
| `E` or `ERROR` | Error during test |
| `s` or `SKIPPED` | Test skipped |
| `x` or `XFAIL` | Expected failure |
| `X` or `XPASS` | Unexpected pass |

---

## Hands-on Exercise

### Your Task

Create tests for this function:

```python
# calculator.py
def calculate(a: int, b: int, operation: str) -> int:
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a // b
    else:
        raise ValueError(f"Unknown operation: {operation}")
```

<details>
<summary>✅ Solution</summary>

```python
# tests/test_calculator.py
import pytest
from calculator import calculate

class TestCalculate:
    def test_add(self):
        assert calculate(2, 3, "add") == 5
    
    def test_subtract(self):
        assert calculate(5, 3, "subtract") == 2
    
    def test_multiply(self):
        assert calculate(4, 3, "multiply") == 12
    
    def test_divide(self):
        assert calculate(10, 2, "divide") == 5
    
    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="divide by zero"):
            calculate(10, 0, "divide")
    
    def test_unknown_operation(self):
        with pytest.raises(ValueError, match="Unknown operation"):
            calculate(1, 2, "power")
    
    def test_negative_numbers(self):
        assert calculate(-2, 3, "add") == 1
        assert calculate(-2, -3, "multiply") == 6

# Run: pytest tests/test_calculator.py -v
```
</details>

---

## Summary

✅ **Install pytest** with `pip install pytest`
✅ **Name tests** with `test_` prefix
✅ **Use simple assertions** with `assert`
✅ **Test exceptions** with `pytest.raises`
✅ **Run specific tests** with `-k` pattern matching
✅ **Configure** in `pytest.ini` or `pyproject.toml`

**Next:** [Writing Effective Tests](./02-writing-effective-tests.md)

---

## Further Reading

- [pytest Documentation](https://docs.pytest.org/)
- [pytest Assertions](https://docs.pytest.org/en/stable/how-to/assert.html)

<!-- 
Sources Consulted:
- pytest Docs: https://docs.pytest.org/
-->
