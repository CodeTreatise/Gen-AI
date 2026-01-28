---
title: "Exception Handling"
---

# Exception Handling

## Introduction

Exception handling lets your code respond gracefully to errors instead of crashing. Python's try/except blocks provide structured error handling.

### What We'll Cover

- try/except blocks
- Multiple exception types
- else and finally clauses
- Exception chaining
- Re-raising exceptions

### Prerequisites

- Python basics

---

## Basic try/except

### Simple Exception Handling

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
    result = 0

print(f"Result: {result}")  # Result: 0
```

### Catching the Exception Object

```python
try:
    value = int("not a number")
except ValueError as e:
    print(f"Error: {e}")
    # Error: invalid literal for int() with base 10: 'not a number'
```

---

## Multiple Exceptions

### Multiple except Blocks

```python
def get_item(data, key):
    try:
        return data[key]
    except KeyError:
        return f"Key '{key}' not found"
    except TypeError:
        return "Invalid data type"
    except Exception as e:
        return f"Unexpected error: {e}"

print(get_item({"a": 1}, "b"))   # Key 'b' not found
print(get_item(None, "a"))       # Invalid data type
print(get_item([1, 2], "a"))     # Unexpected error: list indices must be integers
```

### Catching Multiple Types Together

```python
try:
    result = risky_operation()
except (ValueError, TypeError) as e:
    print(f"Value or type error: {e}")
except (ConnectionError, TimeoutError) as e:
    print(f"Network error: {e}")
```

---

## else and finally

### else Clause

The `else` block runs only if no exception occurred:

```python
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return None
    else:
        # Only runs if no exception
        print(f"Division successful: {result}")
        return result
    finally:
        # Always runs
        print("Operation complete")

divide(10, 2)
# Division successful: 5.0
# Operation complete

divide(10, 0)
# Cannot divide by zero
# Operation complete
```

### finally for Cleanup

```python
def read_file(path):
    file = None
    try:
        file = open(path, 'r')
        return file.read()
    except FileNotFoundError:
        return None
    finally:
        # Cleanup happens even if exception occurs
        if file:
            file.close()
```

### Context Managers Are Better

```python
# ✅ Preferred: Context manager handles cleanup
def read_file(path):
    try:
        with open(path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return None
```

---

## Exception Chaining

### Automatic Chaining (During Handling)

```python
def process():
    try:
        data = {"key": "value"}
        value = data["missing"]
    except KeyError:
        # This creates a chain: ValueError -> KeyError
        raise ValueError("Missing required key")

try:
    process()
except ValueError as e:
    print(f"Error: {e}")
    print(f"Cause: {e.__cause__}")  # None (implicit chain)
    print(f"Context: {e.__context__}")  # KeyError
```

### Explicit Chaining with `from`

```python
def validate_config(config):
    try:
        api_key = config["api_key"]
    except KeyError as e:
        raise ValueError("Missing API key in configuration") from e

try:
    validate_config({})
except ValueError as e:
    print(f"Error: {e}")
    print(f"Caused by: {e.__cause__}")  # KeyError('api_key')
```

### Suppressing Chain with `from None`

```python
def get_user(user_id):
    try:
        return users[user_id]
    except KeyError:
        # Don't show internal KeyError, just raise clean error
        raise ValueError(f"User {user_id} not found") from None

try:
    get_user(999)
except ValueError as e:
    print(f"Error: {e}")
    print(f"Cause: {e.__cause__}")  # None - chain suppressed
```

---

## Re-raising Exceptions

### Bare raise

```python
def process_data(data):
    try:
        result = transform(data)
    except ValueError:
        print("Logging: ValueError occurred")
        raise  # Re-raises the same exception

try:
    process_data(None)
except ValueError as e:
    print(f"Caught: {e}")
```

### Log and Re-raise Pattern

```python
import logging

logger = logging.getLogger(__name__)

def api_call(url):
    try:
        response = make_request(url)
        return response.json()
    except Exception as e:
        logger.exception(f"API call failed: {url}")
        raise  # Preserves full traceback
```

---

## Best Practices

### 1. Be Specific

```python
# ❌ Too broad
try:
    do_something()
except Exception:
    pass

# ✅ Specific
try:
    do_something()
except ValueError:
    handle_value_error()
except TypeError:
    handle_type_error()
```

### 2. Don't Silence Errors

```python
# ❌ Silent failure
try:
    important_operation()
except Exception:
    pass  # Error disappears!

# ✅ At minimum, log it
try:
    important_operation()
except Exception:
    logger.exception("Operation failed")
    raise
```

### 3. Keep try Blocks Small

```python
# ❌ Too much in try block
try:
    data = load_data()
    processed = process(data)
    result = transform(processed)
    save(result)
except Exception:
    print("Something failed")  # Which operation?

# ✅ Focused try blocks
data = load_data()
try:
    processed = process(data)
except ValueError as e:
    print(f"Processing failed: {e}")
    raise
result = transform(processed)
save(result)
```

### 4. Use Context Managers for Resources

```python
# ✅ Automatic cleanup
with open("file.txt") as f:
    data = f.read()

# ✅ Database connections
with db.connect() as conn:
    conn.execute(query)
```

---

## Hands-on Exercise

### Your Task

```python
# Create a function that:
# 1. Reads a JSON file
# 2. Extracts a required field
# 3. Returns None on any error, logging the issue
# 4. Uses proper exception chaining
```

<details>
<summary>✅ Solution</summary>

```python
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_config_field(filepath: str, field: str) -> str | None:
    """Read a field from a JSON config file.
    
    Args:
        filepath: Path to JSON file.
        field: Field name to extract.
    
    Returns:
        Field value or None if any error occurs.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return None
    
    try:
        value = data[field]
    except KeyError:
        logger.warning(f"Field '{field}' not found in config")
        return None
    except TypeError:
        logger.error("Config is not a dictionary")
        return None
    else:
        logger.info(f"Successfully read {field}")
        return value

# Test
import tempfile
import os

# Create test file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump({"api_key": "secret123"}, f)
    temp_path = f.name

try:
    # Test successful read
    result = read_config_field(temp_path, "api_key")
    print(f"Got: {result}")  # Got: secret123
    
    # Test missing field
    result = read_config_field(temp_path, "missing")
    print(f"Got: {result}")  # Got: None
    
    # Test missing file
    result = read_config_field("nonexistent.json", "key")
    print(f"Got: {result}")  # Got: None
finally:
    os.unlink(temp_path)
```
</details>

---

## Summary

✅ **try/except** catches and handles specific exceptions
✅ **else** runs only when no exception occurred
✅ **finally** always runs (for cleanup)
✅ **`from e`** explicitly chains exceptions
✅ **`from None`** suppresses the chain
✅ **Bare `raise`** re-raises the current exception

**Next:** [Built-in Exceptions](./02-built-in-exceptions.md)

---

## Further Reading

- [Python Exceptions Tutorial](https://docs.python.org/3/tutorial/errors.html)
- [Built-in Exceptions](https://docs.python.org/3/library/exceptions.html)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/tutorial/errors.html
-->
