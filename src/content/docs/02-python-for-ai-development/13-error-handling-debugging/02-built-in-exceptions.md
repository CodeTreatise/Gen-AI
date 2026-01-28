---
title: "Built-in Exceptions"
---

# Built-in Exceptions

## Introduction

Python has a rich hierarchy of built-in exceptions. Understanding them helps you handle errors appropriately and create meaningful custom exceptions.

### What We'll Cover

- Common exception types
- Exception hierarchy
- Custom exceptions
- When to use each type

### Prerequisites

- Basic exception handling

---

## Exception Hierarchy

```
BaseException
├── SystemExit
├── KeyboardInterrupt
├── GeneratorExit
└── Exception
    ├── StopIteration
    ├── ArithmeticError
    │   ├── ZeroDivisionError
    │   ├── FloatingPointError
    │   └── OverflowError
    ├── LookupError
    │   ├── KeyError
    │   └── IndexError
    ├── OSError
    │   ├── FileNotFoundError
    │   ├── PermissionError
    │   └── ConnectionError
    ├── ValueError
    ├── TypeError
    ├── AttributeError
    └── ...more
```

> **Note:** Catch `Exception`, not `BaseException`, to avoid catching system exits.

---

## Common Exceptions

### ValueError

Invalid value for the operation (correct type, wrong value):

```python
int("not a number")   # ValueError
int("42")             # OK

list([1, 2]).index(5) # ValueError (5 not in list)
```

### TypeError

Wrong type for the operation:

```python
"hello" + 5           # TypeError: can't add str and int
len(42)               # TypeError: int has no len()
sorted(42)            # TypeError: 'int' object is not iterable
```

### KeyError

Dictionary key not found:

```python
data = {"name": "Alice"}
data["age"]           # KeyError: 'age'

# Use .get() for safe access
data.get("age", None) # Returns None, no error
```

### IndexError

Sequence index out of range:

```python
items = [1, 2, 3]
items[5]              # IndexError: list index out of range
items[-10]            # IndexError

# Safe alternative
items[5] if len(items) > 5 else None
```

### AttributeError

Object doesn't have the attribute:

```python
"hello".non_existent  # AttributeError
None.split()          # AttributeError: 'NoneType' has no attribute 'split'
```

### FileNotFoundError

File or directory doesn't exist:

```python
open("nonexistent.txt")  # FileNotFoundError
```

### ImportError / ModuleNotFoundError

Can't import module:

```python
from nonexistent import something  # ModuleNotFoundError
from os import nonexistent_func    # ImportError
```

---

## Specialized Exceptions

### ConnectionError Family

```python
# Base class for network errors
ConnectionError
├── ConnectionRefusedError  # Server refused connection
├── ConnectionResetError    # Connection reset by peer
├── ConnectionAbortedError  # Connection aborted
└── BrokenPipeError        # Broken pipe
```

```python
import requests

try:
    response = requests.get("http://localhost:9999")
except ConnectionRefusedError:
    print("Server not running")
except ConnectionError as e:
    print(f"Network error: {e}")
```

### OSError Family

```python
OSError
├── FileNotFoundError     # File doesn't exist
├── FileExistsError       # File already exists
├── PermissionError       # Permission denied
├── IsADirectoryError     # Expected file, got directory
└── NotADirectoryError    # Expected directory, got file
```

```python
try:
    with open("/etc/passwd", "w") as f:
        f.write("hacked")
except PermissionError:
    print("Cannot write to system file")
except FileNotFoundError:
    print("File not found")
```

---

## Custom Exceptions

### Basic Custom Exception

```python
class ValidationError(Exception):
    """Raised when validation fails."""
    pass

def validate_age(age: int) -> None:
    if age < 0:
        raise ValidationError("Age cannot be negative")
    if age > 150:
        raise ValidationError("Age seems unrealistic")

try:
    validate_age(-5)
except ValidationError as e:
    print(f"Invalid: {e}")  # Invalid: Age cannot be negative
```

### Exception with Extra Data

```python
class APIError(Exception):
    """API request failed."""
    
    def __init__(self, message: str, status_code: int, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response or {}
    
    def __str__(self):
        return f"{self.args[0]} (HTTP {self.status_code})"

try:
    raise APIError("Not Found", 404, {"detail": "User not found"})
except APIError as e:
    print(f"Error: {e}")              # Error: Not Found (HTTP 404)
    print(f"Status: {e.status_code}") # Status: 404
    print(f"Detail: {e.response}")    # Detail: {'detail': 'User not found'}
```

### Exception Hierarchy

```python
# Create a hierarchy for your application
class AppError(Exception):
    """Base exception for the application."""
    pass

class ConfigError(AppError):
    """Configuration-related errors."""
    pass

class DataError(AppError):
    """Data processing errors."""
    pass

class ValidationError(DataError):
    """Data validation errors."""
    pass

class NetworkError(AppError):
    """Network-related errors."""
    pass

# Now you can catch at different levels:
try:
    process()
except ValidationError:
    # Specific handling
    pass
except DataError:
    # Broader handling
    pass
except AppError:
    # Catch all app errors
    pass
```

---

## Choosing the Right Exception

### Decision Guide

| Situation | Exception to Raise |
|-----------|-------------------|
| Wrong argument type | `TypeError` |
| Wrong argument value | `ValueError` |
| Missing required key | `KeyError` |
| Invalid index/slice | `IndexError` |
| File doesn't exist | `FileNotFoundError` |
| Network failure | `ConnectionError` |
| Permission denied | `PermissionError` |
| Object has no attribute | `AttributeError` |
| Custom business logic | Custom exception |

### When to Create Custom Exceptions

```python
# ✅ Good: Domain-specific error
class InsufficientFundsError(Exception):
    """User doesn't have enough balance."""
    pass

# ❌ Bad: Could use ValueError
class NumberTooSmallError(Exception):
    pass

# Use ValueError instead:
raise ValueError("Number must be positive")
```

---

## Exception Inspection

### Getting Exception Info

```python
import sys

try:
    1 / 0
except ZeroDivisionError:
    exc_type, exc_value, exc_tb = sys.exc_info()
    print(f"Type: {exc_type}")     # <class 'ZeroDivisionError'>
    print(f"Value: {exc_value}")   # division by zero
    print(f"Traceback: {exc_tb}")  # <traceback object>
```

### Exception Attributes

```python
try:
    raise ValueError("Something went wrong")
except ValueError as e:
    print(f"args: {e.args}")           # ('Something went wrong',)
    print(f"str: {str(e)}")            # Something went wrong
    print(f"type: {type(e).__name__}") # ValueError
```

---

## Hands-on Exercise

### Your Task

```python
# Create a user validation system:
# 1. Create custom exceptions: UserError, ValidationError, NotFoundError
# 2. Create a User class with validation
# 3. Handle different error cases
```

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass
import re

# 1. Custom exceptions
class UserError(Exception):
    """Base exception for user-related errors."""
    pass

class ValidationError(UserError):
    """User data validation failed."""
    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field

class NotFoundError(UserError):
    """User not found."""
    def __init__(self, user_id: int):
        super().__init__(f"User {user_id} not found")
        self.user_id = user_id

# 2. User class with validation
@dataclass
class User:
    id: int
    name: str
    email: str
    age: int
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        if not self.name or len(self.name) < 2:
            raise ValidationError("name", "must be at least 2 characters")
        
        if not re.match(r"[^@]+@[^@]+\.[^@]+", self.email):
            raise ValidationError("email", "invalid email format")
        
        if not 0 < self.age < 150:
            raise ValidationError("age", "must be between 1 and 149")

# User database simulation
users_db: dict[int, User] = {}

def get_user(user_id: int) -> User:
    if user_id not in users_db:
        raise NotFoundError(user_id)
    return users_db[user_id]

def create_user(name: str, email: str, age: int) -> User:
    user_id = len(users_db) + 1
    user = User(id=user_id, name=name, email=email, age=age)
    users_db[user_id] = user
    return user

# 3. Handle different error cases
def main():
    # Test validation errors
    print("=== Testing Validation ===")
    
    try:
        create_user("A", "test@example.com", 25)
    except ValidationError as e:
        print(f"Validation failed - {e.field}: {e}")
    
    try:
        create_user("Alice", "invalid-email", 25)
    except ValidationError as e:
        print(f"Validation failed - {e.field}: {e}")
    
    try:
        create_user("Alice", "alice@example.com", 200)
    except ValidationError as e:
        print(f"Validation failed - {e.field}: {e}")
    
    # Test successful creation
    print("\n=== Testing Success ===")
    try:
        user = create_user("Alice", "alice@example.com", 25)
        print(f"Created: {user}")
    except UserError as e:
        print(f"Failed: {e}")
    
    # Test not found
    print("\n=== Testing Not Found ===")
    try:
        get_user(999)
    except NotFoundError as e:
        print(f"Error: {e}, ID was: {e.user_id}")
    
    # Catch all user errors
    print("\n=== Catching All User Errors ===")
    try:
        create_user("", "bad", -1)
    except UserError as e:
        print(f"Some user error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
```

**Output:**
```
=== Testing Validation ===
Validation failed - name: name: must be at least 2 characters
Validation failed - email: email: invalid email format
Validation failed - age: age: must be between 1 and 149

=== Testing Success ===
Created: User(id=1, name='Alice', email='alice@example.com', age=25)

=== Testing Not Found ===
Error: User 999 not found, ID was: 999

=== Catching All User Errors ===
Some user error: ValidationError: name: must be at least 2 characters
```
</details>

---

## Summary

✅ **ValueError** for wrong values, **TypeError** for wrong types
✅ **KeyError** and **IndexError** for lookup failures
✅ **FileNotFoundError** for missing files
✅ **ConnectionError** family for network issues
✅ **Custom exceptions** for domain-specific errors
✅ **Exception hierarchies** enable flexible catching

**Next:** [Logging](./03-logging.md)

---

## Further Reading

- [Built-in Exceptions](https://docs.python.org/3/library/exceptions.html)
- [Exception Hierarchy](https://docs.python.org/3/library/exceptions.html#exception-hierarchy)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/exceptions.html
-->
