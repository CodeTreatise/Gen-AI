---
title: "Assertions"
---

# Assertions

## Introduction

Assertions are statements that verify assumptions in your code. They're a debugging aid that catches programming errors during development.

### What We'll Cover

- assert statements
- When to use assertions
- Assertions vs exceptions
- Disabling assertions

### Prerequisites

- Exception handling
- Basic debugging

---

## Basic assert Statement

### Syntax

```python
assert condition, "Optional error message"
```

### How It Works

```python
# If condition is True, nothing happens
assert 2 + 2 == 4

# If condition is False, raises AssertionError
assert 2 + 2 == 5
# AssertionError

# With message
assert 2 + 2 == 5, "Math is broken!"
# AssertionError: Math is broken!
```

### Equivalent Code

```python
# assert condition, message
# is equivalent to:
if not condition:
    raise AssertionError(message)
```

---

## When to Use Assertions

### 1. Check Internal Invariants

```python
def calculate_average(numbers):
    assert len(numbers) > 0, "List cannot be empty"
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    
    # Invariant: average should be within range
    assert min(numbers) <= average <= max(numbers), \
        f"Average {average} outside range [{min(numbers)}, {max(numbers)}]"
    
    return average
```

### 2. Document Assumptions

```python
def process_user(user):
    # Document assumption about data structure
    assert "id" in user, "User must have 'id' field"
    assert "email" in user, "User must have 'email' field"
    assert isinstance(user["id"], int), "User id must be integer"
    
    # Process user...
```

### 3. Catch Programmer Errors

```python
def set_mode(mode):
    # Catch programming mistakes
    assert mode in ("read", "write", "append"), \
        f"Invalid mode: {mode}. Use 'read', 'write', or 'append'"
    
    # Set mode...
```

### 4. Verify Post-conditions

```python
def sort_list(items):
    result = sorted(items)
    
    # Verify our assumption about sorted()
    assert len(result) == len(items), "Sorting changed list length!"
    assert all(result[i] <= result[i+1] for i in range(len(result)-1)), \
        "Result is not sorted!"
    
    return result
```

---

## When NOT to Use Assertions

### 1. Don't Validate User Input

```python
# ❌ Wrong: User input should raise proper exception
def process_age(age):
    assert age >= 0, "Age cannot be negative"
    return age

# ✅ Right: Raise ValueError for invalid input
def process_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age
```

### 2. Don't Use for Expected Errors

```python
# ❌ Wrong: File might not exist (expected condition)
def read_config(path):
    assert os.path.exists(path), f"Config not found: {path}"
    return load_config(path)

# ✅ Right: Handle as normal error case
def read_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    return load_config(path)
```

### 3. Don't Put Side Effects in Assertions

```python
# ❌ Wrong: Side effect in assertion (won't run with -O)
assert users.pop() is not None

# ✅ Right: Separate action from check
user = users.pop()
assert user is not None
```

---

## Assertions vs Exceptions

| Feature | Assertions | Exceptions |
|---------|------------|------------|
| **Purpose** | Catch programmer errors | Handle runtime conditions |
| **Can disable** | Yes (-O flag) | No |
| **User input** | Never | Yes |
| **Production** | Can be disabled | Always active |
| **Recovery** | Not expected | Often possible |

### Decision Guide

```python
# Use ASSERTION when:
# - Condition should NEVER be false if code is correct
# - Failure indicates a bug in the program
# - It's a development/debugging aid

assert len(self._items) >= 0, "Length cannot be negative"

# Use EXCEPTION when:
# - Condition might be false in normal operation
# - External factors can cause failure
# - Caller should handle the error

if not user_input.strip():
    raise ValueError("Input cannot be empty")
```

---

## Disabling Assertions

### Using -O Flag

```bash
# Normal execution - assertions run
python script.py

# Optimized execution - assertions skipped
python -O script.py

# More optimization
python -OO script.py
```

### How It Works

```python
# With -O flag, __debug__ is False
if __debug__:
    # This code is removed entirely with -O
    print("Debug mode")

# Assertions are equivalent to:
if __debug__:
    if not condition:
        raise AssertionError(message)
```

### Check Debug Mode

```python
def process(data):
    if __debug__:
        print(f"Processing {len(data)} items")
        validate_data(data)  # Only in debug mode
    
    # Production code
    return transform(data)
```

---

## Best Practices

### 1. Keep Assertions Simple

```python
# ✅ Good: Simple, clear check
assert user is not None, "User required"

# ❌ Bad: Complex logic in assertion
assert validate_user(user) and check_permissions(user) and user.active
```

### 2. Provide Helpful Messages

```python
# ❌ Bad: No message
assert x > 0

# ✅ Good: Descriptive message
assert x > 0, f"x must be positive, got {x}"
```

### 3. Don't Use Tuples

```python
# ❌ Bug: Non-empty tuple is always truthy!
assert (condition, "message")  # Always passes!

# ✅ Correct
assert condition, "message"
```

### 4. Use for Development Checks

```python
class Queue:
    def __init__(self):
        self._items = []
    
    def enqueue(self, item):
        self._items.append(item)
        # Invariant check during development
        assert self._check_invariant(), "Queue invariant violated"
    
    def _check_invariant(self):
        """Check internal consistency."""
        return isinstance(self._items, list)
```

---

## Hands-on Exercise

### Your Task

```python
# Add appropriate assertions to this function:
# 1. Check preconditions
# 2. Check invariants
# 3. Check postconditions

def calculate_discount(price, discount_percent):
    discounted = price * (1 - discount_percent / 100)
    return discounted
```

<details>
<summary>✅ Solution</summary>

```python
def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate discounted price.
    
    Args:
        price: Original price (must be positive)
        discount_percent: Discount percentage (0-100)
    
    Returns:
        Discounted price
    """
    # Preconditions: Check assumptions about inputs
    assert isinstance(price, (int, float)), \
        f"Price must be numeric, got {type(price)}"
    assert isinstance(discount_percent, (int, float)), \
        f"Discount must be numeric, got {type(discount_percent)}"
    assert price >= 0, \
        f"Price must be non-negative, got {price}"
    assert 0 <= discount_percent <= 100, \
        f"Discount must be 0-100, got {discount_percent}"
    
    # Calculate
    discounted = price * (1 - discount_percent / 100)
    
    # Postconditions: Verify results make sense
    assert discounted >= 0, \
        f"Discounted price cannot be negative: {discounted}"
    assert discounted <= price, \
        f"Discounted price {discounted} > original {price}"
    
    # Invariant: discount should reduce or maintain price
    if discount_percent > 0:
        assert discounted < price, \
            "Positive discount should reduce price"
    
    return discounted

# Test valid cases
print(calculate_discount(100, 20))  # 80.0
print(calculate_discount(50, 0))    # 50.0
print(calculate_discount(100, 100)) # 0.0

# These would fail assertions:
# calculate_discount(-100, 20)  # AssertionError: Price must be non-negative
# calculate_discount(100, 150)  # AssertionError: Discount must be 0-100
```
</details>

---

## Summary

✅ **Assertions** catch programmer errors during development
✅ **Don't use** for user input validation
✅ **Can be disabled** with `-O` flag
✅ **Include messages** that explain the problem
✅ **No side effects** in assertion conditions
✅ **Use exceptions** for expected error conditions

**Back to:** [Error Handling & Debugging Overview](./00-error-handling-debugging.md)

---

## Further Reading

- [Python assert statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement)
- [Programming with Assertions](https://wiki.python.org/moin/UsingAssertionsEffectively)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement
-->
