---
title: "Writing Effective Tests"
---

# Writing Effective Tests

## Introduction

Well-written tests are readable, maintainable, and catch real bugs. Following established patterns helps create tests that serve as documentation and prevent regressions.

### What We'll Cover

- Arrange-Act-Assert pattern
- Test isolation
- Testing edge cases
- Parameterized tests
- Test naming conventions

### Prerequisites

- pytest fundamentals

---

## Arrange-Act-Assert Pattern

### The AAA Structure

```python
def test_user_creation():
    # Arrange - Set up test data and conditions
    name = "Alice"
    email = "alice@example.com"
    
    # Act - Perform the action being tested
    user = User(name=name, email=email)
    
    # Assert - Verify the expected outcome
    assert user.name == "Alice"
    assert user.email == "alice@example.com"
```

### Clear Separation

```python
def test_order_total_with_discount():
    # Arrange
    items = [
        {"name": "Widget", "price": 10.00, "quantity": 2},
        {"name": "Gadget", "price": 25.00, "quantity": 1},
    ]
    discount_percent = 10
    
    # Act
    order = Order(items)
    order.apply_discount(discount_percent)
    
    # Assert
    assert order.subtotal == 45.00
    assert order.discount == 4.50
    assert order.total == 40.50
```

---

## Test Isolation

### Each Test is Independent

```python
# ❌ Bad: Tests depend on each other
class TestBadIsolation:
    items = []
    
    def test_add_item(self):
        self.items.append("item1")
        assert len(self.items) == 1  # Might fail!
    
    def test_remove_item(self):
        self.items.remove("item1")  # Depends on previous test
        assert len(self.items) == 0

# ✅ Good: Each test is self-contained
class TestGoodIsolation:
    def test_add_item(self):
        items = []
        items.append("item1")
        assert len(items) == 1
    
    def test_remove_item(self):
        items = ["item1"]
        items.remove("item1")
        assert len(items) == 0
```

### Use Fixtures for Shared Setup

```python
import pytest

@pytest.fixture
def sample_items():
    return ["item1", "item2", "item3"]

def test_add_item(sample_items):
    sample_items.append("item4")
    assert len(sample_items) == 4

def test_remove_item(sample_items):
    sample_items.remove("item1")
    assert len(sample_items) == 2
```

---

## Testing Edge Cases

### Common Edge Cases

```python
def test_empty_input():
    assert process([]) == []
    assert calculate_average([]) is None

def test_single_element():
    assert process([1]) == [2]
    assert calculate_average([5]) == 5.0

def test_boundary_values():
    assert is_valid_age(0) == True   # Minimum
    assert is_valid_age(150) == True  # Maximum
    assert is_valid_age(-1) == False  # Below minimum
    assert is_valid_age(151) == False # Above maximum

def test_special_characters():
    assert sanitize("hello<script>") == "hello"
    assert sanitize("") == ""
    assert sanitize(None) is None

def test_unicode():
    assert process("こんにちは") == "こんにちは"
    assert len_chars("café") == 4
```

### Error Conditions

```python
import pytest

def test_invalid_input_type():
    with pytest.raises(TypeError):
        process("not a list")

def test_missing_required_field():
    with pytest.raises(ValueError, match="name is required"):
        create_user(email="test@example.com")

def test_network_timeout():
    with pytest.raises(TimeoutError):
        fetch_with_timeout(url, timeout=0.001)
```

---

## Parameterized Tests

### Basic Parameterization

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (0, 0),
    (-1, -2),
])
def test_double(input, expected):
    assert double(input) == expected
```

### Multiple Parameters

```python
@pytest.mark.parametrize("a,b,expected", [
    (1, 1, 2),
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

### Named Test Cases

```python
@pytest.mark.parametrize("email,is_valid", [
    pytest.param("user@example.com", True, id="valid_email"),
    pytest.param("user@domain.co.uk", True, id="valid_subdomain"),
    pytest.param("user.name@example.com", True, id="valid_with_dot"),
    pytest.param("invalid", False, id="no_at_symbol"),
    pytest.param("@example.com", False, id="no_local_part"),
    pytest.param("user@", False, id="no_domain"),
])
def test_email_validation(email, is_valid):
    assert validate_email(email) == is_valid
```

### Combining Parameters

```python
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [10, 20])
def test_multiply_combinations(x, y):
    # Tests: (1,10), (1,20), (2,10), (2,20)
    result = x * y
    assert result == x * y
```

---

## Test Naming

### Descriptive Names

```python
# ❌ Bad: Unclear what's being tested
def test_user():
    pass

def test_1():
    pass

def test_process():
    pass

# ✅ Good: Clear and descriptive
def test_user_creation_with_valid_email():
    pass

def test_user_creation_fails_with_invalid_email():
    pass

def test_process_returns_empty_list_when_input_is_empty():
    pass
```

### Pattern: test_[unit]_[scenario]_[expected]

```python
def test_calculate_total_with_discount_returns_reduced_price():
    pass

def test_validate_password_with_short_input_raises_error():
    pass

def test_send_email_with_invalid_address_returns_false():
    pass
```

### Given-When-Then Style

```python
def test_given_empty_cart_when_adding_item_then_cart_has_one_item():
    # Given
    cart = ShoppingCart()
    
    # When
    cart.add_item("Widget")
    
    # Then
    assert cart.item_count == 1
```

---

## Test Organization

### Group Related Tests

```python
class TestUserRegistration:
    def test_valid_registration(self):
        pass
    
    def test_duplicate_email_fails(self):
        pass
    
    def test_weak_password_fails(self):
        pass

class TestUserLogin:
    def test_valid_login(self):
        pass
    
    def test_wrong_password_fails(self):
        pass
    
    def test_locked_account_fails(self):
        pass
```

### One Concept Per Test

```python
# ❌ Bad: Testing multiple things
def test_user():
    user = User("Alice", "alice@example.com")
    assert user.name == "Alice"
    assert user.email == "alice@example.com"
    assert user.is_active == True
    user.deactivate()
    assert user.is_active == False
    user.activate()
    assert user.is_active == True

# ✅ Good: Focused tests
def test_user_creation_sets_name_and_email():
    user = User("Alice", "alice@example.com")
    assert user.name == "Alice"
    assert user.email == "alice@example.com"

def test_new_user_is_active_by_default():
    user = User("Alice", "alice@example.com")
    assert user.is_active == True

def test_deactivate_sets_user_inactive():
    user = User("Alice", "alice@example.com")
    user.deactivate()
    assert user.is_active == False
```

---

## Hands-on Exercise

### Your Task

Write tests for this password validator:

```python
def validate_password(password: str) -> tuple[bool, list[str]]:
    """Validate password and return (is_valid, error_messages)."""
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters")
    if not any(c.isupper() for c in password):
        errors.append("Password must contain uppercase letter")
    if not any(c.islower() for c in password):
        errors.append("Password must contain lowercase letter")
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain a digit")
    
    return len(errors) == 0, errors
```

<details>
<summary>✅ Solution</summary>

```python
import pytest
from password import validate_password

class TestPasswordValidation:
    def test_valid_password_returns_true(self):
        is_valid, errors = validate_password("SecurePass1")
        assert is_valid is True
        assert errors == []
    
    @pytest.mark.parametrize("password,expected_error", [
        ("Short1", "at least 8 characters"),
        ("alllowercase1", "uppercase letter"),
        ("ALLUPPERCASE1", "lowercase letter"),
        ("NoDigitsHere", "contain a digit"),
    ])
    def test_invalid_password_returns_specific_error(self, password, expected_error):
        is_valid, errors = validate_password(password)
        assert is_valid is False
        assert any(expected_error in e for e in errors)
    
    def test_empty_password_returns_all_errors(self):
        is_valid, errors = validate_password("")
        assert is_valid is False
        assert len(errors) == 4
    
    def test_password_with_special_characters_is_valid(self):
        is_valid, errors = validate_password("Secure@Pass1!")
        assert is_valid is True
    
    def test_exactly_eight_characters_is_valid(self):
        is_valid, errors = validate_password("Abcdefg1")
        assert is_valid is True
    
    def test_seven_characters_is_invalid(self):
        is_valid, errors = validate_password("Abcdef1")
        assert is_valid is False
        assert "8 characters" in errors[0]
```
</details>

---

## Summary

✅ **AAA pattern** structures tests clearly
✅ **Isolated tests** don't depend on each other
✅ **Edge cases** test boundaries and special inputs
✅ **Parameterized tests** reduce duplication
✅ **Descriptive names** explain what's tested
✅ **One concept per test** keeps tests focused

**Next:** [Fixtures](./03-fixtures.md)

---

## Further Reading

- [pytest Best Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [Effective Python Testing](https://realpython.com/pytest-python-testing/)

<!-- 
Sources Consulted:
- pytest Docs: https://docs.pytest.org/
-->
