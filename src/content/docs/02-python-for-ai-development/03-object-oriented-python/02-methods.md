---
title: "Methods"
---

# Methods

## Introduction

Python classes support different types of methods: instance methods, class methods, and static methods. Understanding when to use each type is key to writing clean, maintainable code.

### What We'll Cover

- Instance methods
- Class methods (@classmethod)
- Static methods (@staticmethod)
- Properties (@property)
- Getters and setters

### Prerequisites

- Classes and objects

---

## Instance Methods

### Basic Instance Methods

```python
class Dog:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    # Instance method - operates on instance data
    def bark(self) -> str:
        return f"{self.name} says woof!"
    
    def birthday(self) -> None:
        self.age += 1
        print(f"{self.name} is now {self.age}!")

dog = Dog("Buddy", 5)
print(dog.bark())      # "Buddy says woof!"
dog.birthday()         # "Buddy is now 6!"
```

### Method Chaining

```python
class QueryBuilder:
    def __init__(self):
        self.query = ""
    
    def select(self, *columns) -> "QueryBuilder":
        self.query = f"SELECT {', '.join(columns)}"
        return self  # Return self for chaining
    
    def from_table(self, table: str) -> "QueryBuilder":
        self.query += f" FROM {table}"
        return self
    
    def where(self, condition: str) -> "QueryBuilder":
        self.query += f" WHERE {condition}"
        return self
    
    def build(self) -> str:
        return self.query

# Method chaining
query = (QueryBuilder()
    .select("id", "name", "email")
    .from_table("users")
    .where("active = true")
    .build())

print(query)
# "SELECT id, name, email FROM users WHERE active = true"
```

---

## Class Methods

### Basic Class Methods

```python
class Date:
    def __init__(self, year: int, month: int, day: int):
        self.year = year
        self.month = month
        self.day = day
    
    # Class method - receives class as first argument
    @classmethod
    def from_string(cls, date_string: str) -> "Date":
        """Create Date from 'YYYY-MM-DD' string."""
        year, month, day = map(int, date_string.split("-"))
        return cls(year, month, day)  # cls is the class itself
    
    @classmethod
    def today(cls) -> "Date":
        """Create Date for today."""
        import datetime
        t = datetime.date.today()
        return cls(t.year, t.month, t.day)
    
    def __repr__(self):
        return f"Date({self.year}, {self.month}, {self.day})"

# Using class methods as alternative constructors
d1 = Date(2024, 1, 15)
d2 = Date.from_string("2024-06-20")
d3 = Date.today()

print(d1)  # Date(2024, 1, 15)
print(d2)  # Date(2024, 6, 20)
```

### Why Use cls?

```python
class Animal:
    @classmethod
    def create(cls, name: str) -> "Animal":
        return cls(name)  # Works with subclasses!

class Dog(Animal):
    def __init__(self, name: str):
        self.name = name

# Creates Dog, not Animal
dog = Dog.create("Buddy")
print(type(dog))  # <class '__main__.Dog'>
```

### Class Method Use Cases

```python
class User:
    _registry: list["User"] = []
    
    def __init__(self, name: str):
        self.name = name
        User._registry.append(self)
    
    @classmethod
    def get_all(cls) -> list["User"]:
        """Get all registered users."""
        return cls._registry.copy()
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear user registry."""
        cls._registry.clear()
    
    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Create user from dictionary."""
        return cls(data["name"])
```

---

## Static Methods

### Basic Static Methods

```python
class MathUtils:
    @staticmethod
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    @staticmethod
    def is_even(n: int) -> bool:
        """Check if number is even."""
        return n % 2 == 0

# Called on class (no instance needed)
print(MathUtils.add(5, 3))      # 8
print(MathUtils.is_even(4))     # True

# Can also call on instance (less common)
m = MathUtils()
print(m.is_even(5))             # False
```

### When to Use Static Methods

```python
class Validator:
    def __init__(self, data: dict):
        self.data = data
        self.errors = []
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Check if email format is valid."""
        import re
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))
    
    @staticmethod
    def is_valid_phone(phone: str) -> bool:
        """Check if phone format is valid."""
        import re
        pattern = r"^\d{3}-\d{3}-\d{4}$"
        return bool(re.match(pattern, phone))
    
    def validate(self) -> bool:
        """Validate all data."""
        if not self.is_valid_email(self.data.get("email", "")):
            self.errors.append("Invalid email")
        if not self.is_valid_phone(self.data.get("phone", "")):
            self.errors.append("Invalid phone")
        return len(self.errors) == 0

# Static methods are utility functions related to the class
print(Validator.is_valid_email("test@example.com"))  # True
```

---

## Comparison: Instance vs Class vs Static

| Feature | Instance Method | Class Method | Static Method |
|---------|----------------|--------------|---------------|
| First parameter | `self` (instance) | `cls` (class) | None |
| Access instance data | ✅ | ❌ | ❌ |
| Access class data | ✅ | ✅ | ❌ |
| Can modify instance | ✅ | ❌ | ❌ |
| Can modify class | ✅ | ✅ | ❌ |
| Works with inheritance | ✅ | ✅ | Limited |
| Requires instance | ✅ | ❌ | ❌ |

```python
class Demo:
    class_attr = "class"
    
    def __init__(self, value):
        self.instance_attr = value
    
    def instance_method(self):
        # Can access both
        return f"{self.instance_attr} - {Demo.class_attr}"
    
    @classmethod
    def class_method(cls):
        # Can only access class
        return cls.class_attr
    
    @staticmethod
    def static_method():
        # Cannot access class or instance
        return "static"
```

---

## Properties

### Basic Property

```python
class Circle:
    def __init__(self, radius: float):
        self._radius = radius
    
    @property
    def radius(self) -> float:
        """Get radius."""
        return self._radius
    
    @property
    def diameter(self) -> float:
        """Calculate diameter."""
        return self._radius * 2
    
    @property
    def area(self) -> float:
        """Calculate area."""
        import math
        return math.pi * self._radius ** 2

c = Circle(5)
print(c.radius)    # 5 (looks like attribute, but calls method)
print(c.diameter)  # 10
print(c.area)      # 78.54...
```

### Property with Setter

```python
class Temperature:
    def __init__(self, celsius: float = 0):
        self._celsius = celsius
    
    @property
    def celsius(self) -> float:
        return self._celsius
    
    @celsius.setter
    def celsius(self, value: float) -> None:
        if value < -273.15:
            raise ValueError("Below absolute zero!")
        self._celsius = value
    
    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        self.celsius = (value - 32) * 5/9

temp = Temperature(25)
print(temp.celsius)     # 25
print(temp.fahrenheit)  # 77.0

temp.fahrenheit = 100
print(temp.celsius)     # 37.78...

# Validation works
# temp.celsius = -300  # ValueError!
```

### Property with Deleter

```python
class User:
    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        if not value:
            raise ValueError("Name cannot be empty")
        self._name = value
    
    @name.deleter
    def name(self) -> None:
        print("Deleting name...")
        self._name = "Anonymous"

user = User("Alice")
del user.name  # Calls deleter
print(user.name)  # "Anonymous"
```

---

## Read-Only Properties

```python
class Person:
    def __init__(self, first: str, last: str, birth_year: int):
        self._first = first
        self._last = last
        self._birth_year = birth_year
    
    @property
    def full_name(self) -> str:
        """Read-only: computed from first and last."""
        return f"{self._first} {self._last}"
    
    @property
    def age(self) -> int:
        """Read-only: computed from birth year."""
        from datetime import date
        return date.today().year - self._birth_year

person = Person("John", "Doe", 1990)
print(person.full_name)  # "John Doe"
print(person.age)        # 34 (varies by year)

# Cannot set read-only property
# person.full_name = "Jane Doe"  # AttributeError
```

---

## Cached Properties

```python
from functools import cached_property
import time

class DataProcessor:
    def __init__(self, data: list):
        self._data = data
    
    @cached_property
    def processed(self) -> list:
        """Expensive computation, cached after first access."""
        print("Processing data...")
        time.sleep(1)  # Simulate expensive operation
        return [x * 2 for x in self._data]

dp = DataProcessor([1, 2, 3, 4, 5])
print(dp.processed)  # Processing data... [2, 4, 6, 8, 10]
print(dp.processed)  # [2, 4, 6, 8, 10] (no recomputation)
```

---

## Hands-on Exercise

### Your Task

Create a `BankAccount` class with:

```python
# Requirements:
# 1. Instance attributes: owner, _balance (private)
# 2. Property: balance (read-only)
# 3. Methods: deposit(), withdraw() with validation
# 4. Class method: from_dict() to create from dictionary
# 5. Static method: validate_amount() to check positive amount

# Example usage:
account = BankAccount("Alice", 100)
account.deposit(50)
print(account.balance)  # 150
account.withdraw(30)    # 120

account2 = BankAccount.from_dict({"owner": "Bob", "balance": 200})
```

<details>
<summary>✅ Solution</summary>

```python
class BankAccount:
    """A simple bank account with validation."""
    
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self._balance = balance
    
    @property
    def balance(self) -> float:
        """Read-only balance."""
        return self._balance
    
    @staticmethod
    def validate_amount(amount: float) -> None:
        """Validate that amount is positive."""
        if amount <= 0:
            raise ValueError("Amount must be positive")
    
    def deposit(self, amount: float) -> float:
        """Deposit money into account."""
        self.validate_amount(amount)
        self._balance += amount
        return self._balance
    
    def withdraw(self, amount: float) -> float:
        """Withdraw money from account."""
        self.validate_amount(amount)
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        return self._balance
    
    @classmethod
    def from_dict(cls, data: dict) -> "BankAccount":
        """Create account from dictionary."""
        return cls(
            owner=data["owner"],
            balance=data.get("balance", 0)
        )
    
    def __repr__(self) -> str:
        return f"BankAccount({self.owner!r}, {self._balance})"

# Test
account = BankAccount("Alice", 100)
account.deposit(50)
print(account.balance)  # 150
account.withdraw(30)
print(account.balance)  # 120

account2 = BankAccount.from_dict({"owner": "Bob", "balance": 200})
print(account2)  # BankAccount('Bob', 200)
```
</details>

---

## Summary

✅ **Instance methods** operate on instance data (`self`)
✅ **Class methods** operate on class data (`cls`)
✅ **Static methods** are utility functions in class namespace
✅ **Properties** make methods look like attributes
✅ Use properties for computed values and validation
✅ `@cached_property` caches expensive computations

**Next:** [Inheritance](./03-inheritance.md)

---

## Further Reading

- [Class Methods](https://docs.python.org/3/library/functions.html#classmethod)
- [Property](https://docs.python.org/3/library/functions.html#property)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/functions.html
-->
