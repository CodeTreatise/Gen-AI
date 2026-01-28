---
title: "Pydantic"
---

# Pydantic

## Introduction

Pydantic provides runtime data validation using Python type hints. It's the foundation for FastAPI and essential for working with APIs and configuration.

### What We'll Cover

- BaseModel basics
- Field definitions
- Validation and coercion
- Nested models
- Settings management

### Prerequisites

- Type hints
- Classes and OOP

---

## Installation

```bash
pip install pydantic
# Or with email validation
pip install pydantic[email]
```

---

## BaseModel Basics

### Simple Model

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
    age: int

# Create from keyword arguments
user = User(name="Alice", email="alice@example.com", age=30)
print(user.name)   # Alice
print(user.age)    # 30

# Create from dict
data = {"name": "Bob", "email": "bob@example.com", "age": 25}
user = User(**data)
# Or
user = User.model_validate(data)
```

### Automatic Type Coercion

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    quantity: int

# Pydantic converts types when possible
item = Item(name="Widget", price="19.99", quantity="5")
print(item.price)     # 19.99 (float, not str!)
print(item.quantity)  # 5 (int, not str!)
```

### Validation Errors

```python
from pydantic import BaseModel, ValidationError

class User(BaseModel):
    name: str
    age: int

try:
    user = User(name="Alice", age="not a number")
except ValidationError as e:
    print(e)
    # 1 validation error for User
    # age
    #   Input should be a valid integer [type=int_parsing]
```

---

## Field Definitions

### Default Values

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str
    email: str
    age: int = 18  # Simple default
    active: bool = True
    role: str = Field(default="user")
```

### Field with Constraints

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    age: int = Field(ge=0, le=150)  # >= 0 and <= 150
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    score: float = Field(gt=0, lt=100)  # > 0 and < 100
```

### Field Descriptions

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(
        description="The user's full name",
        examples=["Alice Smith"]
    )
    age: int = Field(
        ge=0,
        description="Age in years",
        examples=[25, 30]
    )
```

### Computed Fields

```python
from pydantic import BaseModel, computed_field

class Rectangle(BaseModel):
    width: float
    height: float
    
    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height

rect = Rectangle(width=10, height=5)
print(rect.area)  # 50.0
```

---

## Validation

### Custom Validators

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str
    email: str
    
    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()
    
    @field_validator("email")
    @classmethod
    def email_must_be_lowercase(cls, v: str) -> str:
        return v.lower()

user = User(name="  Alice  ", email="ALICE@EXAMPLE.COM")
print(user.name)   # "Alice" (stripped)
print(user.email)  # "alice@example.com" (lowercase)
```

### Model Validators

```python
from pydantic import BaseModel, model_validator

class DateRange(BaseModel):
    start: str
    end: str
    
    @model_validator(mode="after")
    def check_dates(self) -> "DateRange":
        if self.start > self.end:
            raise ValueError("start must be before end")
        return self

# Valid
range1 = DateRange(start="2024-01-01", end="2024-12-31")

# Invalid - raises ValidationError
# DateRange(start="2024-12-31", end="2024-01-01")
```

---

## Nested Models

### Basic Nesting

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    name: str
    address: Address

user = User(
    name="Alice",
    address={
        "street": "123 Main St",
        "city": "New York",
        "country": "USA"
    }
)

print(user.address.city)  # New York
```

### Lists of Models

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    id: int
    items: list[Item]
    
    @computed_field
    @property
    def total(self) -> float:
        return sum(item.price for item in self.items)

order = Order(
    id=1,
    items=[
        {"name": "Widget", "price": 9.99},
        {"name": "Gadget", "price": 19.99}
    ]
)

print(order.total)  # 29.98
```

---

## Serialization

### To Dict and JSON

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    password: str

user = User(name="Alice", age=30, password="secret")

# To dictionary
user_dict = user.model_dump()
print(user_dict)  # {'name': 'Alice', 'age': 30, 'password': 'secret'}

# Exclude fields
user_dict = user.model_dump(exclude={"password"})
print(user_dict)  # {'name': 'Alice', 'age': 30}

# To JSON string
user_json = user.model_dump_json()
print(user_json)  # '{"name":"Alice","age":30,"password":"secret"}'
```

### From JSON

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

json_str = '{"name": "Alice", "age": 30}'
user = User.model_validate_json(json_str)
print(user)  # name='Alice' age=30
```

---

## Settings Management

### Environment Variables

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "MyApp"
    debug: bool = False
    database_url: str
    api_key: str
    
    class Config:
        env_file = ".env"

# Reads from environment variables
settings = Settings()
# Reads APP_NAME, DEBUG, DATABASE_URL, API_KEY
```

### .env File

```bash
# .env file
DATABASE_URL=postgresql://localhost/mydb
API_KEY=secret123
DEBUG=true
```

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    api_key: str
    debug: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
print(settings.database_url)  # postgresql://localhost/mydb
print(settings.debug)         # True
```

---

## Common Patterns

### API Response Models

```python
from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar("T")

class APIResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: str | None = None

class User(BaseModel):
    id: int
    name: str

# Type-safe API responses
response: APIResponse[User] = APIResponse(
    success=True,
    data=User(id=1, name="Alice")
)
```

### Request/Response Models

```python
from pydantic import BaseModel, Field

class UserCreate(BaseModel):
    """Request model for creating user."""
    name: str = Field(min_length=2)
    email: str
    password: str = Field(min_length=8)

class UserResponse(BaseModel):
    """Response model (no password)."""
    id: int
    name: str
    email: str
```

---

## Hands-on Exercise

### Your Task

```python
# Create Pydantic models for:
# 1. A Product with name, price, and optional description
# 2. A Cart with items (list of products) and computed total
# 3. Validate price is positive
```

<details>
<summary>✅ Solution</summary>

```python
from pydantic import BaseModel, Field, field_validator, computed_field

class Product(BaseModel):
    name: str = Field(min_length=1)
    price: float = Field(gt=0, description="Price must be positive")
    description: str | None = None
    
    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

class CartItem(BaseModel):
    product: Product
    quantity: int = Field(ge=1, default=1)
    
    @computed_field
    @property
    def subtotal(self) -> float:
        return self.product.price * self.quantity

class Cart(BaseModel):
    items: list[CartItem] = []
    
    @computed_field
    @property
    def total(self) -> float:
        return sum(item.subtotal for item in self.items)
    
    def add_item(self, product: Product, quantity: int = 1) -> None:
        self.items.append(CartItem(product=product, quantity=quantity))

# Test
widget = Product(name="Widget", price=9.99, description="A useful widget")
gadget = Product(name="Gadget", price=19.99)

cart = Cart()
cart.add_item(widget, quantity=2)
cart.add_item(gadget, quantity=1)

print(f"Items: {len(cart.items)}")
print(f"Total: ${cart.total:.2f}")

# Serialization
print(cart.model_dump_json(indent=2))
```
</details>

---

## Summary

✅ **BaseModel** provides automatic validation
✅ **Field** adds constraints and metadata
✅ **Validators** customize validation logic
✅ **Nested models** handle complex data
✅ **model_dump/model_dump_json** for serialization
✅ **BaseSettings** reads environment variables

**Next:** [Type Checking Tools](./05-type-checking-tools.md)

---

## Further Reading

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

<!-- 
Sources Consulted:
- Pydantic Docs: https://docs.pydantic.dev/
-->
