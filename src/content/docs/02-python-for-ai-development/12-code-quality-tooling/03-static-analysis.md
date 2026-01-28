---
title: "Static Analysis"
---

# Static Analysis

## Introduction

Static analysis examines code without executing it to find type errors, bugs, and issues that linting might miss. Type checkers like mypy and pyright catch errors before runtime.

### What We'll Cover

- Type checking with mypy
- pyright for VS Code integration
- Configuration options
- Gradual typing strategies

### Prerequisites

- Type hints basics
- Linting setup

---

## Why Static Analysis?

### Catches Type Errors Early

```python
def calculate_total(prices: list[float]) -> float:
    return sum(prices)

# Runtime error without type checking:
result = calculate_total("not a list")  # Would fail at runtime

# mypy catches this immediately:
# error: Argument 1 has incompatible type "str"; expected "list[float]"
```

### Self-Documenting Code

```python
# Without types: what does this return?
def process(data):
    ...

# With types: clear contract
def process(data: list[dict[str, Any]]) -> ProcessResult:
    ...
```

---

## mypy: The Standard Type Checker

### Installation

```bash
pip install mypy
```

### Basic Usage

```bash
# Check file
mypy myfile.py

# Check directory
mypy src/

# Check with more detail
mypy src/ --show-error-codes
```

### Example Errors

```python
# example.py
def greet(name: str) -> str:
    return "Hello, " + name

greet(123)  # Passing int instead of str
```

```bash
mypy example.py
# example.py:4: error: Argument 1 to "greet" has incompatible type "int"; expected "str"  [arg-type]
```

### Configuration

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
show_error_codes = true

# Per-module configuration
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[[tool.mypy.overrides]]
module = "migrations.*"
ignore_errors = true
```

### Strict Mode Options

```toml
[tool.mypy]
# Individual options that make up "strict"
disallow_untyped_defs = true        # Functions must have types
disallow_any_generics = true        # No bare List, Dict
check_untyped_defs = true           # Check inside untyped functions
disallow_untyped_calls = true       # Can't call untyped functions
warn_return_any = true              # Warn when returning Any
```

---

## pyright: VS Code Integration

### Why pyright?

- Powers Pylance in VS Code
- Faster than mypy
- Stricter by default

### Installation

```bash
pip install pyright
```

### Usage

```bash
# Check project
pyright

# Check specific file
pyright myfile.py
```

### Configuration

```json
// pyrightconfig.json
{
    "include": ["src"],
    "exclude": ["**/node_modules", "**/__pycache__"],
    "pythonVersion": "3.11",
    "typeCheckingMode": "strict",
    "reportMissingTypeStubs": "warning"
}
```

Or in pyproject.toml:

```toml
[tool.pyright]
include = ["src"]
pythonVersion = "3.11"
typeCheckingMode = "strict"
```

### VS Code Settings

```json
// .vscode/settings.json
{
    "python.analysis.typeCheckingMode": "basic",  // or "strict"
    "python.analysis.diagnosticSeverityOverrides": {
        "reportMissingTypeStubs": "warning",
        "reportUnknownMemberType": "none"
    }
}
```

---

## Common Type Checking Patterns

### Handling Optional Values

```python
from typing import Optional

def get_user(user_id: int) -> Optional[User]:
    # May return None
    ...

# mypy error: Item "None" has no attribute "name"
user = get_user(1)
print(user.name)

# Correct: Check for None
user = get_user(1)
if user is not None:
    print(user.name)

# Or use assertion
user = get_user(1)
assert user is not None
print(user.name)
```

### Type Narrowing

```python
from typing import Union

def process(value: Union[str, int]) -> str:
    if isinstance(value, str):
        # mypy knows value is str here
        return value.upper()
    else:
        # mypy knows value is int here
        return str(value)
```

### Handling External Libraries

```bash
# Install type stubs
pip install types-requests
pip install pandas-stubs
```

```toml
# Or ignore missing stubs
[tool.mypy]
ignore_missing_imports = true
```

---

## Gradual Typing Strategies

### Strategy 1: Start Strict on New Code

```toml
# mypy.ini or pyproject.toml
[tool.mypy]
# Strict for new code
strict = true

# Relax for legacy
[[tool.mypy.overrides]]
module = "legacy.*"
ignore_errors = true
```

### Strategy 2: Per-File Strictness

```python
# mypy: strict
# ^ This comment enables strict mode for this file only

def typed_function(x: int) -> str:
    return str(x)
```

### Strategy 3: Incremental Adoption

1. Start with `ignore_missing_imports = true`
2. Add types to public APIs first
3. Enable stricter options over time
4. Use `reveal_type()` for debugging

```python
from typing import reveal_type

def mystery_function(x):
    result = some_operation(x)
    reveal_type(result)  # mypy shows inferred type
    return result
```

---

## Type Stubs

### What Are Stubs?

`.pyi` files that provide type information for untyped libraries.

```python
# mymodule.pyi - stub file
def process(data: list[str]) -> dict[str, int]: ...

class Processor:
    def __init__(self, config: Config) -> None: ...
    def run(self) -> Result: ...
```

### Installing Stubs

```bash
# From typeshed (included with mypy)
pip install types-requests
pip install types-PyYAML
pip install types-redis

# Check available stubs
pip search types-*  # (may require alternative index)
```

### Creating Custom Stubs

```
myproject/
├── src/
│   └── mymodule.py
├── stubs/
│   └── external_lib.pyi  # Stub for untyped library
└── pyproject.toml
```

```toml
[tool.mypy]
mypy_path = "stubs"
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Type Check

on: [push, pull_request]

jobs:
  mypy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install mypy
          pip install -r requirements.txt
      
      - name: Run mypy
        run: mypy src/
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies:
          - types-requests
          - pydantic
```

---

## Hands-on Exercise

### Your Task

```bash
# Set up type checking:
# 1. Create a Python file with type hints
# 2. Add some type errors intentionally
# 3. Configure mypy
# 4. Run mypy and fix the errors
```

<details>
<summary>✅ Solution</summary>

```bash
# 1. Create file with types
cat > typed_example.py << 'EOF'
from typing import Optional

class User:
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

def get_user(user_id: int) -> Optional[User]:
    if user_id > 0:
        return User("Alice", 30)
    return None

def greet_user(user: User) -> str:
    return f"Hello, {user.name}!"

# Intentional errors:
result1: str = 123  # Wrong type
result2 = greet_user(None)  # Passing None
user = get_user(1)
print(user.name)  # user might be None
EOF

# 2. Configure mypy
cat >> pyproject.toml << 'EOF'

[tool.mypy]
python_version = "3.11"
strict = true
show_error_codes = true
EOF

# 3. Run mypy
pip install mypy
mypy typed_example.py

# 4. Fix the file
cat > typed_example_fixed.py << 'EOF'
from typing import Optional

class User:
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

def get_user(user_id: int) -> Optional[User]:
    if user_id > 0:
        return User("Alice", 30)
    return None

def greet_user(user: User) -> str:
    return f"Hello, {user.name}!"

# Fixed:
result1: str = "123"  # Correct type
user = get_user(1)
if user is not None:  # Handle None case
    result2 = greet_user(user)
    print(user.name)
EOF

# Verify fix
mypy typed_example_fixed.py
echo "Success! No type errors."
```
</details>

---

## Summary

✅ **mypy** catches type errors at development time
✅ **pyright** provides fast checking and VS Code integration
✅ **Strict mode** enables all type checks
✅ **Type stubs** add types to untyped libraries
✅ **Gradual typing** allows incremental adoption
✅ **CI integration** ensures type safety in PRs

**Next:** [Pre-commit Hooks](./04-pre-commit-hooks.md)

---

## Further Reading

- [mypy Documentation](https://mypy.readthedocs.io/)
- [pyright Documentation](https://github.com/microsoft/pyright)
- [Typing Best Practices](https://typing.readthedocs.io/)

<!-- 
Sources Consulted:
- mypy Docs: https://mypy.readthedocs.io/
-->
