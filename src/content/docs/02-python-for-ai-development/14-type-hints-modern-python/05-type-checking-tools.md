---
title: "Type Checking Tools"
---

# Type Checking Tools

## Introduction

Type checkers analyze your code without running it to find type errors. They're essential for catching bugs early and ensuring type safety.

### What We'll Cover

- mypy for static type checking
- pyright and VS Code Pylance
- CI/CD integration
- Gradual typing adoption

### Prerequisites

- Type hints basics
- Common types

---

## mypy

### Installation

```bash
pip install mypy
```

### Basic Usage

```bash
# Check single file
mypy script.py

# Check directory
mypy src/

# Check with more info
mypy src/ --show-error-codes
```

### Example

```python
# example.py
def greet(name: str) -> str:
    return "Hello, " + name

greet(123)  # Type error!
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
plugins = ["pydantic.mypy"]

# Per-module settings
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[[tool.mypy.overrides]]
module = "third_party_lib.*"
ignore_missing_imports = true
```

### Strict Mode

```toml
[tool.mypy]
strict = true
# Equivalent to enabling:
# disallow_untyped_defs = true
# disallow_any_generics = true
# check_untyped_defs = true
# warn_return_any = true
# ... and more
```

---

## pyright

### Installation

```bash
pip install pyright
```

### Usage

```bash
# Check project
pyright

# Check specific files
pyright src/main.py
```

### Configuration

```json
// pyrightconfig.json
{
    "include": ["src"],
    "exclude": ["**/node_modules", "**/__pycache__"],
    "pythonVersion": "3.11",
    "typeCheckingMode": "strict",
    "reportMissingImports": true,
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

---

## VS Code Pylance

### Setup

1. Install Pylance extension
2. Configure settings:

```json
// .vscode/settings.json
{
    "python.analysis.typeCheckingMode": "basic",  // or "strict"
    "python.analysis.diagnosticSeverityOverrides": {
        "reportMissingTypeStubs": "warning",
        "reportUnknownMemberType": "none"
    },
    "python.analysis.autoImportCompletions": true,
    "python.analysis.inlayHints.variableTypes": true,
    "python.analysis.inlayHints.functionReturnTypes": true
}
```

### Type Checking Modes

| Mode | Description |
|------|-------------|
| `off` | No type checking |
| `basic` | Catch common errors |
| `standard` | More thorough checking |
| `strict` | Full type checking |

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/typecheck.yml
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

  pyright:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install pyright
      
      - name: Run pyright
        run: pyright
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

## Type Stubs

### Installing Stubs

```bash
# Search for stubs
pip search types-*  # May need alternate method

# Install common stubs
pip install types-requests
pip install types-PyYAML
pip install types-redis
pip install types-python-dateutil
```

### Bundled Stubs

Many packages include type information:
- pydantic
- FastAPI
- httpx
- numpy (numpy.typing)
- pandas (pandas-stubs)

### Stub Files

```python
# mymodule.pyi (stub file)
def process(data: list[str]) -> dict[str, int]: ...

class Handler:
    def __init__(self, config: dict[str, str]) -> None: ...
    def handle(self, request: bytes) -> bytes: ...
```

---

## Gradual Typing

### Strategy 1: Start Strict on New Code

```toml
[tool.mypy]
strict = true

[[tool.mypy.overrides]]
module = "legacy.*"
ignore_errors = true
```

### Strategy 2: Type Public APIs First

```python
# Public API - fully typed
def process_data(
    items: list[dict[str, str]],
    options: Options
) -> ProcessResult:
    return _internal_process(items, options)

# Internal - can add types later
def _internal_process(items, options):
    ...
```

### Strategy 3: Use reveal_type

```python
from typing import reveal_type

def mystery(x):
    result = some_operation(x)
    reveal_type(result)  # mypy shows inferred type
    return result
```

### Strategy 4: Incremental Strictness

```toml
# Start relaxed
[tool.mypy]
python_version = "3.11"
warn_return_any = true
ignore_missing_imports = true

# Gradually enable more checks
# disallow_untyped_defs = true  # Enable later
# strict = true                  # Final goal
```

---

## Common Patterns

### Ignoring Specific Errors

```python
# Ignore specific line
result = untyped_function()  # type: ignore

# With error code
result = untyped_function()  # type: ignore[no-untyped-call]

# Ignore missing imports for module
import some_untyped_lib  # type: ignore[import]
```

### Type Comments (Legacy)

```python
# For Python 2 compatibility or special cases
def process(x, y):
    # type: (int, str) -> bool
    return True
```

### Assertions for Type Narrowing

```python
from typing import Optional

def process(value: Optional[str]) -> str:
    assert value is not None  # Type narrower
    return value.upper()  # No type error now
```

---

## Comparison

| Feature | mypy | pyright |
|---------|------|---------|
| Speed | Slower | Faster |
| Strictness | Configurable | Stricter default |
| IDE Integration | Various | VS Code (Pylance) |
| Plugin System | Yes | Limited |
| Watch Mode | Yes | Yes |
| Error Messages | Detailed | Very detailed |

### When to Use Each

```
mypy:
├── Established projects
├── Need plugin support
├── CI/CD pipelines
└── Compatibility with various IDEs

pyright:
├── VS Code users
├── Fast local checking
├── Stricter analysis wanted
└── Better completion/hover
```

---

## Hands-on Exercise

### Your Task

```bash
# Set up type checking for a project:
# 1. Create a Python module with mixed typed/untyped code
# 2. Configure mypy for gradual typing
# 3. Run type checker and fix errors
# 4. Add to pre-commit
```

<details>
<summary>✅ Solution</summary>

```bash
# 1. Create project
mkdir typecheck-demo && cd typecheck-demo

# Create module with issues
cat > app.py << 'EOF'
from typing import Optional

def greet(name: str) -> str:
    return "Hello, " + name

def process(data):  # Missing types
    return [x * 2 for x in data]

def find_user(user_id: int) -> Optional[dict]:
    users = {1: {"name": "Alice"}}
    return users.get(user_id)

# Type errors:
greet(123)  # Wrong type
result = find_user(1)
print(result.upper())  # result might be None!
EOF

# 2. Configure mypy
cat > pyproject.toml << 'EOF'
[tool.mypy]
python_version = "3.11"
warn_return_any = true
show_error_codes = true

# Start relaxed
disallow_untyped_defs = false

# Ignore third-party
ignore_missing_imports = true
EOF

# 3. Run mypy
pip install mypy
mypy app.py

# Fix errors
cat > app_fixed.py << 'EOF'
from typing import Optional

def greet(name: str) -> str:
    return "Hello, " + name

def process(data: list[int]) -> list[int]:
    return [x * 2 for x in data]

def find_user(user_id: int) -> Optional[dict[str, str]]:
    users = {1: {"name": "Alice"}}
    return users.get(user_id)

# Fixed:
greet("Alice")  # Correct type

result = find_user(1)
if result is not None:  # Handle None case
    print(result["name"].upper())
EOF

mypy app_fixed.py  # Should pass

# 4. Pre-commit config
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--config-file=pyproject.toml]
EOF

pip install pre-commit
pre-commit install
echo "Type checking setup complete!"
```
</details>

---

## Summary

✅ **mypy** is the standard Python type checker
✅ **pyright/Pylance** provides fast VS Code integration
✅ **Type stubs** add types to untyped libraries
✅ **Gradual typing** lets you add types incrementally
✅ **CI integration** ensures type safety in PRs
✅ **`# type: ignore`** suppresses specific errors

**Next:** [Modern Python Features](./06-modern-python-features.md)

---

## Further Reading

- [mypy Documentation](https://mypy.readthedocs.io/)
- [pyright Documentation](https://github.com/microsoft/pyright)
- [Type Stubs](https://github.com/python/typeshed)

<!-- 
Sources Consulted:
- mypy Docs: https://mypy.readthedocs.io/
-->
