---
title: "Linting"
---

# Linting

## Introduction

Linters analyze code for errors, style violations, and potential bugs without running it. They catch problems early, enforce standards, and improve code quality.

### What We'll Cover

- Ruff for fast linting
- Pylint for comprehensive analysis
- Flake8 for style checking
- Bandit for security linting
- Rule configuration

### Prerequisites

- Basic Python knowledge
- Code formatting setup

---

## Why Linting?

### Catches Bugs Before Runtime

```python
# Linter catches these errors before you run the code:

def process(data):
    resutl = []  # Typo: undefined variable 'resutl'
    for item in data:
        result.append(item)  # Uses undefined 'result'
    return resutl

# F841: local variable 'resutl' assigned but never used
# F821: undefined name 'result'
```

### Enforces Best Practices

```python
# Linter warns about bad patterns:

import os  # F401: 'os' imported but unused

def get_data(items=[]):  # B006: Mutable default argument
    return items

if x == None:  # E711: Comparison to None (use 'is None')
    pass
```

---

## Ruff: The Modern Choice

### Why Ruff?

- 10-100x faster than other linters
- Replaces Flake8, isort, and many plugins
- Easy configuration
- Auto-fix support

### Installation

```bash
pip install ruff
```

### Basic Usage

```bash
# Check for issues
ruff check .

# Auto-fix what's possible
ruff check . --fix

# Check specific file
ruff check myfile.py
```

### Example Output

```
example.py:1:8: F401 [*] `os` imported but unused
example.py:3:16: B006 Do not use mutable data structures for argument defaults
example.py:7:4: E711 [*] Comparison to `None` should be `cond is None`
Found 3 errors.
[*] 2 fixable with the `--fix` option.
```

### Configuration

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"

# Enable rule categories
select = [
    "E",     # pycodestyle errors
    "F",     # pyflakes
    "I",     # isort
    "N",     # pep8-naming
    "W",     # pycodestyle warnings
    "UP",    # pyupgrade
    "B",     # flake8-bugbear
    "C4",    # flake8-comprehensions
    "SIM",   # flake8-simplify
    "ARG",   # flake8-unused-arguments
    "PTH",   # flake8-use-pathlib
]

# Ignore specific rules
ignore = [
    "E501",  # Line too long (formatter handles this)
]

# Per-file ignores
[tool.ruff.per-file-ignores]
"tests/*" = ["ARG"]  # Allow unused args in tests
"__init__.py" = ["F401"]  # Allow unused imports in __init__
```

### Common Rule Categories

| Code | Category | Examples |
|------|----------|----------|
| `E` | pycodestyle errors | Whitespace, indentation |
| `F` | pyflakes | Unused imports, undefined names |
| `I` | isort | Import sorting |
| `N` | pep8-naming | Naming conventions |
| `B` | flake8-bugbear | Common bugs, design problems |
| `UP` | pyupgrade | Upgrade to modern Python |
| `SIM` | flake8-simplify | Simplifiable code |

---

## Pylint: Comprehensive Analysis

### When to Use Pylint

- Legacy projects
- Maximum strictness needed
- Advanced analysis (code complexity)

### Installation

```bash
pip install pylint
```

### Usage

```bash
# Check file
pylint myfile.py

# Check directory
pylint src/

# Generate configuration
pylint --generate-rcfile > .pylintrc
```

### Example Output

```
************* Module example
example.py:1:0: W0611: Unused import os (unused-import)
example.py:3:0: W0102: Dangerous default value [] (dangerous-default-value)
example.py:7:0: C0121: Comparison 'x == None' should be 'x is None' (singleton-comparison)

-----------------------------------
Your code has been rated at 4.00/10
```

### Configuration

```toml
# pyproject.toml
[tool.pylint.main]
py-version = "3.11"
jobs = 0  # Auto-detect CPU cores

[tool.pylint.messages_control]
disable = [
    "C0114",  # missing-module-docstring
    "C0115",  # missing-class-docstring
    "C0116",  # missing-function-docstring
]

[tool.pylint.format]
max-line-length = 88
```

---

## Flake8: Simple Style Checking

### When to Use Flake8

- Simple projects
- Plugin ecosystem needed
- Familiar from existing projects

### Installation

```bash
pip install flake8
```

### Usage

```bash
flake8 src/
flake8 myfile.py
```

### Configuration

```ini
# setup.cfg or .flake8
[flake8]
max-line-length = 88
extend-ignore = E203, E501
exclude = 
    .git,
    __pycache__,
    .venv
```

> **Note:** Ruff replaces Flake8 for most use cases with better speed and configuration.

---

## Bandit: Security Linting

### Purpose

Bandit finds common security issues in Python code:
- Hardcoded passwords
- SQL injection risks
- Insecure function usage
- Dangerous imports

### Installation

```bash
pip install bandit
```

### Usage

```bash
# Scan directory
bandit -r src/

# Exclude test files
bandit -r src/ --exclude tests

# Output as JSON
bandit -r src/ -f json -o security-report.json
```

### Example Findings

```python
# Bandit catches these issues:

password = "hardcoded_password123"  # B105: Hardcoded password

import pickle
pickle.loads(user_input)  # B301: Pickle insecure deserialization

subprocess.call(user_input, shell=True)  # B602: Shell injection
```

### Configuration

```toml
# pyproject.toml
[tool.bandit]
exclude_dirs = ["tests", "venv"]
skips = ["B101"]  # Skip assert warnings
```

---

## Integrating Multiple Linters

### Recommended Setup (2024+)

```toml
# pyproject.toml - Ruff does most things
[tool.ruff]
select = [
    "E", "F", "I", "N", "W", "UP",
    "B", "C4", "SIM", "S",  # "S" includes security checks
]
```

```bash
# Run Ruff for most checks
ruff check . --fix

# Run Bandit for deeper security analysis
bandit -r src/
```

### CI/CD Integration

```yaml
# GitHub Actions
name: Lint

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install ruff bandit
      
      - name: Run Ruff
        run: ruff check .
      
      - name: Run Bandit
        run: bandit -r src/ --exit-zero
```

---

## Ignoring Warnings

### Inline Suppression

```python
import os  # noqa: F401 - needed for side effects

password = get_password()  # noqa: S105 - not hardcoded
```

### File-Level Suppression

```python
# ruff: noqa: F401
from mypackage import a, b, c, d, e  # All unused import warnings ignored
```

### Configuration-Level

```toml
[tool.ruff.per-file-ignores]
"tests/*" = ["S101"]  # Allow assert in tests
"__init__.py" = ["F401"]  # Allow re-exports
"migrations/*" = ["E501"]  # Allow long lines
```

---

## Hands-on Exercise

### Your Task

```bash
# Set up linting for a project:
# 1. Create a file with various issues
# 2. Configure Ruff with comprehensive rules
# 3. Run linting and fix issues
# 4. Run Bandit for security check
```

<details>
<summary>✅ Solution</summary>

```bash
# 1. Create file with issues
cat > buggy_code.py << 'EOF'
import os
import sys
import pickle

password = "admin123"

def process_data(items=[]):
    result = []
    for item in items:
        if item == None:
            continue
        result.append(item)
    return result

def execute_command(cmd):
    import subprocess
    subprocess.call(cmd, shell=True)

def load_data(data):
    return pickle.loads(data)
EOF

# 2. Configure Ruff
cat > pyproject.toml << 'EOF'
[tool.ruff]
line-length = 88
target-version = "py311"
select = [
    "E",   # pycodestyle
    "F",   # pyflakes
    "B",   # bugbear
    "S",   # security (bandit rules)
    "SIM", # simplify
]

[tool.ruff.per-file-ignores]
"tests/*" = ["S101"]
EOF

# 3. Run linting
pip install ruff bandit
echo "=== Ruff check ==="
ruff check buggy_code.py

echo ""
echo "=== Auto-fix ==="
ruff check buggy_code.py --fix

echo ""
echo "=== Remaining issues ==="
ruff check buggy_code.py

# 4. Security check
echo ""
echo "=== Bandit security scan ==="
bandit buggy_code.py
```

**Expected Ruff output:**
```
buggy_code.py:1:8: F401 [*] `os` imported but unused
buggy_code.py:2:8: F401 [*] `sys` imported but unused
buggy_code.py:5:12: S105 Possible hardcoded password
buggy_code.py:7:21: B006 Mutable default argument
buggy_code.py:10:12: E711 [*] Comparison to `None`
buggy_code.py:17:5: S602 `subprocess.call` with `shell=True`
buggy_code.py:20:12: S301 `pickle.loads` is insecure
```
</details>

---

## Summary

✅ **Ruff** is the fastest, most comprehensive modern linter
✅ **Pylint** offers deepest analysis but slower
✅ **Bandit** catches security vulnerabilities
✅ **Auto-fix** (`--fix`) corrects many issues automatically
✅ **noqa comments** suppress specific warnings
✅ **CI integration** ensures all code is linted

**Next:** [Static Analysis](./03-static-analysis.md)

---

## Further Reading

- [Ruff Rules Reference](https://docs.astral.sh/ruff/rules/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Pylint Documentation](https://pylint.pycqa.org/)

<!-- 
Sources Consulted:
- Ruff Docs: https://docs.astral.sh/ruff/
- Bandit Docs: https://bandit.readthedocs.io/
-->
