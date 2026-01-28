---
title: "Pre-commit Hooks"
---

# Pre-commit Hooks

## Introduction

Pre-commit hooks run quality checks automatically before each commit. They prevent bad code from entering your repository and ensure consistent standards across the team.

### What We'll Cover

- Installing pre-commit
- Configuration file setup
- Common hooks
- CI integration
- Custom hooks

### Prerequisites

- Git basics
- Code formatting and linting tools

---

## What Are Pre-commit Hooks?

### Git Hooks

Git provides hooks that run scripts at different points:
- `pre-commit`: Before commit is created
- `pre-push`: Before pushing to remote
- `commit-msg`: After commit message is written

### pre-commit Framework

The `pre-commit` framework manages these hooks:
- Easy configuration via YAML
- Automatic tool installation
- Runs only on changed files
- Caches environments for speed

---

## Installation

### Install pre-commit

```bash
pip install pre-commit
# or
pipx install pre-commit
# or
brew install pre-commit  # macOS
```

### Initialize in Repository

```bash
cd your-project

# Create config file
pre-commit sample-config > .pre-commit-config.yaml

# Install git hooks
pre-commit install

# Now hooks run on every commit!
```

---

## Configuration File

### Basic .pre-commit-config.yaml

```yaml
# .pre-commit-config.yaml
repos:
  # Ruff for linting and formatting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  # Standard pre-commit hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

### Complete Configuration

```yaml
# .pre-commit-config.yaml
default_language_version:
  python: python3.11

repos:
  # Ruff - linting and formatting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  # mypy - type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies:
          - types-requests
          - pydantic

  # Standard hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-json
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-merge-conflict
      - id: debug-statements
      - id: detect-private-key

  # Security scanning
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.7
    hooks:
      - id: bandit
        args: [-r, src/, --skip, B101]

  # Commit message format
  - repo: https://github.com/commitizen-tools/commitizen
    rev: v3.14.1
    hooks:
      - id: commitizen
```

---

## Common Hooks

### Ruff (Linting + Formatting)

```yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.3.0
  hooks:
    - id: ruff
      args: [--fix]  # Auto-fix issues
    - id: ruff-format
```

### Black (Formatting)

```yaml
- repo: https://github.com/psf/black
  rev: 24.2.0
  hooks:
    - id: black
```

### mypy (Type Checking)

```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.8.0
  hooks:
    - id: mypy
      additional_dependencies:
        - types-requests
        - pydantic
        - numpy
```

### isort (Import Sorting)

```yaml
- repo: https://github.com/pycqa/isort
  rev: 5.13.2
  hooks:
    - id: isort
```

### Built-in Hooks

```yaml
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.5.0
  hooks:
    # Whitespace
    - id: trailing-whitespace
    - id: end-of-file-fixer
    
    # File checks
    - id: check-yaml
    - id: check-toml
    - id: check-json
    - id: check-xml
    
    # Safety checks
    - id: check-added-large-files
    - id: check-merge-conflict
    - id: debug-statements      # No print/pdb in code
    - id: detect-private-key
    
    # Python specific
    - id: check-ast             # Valid Python syntax
    - id: fix-encoding-pragma   # UTF-8 encoding
    - id: name-tests-test       # Test files named test_*
```

---

## Running Hooks

### Automatic (on commit)

```bash
git add myfile.py
git commit -m "Add feature"
# Hooks run automatically!
```

### Manual Runs

```bash
# Run on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files

# Run with verbose output
pre-commit run --verbose
```

### Update Hooks

```bash
# Update all hooks to latest versions
pre-commit autoupdate
```

---

## Skipping Hooks

### Skip for One Commit

```bash
# Skip all hooks
git commit -m "WIP" --no-verify

# Or
SKIP=ruff git commit -m "Skip ruff check"
```

### Skip Specific Hooks

```bash
# Skip multiple hooks
SKIP=ruff,mypy git commit -m "Skip checks"
```

> **Warning:** Only skip hooks when necessary. Document why in commit message.

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/pre-commit.yml
name: Pre-commit

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - uses: pre-commit/action@v3.0.1
```

### Caching in CI

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pre-commit
    key: pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}
```

---

## Custom Hooks

### Local Hook

```yaml
repos:
  - repo: local
    hooks:
      - id: check-debug
        name: Check for debug statements
        entry: python -c "import sys; sys.exit(1 if 'breakpoint()' in open(sys.argv[1]).read() else 0)"
        language: python
        types: [python]
```

### Script Hook

```yaml
repos:
  - repo: local
    hooks:
      - id: run-tests
        name: Run unit tests
        entry: pytest tests/ -x
        language: system
        pass_filenames: false
        always_run: true
```

### Requirements Check

```yaml
repos:
  - repo: local
    hooks:
      - id: pip-check
        name: Check requirements consistency
        entry: pip check
        language: system
        pass_filenames: false
```

---

## Best Practices

### 1. Fast Hooks First

```yaml
repos:
  # Fast hooks run first
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace  # ~instant
      
  # Slower hooks later
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy  # Can be slow
```

### 2. Pin Versions

```yaml
# Good: Pinned version
rev: v0.3.0

# Bad: Unpredictable
rev: main
```

### 3. Document Skips

```yaml
- id: bandit
  args: [-r, src/, --skip, B101]  # B101: assert used for tests
```

### 4. Run on PR

Ensure CI runs pre-commit to catch bypassed commits.

---

## Hands-on Exercise

### Your Task

```bash
# Set up pre-commit for a project:
# 1. Create a new project with git
# 2. Set up .pre-commit-config.yaml
# 3. Install the hooks
# 4. Create a file with issues and try to commit
# 5. Fix and successfully commit
```

<details>
<summary>✅ Solution</summary>

```bash
# 1. Create project
mkdir precommit-demo && cd precommit-demo
git init

# 2. Create config
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: debug-statements
EOF

# 3. Install hooks
pip install pre-commit
pre-commit install

# 4. Create problematic file
cat > bad_code.py << 'EOF'
import os
import sys   

def  test():
    x=1+2
    print(x)
    breakpoint()
EOF

# Try to commit
git add .
git commit -m "Add code"
# FAILS! Pre-commit catches issues

# 5. Fix the file
cat > bad_code.py << 'EOF'
def test():
    x = 1 + 2
    print(x)
EOF

# Commit again
git add .
git commit -m "Add code"
# SUCCESS!

echo "Pre-commit hooks working!"
```
</details>

---

## Summary

✅ **pre-commit** automates quality checks before commits
✅ **`.pre-commit-config.yaml`** defines hooks to run
✅ **`pre-commit install`** activates hooks in repository
✅ **Ruff hooks** handle linting and formatting
✅ **`--no-verify`** skips hooks (use sparingly)
✅ **CI integration** catches bypassed commits

**Next:** [Documentation](./05-documentation.md)

---

## Further Reading

- [Pre-commit Documentation](https://pre-commit.com/)
- [Available Hooks](https://pre-commit.com/hooks.html)
- [pre-commit CI](https://pre-commit.ci/)

<!-- 
Sources Consulted:
- Pre-commit Docs: https://pre-commit.com/
-->
