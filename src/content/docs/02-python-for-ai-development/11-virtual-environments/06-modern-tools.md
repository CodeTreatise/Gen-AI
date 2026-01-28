---
title: "Modern Tools"
---

# Modern Tools

## Introduction

Modern Python packaging tools improve on pip with better dependency resolution, lock files, and project configuration. This lesson covers Poetry, pipenv, and the fast new `uv` tool.

### What We'll Cover

- pyproject.toml standard
- Poetry for dependency management
- pipenv basics
- uv for fast installs

### Prerequisites

- pip and venv knowledge
- Package management concepts

---

## pyproject.toml

### The Modern Standard

`pyproject.toml` is the new standard for Python project configuration, replacing `setup.py`, `setup.cfg`, and multiple config files.

```toml
# pyproject.toml
[project]
name = "my-project"
version = "0.1.0"
description = "My awesome project"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"}
]
dependencies = [
    "flask>=2.0",
    "httpx>=0.27",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "mypy>=1.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

### Installing from pyproject.toml

```bash
# Install project
pip install .

# Install with dev dependencies
pip install ".[dev]"

# Editable install for development
pip install -e ".[dev]"
```

---

## Poetry

### Installation

```bash
# Official installer
curl -sSL https://install.python-poetry.org | python3 -

# Or with pipx (recommended)
pipx install poetry

# Verify
poetry --version
```

### Creating a New Project

```bash
# New project from scratch
poetry new my-project

# In existing directory
cd my-project
poetry init
```

### Project Structure

```
my-project/
├── pyproject.toml      # Project configuration
├── poetry.lock         # Lock file (commit this!)
├── src/
│   └── my_project/
│       └── __init__.py
└── tests/
    └── __init__.py
```

### pyproject.toml (Poetry Format)

```toml
[tool.poetry]
name = "my-project"
version = "0.1.0"
description = "My project"
authors = ["Your Name <you@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
flask = "^2.3"
httpx = "^0.27"
pydantic = "^2.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
black = "^24.0"
mypy = "^1.8"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### Common Commands

```bash
# Add dependency
poetry add flask

# Add dev dependency
poetry add pytest --group dev

# Remove dependency
poetry remove flask

# Install all dependencies
poetry install

# Install without dev
poetry install --only main

# Update dependencies
poetry update

# Show installed packages
poetry show

# Run commands in venv
poetry run python app.py
poetry run pytest

# Activate venv shell
poetry shell
```

### Lock Files

```bash
# poetry.lock is generated automatically
# Always commit it!

# Regenerate lock file
poetry lock

# Install exact versions from lock
poetry install
```

---

## pipenv

### Installation

```bash
pip install pipenv
# or
pipx install pipenv
```

### Basic Usage

```bash
# Create environment and Pipfile
cd my-project
pipenv install

# Add dependency
pipenv install flask

# Add dev dependency
pipenv install pytest --dev

# Activate shell
pipenv shell

# Run command in venv
pipenv run python app.py

# Generate lock file
pipenv lock

# Install from lock file
pipenv sync
```

### Pipfile

```toml
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
flask = ">=2.0"
httpx = "*"

[dev-packages]
pytest = "*"
black = "*"

[requires]
python_version = "3.11"
```

---

## uv - The Fast Package Manager

### Installation

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify
uv --version
```

### Why uv?

- Written in Rust, extremely fast (10-100x faster than pip)
- Drop-in replacement for pip and pip-tools
- Compatible with existing requirements.txt
- Creates virtual environments

### Basic Usage

```bash
# Create venv
uv venv

# Activate
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install packages (much faster than pip!)
uv pip install flask numpy pandas

# Install from requirements
uv pip install -r requirements.txt

# Freeze
uv pip freeze > requirements.txt

# Compile requirements (like pip-compile)
uv pip compile requirements.in -o requirements.txt

# Sync environment (like pip-sync)
uv pip sync requirements.txt
```

### Speed Comparison

```bash
# pip
time pip install pandas scikit-learn matplotlib
# real    0m35.234s

# uv
time uv pip install pandas scikit-learn matplotlib
# real    0m2.156s

# ~16x faster!
```

---

## Tool Comparison

| Feature | pip+venv | Poetry | pipenv | uv |
|---------|----------|--------|--------|-----|
| **Speed** | Moderate | Slow | Slow | Fast |
| **Lock file** | Manual | ✅ | ✅ | ✅ |
| **Dependency resolution** | Basic | Excellent | Good | Excellent |
| **pyproject.toml** | ✅ | ✅ | ❌ | ✅ |
| **Publishing** | Manual | ✅ | ❌ | ❌ |
| **Learning curve** | Low | Medium | Low | Low |

### When to Use Each

```
pip + venv:
├── Simple projects
├── CI/CD scripts
└── When speed matters (with uv)

Poetry:
├── Libraries you'll publish
├── Complex dependency trees
├── Team projects

pipenv:
├── Application development
├── Teams familiar with it
└── Simpler than Poetry

uv:
├── Speed-critical workflows
├── CI/CD pipelines
├── Drop-in pip replacement
```

---

## Migration Paths

### pip → Poetry

```bash
# In existing project
cd my-project
poetry init

# Import from requirements.txt
cat requirements.txt | xargs poetry add
```

### pip → uv

```bash
# Just use uv commands instead of pip
uv pip install -r requirements.txt
uv pip freeze > requirements.txt

# No project changes needed!
```

### Poetry → pip

```bash
# Export requirements
poetry export -f requirements.txt > requirements.txt
poetry export -f requirements.txt --with dev > requirements-dev.txt

# Now use pip/venv
python -m venv .venv
pip install -r requirements-dev.txt
```

---

## Hands-on Exercise

### Your Task

```bash
# Try different tools:
# 1. Create a project with Poetry
# 2. Add dependencies
# 3. Export to requirements.txt
# 4. Install with uv and compare speed
```

<details>
<summary>✅ Solution</summary>

```bash
# 1. Create Poetry project
poetry new modern-tools-demo
cd modern-tools-demo

# 2. Add dependencies
poetry add flask httpx pydantic
poetry add pytest black mypy --group dev

# View pyproject.toml
cat pyproject.toml

# 3. Export to requirements.txt
poetry export -f requirements.txt --output requirements.txt
poetry export -f requirements.txt --with dev --output requirements-dev.txt

cat requirements.txt

# 4. Compare speed: pip vs uv
cd ..
mkdir speed-test && cd speed-test
python -m venv .venv-pip
source .venv-pip/bin/activate

# Time pip
echo "=== pip install ==="
time pip install -r ../modern-tools-demo/requirements.txt
deactivate

# Time uv
uv venv .venv-uv
source .venv-uv/bin/activate
echo "=== uv pip install ==="
time uv pip install -r ../modern-tools-demo/requirements.txt
deactivate

# Compare results
echo "Check the times above - uv should be much faster!"
```
</details>

---

## Summary

✅ **pyproject.toml** is the modern project config standard
✅ **Poetry** offers excellent dependency management + publishing
✅ **pipenv** is simpler alternative with Pipfile
✅ **uv** is blazing fast pip replacement
✅ **Lock files** ensure reproducible environments
✅ Choose tool based on project needs

**Back to:** [Virtual Environments Overview](./00-virtual-environments.md)

---

## Further Reading

- [Poetry Documentation](https://python-poetry.org/docs/)
- [pipenv Documentation](https://pipenv.pypa.io/)
- [uv Documentation](https://github.com/astral-sh/uv)
- [pyproject.toml Specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)

<!-- 
Sources Consulted:
- Poetry Docs: https://python-poetry.org/docs/
- uv GitHub: https://github.com/astral-sh/uv
-->
