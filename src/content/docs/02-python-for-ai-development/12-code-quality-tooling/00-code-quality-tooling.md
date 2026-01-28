---
title: "Code Quality & Tooling"
---

# Code Quality & Tooling

## Overview

Professional Python development requires consistent code quality, proper formatting, and maintainable codebases. Modern tooling automates style enforcement, catches bugs early, and ensures team consistency.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-code-formatting.md) | Code Formatting | Black, Ruff formatter, isort |
| [02](./02-linting.md) | Linting | Ruff linter, Pylint, Flake8, Bandit |
| [03](./03-static-analysis.md) | Static Analysis | mypy, pyright, type checking |
| [04](./04-pre-commit-hooks.md) | Pre-commit Hooks | Automation, CI integration |
| [05](./05-documentation.md) | Documentation | Docstrings, Sphinx, mkdocs |
| [06](./06-code-organization.md) | Code Organization | Project structure, imports |

---

## Quick Start: Minimal Quality Setup

```bash
# Install essential tools
pip install ruff

# pyproject.toml - minimal config
cat >> pyproject.toml << 'EOF'
[tool.ruff]
line-length = 88
select = ["E", "F", "I", "W"]

[tool.ruff.format]
quote-style = "double"
EOF

# Format and lint
ruff format .
ruff check . --fix
```

---

## Tool Comparison

| Tool | Purpose | Speed | Best For |
|------|---------|-------|----------|
| **Ruff** | Linter + Formatter | ⚡ Fastest | All projects (2024+) |
| **Black** | Formatter | Fast | Opinionated formatting |
| **Pylint** | Linter | Slow | Comprehensive analysis |
| **Flake8** | Linter | Medium | Simple style checking |
| **mypy** | Type checker | Medium | Type safety |
| **Bandit** | Security linter | Fast | Security scanning |

> **🤖 AI Context:** Consistent code formatting makes LLM-generated code easier to review and integrate.

---

## Recommended Configuration

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "W",    # pycodestyle warnings
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "SIM",  # flake8-simplify
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.11"
strict = true
```

---

## Prerequisites

Before starting this lesson:
- Virtual environment basics
- Package management with pip
- Basic command line usage

---

## Start Learning

Begin with [Code Formatting](./01-code-formatting.md) to set up automatic formatting.

---

## Further Reading

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
