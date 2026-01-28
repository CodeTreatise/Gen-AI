---
title: "Virtual Environments"
---

# Virtual Environments

## Overview

Virtual environments are essential for Python development. They isolate project dependencies, prevent version conflicts, and ensure reproducible setups across machines.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-why-virtual-environments.md) | Why Virtual Environments | Isolation, conflicts, reproducibility |
| [02](./02-venv.md) | venv Module | Creating, activating, managing |
| [03](./03-pip.md) | pip Package Manager | Install, freeze, requirements |
| [04](./04-requirements.md) | requirements.txt | Pinning, dev deps, formats |
| [05](./05-conda.md) | Conda Environments | Conda vs pip, environment.yml |
| [06](./06-modern-tools.md) | Modern Tools | Poetry, uv, pyproject.toml |

---

## Quick Start

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install packages
pip install requests numpy

# Save dependencies
pip freeze > requirements.txt

# Deactivate
deactivate
```

---

## Why It Matters

| Problem | Without venv | With venv |
|---------|--------------|-----------|
| **Project A needs numpy 1.x** | Conflict! | ✅ Isolated |
| **Project B needs numpy 2.x** | Conflict! | ✅ Isolated |
| **Sharing project** | "Works on my machine" | ✅ Reproducible |
| **System Python** | Risk of breaking OS tools | ✅ Protected |

---

## Prerequisites

Before starting this lesson:
- Python basics
- Command line familiarity
- Package installation concepts

---

## Start Learning

Begin with [Why Virtual Environments](./01-why-virtual-environments.md) to understand the motivation.

---

## Further Reading

- [Python venv Docs](https://docs.python.org/3/library/venv.html)
- [pip Documentation](https://pip.pypa.io/en/stable/)
- [Poetry Documentation](https://python-poetry.org/docs/)
