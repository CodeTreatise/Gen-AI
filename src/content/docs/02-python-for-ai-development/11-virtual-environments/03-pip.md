---
title: "pip Package Manager"
---

# pip Package Manager

## Introduction

`pip` is Python's package installer. It downloads and installs packages from PyPI (Python Package Index) and manages your project's dependencies.

### What We'll Cover

- Installing packages
- Upgrading and uninstalling
- Requirements files
- Package information
- Best practices

### Prerequisites

- Virtual environment knowledge

---

## Installing Packages

### Basic Installation

```bash
# Install latest version
pip install requests

# Install specific version
pip install requests==2.31.0

# Install minimum version
pip install requests>=2.28.0

# Install version range
pip install "requests>=2.28.0,<3.0.0"
```

### Multiple Packages

```bash
# Multiple in one command
pip install numpy pandas scikit-learn

# From requirements file
pip install -r requirements.txt
```

### Installation Sources

```bash
# From PyPI (default)
pip install package-name

# From GitHub
pip install git+https://github.com/user/repo.git

# From GitHub specific branch/tag
pip install git+https://github.com/user/repo.git@main
pip install git+https://github.com/user/repo.git@v1.0.0

# From local directory (editable/development mode)
pip install -e .
pip install -e ./path/to/package

# From wheel file
pip install package-1.0.0-py3-none-any.whl

# From tarball
pip install package-1.0.0.tar.gz
```

---

## Upgrading Packages

### Upgrade Single Package

```bash
# Upgrade to latest
pip install --upgrade requests
pip install -U requests  # Short form

# Upgrade to specific version
pip install --upgrade requests==2.31.0
```

### Upgrade All Packages

```bash
# List outdated packages
pip list --outdated

# Upgrade all (requires manual loop)
pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U

# Or use pip-review tool
pip install pip-review
pip-review --auto
```

### Upgrade pip Itself

```bash
pip install --upgrade pip
# Or
python -m pip install --upgrade pip
```

---

## Uninstalling Packages

### Basic Uninstall

```bash
# Single package
pip uninstall requests

# Multiple packages
pip uninstall requests numpy pandas

# Skip confirmation
pip uninstall -y requests
```

### Uninstall with Dependencies

```bash
# pip doesn't auto-remove dependencies
# Use pip-autoremove for that
pip install pip-autoremove
pip-autoremove requests -y
```

---

## Listing Packages

### List All Installed

```bash
# Simple list
pip list

# Detailed format
pip list --format=columns
pip list --format=freeze
pip list --format=json
```

### List Outdated

```bash
pip list --outdated
```

**Output:**
```
Package    Version Latest Type
---------- ------- ------ -----
requests   2.28.0  2.31.0 wheel
numpy      1.24.0  1.26.0 wheel
```

### List Not Required by Others

```bash
pip list --not-required
```

---

## Package Information

### Show Package Details

```bash
pip show requests
```

**Output:**
```
Name: requests
Version: 2.31.0
Summary: Python HTTP for Humans.
Home-page: https://requests.readthedocs.io
Author: Kenneth Reitz
License: Apache 2.0
Location: /home/user/.venv/lib/python3.11/site-packages
Requires: certifi, charset-normalizer, idna, urllib3
Required-by: httpx
```

### Show Files Installed

```bash
pip show --files requests
```

### Check Dependencies

```bash
pip check
```

**Output (if issues):**
```
package-a 1.0.0 requires numpy>=2.0, but you have numpy 1.24.0.
```

---

## Freeze and Requirements

### Generate requirements.txt

```bash
# Freeze all packages
pip freeze > requirements.txt

# Contents:
# certifi==2024.2.2
# charset-normalizer==3.3.2
# idna==3.6
# requests==2.31.0
# urllib3==2.2.1
```

### Install from requirements.txt

```bash
pip install -r requirements.txt

# With upgrade
pip install -r requirements.txt --upgrade
```

### Freeze Without Editables

```bash
pip freeze --exclude-editable > requirements.txt
```

---

## pip Configuration

### Config File Location

```bash
# Linux/Mac
~/.config/pip/pip.conf
# or
~/.pip/pip.conf

# Windows
%APPDATA%\pip\pip.ini
```

### Common Configuration

```ini
# pip.conf
[global]
timeout = 60
index-url = https://pypi.org/simple
trusted-host = pypi.org

[install]
no-cache-dir = false
```

### Using Index Mirrors

```bash
# Use different index temporarily
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple package

# Or configure permanently in pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## Best Practices

### Always Use Virtual Environments

```bash
# Never do this:
sudo pip install package  # ❌

# Always do this:
python -m venv .venv
source .venv/bin/activate
pip install package  # ✅
```

### Pin Exact Versions for Production

```bash
# Development: OK to be flexible
pip install requests>=2.28.0

# Production: Pin exact versions
pip install requests==2.31.0
```

### Keep pip Updated

```bash
pip install --upgrade pip
```

### Use pip-tools for Better Management

```bash
pip install pip-tools

# Create requirements.in with loose requirements
echo "requests>=2.28" > requirements.in

# Compile to exact versions
pip-compile requirements.in
# Creates requirements.txt with pinned versions

# Sync environment to requirements
pip-sync requirements.txt
```

---

## Troubleshooting

### Common Errors

```bash
# SSL Certificate Error
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package

# Permission Denied
pip install --user package  # Install to user directory

# Version Conflict
pip install package --force-reinstall

# Cache Issues
pip install package --no-cache-dir
```

### Verbose Output

```bash
pip install package -v      # Verbose
pip install package -vv     # More verbose
pip install package -vvv    # Most verbose
```

---

## Hands-on Exercise

### Your Task

```bash
# Practice pip workflow:
# 1. Create and activate a venv
# 2. Install requests with a specific version
# 3. Show package info
# 4. List installed packages
# 5. Upgrade to latest version
# 6. Freeze requirements
# 7. Uninstall and reinstall from requirements
```

<details>
<summary>✅ Solution</summary>

```bash
# 1. Create and activate venv
python -m venv .venv
source .venv/bin/activate

# 2. Install specific version
pip install requests==2.28.0
python -c "import requests; print(requests.__version__)"
# Output: 2.28.0

# 3. Show package info
pip show requests
# Shows version, dependencies, location

# 4. List installed packages
pip list
# Output:
# Package            Version
# requests           2.28.0
# certifi            2024.2.2
# ...

# 5. Upgrade to latest
pip install --upgrade requests
python -c "import requests; print(requests.__version__)"
# Output: 2.31.0 (or current latest)

# 6. Freeze requirements
pip freeze > requirements.txt
cat requirements.txt

# 7. Uninstall and reinstall
pip uninstall -y requests
python -c "import requests"  # Should error

pip install -r requirements.txt
python -c "import requests; print(requests.__version__)"
# Output: 2.31.0 (version from requirements)

deactivate
```
</details>

---

## Summary

✅ **`pip install`** installs packages from PyPI
✅ **`pip install -r requirements.txt`** installs from file
✅ **`pip freeze > requirements.txt`** saves current packages
✅ **`pip show`** displays package information
✅ **`pip list --outdated`** shows updatable packages
✅ **Pin exact versions** for production reproducibility

**Next:** [requirements.txt](./04-requirements.md)

---

## Further Reading

- [pip Documentation](https://pip.pypa.io/en/stable/)
- [PyPI - Python Package Index](https://pypi.org/)

<!-- 
Sources Consulted:
- pip Documentation: https://pip.pypa.io/en/stable/
-->
