---
title: "venv Module"
---

# venv Module

## Introduction

The `venv` module is Python's built-in tool for creating virtual environments. It's simple, reliable, and doesn't require any additional installation.

### What We'll Cover

- Creating virtual environments
- Activating and deactivating
- Location conventions
- Managing environments

### Prerequisites

- Command line basics
- Understanding of why venvs matter

---

## Creating Virtual Environments

### Basic Creation

```bash
# Standard command
python -m venv .venv

# Or with explicit Python version
python3.11 -m venv .venv

# On some systems
python3 -m venv .venv
```

### What Gets Created

```
.venv/
├── bin/                    # Linux/Mac
│   ├── activate            # Bash activation script
│   ├── activate.csh        # C-shell activation
│   ├── activate.fish       # Fish shell activation
│   ├── pip                 # pip for this env
│   ├── pip3
│   ├── python              # Python symlink
│   └── python3
├── Scripts/                # Windows instead of bin/
│   ├── activate
│   ├── activate.bat
│   ├── Activate.ps1
│   ├── pip.exe
│   └── python.exe
├── lib/
│   └── python3.11/
│       └── site-packages/  # Where packages install
├── include/                # C headers for extensions
└── pyvenv.cfg              # Environment config
```

### Creation Options

```bash
# Without pip (faster, smaller)
python -m venv .venv --without-pip

# With system packages accessible
python -m venv .venv --system-site-packages

# Clear existing and recreate
python -m venv .venv --clear

# Upgrade to current Python's venv
python -m venv .venv --upgrade
```

---

## Activating Environments

### Linux / macOS

```bash
# Bash/Zsh
source .venv/bin/activate

# Fish
source .venv/bin/activate.fish

# Csh/Tcsh
source .venv/bin/activate.csh
```

### Windows

```powershell
# PowerShell
.venv\Scripts\Activate.ps1

# Command Prompt
.venv\Scripts\activate.bat

# Git Bash
source .venv/Scripts/activate
```

### Verifying Activation

```bash
# Check Python location
which python        # Linux/Mac
where python        # Windows

# Should show: /path/to/project/.venv/bin/python

# Check pip location
which pip
# Should show: /path/to/project/.venv/bin/pip

# Prompt usually changes
# (venv) user@machine:~/project$
```

---

## Deactivating Environments

### Simple Deactivation

```bash
# Works on all systems
deactivate

# Prompt returns to normal
# user@machine:~/project$
```

### Verifying Deactivation

```bash
which python
# Should show system Python: /usr/bin/python
```

---

## Location Conventions

### Standard Names

| Name | Usage |
|------|-------|
| `.venv` | Most common, hidden directory |
| `venv` | Common, visible directory |
| `.env` | Less common (conflicts with dotenv) |
| `env` | Less common |

### Project Root Location

```
my-project/
├── .venv/              # ✅ Standard location
├── .git/
├── .gitignore          # Contains: .venv/
├── requirements.txt
├── src/
└── tests/
```

### External Location

```bash
# Sometimes useful for shared environments
python -m venv ~/.virtualenvs/myproject

# Or for temporary testing
python -m venv /tmp/test-env
```

---

## Managing Multiple Environments

### Per-Project Pattern

```bash
# Project 1
cd ~/projects/web-app
python -m venv .venv
source .venv/bin/activate
pip install flask

# Project 2
cd ~/projects/ml-project
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn
```

### Quick Switching

```bash
# Create alias in ~/.bashrc or ~/.zshrc
alias activate='source .venv/bin/activate'

# Now just:
cd my-project
activate
```

### Using direnv (Auto-activation)

```bash
# Install direnv, then in project root:
echo 'source .venv/bin/activate' > .envrc
direnv allow

# Now activates automatically when you cd into directory!
```

---

## VS Code Integration

### Selecting Python Interpreter

1. Open Command Palette: `Ctrl+Shift+P` / `Cmd+Shift+P`
2. Type: "Python: Select Interpreter"
3. Choose: `./.venv/bin/python`

### Auto-Detection

VS Code automatically detects `.venv` directories:

```
.vscode/settings.json:
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

### Terminal Auto-Activation

VS Code can auto-activate venvs in integrated terminal:

```json
{
    "python.terminal.activateEnvironment": true
}
```

---

## Common Issues

### Issue: Permission Denied

```bash
# Problem
source .venv/bin/activate
# bash: .venv/bin/activate: Permission denied

# Solution
chmod +x .venv/bin/activate
```

### Issue: Wrong Python Version

```bash
# Check what Python venv uses
cat .venv/pyvenv.cfg
# home = /usr/bin
# version = 3.9.7

# If wrong, recreate with correct Python
rm -rf .venv
python3.11 -m venv .venv
```

### Issue: Execution Policy (Windows)

```powershell
# Problem
.venv\Scripts\Activate.ps1
# Scripts are disabled on this system

# Solution: Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Hands-on Exercise

### Your Task

```bash
# Create a complete project setup:
# 1. Create project directory
# 2. Create virtual environment
# 3. Activate it
# 4. Install a package
# 5. Verify installation
# 6. Create requirements.txt
# 7. Deactivate and verify cleanup
```

<details>
<summary>✅ Solution</summary>

```bash
# 1. Create project directory
mkdir my-ai-project
cd my-ai-project

# 2. Create virtual environment
python -m venv .venv

# 3. Activate it
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Verify activation
echo "Python: $(which python)"
echo "Pip: $(which pip)"

# 4. Install a package
pip install httpx

# 5. Verify installation
python -c "import httpx; print(f'httpx version: {httpx.__version__}')"

# 6. Create requirements.txt
pip freeze > requirements.txt
cat requirements.txt

# 7. Deactivate and verify
deactivate
which python  # Should show system Python

# Bonus: Create .gitignore
echo ".venv/" > .gitignore
echo "__pycache__/" >> .gitignore

# Verify final structure
ls -la
```

**Expected Output:**
```
Python: /home/user/my-ai-project/.venv/bin/python
Pip: /home/user/my-ai-project/.venv/bin/pip
httpx version: 0.27.0

requirements.txt contents:
anyio==4.3.0
certifi==2024.2.2
h11==0.14.0
httpcore==1.0.5
httpx==0.27.0
idna==3.6
sniffio==1.3.1
```
</details>

---

## Summary

✅ **`python -m venv .venv`** creates a virtual environment
✅ **`source .venv/bin/activate`** activates (Linux/Mac)
✅ **`.venv\Scripts\activate`** activates (Windows)
✅ **`deactivate`** exits the virtual environment
✅ **`.venv` in project root** is the convention
✅ **Never commit** the `.venv` directory to git

**Next:** [pip Package Manager](./03-pip.md)

---

## Further Reading

- [venv Documentation](https://docs.python.org/3/library/venv.html)
- [Virtual Environment Tutorial](https://docs.python.org/3/tutorial/venv.html)

<!-- 
Sources Consulted:
- Python venv Docs: https://docs.python.org/3/library/venv.html
-->
