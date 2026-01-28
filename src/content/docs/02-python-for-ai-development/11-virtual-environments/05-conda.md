---
title: "Conda Environments"
---

# Conda Environments

## Introduction

Conda is a package and environment manager popular in data science and ML. Unlike pip, Conda manages both Python packages and system-level dependencies.

### What We'll Cover

- Conda vs pip
- Creating and managing environments
- environment.yml files
- When to use Conda

### Prerequisites

- Basic package management
- Virtual environment concepts

---

## Conda vs pip

### Key Differences

| Feature | pip + venv | Conda |
|---------|-----------|-------|
| **Package source** | PyPI | Anaconda, conda-forge |
| **Python version** | System-dependent | Manages Python itself |
| **Non-Python deps** | No | Yes (CUDA, MKL, etc.) |
| **Environment** | venv module | Built-in |
| **Speed** | Fast | Slower (but improving) |
| **Binary packages** | Wheels | Conda packages |

### When to Use Each

```
Use pip + venv when:
├── Pure Python projects
├── Web development
├── API development
├── Simpler dependency tree
└── CI/CD pipelines (faster)

Use Conda when:
├── Data science / ML projects
├── Need specific Python versions easily
├── Require system libraries (CUDA, MKL)
├── Complex scientific computing
└── Cross-platform binary dependencies
```

---

## Installing Conda

### Miniconda (Recommended)

```bash
# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# macOS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# Verify
conda --version
```

### Anaconda vs Miniconda

| | Anaconda | Miniconda |
|--|----------|-----------|
| **Size** | ~3 GB | ~400 MB |
| **Packages** | 250+ pre-installed | Minimal |
| **Best for** | Beginners | Experienced users |

---

## Creating Environments

### Basic Creation

```bash
# Create with default Python
conda create --name myenv

# Create with specific Python version
conda create --name myenv python=3.11

# Create with packages
conda create --name myenv python=3.11 numpy pandas
```

### Activating and Deactivating

```bash
# Activate
conda activate myenv

# Prompt changes
(myenv) user@machine:~$

# Deactivate
conda deactivate

# Switch to different environment
conda activate other-env
```

### Listing Environments

```bash
conda env list
# or
conda info --envs
```

**Output:**
```
# conda environments:
#
base                     /home/user/miniconda3
myenv                 *  /home/user/miniconda3/envs/myenv
ml-project               /home/user/miniconda3/envs/ml-project
```

### Removing Environments

```bash
conda env remove --name myenv

# Or remove all packages
conda remove --name myenv --all
```

---

## Installing Packages

### From Conda

```bash
# Single package
conda install numpy

# Multiple packages
conda install numpy pandas scikit-learn

# Specific version
conda install numpy=1.24.0

# From specific channel
conda install -c conda-forge package-name
```

### From pip (When Needed)

```bash
# Activate conda env first
conda activate myenv

# Then use pip
pip install package-not-on-conda
```

> **Note:** Prefer `conda install` when available. Use `pip` only for packages not on Conda.

### Searching Packages

```bash
conda search numpy
conda search -c conda-forge httpx
```

---

## environment.yml

### Basic Format

```yaml
# environment.yml
name: myproject
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy=1.26
  - pandas>=2.0
  - scikit-learn
  - pip
  - pip:
    - httpx>=0.27
    - openai
```

### Creating from File

```bash
conda env create -f environment.yml
```

### Exporting Environment

```bash
# Full export (may not be portable)
conda env export > environment.yml

# Cross-platform export (recommended)
conda env export --from-history > environment.yml
```

### Updating from File

```bash
conda env update -f environment.yml --prune
```

---

## Channels

### What Are Channels?

Channels are package repositories. Common ones:

| Channel | Description |
|---------|-------------|
| `defaults` | Official Anaconda packages |
| `conda-forge` | Community packages (larger selection) |
| `pytorch` | PyTorch packages |
| `nvidia` | CUDA packages |

### Configuring Channels

```bash
# Add channel
conda config --add channels conda-forge

# Set channel priority
conda config --set channel_priority strict

# View configuration
conda config --show channels
```

### Channel Priority in environment.yml

```yaml
channels:
  - pytorch         # Highest priority
  - conda-forge
  - defaults        # Lowest priority
```

---

## Best Practices

### 1. Use environment.yml for Projects

```yaml
# environment.yml
name: ml-project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pandas
  - scikit-learn
  - jupyter
  - matplotlib
  - pip
  - pip:
    - openai
    - langchain
```

### 2. Separate Environments per Project

```bash
cd project-a && conda activate project-a-env
cd project-b && conda activate project-b-env
```

### 3. Pin Python Version

```yaml
dependencies:
  - python=3.11  # ✅ Pinned
  # - python     # ❌ Not pinned
```

### 4. Use conda-forge for Latest Packages

```yaml
channels:
  - conda-forge  # Usually has newer versions
  - defaults
```

---

## Mixing Conda and pip

### Safe Pattern

```yaml
# environment.yml
name: myproject
channels:
  - conda-forge
  - defaults
dependencies:
  # Conda packages first
  - python=3.11
  - numpy
  - pandas
  - pytorch
  - pip
  # pip packages last
  - pip:
    - openai
    - langchain
    - httpx
```

### Rules for Mixing

1. Install conda packages first
2. Install pip packages last
3. Once you use pip, avoid conda install for same packages
4. List pip packages in environment.yml

---

## Hands-on Exercise

### Your Task

```bash
# Create a complete ML environment:
# 1. Create environment.yml for an ML project
# 2. Include Python, numpy, pandas, scikit-learn
# 3. Add a pip package (openai)
# 4. Create and activate the environment
# 5. Export the environment
```

<details>
<summary>✅ Solution</summary>

```bash
# 1. Create environment.yml
cat > environment.yml << EOF
name: ml-project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy>=1.24
  - pandas>=2.0
  - scikit-learn>=1.3
  - matplotlib>=3.8
  - jupyter
  - ipykernel
  - pip
  - pip:
    - openai>=1.0
    - httpx>=0.27
EOF

# 2. Create environment
conda env create -f environment.yml

# 3. Activate
conda activate ml-project

# 4. Verify
python -c "import numpy, pandas, sklearn; print('Conda packages work!')"
python -c "import openai, httpx; print('Pip packages work!')"

# 5. Export (cross-platform)
conda env export --from-history > environment-exported.yml
cat environment-exported.yml

# 6. Test by recreating
conda deactivate
conda env remove --name ml-project
conda env create -f environment-exported.yml
conda activate ml-project
python -c "import numpy; print('Recreated successfully!')"
```
</details>

---

## Summary

✅ **Conda** manages Python AND system dependencies
✅ **`conda create`** creates isolated environments
✅ **`conda activate/deactivate`** switches environments
✅ **environment.yml** defines reproducible environments
✅ **conda-forge** has latest community packages
✅ Use **pip inside conda** for PyPI-only packages

**Next:** [Modern Tools](./06-modern-tools.md)

---

## Further Reading

- [Conda Documentation](https://docs.conda.io/)
- [Conda-Forge](https://conda-forge.org/)
- [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

<!-- 
Sources Consulted:
- Conda Docs: https://docs.conda.io/
-->
