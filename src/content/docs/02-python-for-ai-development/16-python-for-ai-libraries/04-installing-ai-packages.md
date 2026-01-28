---
title: "Installing AI Packages"
---

# Installing AI Packages

## Introduction

AI packages often require GPU drivers, CUDA, and careful version management. This lesson covers common installation scenarios and troubleshooting.

### What We'll Cover

- GPU-enabled installations
- CUDA and cuDNN setup
- Platform-specific considerations
- Version compatibility
- Troubleshooting common issues

### Prerequisites

- Virtual environments
- Basic command line

---

## CPU vs GPU Installation

### CPU-Only (Simpler)

```bash
# PyTorch CPU
pip install torch torchvision torchaudio

# TensorFlow CPU
pip install tensorflow
```

### GPU-Enabled (Faster)

```bash
# Check GPU first
nvidia-smi

# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorFlow GPU (included in main package)
pip install tensorflow
```

---

## CUDA and cuDNN

### What They Are

| Component | Purpose |
|-----------|---------|
| **CUDA** | NVIDIA's parallel computing platform |
| **cuDNN** | Deep neural network library for CUDA |
| **NVIDIA Driver** | Hardware driver for GPU |

### Check Your Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# In Python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

### Version Compatibility

| PyTorch | CUDA | cuDNN |
|---------|------|-------|
| 2.2.x | 11.8, 12.1 | 8.x |
| 2.1.x | 11.8, 12.1 | 8.x |
| 2.0.x | 11.7, 11.8 | 8.x |

> **Tip:** Check [PyTorch's website](https://pytorch.org/get-started/locally/) for the exact command for your system.

### Installing CUDA Toolkit

**Ubuntu/Debian:**
```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install cuda-toolkit-12-1
```

**Windows:**
Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

---

## Platform-Specific Installations

### macOS (Apple Silicon)

```bash
# PyTorch with MPS (Metal Performance Shaders)
pip install torch torchvision torchaudio

# Verify MPS
import torch
print(torch.backends.mps.is_available())  # True on M1/M2/M3
```

### Windows

```bash
# Ensure Visual C++ Build Tools installed
# Download from Microsoft

# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Linux

```bash
# Most straightforward
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For AMD GPUs (ROCm)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

---

## Version Compatibility

### Specifying Versions

```bash
# Exact version
pip install torch==2.2.0
pip install transformers==4.38.0

# Compatible versions
pip install "torch>=2.0.0,<3.0.0"
pip install "transformers>=4.30.0"
```

### Common Compatibility Issues

```python
# Version mismatch example
# ❌ Error: transformers requires torch<2.3, but you have 2.4
pip install transformers  # Might fail

# ✅ Solution: Install compatible versions
pip install torch==2.2.0 transformers==4.38.0
```

### Requirements File with Pinned Versions

```text
# requirements.txt
torch==2.2.0
transformers==4.38.0
langchain==0.1.0
openai==1.12.0
numpy==1.26.4
pandas==2.2.0
```

```bash
pip install -r requirements.txt
```

---

## Using Conda

### Why Conda?

- Handles binary dependencies better
- Manages CUDA automatically
- Provides isolated environments

### Installation

```bash
# Create environment with Python
conda create -n ai-env python=3.11

# Activate
conda activate ai-env

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other packages
conda install numpy pandas matplotlib scikit-learn
pip install transformers langchain openai
```

### Mixed Conda and Pip

```bash
# Best practice: Install conda packages first
conda install numpy pandas scipy

# Then pip for packages not in conda
pip install transformers langchain
```

---

## Troubleshooting

### CUDA Not Found

```python
# Error: CUDA not available
import torch
print(torch.cuda.is_available())  # False

# Solutions:
# 1. Check driver
# nvidia-smi

# 2. Reinstall with correct CUDA version
# pip uninstall torch
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Check PATH includes CUDA
# export PATH=/usr/local/cuda-12.1/bin:$PATH
```

### Version Conflicts

```bash
# Error: package X requires Y<2.0 but you have Y==2.1

# Solution 1: Create fresh environment
python -m venv fresh-env
source fresh-env/bin/activate
pip install package-x  # Let it resolve dependencies

# Solution 2: Use pip-tools
pip install pip-tools
pip-compile requirements.in  # Resolves compatible versions
pip-sync requirements.txt
```

### Out of Memory

```python
# Error: CUDA out of memory

# Solution 1: Reduce batch size
batch_size = 16  # Instead of 32

# Solution 2: Use gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 3: Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)

# Solution 4: Clear cache
torch.cuda.empty_cache()
```

### Import Errors

```bash
# Error: No module named 'transformers'

# Solution 1: Check you're in right environment
which python
pip list | grep transformers

# Solution 2: Install in current environment
pip install transformers

# Solution 3: Kernel mismatch in Jupyter
# Restart kernel after installing
```

---

## Verifying Installation

### Complete Check Script

```python
def check_ai_setup():
    """Verify AI development environment."""
    
    # Basic imports
    print("Checking imports...")
    import numpy as np
    import pandas as pd
    print("✅ NumPy and Pandas")
    
    # PyTorch
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Transformers
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
    
    # LangChain
    import langchain
    print(f"✅ LangChain {langchain.__version__}")
    
    # OpenAI
    import openai
    print(f"✅ OpenAI {openai.__version__}")
    
    print("\n🎉 All packages installed correctly!")

if __name__ == "__main__":
    check_ai_setup()
```

---

## Hands-on Exercise

### Your Task

Create a complete AI development environment:

1. Create virtual environment
2. Install GPU-enabled PyTorch
3. Install transformers, langchain, openai
4. Verify all installations

<details>
<summary>✅ Solution</summary>

```bash
#!/bin/bash
# setup_ai_env.sh

# Create environment
python -m venv ai-env
source ai-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install AI packages
pip install transformers
pip install langchain langchain-openai
pip install openai
pip install numpy pandas matplotlib scikit-learn

# Create verification script
cat > verify.py << 'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

import transformers
print(f"Transformers: {transformers.__version__}")

import langchain
print(f"LangChain: {langchain.__version__}")

import openai
print(f"OpenAI: {openai.__version__}")

print("\n✅ Environment ready!")
EOF

# Run verification
python verify.py

# Save requirements
pip freeze > requirements.txt
echo "Requirements saved to requirements.txt"
```
</details>

---

## Summary

✅ **GPU installation** requires matching CUDA versions
✅ **Check compatibility** before installing packages
✅ **Use conda** for complex binary dependencies
✅ **Pin versions** in requirements.txt
✅ **Verify installation** with test scripts
✅ **Troubleshoot** with nvidia-smi and version checks

**Next:** [Development Environment](./05-development-environment.md)

---

## Further Reading

- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [TensorFlow GPU Guide](https://www.tensorflow.org/install/gpu)
- [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

<!-- 
Sources Consulted:
- PyTorch Docs: https://pytorch.org/get-started/locally/
-->
