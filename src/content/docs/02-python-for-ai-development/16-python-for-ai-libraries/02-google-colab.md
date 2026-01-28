---
title: "Google Colab"
---

# Google Colab

## Introduction

Google Colab provides free access to GPUs and TPUs for running Jupyter notebooks in the cloud. It's ideal for AI experimentation without local hardware.

### What We'll Cover

- Accessing free GPU/TPU
- Colab-specific features
- Mounting Google Drive
- Installing packages
- Sharing notebooks
- Understanding limitations

### Prerequisites

- Google account
- Jupyter basics

---

## Getting Started

### Access Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with Google account
3. Create new notebook or open existing

### Interface

```
┌─────────────────────────────────────────────┐
│ 📄 Untitled.ipynb  ⭐ │ File Edit View ...  │
├─────────────────────────────────────────────┤
│ ➕ Code  ➕ Text    │ 🔗 Connect ▼  RAM │ │
├─────────────────────────────────────────────┤
│ [ ] │ # Your code here                     │
│     │                                       │
└─────────────────────────────────────────────┘
```

---

## GPU/TPU Access

### Enable GPU Runtime

1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. Click Save

### Verify GPU

```python
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Output:**
```
CUDA Available: True
GPU: Tesla T4
```

### Check GPU Memory

```python
!nvidia-smi
```

### Enable TPU

```python
# For TensorFlow
import tensorflow as tf

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f"TPU: {tpu.cluster_spec().as_dict()['worker']}")
except ValueError:
    print("TPU not available")
```

---

## Colab-Specific Features

### Forms for Parameters

```python
#@title Model Configuration
learning_rate = 0.001 #@param {type:"number"}
batch_size = 32 #@param {type:"slider", min:8, max:128, step:8}
model_name = "bert-base" #@param ["bert-base", "bert-large", "roberta"]
use_gpu = True #@param {type:"boolean"}
```

### Interactive Widgets

```python
#@title Enter your API key { run: "auto", display-mode: "form" }
api_key = "" #@param {type:"string"}
```

### Collapsible Sections

```python
#@title Setup (click to expand)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

---

## Mounting Google Drive

### Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

A popup asks for authorization.

### Access Files

```python
# List Drive contents
!ls /content/drive/MyDrive/

# Read file from Drive
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/data/dataset.csv')

# Save to Drive
df.to_csv('/content/drive/MyDrive/results/output.csv')
```

### Unmount Drive

```python
drive.flush_and_unmount()
```

---

## Installing Packages

### pip Install

```python
!pip install transformers
!pip install langchain openai
!pip install -q package  # Quiet mode

# Specific versions
!pip install torch==2.0.0
```

### Install from GitHub

```python
!pip install git+https://github.com/user/repo.git
```

### Requirements File

```python
# Write requirements
%%writefile requirements.txt
transformers>=4.30.0
langchain>=0.1.0
openai>=1.0.0

# Install
!pip install -r requirements.txt
```

### Package Persistence

> **Note:** Packages are reset when runtime restarts. Add install commands at the top of your notebook.

---

## File Operations

### Upload Files

```python
from google.colab import files

uploaded = files.upload()

# Access uploaded file
import io
import pandas as pd

df = pd.read_csv(io.BytesIO(uploaded['data.csv']))
```

### Download Files

```python
from google.colab import files

# Save and download
df.to_csv('results.csv', index=False)
files.download('results.csv')
```

### Working with URLs

```python
!wget https://example.com/dataset.csv
!curl -O https://example.com/model.pt
```

---

## Sharing Notebooks

### Share Options

1. Click "Share" button (top right)
2. Options:
   - **View**: Others can see
   - **Comment**: Others can comment
   - **Edit**: Others can modify

### Save to GitHub

1. File → Save a copy in GitHub
2. Select repository and commit message

### Save to Drive

File → Save a copy in Drive

---

## Colab Limitations

### Resource Limits

| Resource | Free Tier | Colab Pro |
|----------|-----------|-----------|
| GPU access | Limited | Priority |
| Session length | ~12 hours | 24 hours |
| RAM | ~12 GB | 25-52 GB |
| GPU type | T4 | T4, V100, A100 |
| Idle timeout | 90 min | 24 hours |

### Handling Disconnections

```python
# Save checkpoints frequently
import pickle

# Save model state
with open('/content/drive/MyDrive/checkpoint.pkl', 'wb') as f:
    pickle.dump(model_state, f)

# Load on reconnect
with open('/content/drive/MyDrive/checkpoint.pkl', 'rb') as f:
    model_state = pickle.load(f)
```

### Keep Alive (Not Recommended)

```javascript
// In browser console - but may violate ToS
function ClickConnect(){
    console.log("Keeping alive");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

> **Warning:** Using keep-alive scripts may result in account restrictions.

---

## Colab vs Local Jupyter

| Feature | Colab | Local Jupyter |
|---------|-------|---------------|
| Setup | None | Install required |
| GPU | Free (limited) | Your hardware |
| Persistence | Session-based | Permanent |
| Storage | Google Drive | Local disk |
| Collaboration | Built-in | Manual setup |
| Internet | Required | Optional |

---

## Hands-on Exercise

### Your Task

Create a Colab notebook that:

1. Mounts Google Drive
2. Checks for GPU availability
3. Installs transformers library
4. Loads a pretrained model
5. Saves results to Drive

<details>
<summary>✅ Solution</summary>

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Check GPU
import torch

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("⚠️ No GPU, using CPU")
    device = "cpu"

# Cell 3: Install packages
!pip install -q transformers

# Cell 4: Load model
from transformers import pipeline

classifier = pipeline("sentiment-analysis", device=0 if device == "cuda" else -1)
print("Model loaded!")

# Cell 5: Run inference
texts = [
    "I love this product!",
    "This is terrible.",
    "It's okay, nothing special."
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(f"{text}")
    print(f"  → {result['label']} ({result['score']:.2%})")
    print()

# Cell 6: Save results
import pandas as pd

df = pd.DataFrame({
    'text': texts,
    'label': [r['label'] for r in results],
    'confidence': [r['score'] for r in results]
})

output_path = '/content/drive/MyDrive/sentiment_results.csv'
df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
```
</details>

---

## Summary

✅ **Free GPU/TPU** with runtime settings
✅ **Mount Google Drive** for persistent storage
✅ **Install packages** with `!pip install`
✅ **Forms** create interactive parameters
✅ **Share notebooks** for collaboration
✅ **Session limits** require checkpoint saving

**Next:** [Essential AI Packages](./03-essential-ai-packages.md)

---

## Further Reading

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Colab Pro Features](https://colab.research.google.com/signup)

<!-- 
Sources Consulted:
- Google Colab: https://colab.research.google.com/
-->
