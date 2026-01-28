---
title: "Jupyter Notebooks"
---

# Jupyter Notebooks

## Introduction

Jupyter Notebooks provide an interactive environment for writing and executing Python code alongside documentation. They're the standard for data science and AI experimentation.

### What We'll Cover

- Installing Jupyter
- Notebook interface
- Code and markdown cells
- Kernel management
- Magic commands
- Exporting notebooks

### Prerequisites

- Python basics
- Virtual environments

---

## Installation

### Install Jupyter

```bash
# Create and activate virtual environment
python -m venv ai-env
source ai-env/bin/activate

# Install Jupyter Notebook
pip install notebook

# Or install JupyterLab (recommended)
pip install jupyterlab
```

### Start Jupyter

```bash
# Classic notebook
jupyter notebook

# JupyterLab (modern interface)
jupyter lab
```

The browser opens automatically to `http://localhost:8888`

---

## Notebook Interface

### Creating a Notebook

1. Click "New" → "Python 3"
2. Or File → New → Notebook

### Interface Components

```
┌─────────────────────────────────────────────┐
│ File  Edit  View  Insert  Cell  Kernel  ... │ ← Menu
├─────────────────────────────────────────────┤
│ 💾  ➕  ✂️  📋  ▶️  ⏹  ⟳  | Code ▼  │ ← Toolbar
├─────────────────────────────────────────────┤
│ In [1]: │ print("Hello, World!")         │   │
│         │                                 │   │ ← Code Cell
│ Out[1]: │ Hello, World!                   │   │
├─────────────────────────────────────────────┤
│ In [2]: │ # Your next cell...            │   │
└─────────────────────────────────────────────┘
```

---

## Cell Types

### Code Cells

```python
# This is a code cell
import numpy as np

data = np.array([1, 2, 3, 4, 5])
print(f"Mean: {data.mean()}")
print(f"Sum: {data.sum()}")
```

**Output:**
```
Mean: 3.0
Sum: 15
```

### Markdown Cells

```markdown
# Heading 1
## Heading 2

This is **bold** and *italic* text.

- Bullet point 1
- Bullet point 2

```python
# Code blocks in markdown
print("Formatted code")
```

### Cell Operations

| Shortcut | Action |
|----------|--------|
| `Shift+Enter` | Run cell, move to next |
| `Ctrl+Enter` | Run cell, stay in place |
| `Alt+Enter` | Run cell, insert below |
| `A` | Insert cell above (command mode) |
| `B` | Insert cell below (command mode) |
| `DD` | Delete cell (command mode) |
| `M` | Change to markdown |
| `Y` | Change to code |

---

## Kernel Management

### What is a Kernel?

The kernel is the Python process that executes your code.

### Kernel Operations

```
Kernel → Restart                 # Restart Python
Kernel → Restart & Clear Output  # Restart and clear
Kernel → Restart & Run All       # Fresh run of all cells
Kernel → Interrupt               # Stop running code
```

### Checking Kernel State

```python
# Cell execution order matters!
In [1]: x = 10
In [2]: y = 20
In [3]: print(x + y)  # 30

# If you run [3] before [1] and [2], you get NameError
```

### Multiple Kernels

```bash
# Install additional kernels
pip install ipykernel

# Add kernel to Jupyter
python -m ipykernel install --user --name=myenv --display-name="My Project"
```

---

## Magic Commands

### Line Magics (%)

```python
# Time single statement
%time sum(range(1000000))
# Output: CPU times: user 31.2 ms, sys: 0 ns, total: 31.2 ms

# Time with multiple runs
%timeit sum(range(1000))
# Output: 25.3 µs ± 234 ns per loop

# List variables
%who     # Names only
%whos    # With details

# Current directory
%pwd

# Change directory
%cd /path/to/directory

# Run Python file
%run script.py

# Load file into cell
%load script.py
```

### Cell Magics (%%)

```python
%%time
# Time entire cell
for i in range(1000):
    result = i ** 2
```

```python
%%writefile script.py
# Write cell contents to file
def hello():
    print("Hello from file!")
```

```python
%%bash
# Run bash commands
echo "Hello from bash"
ls -la
```

### Matplotlib Integration

```python
# Enable inline plotting
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.title("Sine Wave")
plt.show()
```

---

## Rich Output

### Displaying Data

```python
import pandas as pd

# DataFrames display nicely
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [85, 92, 78]
})
df  # Just the variable name shows formatted table
```

### Images

```python
from IPython.display import Image, display

# Display image from file
display(Image('chart.png'))

# Display from URL
display(Image(url='https://example.com/image.png'))
```

### HTML and Markdown

```python
from IPython.display import HTML, Markdown

display(HTML('<h1 style="color: blue;">Blue Heading</h1>'))
display(Markdown('**Bold** and *italic* text'))
```

---

## Exporting Notebooks

### Export Formats

```bash
# To Python script
jupyter nbconvert --to script notebook.ipynb

# To HTML
jupyter nbconvert --to html notebook.ipynb

# To PDF (requires LaTeX)
jupyter nbconvert --to pdf notebook.ipynb

# To Markdown
jupyter nbconvert --to markdown notebook.ipynb
```

### From JupyterLab

File → Export Notebook As → Choose format

### Clearing Outputs

```bash
# Clear all outputs before sharing
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

---

## Best Practices

### 1. Run Cells in Order

```python
# Always test with Kernel → Restart & Run All
# Ensures reproducibility
```

### 2. Use Markdown Sections

```markdown
# 1. Data Loading
Description of this section...

# 2. Data Preprocessing
Steps we'll take...

# 3. Model Training
Training approach...
```

### 3. Keep Cells Focused

```python
# ❌ Bad: Too much in one cell
data = load_data()
processed = preprocess(data)
model = train_model(processed)
results = evaluate(model)
plot_results(results)

# ✅ Good: One concept per cell
# Cell 1: Load data
data = load_data()

# Cell 2: Preprocess
processed = preprocess(data)

# Cell 3: Train model
model = train_model(processed)
```

---

## Hands-on Exercise

### Your Task

Create a Jupyter notebook that:

1. Imports numpy and creates an array
2. Calculates statistics (mean, std, min, max)
3. Creates a simple plot
4. Documents each step with markdown

<details>
<summary>✅ Solution</summary>

```markdown
# Cell 1 (Markdown)
# Data Analysis Example
This notebook demonstrates basic data analysis with NumPy.

# Cell 2 (Code)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Cell 3 (Markdown)
## Generate Sample Data

# Cell 4 (Code)
np.random.seed(42)
data = np.random.randn(1000)
print(f"Generated {len(data)} data points")

# Cell 5 (Markdown)
## Calculate Statistics

# Cell 6 (Code)
stats = {
    'Mean': data.mean(),
    'Std Dev': data.std(),
    'Min': data.min(),
    'Max': data.max()
}

for name, value in stats.items():
    print(f"{name}: {value:.4f}")

# Cell 7 (Markdown)
## Visualize Distribution

# Cell 8 (Code)
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(data.mean(), color='red', linestyle='--', label='Mean')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Random Data')
plt.legend()
plt.show()

# Cell 9 (Markdown)
## Conclusions
The data follows a normal distribution with mean ≈ 0 and std ≈ 1.
```
</details>

---

## Summary

✅ **Install Jupyter** with `pip install jupyterlab`
✅ **Code cells** execute Python
✅ **Markdown cells** document your work
✅ **Magic commands** (`%time`, `%%writefile`) extend functionality
✅ **Kernels** manage Python state
✅ **Export** to HTML, PDF, or Python scripts

**Next:** [Google Colab](./02-google-colab.md)

---

## Further Reading

- [Jupyter Documentation](https://jupyter.org/documentation)
- [JupyterLab Guide](https://jupyterlab.readthedocs.io/)
- [IPython Magic Commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html)

<!-- 
Sources Consulted:
- Jupyter Docs: https://jupyter.org/documentation
-->
