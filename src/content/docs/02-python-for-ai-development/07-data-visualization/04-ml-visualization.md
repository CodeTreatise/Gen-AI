---
title: "Visualization for AI/ML"
---

# Visualization for AI/ML

## Introduction

Effective visualization is crucial at every stage of machine learning—from understanding your data to evaluating model performance. This lesson covers essential ML visualization techniques.

### What We'll Cover

- Confusion matrices
- ROC curves and AUC
- Learning curves
- Feature importance
- Correlation analysis
- Model comparison

### Prerequisites

- Matplotlib and Seaborn basics
- Basic ML concepts

---

## Confusion Matrix

### Basic Confusion Matrix

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Sample predictions
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                               display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix')
plt.show()
```

### Multi-class Confusion Matrix

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Multi-class example
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 1]
y_pred = [0, 2, 2, 0, 1, 1, 0, 1, 2, 0]

cm = confusion_matrix(y_true, y_pred)
labels = ['Class A', 'Class B', 'Class C']

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', ax=ax, values_format='d')
ax.set_title('Multi-class Confusion Matrix')
plt.show()
```

### Normalized Confusion Matrix

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]

# Normalize by row (true labels)
cm = confusion_matrix(y_true, y_pred, normalize='true')

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues', ax=ax, values_format='.2%')
ax.set_title('Normalized Confusion Matrix')
plt.show()
```

---

## ROC Curves and AUC

### Single ROC Curve

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# Sample probability predictions
y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
y_scores = [0.1, 0.3, 0.6, 0.8, 0.2, 0.7, 0.4, 0.9, 0.75, 0.35]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, 
        label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
plt.show()
```

### Multiple ROC Curves

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# Simulate different models
np.random.seed(42)
y_true = np.random.randint(0, 2, 100)

models = {
    'Model A': y_true * 0.7 + np.random.rand(100) * 0.3,
    'Model B': y_true * 0.5 + np.random.rand(100) * 0.5,
    'Model C': y_true * 0.3 + np.random.rand(100) * 0.7
}

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['darkorange', 'green', 'blue']
for (name, scores), color in zip(models.items(), colors):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, 
            label=f'{name} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.50)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Model Comparison')
ax.legend(loc='lower right')
plt.show()
```

---

## Learning Curves

### Training and Validation Loss

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulate training history
epochs = 50
train_loss = 1.0 * np.exp(-np.arange(epochs) / 10) + np.random.rand(epochs) * 0.1
val_loss = 1.2 * np.exp(-np.arange(epochs) / 15) + np.random.rand(epochs) * 0.15

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, epochs + 1), train_loss, label='Training Loss')
ax.plot(range(1, epochs + 1), val_loss, label='Validation Loss')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Learning Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### Loss and Accuracy Combined

```python
import matplotlib.pyplot as plt
import numpy as np

epochs = 50
train_loss = 1.0 * np.exp(-np.arange(epochs) / 10) + 0.1
val_loss = 1.2 * np.exp(-np.arange(epochs) / 15) + 0.15
train_acc = 1 - train_loss * 0.8
val_acc = 1 - val_loss * 0.7

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(range(1, epochs + 1), train_loss, label='Train')
ax1.plot(range(1, epochs + 1), val_loss, label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy
ax2.plot(range(1, epochs + 1), train_acc, label='Train')
ax2.plot(range(1, epochs + 1), val_acc, label='Validation')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Curves')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Detecting Overfitting

```python
import matplotlib.pyplot as plt
import numpy as np

epochs = 100
# Overfitting scenario
train_loss = 1.0 * np.exp(-np.arange(epochs) / 10) + 0.01
val_loss = np.concatenate([
    1.2 * np.exp(-np.arange(30) / 15),
    0.3 + np.arange(70) * 0.005
])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, epochs + 1), train_loss, label='Training Loss')
ax.plot(range(1, epochs + 1), val_loss, label='Validation Loss')

# Mark overfitting point
ax.axvline(x=30, color='red', linestyle='--', alpha=0.5)
ax.annotate('Overfitting begins',
            xy=(30, val_loss[30]),
            xytext=(45, 0.5),
            arrowprops=dict(arrowstyle='->', color='red'))

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Overfitting Detection')
ax.legend()
plt.show()
```

---

## Feature Importance

### Bar Chart

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulated feature importances
features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 
            'Feature E', 'Feature F', 'Feature G', 'Feature H']
importances = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]

# Sort by importance
sorted_idx = np.argsort(importances)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([features[i] for i in sorted_idx], 
        [importances[i] for i in sorted_idx],
        color='steelblue')

ax.set_xlabel('Importance')
ax.set_title('Feature Importance')
plt.tight_layout()
plt.show()
```

### Permutation Importance with Error Bars

```python
import matplotlib.pyplot as plt
import numpy as np

features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
importances = [0.25, 0.20, 0.15, 0.12, 0.10]
std = [0.03, 0.02, 0.04, 0.02, 0.01]

sorted_idx = np.argsort(importances)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([features[i] for i in sorted_idx], 
        [importances[i] for i in sorted_idx],
        xerr=[std[i] for i in sorted_idx],
        color='steelblue',
        capsize=5)

ax.set_xlabel('Importance')
ax.set_title('Feature Importance (with std)')
plt.tight_layout()
plt.show()
```

---

## Correlation Analysis

### Correlation Heatmap

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Create sample data
np.random.seed(42)
n = 100
data = pd.DataFrame({
    'feature_a': np.random.randn(n),
    'feature_b': np.random.randn(n),
    'feature_c': np.random.randn(n),
    'feature_d': np.random.randn(n),
    'target': np.random.randn(n)
})
data['feature_e'] = data['feature_a'] * 0.8 + np.random.randn(n) * 0.2

corr = data.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle mask
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```

---

## Model Comparison

### Performance Metrics Bar Chart

```python
import matplotlib.pyplot as plt
import numpy as np

models = ['Logistic Reg', 'Random Forest', 'XGBoost', 'Neural Net']
accuracy = [0.82, 0.88, 0.91, 0.89]
precision = [0.80, 0.86, 0.90, 0.87]
recall = [0.78, 0.85, 0.88, 0.86]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, accuracy, width, label='Accuracy')
bars2 = ax.bar(x, precision, width, label='Precision')
bars3 = ax.bar(x + width, recall, width, label='Recall')

ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0.7, 1.0)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
```

---

## Hands-on Exercise

### Your Task

```python
# Create a comprehensive ML evaluation dashboard:
# 1. Confusion matrix (top-left)
# 2. ROC curve (top-right)
# 3. Learning curves (bottom-left)
# 4. Feature importance (bottom-right)
#
# Use simulated data or sklearn's make_classification
```

<details>
<summary>✅ Solution</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

np.random.seed(42)

# Simulated data
y_true = np.random.randint(0, 2, 100)
y_pred = (y_true * 0.7 + np.random.rand(100) * 0.3 > 0.5).astype(int)
y_scores = y_true * 0.7 + np.random.rand(100) * 0.3

# Training history
epochs = 50
train_loss = 1.0 * np.exp(-np.arange(epochs) / 10) + 0.1
val_loss = 1.2 * np.exp(-np.arange(epochs) / 15) + 0.15

# Feature importance
features = ['Age', 'Income', 'Credit Score', 'Tenure', 'Balance']
importances = [0.25, 0.22, 0.18, 0.20, 0.15]

# Create dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                               display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
axes[0, 1].plot(fpr, tpr, 'darkorange', lw=2, 
                label=f'ROC (AUC = {roc_auc:.2f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend()

# 3. Learning Curves
axes[1, 0].plot(range(1, epochs + 1), train_loss, label='Train')
axes[1, 0].plot(range(1, epochs + 1), val_loss, label='Validation')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Learning Curves')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Feature Importance
sorted_idx = np.argsort(importances)
axes[1, 1].barh([features[i] for i in sorted_idx], 
                [importances[i] for i in sorted_idx],
                color='steelblue')
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_title('Feature Importance')

plt.tight_layout()
fig.savefig('ml_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved as ml_dashboard.png")
```
</details>

---

## Summary

✅ **Confusion matrices** show classification performance details
✅ **ROC curves** with AUC compare model discrimination ability
✅ **Learning curves** detect overfitting and underfitting
✅ **Feature importance** identifies key predictors
✅ **Correlation heatmaps** reveal feature relationships
✅ Create **dashboards** for comprehensive model evaluation

**Next:** [Interactive Plotly](./05-interactive-plotly.md)

---

## Further Reading

- [Scikit-learn Visualization](https://scikit-learn.org/stable/visualizations.html)
- [Model Evaluation Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)

<!-- 
Sources Consulted:
- Scikit-learn: https://scikit-learn.org/stable/visualizations.html
-->
