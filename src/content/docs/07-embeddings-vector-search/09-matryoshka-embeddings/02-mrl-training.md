---
title: "How MRL Training Works"
---

# How MRL Training Works

## Introduction

The magic of Matryoshka embeddings doesn't come from special architectures or post-processing—it comes from a specific training objective. **Matryoshka Representation Learning (MRL)** modifies the loss function to ensure that every prefix of an embedding is independently useful.

This lesson explores the training mechanism that creates coarse-to-fine embeddings, why it works mathematically, and how you can apply it to train your own Matryoshka models.

### What We'll Cover

- The multi-scale loss function that enables MRL
- Why early dimensions become information-dense
- Practical implementation using Sentence Transformers
- Comparison with standard training objectives

### Prerequisites

- Understanding of embedding training concepts
- Familiarity with loss functions (contrastive loss, triplet loss)
- Basic PyTorch or deep learning knowledge helpful

---

## The Core Insight: Multi-Scale Loss

### Standard Embedding Training

In conventional embedding training, we compute loss only on the full embedding:

```python
def standard_contrastive_loss(model, anchor, positive, negative):
    """Standard training: optimize only full embeddings."""
    emb_anchor = model.encode(anchor)      # [batch, 768]
    emb_positive = model.encode(positive)  # [batch, 768]
    emb_negative = model.encode(negative)  # [batch, 768]
    
    # Compute similarity on full embeddings only
    pos_sim = cosine_similarity(emb_anchor, emb_positive)
    neg_sim = cosine_similarity(emb_anchor, emb_negative)
    
    return contrastive_loss(pos_sim, neg_sim)
```

The model only receives gradients based on how well the *full* embedding separates positives from negatives. It has no incentive to organize information in any particular order.

### MRL: Loss at Multiple Dimensions

Matryoshka Representation Learning changes this by computing loss at multiple dimension cutoffs:

```python
def matryoshka_loss(model, anchor, positive, negative, dims=[64, 128, 256, 512, 768]):
    """MRL training: optimize at multiple dimension levels."""
    emb_anchor = model.encode(anchor)      # [batch, 768]
    emb_positive = model.encode(positive)  # [batch, 768]
    emb_negative = model.encode(negative)  # [batch, 768]
    
    total_loss = 0.0
    
    for dim in dims:
        # Truncate to current dimension
        a = emb_anchor[:, :dim]
        p = emb_positive[:, :dim]
        n = emb_negative[:, :dim]
        
        # Normalize truncated vectors
        a = F.normalize(a, dim=-1)
        p = F.normalize(p, dim=-1)
        n = F.normalize(n, dim=-1)
        
        # Compute loss at this dimension level
        pos_sim = (a * p).sum(dim=-1)
        neg_sim = (a * n).sum(dim=-1)
        
        loss_at_dim = contrastive_loss(pos_sim, neg_sim)
        total_loss += loss_at_dim
    
    return total_loss / len(dims)
```

Now the model receives gradients from *every* dimension level. To minimize total loss, it must make each prefix independently effective.

---

## Why Early Dimensions Become Dense

### The Gradient Flow

Consider what happens during backpropagation:

```
Dimension:     1-64   65-128  129-256  257-512  513-768
               ────── ─────── ──────── ──────── ────────

Gradient from 64-dim loss:    ✓       ✗        ✗        ✗        ✗
Gradient from 128-dim loss:   ✓       ✓        ✗        ✗        ✗
Gradient from 256-dim loss:   ✓       ✓        ✓        ✗        ✗
Gradient from 512-dim loss:   ✓       ✓        ✓        ✓        ✗
Gradient from full loss:      ✓       ✓        ✓        ✓        ✓
                             ─────   ─────    ─────    ─────    ─────
Total gradient signals:        5       4        3        2        1
```

The first 64 dimensions receive gradients from *all* loss terms, while the last 256 dimensions only receive gradients from the full-dimension loss. This asymmetry creates pressure:

- **Early dimensions** must work hard to satisfy multiple objectives
- **Later dimensions** only need to improve upon what earlier dimensions established

The result: early dimensions become packed with the most discriminative information.

### Information Theory Perspective

From an information theory viewpoint, MRL encourages the model to encode a **rate-distortion optimal** representation:

- At low dimensions, capture the most important bits (broad semantics)
- Each additional dimension adds progressively finer detail
- This is similar to how JPEG progressive encoding works for images

---

## Implementation with Sentence Transformers

The `sentence-transformers` library provides a ready-to-use `MatryoshkaLoss` wrapper:

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from datasets import Dataset

# Load base model
model = SentenceTransformer("microsoft/mpnet-base")

# Define base loss (any contrastive loss works)
base_loss = losses.MultipleNegativesRankingLoss(model)

# Wrap with MatryoshkaLoss
matryoshka_loss = losses.MatryoshkaLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=[768, 512, 256, 128, 64],  # Dimension checkpoints
    matryoshka_weights=[1, 1, 1, 1, 1]          # Optional: weight each level
)

# Prepare training data
train_dataset = Dataset.from_dict({
    "anchor": ["What is machine learning?", "How do neural networks work?"],
    "positive": ["ML is a subset of AI that learns from data.", "Neural nets are inspired by biological neurons."],
})

# Train
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=matryoshka_loss,
)
trainer.train()
```

### Choosing Dimension Checkpoints

The `matryoshka_dims` parameter defines where loss is computed:

| Strategy | Dimensions | Use Case |
|----------|------------|----------|
| Powers of 2 | [768, 512, 256, 128, 64] | General purpose |
| Dense low-end | [768, 384, 192, 96, 48, 24] | Optimized for small embeddings |
| Sparse | [768, 256, 64] | Faster training, less fine-grained |
| Custom | [768, 512, 384, 256] | Optimized for specific target dimensions |

> **Tip:** Include your expected deployment dimension in the list. If you plan to use 256-dim embeddings in production, ensure 256 is a checkpoint.

---

## Weighted Matryoshka Loss

Not all dimensions need equal weight. You can emphasize certain scales:

```python
# Emphasize smaller dimensions for aggressive compression
matryoshka_loss = losses.MatryoshkaLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=[768, 512, 256, 128, 64],
    matryoshka_weights=[1.0, 1.0, 1.5, 2.0, 2.0]  # Higher weight on small dims
)
```

Weight strategies:

| Goal | Weight Pattern | Example |
|------|----------------|---------|
| Balanced | All 1s | [1, 1, 1, 1, 1] |
| Prioritize small | Increasing | [1, 1, 1.5, 2, 2.5] |
| Prioritize full | Decreasing | [2, 1.5, 1, 0.5, 0.5] |
| Focus on 256 | Spike | [1, 1, 3, 1, 1] |

---

## 2D Matryoshka: Dimensions + Layers

Beyond dimension reduction, **Matryoshka2dLoss** also enables *layer* reduction:

```python
from sentence_transformers.losses import Matryoshka2dLoss

# Combine dimension AND layer reduction
loss = Matryoshka2dLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=[768, 512, 256, 128, 64]
)
```

This allows you to:
1. Use fewer transformer layers for faster inference
2. Use fewer dimensions for smaller embeddings

The result is a 2D tradeoff space:

```
           Layers →
        │  12    8     4    
        │ ───── ───── ─────
   768  │ Best  Good  OK
Dims 512│ Good  Good  OK
  ↓  256│ Good  OK    Fast
     128│ OK    Fast  Fastest
```

---

## Training Dynamics

### What Happens During Training

1. **Early epochs**: Model struggles to satisfy all dimension levels
2. **Mid training**: Lower dimensions stabilize first (fewer parameters to optimize)
3. **Late training**: Higher dimensions refine upon lower-dimension foundations

You can monitor this by tracking loss at each dimension level separately:

```python
# Monitor training progress per dimension
def log_matryoshka_metrics(model, eval_data, dims=[64, 128, 256, 512, 768]):
    """Log quality metrics at each dimension level."""
    metrics = {}
    for dim in dims:
        model.truncate_dim = dim  # Temporarily set truncation
        score = evaluate_retrieval(model, eval_data)
        metrics[f"recall@10_dim{dim}"] = score
    return metrics
```

### Expected Training Curves

```
Loss
 │
 │   ****                      64-dim loss (stabilizes first)
 │  *    ****
 │ *         ****
 │*              ****  ─────── 128-dim loss
 │                   ****
 │                       **** 256-dim loss
 │                           ****
 │                               **** Full-dim loss (last to stabilize)
 └────────────────────────────────────── Epochs
```

---

## Comparison: MRL vs. Other Approaches

### MRL vs. Post-hoc PCA

| Aspect | MRL (Matryoshka) | PCA |
|--------|------------------|-----|
| When applied | During training | After training |
| Information structure | Learned hierarchy | Statistical projection |
| Computation | No extra cost at inference | Requires matrix multiplication |
| Quality at 256 dims | ~98-99% of full | ~85-95% of full |
| Model-specific | Yes | No (any embedding) |

### MRL vs. Quantization

| Aspect | MRL | Quantization |
|--------|-----|--------------|
| What it reduces | Number of dimensions | Precision per dimension |
| Typical savings | 2-8x | 4-8x |
| Can combine? | ✅ Yes! | ✅ Yes! |
| Quality impact | Gradual degradation | Step-function degradation |

Combining both:
```
Full: 768 dims × 32-bit = 98,304 bits per vector
Matryoshka (256 dims): 256 × 32-bit = 32,768 bits (3x smaller)
+ Quantization (8-bit): 256 × 8-bit = 2,048 bits (48x smaller!)
```

---

## Advanced: Custom MRL Training Loop

For maximum control, implement the training loop directly:

```python
import torch
import torch.nn.functional as F

class MatryoshkaTrainer:
    def __init__(self, model, dims=[768, 512, 256, 128, 64], weights=None):
        self.model = model
        self.dims = dims
        self.weights = weights or [1.0] * len(dims)
    
    def compute_loss(self, anchors, positives, negatives):
        """Compute multi-scale contrastive loss."""
        # Get full embeddings
        emb_a = self.model.encode(anchors, convert_to_tensor=True)
        emb_p = self.model.encode(positives, convert_to_tensor=True)
        emb_n = self.model.encode(negatives, convert_to_tensor=True)
        
        total_loss = 0.0
        
        for dim, weight in zip(self.dims, self.weights):
            # Truncate and normalize
            a = F.normalize(emb_a[:, :dim], dim=-1)
            p = F.normalize(emb_p[:, :dim], dim=-1)
            n = F.normalize(emb_n[:, :dim], dim=-1)
            
            # Compute similarities
            pos_sim = torch.sum(a * p, dim=-1)
            neg_sim = torch.sum(a * n, dim=-1)
            
            # Margin-based loss
            loss = F.relu(0.2 - pos_sim + neg_sim).mean()
            
            total_loss += weight * loss
        
        return total_loss / sum(self.weights)
    
    def train_step(self, batch, optimizer):
        """Single training step."""
        optimizer.zero_grad()
        loss = self.compute_loss(batch["anchor"], batch["positive"], batch["negative"])
        loss.backward()
        optimizer.step()
        return loss.item()
```

---

## Common Training Pitfalls

### ❌ Forgetting to Normalize After Truncation

The model outputs normalized full embeddings, but truncation breaks normalization:

```python
# WRONG: Truncated vector has ||v|| < 1
truncated = full_embedding[:256]  # Not unit length!

# CORRECT: Re-normalize after truncation
truncated = F.normalize(full_embedding[:256], dim=-1)
```

### ❌ Too Few Dimension Checkpoints

```python
# WRONG: Model won't optimize well for dimensions between checkpoints
dims = [768, 64]  # Nothing between!

# BETTER: Include intermediate points
dims = [768, 512, 256, 128, 64]
```

### ❌ Imbalanced Weights Causing Instability

```python
# WRONG: Extreme weights cause training instability
weights = [0.1, 0.1, 0.1, 5.0, 10.0]

# BETTER: Moderate variation
weights = [1.0, 1.0, 1.2, 1.5, 1.5]
```

---

## Summary

✅ **MRL computes loss at multiple dimension cutoffs** during training  
✅ **Early dimensions receive more gradient signals**, becoming information-dense  
✅ **Sentence Transformers provides `MatryoshkaLoss`** for easy implementation  
✅ **Dimension checkpoints should include your target deployment dimension**  
✅ **Weights can emphasize specific dimension levels** based on your use case  
✅ **2D Matryoshka** extends the concept to layer reduction

---

## Hands-On Exercise

### Your Task

Train a small Matryoshka model and compare it to a standard model:

1. Use the same base model (e.g., `microsoft/mpnet-base`)
2. Train one version with `MatryoshkaLoss`, one with standard loss
3. Evaluate both at 64, 128, 256, 512, and 768 dimensions
4. Plot the quality curves and observe the difference

### Expected Results

- Standard model: Quality drops sharply below full dimensions
- Matryoshka model: Quality degrades gracefully, maintaining ~95%+ at 256 dims

<details>
<summary>✅ Solution Approach</summary>

```python
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer
from datasets import load_dataset

# Load training data (e.g., NLI dataset)
dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train[:10000]")

# Model 1: Standard training
model_std = SentenceTransformer("microsoft/mpnet-base")
loss_std = losses.TripletLoss(model_std)
trainer_std = SentenceTransformerTrainer(model=model_std, train_dataset=dataset, loss=loss_std)
trainer_std.train()

# Model 2: Matryoshka training
model_mrl = SentenceTransformer("microsoft/mpnet-base")
base_loss = losses.TripletLoss(model_mrl)
loss_mrl = losses.MatryoshkaLoss(model_mrl, base_loss, [768, 512, 256, 128, 64])
trainer_mrl = SentenceTransformerTrainer(model=model_mrl, train_dataset=dataset, loss=loss_mrl)
trainer_mrl.train()

# Evaluate both at different dimensions
# ... (evaluation code)
```

</details>

---

**Next:** [Supported Models →](./03-supported-models.md)

---

<!-- 
Sources Consulted:
- arXiv 2205.13147: Matryoshka Representation Learning
- Sentence Transformers MatryoshkaLoss: https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss
- Sentence Transformers Matryoshka training examples: https://sbert.net/examples/training/matryoshka/README.html
-->
