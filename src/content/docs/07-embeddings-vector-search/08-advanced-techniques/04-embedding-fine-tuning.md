---
title: "Embedding Fine-Tuning"
---

# Embedding Fine-Tuning

## Introduction

Pre-trained embedding models work well for general text, but fine-tuning on your domain data can significantly improve retrieval accuracy. Sentence Transformers v3 provides a streamlined training API that makes fine-tuning accessible.

> **🤖 AI Context:** Fine-tuning teaches the model that "customer churn" and "subscriber attrition" should be close in embedding space for your business context, even if a generic model doesn't understand your domain terminology.

---

## When to Fine-Tune

| Scenario | Fine-Tune? | Why |
|----------|------------|-----|
| General web search | No | Pre-trained models are optimized for this |
| Domain-specific jargon | Yes | Medical, legal, technical terms |
| Internal acronyms/terminology | Yes | Company-specific language |
| Low retrieval accuracy | Maybe | Test first, fine-tune if needed |
| Multilingual with rare languages | Yes | Improve underrepresented languages |

**Cost-benefit analysis:**
```
Fine-tuning cost:
├── Data preparation: 2-8 hours
├── Training: 1-4 hours GPU time
├── Evaluation: 1-2 hours
└── Total: ~$50-200 cloud compute

Potential benefit:
├── 5-15% retrieval accuracy improvement
├── Better domain understanding
└── Reduced need for query expansion
```

---

## Training Data Formats

Sentence Transformers supports multiple data formats:

### Pairs (Query, Positive)

```python
# Simple positive pairs - minimal but effective
pairs_data = [
    {"query": "What is machine learning?", 
     "positive": "Machine learning is a subset of AI that enables systems to learn from data."},
    {"query": "How do neural networks work?",
     "positive": "Neural networks are computing systems inspired by biological neural networks."},
]
```

### Triplets (Query, Positive, Negative)

```python
# Triplets with hard negatives - more powerful
triplets_data = [
    {
        "query": "Python async programming",
        "positive": "Asyncio provides infrastructure for writing single-threaded concurrent code using coroutines.",
        "negative": "Python is a programming language known for its simple syntax."  # Related but not relevant
    },
]
```

### Scored Pairs (Query, Document, Score)

```python
# Pairs with similarity scores 0-1
scored_data = [
    {"query": "machine learning", "document": "ML algorithms learn from data", "score": 0.9},
    {"query": "machine learning", "document": "Machines are used in factories", "score": 0.1},
]
```

---

## Dataset Preparation

```python
from datasets import Dataset

def prepare_training_data(examples: list[dict]) -> Dataset:
    """Prepare data for Sentence Transformers training."""
    
    # Convert to HuggingFace Dataset format
    if "negative" in examples[0]:
        # Triplet format
        dataset = Dataset.from_dict({
            "anchor": [e["query"] for e in examples],
            "positive": [e["positive"] for e in examples],
            "negative": [e["negative"] for e in examples],
        })
    else:
        # Pairs format
        dataset = Dataset.from_dict({
            "anchor": [e["query"] for e in examples],
            "positive": [e["positive"] for e in examples],
        })
    
    return dataset

# Generate hard negatives with BM25
from rank_bm25 import BM25Okapi

def add_hard_negatives(
    queries: list[str],
    positives: list[str],
    corpus: list[str],
    negatives_per_query: int = 1
) -> list[dict]:
    """Add hard negatives using BM25 to find similar but wrong docs."""
    
    # Build BM25 index
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    triplets = []
    positive_set = set(positives)
    
    for query, positive in zip(queries, positives):
        # Find documents similar to query but not the positive
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Sort by score descending
        ranked_indices = sorted(
            range(len(corpus)),
            key=lambda i: scores[i],
            reverse=True
        )
        
        # Pick top docs that aren't the positive
        negatives = []
        for idx in ranked_indices:
            if corpus[idx] not in positive_set and len(negatives) < negatives_per_query:
                negatives.append(corpus[idx])
        
        for neg in negatives:
            triplets.append({
                "query": query,
                "positive": positive,
                "negative": neg
            })
    
    return triplets
```

---

## Loss Functions

Choose the right loss function for your data:

| Loss Function | Data Format | Use Case |
|---------------|-------------|----------|
| `MultipleNegativesRankingLoss` | Pairs | Most common, uses in-batch negatives |
| `TripletLoss` | Triplets | When you have explicit negatives |
| `CoSENTLoss` | Scored pairs | When you have graded relevance |
| `ContrastiveLoss` | Binary pairs | Similar/dissimilar classification |

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import (
    MultipleNegativesRankingLoss,
    TripletLoss,
    CoSENTLoss
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Most common: Multiple Negatives Ranking Loss
# Uses other positives in the batch as negatives
mnrl_loss = MultipleNegativesRankingLoss(model)

# Triplet Loss: explicit anchor, positive, negative
triplet_loss = TripletLoss(model, distance_metric=TripletLoss.EUCLIDEAN)

# CoSENT Loss: for scored pairs
cosent_loss = CoSENTLoss(model)
```

---

## Training with Sentence Transformers v3

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator
from datasets import Dataset

# 1. Load base model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Prepare dataset
train_data = [
    {"anchor": "What is RAG?", "positive": "Retrieval Augmented Generation combines search with LLMs."},
    {"anchor": "How do embeddings work?", "positive": "Embeddings map text to dense vectors in semantic space."},
    # ... more examples
]
train_dataset = Dataset.from_dict({
    "anchor": [d["anchor"] for d in train_data],
    "positive": [d["positive"] for d in train_data],
})

# 3. Set up loss function
loss = MultipleNegativesRankingLoss(model)

# 4. Configure training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Use mixed precision if GPU supports it
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
)

# 5. Create trainer and train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)

trainer.train()

# 6. Save the fine-tuned model
model.save("./fine-tuned-model")
```

---

## Evaluation

```python
from sentence_transformers.evaluation import (
    TripletEvaluator,
    EmbeddingSimilarityEvaluator,
    InformationRetrievalEvaluator
)

# Triplet Evaluator - measures if positive is closer than negative
eval_triplets = Dataset.from_dict({
    "anchor": ["What is ML?", "How to train a model?"],
    "positive": ["Machine learning explanation", "Model training guide"],
    "negative": ["Cooking recipes", "Travel destinations"],
})

triplet_evaluator = TripletEvaluator(
    anchors=eval_triplets["anchor"],
    positives=eval_triplets["positive"],
    negatives=eval_triplets["negative"],
    name="eval"
)

# Information Retrieval Evaluator - measures retrieval metrics
from sentence_transformers.evaluation import InformationRetrievalEvaluator

queries = {"q1": "What is machine learning?", "q2": "How do neural networks work?"}
corpus = {
    "d1": "Machine learning is a subset of AI.",
    "d2": "Neural networks are inspired by the brain.",
    "d3": "Cooking requires ingredients.",
}
relevant_docs = {
    "q1": {"d1"},  # q1 is relevant to d1
    "q2": {"d2"},  # q2 is relevant to d2
}

ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="ir_eval"
)

# Run evaluation
triplet_score = triplet_evaluator(model)
ir_results = ir_evaluator(model)
```

---

## Practical Tips

### Data Quality > Quantity

```python
# 500 high-quality pairs beat 5000 noisy ones
def filter_quality_examples(examples: list[dict]) -> list[dict]:
    """Filter training examples for quality."""
    
    quality_examples = []
    
    for ex in examples:
        query = ex["query"]
        positive = ex["positive"]
        
        # Skip if too short
        if len(query.split()) < 3 or len(positive.split()) < 10:
            continue
        
        # Skip if query appears verbatim in positive (too easy)
        if query.lower() in positive.lower():
            continue
        
        quality_examples.append(ex)
    
    return quality_examples
```

### Start with Small Models

```python
# Start with smaller model for faster iteration
models_by_size = [
    "all-MiniLM-L6-v2",      # 22M params, fast
    "all-mpnet-base-v2",      # 109M params, balanced
    "e5-large-v2",            # 335M params, high quality
]

# Fine-tune small model first to validate data/approach
# Then scale up if needed
```

### Monitor Training

```python
# Add to training args for better monitoring
args = SentenceTransformerTrainingArguments(
    output_dir="./fine-tuned-model",
    # ... other args ...
    logging_steps=10,
    eval_steps=100,
    report_to="tensorboard",  # Or "wandb"
)
```

---

## When NOT to Fine-Tune

| Situation | Better Alternative |
|-----------|-------------------|
| < 500 training examples | Use better prompts/chunking |
| General domain | Use pre-trained model |
| Time-sensitive deployment | Query expansion instead |
| Rapidly changing content | Fine-tune on stable subset only |

---

## Summary

✅ **Fine-tune** when domain terminology differs from general language

✅ **Triplets with hard negatives** produce best results

✅ **MultipleNegativesRankingLoss** is most common for pairs

✅ **Quality > quantity** for training data

✅ **Start small**, validate approach, then scale up

**Next:** [Cross-Encoder Reranking](./05-cross-encoder-reranking.md)

---

<!-- 
Sources Consulted:
- Sentence Transformers Training: https://sbert.net/docs/sentence_transformer/training_overview.html
- HuggingFace Blog: https://huggingface.co/blog/train-sentence-transformers
-->
