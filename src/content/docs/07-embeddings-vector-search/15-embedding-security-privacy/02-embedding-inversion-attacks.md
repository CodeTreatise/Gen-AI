---
title: "Embedding Inversion Attacks"
---

# Embedding Inversion Attacks

## Introduction

Can you recover the original text from an embedding? Research says yes—partially. Embedding inversion attacks demonstrate that embeddings leak significant information about their source text, challenging the assumption that vector representations provide any form of privacy protection.

This lesson examines the research on embedding inversion, what attackers can realistically recover, and effective defense strategies.

### What We'll Cover

- Research findings on embedding inversion
- How inversion attacks work
- What information is recoverable
- Defense strategies that actually work

### Prerequisites

- Understanding of [PII in embeddings](./01-pii-in-embeddings.md)
- Basic knowledge of machine learning

---

## The Research: Information Leakage in Embedding Models

### Key Paper: Song & Raghunathan (2020)

The seminal paper "Information Leakage in Embedding Models" demonstrated three classes of attacks:

```
┌─────────────────────────────────────────────────────────────────┐
│              Embedding Attack Classes                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CLASS 1: EMBEDDING INVERSION                                   │
│  ├── Goal: Recover input words from embedding                  │
│  ├── Method: Train decoder neural network                      │
│  ├── Result: 50-70% of words recovered (F1 0.5-0.7)           │
│  └── Implication: Significant text reconstruction possible     │
│                                                                 │
│  CLASS 2: ATTRIBUTE INFERENCE                                   │
│  ├── Goal: Infer sensitive attributes from embedding           │
│  ├── Method: Train classifier on labeled embeddings            │
│  ├── Result: High accuracy for authorship, demographics        │
│  └── Implication: Embeddings reveal personal characteristics   │
│                                                                 │
│  CLASS 3: MEMBERSHIP INFERENCE                                  │
│  ├── Goal: Determine if text was in training data             │
│  ├── Method: Analyze embedding patterns                        │
│  ├── Result: Above-chance detection for rare training data     │
│  └── Implication: Training data can be identified             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## How Inversion Attacks Work

### The Attack Model

```
┌─────────────────────────────────────────────────────────────────┐
│              Embedding Inversion Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ATTACKER SETUP:                                                │
│                                                                 │
│  1. Access to same embedding model (or similar)                │
│  2. Large corpus of text                                        │
│  3. Training compute                                            │
│                                                                 │
│  ATTACK PROCESS:                                                │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Train Data  │───▶│  Embedding  │───▶│   Decoder   │         │
│  │   Corpus    │    │    Model    │    │   Network   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
│  Training Phase:                                                │
│  • Create (text, embedding) pairs from public corpus           │
│  • Train decoder: embedding → bag of words                     │
│  • Decoder learns to predict words from vector positions       │
│                                                                 │
│  Attack Phase:                                                  │
│  • Obtain target embedding (stolen/leaked)                     │
│  • Feed through trained decoder                                │
│  • Recover predicted words                                     │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Stolen    │───▶│   Trained   │───▶│  Recovered  │         │
│  │  Embedding  │    │   Decoder   │    │    Words    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Simplified Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class EmbeddingInverter(nn.Module):
    """
    Neural network to invert embeddings to bag-of-words.
    Simplified for illustration.
    """
    def __init__(self, embedding_dim: int, vocab_size: int, hidden_dim: int = 1024):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid()  # Multi-label: which words are present
        )
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Predict which words are in the original text."""
        return self.network(embedding)

class InversionAttack:
    """
    Demonstrates embedding inversion attack.
    """
    def __init__(self, embedding_model, vocab: list):
        self.embed_model = embedding_model
        self.vocab = vocab
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.inverter = EmbeddingInverter(
            embedding_dim=embedding_model.get_embedding_dim(),
            vocab_size=len(vocab)
        )
    
    def train_inverter(self, training_texts: list, epochs: int = 10):
        """Train the inverter on (text, embedding) pairs."""
        optimizer = torch.optim.Adam(self.inverter.parameters())
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for text in training_texts:
                # Create embedding
                embedding = torch.tensor(self.embed_model.encode(text))
                
                # Create target: multi-hot vector of present words
                target = torch.zeros(len(self.vocab))
                for word in text.lower().split():
                    if word in self.word_to_idx:
                        target[self.word_to_idx[word]] = 1
                
                # Train step
                optimizer.zero_grad()
                prediction = self.inverter(embedding)
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(training_texts):.4f}")
    
    def invert(self, embedding: np.ndarray, top_k: int = 20) -> list:
        """Recover words from an embedding."""
        self.inverter.eval()
        with torch.no_grad():
            prediction = self.inverter(torch.tensor(embedding))
            
            # Get top-k predicted words
            top_indices = torch.argsort(prediction, descending=True)[:top_k]
            recovered_words = [self.vocab[i] for i in top_indices]
            
            return recovered_words

# The recovered words reveal significant content
# about the original text, even without exact ordering
```

---

## What Can Attackers Recover?

### Empirical Results

| Embedding Model | F1 Score | Precision | Recall | Notes |
|-----------------|----------|-----------|--------|-------|
| Skip-gram | 0.68 | 0.72 | 0.64 | Word embeddings |
| InferSent | 0.54 | 0.58 | 0.51 | Sentence embeddings |
| USE | 0.52 | 0.56 | 0.48 | Universal Sentence Encoder |
| BERT | 0.49 | 0.53 | 0.46 | Contextual embeddings |

**Key finding:** Simpler, older models are more vulnerable. But even BERT-based embeddings leak ~50% of words.

### What Information Leaks

```
┌─────────────────────────────────────────────────────────────────┐
│              Information Recovery by Type                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HIGH RECOVERY (60-80%):                                        │
│  • Content words (nouns, verbs, adjectives)                    │
│  • Topic-specific terminology                                   │
│  • Named entities (partially)                                   │
│  • Rare/distinctive words                                       │
│                                                                 │
│  MEDIUM RECOVERY (40-60%):                                      │
│  • Common function words                                        │
│  • Pronouns and articles                                        │
│  • Word order (limited)                                         │
│                                                                 │
│  LOW RECOVERY (20-40%):                                         │
│  • Exact phrasing                                               │
│  • Punctuation                                                  │
│  • Formatting                                                   │
│                                                                 │
│  NOT RECOVERED:                                                 │
│  • Exact character-level content                               │
│  • Word order preservation                                      │
│  • Paragraph structure                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Example Recovery

```
Original text:
"Dr. Sarah Johnson from Seattle General Hospital diagnosed 
patient John Smith with Type 2 Diabetes on March 15, 2024."

Recovered words (unordered):
["hospital", "diagnosed", "patient", "diabetes", "doctor", 
 "seattle", "general", "type", "march", "2024", "john", "smith"]

What an attacker learns:
• Medical context (hospital, diagnosed, patient, diabetes)
• Geographic location (seattle)
• Names mentioned (john, smith)
• Time reference (march, 2024)
• Condition type (diabetes, type)
```

---

## Attribute Inference Attacks

### Beyond Word Recovery

Embeddings also reveal attributes about the text author or content:

```python
def attribute_inference_attack(embedding: np.ndarray, classifier) -> dict:
    """
    Infer sensitive attributes from embedding.
    Attackers train classifiers on labeled embeddings.
    """
    attributes = {}
    
    # Gender inference
    attributes['gender'] = classifier['gender'].predict([embedding])[0]
    
    # Age range inference
    attributes['age_range'] = classifier['age'].predict([embedding])[0]
    
    # Location inference
    attributes['region'] = classifier['location'].predict([embedding])[0]
    
    # Political leaning (from writing style)
    attributes['political'] = classifier['political'].predict([embedding])[0]
    
    return attributes

# Research shows high accuracy for these inferences
# even without explicit mentions in the text
```

### Accuracy of Attribute Inference

| Attribute | Accuracy | Training Data Needed |
|-----------|----------|---------------------|
| Author identity (10 authors) | 95%+ | 100 samples/author |
| Author identity (100 authors) | 80%+ | 50 samples/author |
| Gender | 75-85% | 10,000 samples |
| Age range (±10 years) | 65-75% | 10,000 samples |
| Native language | 80-90% | 5,000 samples |

---

## Defense Strategies

### What Doesn't Work

| "Defense" | Why It Fails |
|-----------|--------------|
| Obfuscation | Attackers train on obfuscated data too |
| Dimensionality reduction | Semantic info preserved |
| Quantization | Attackers adapt to quantized embeddings |
| Hoping attackers won't try | Not a security strategy |

### What Actually Works

#### Defense 1: Access Control (Most Important)

```python
from functools import wraps

def require_embedding_access(func):
    """Decorator to enforce access control on embedding operations."""
    @wraps(func)
    def wrapper(user, *args, **kwargs):
        # Check user has embedding access permissions
        if not user.has_permission('embedding:read'):
            raise PermissionError("User lacks embedding access")
        
        # Log access for audit
        audit_log.record({
            'user': user.id,
            'action': 'embedding_access',
            'function': func.__name__,
            'timestamp': datetime.now()
        })
        
        return func(user, *args, **kwargs)
    return wrapper

@require_embedding_access
def retrieve_embeddings(user, document_ids: list) -> list:
    """Only authorized users can access raw embeddings."""
    return vector_db.get_embeddings(document_ids)
```

> **Key insight:** If embeddings can be stolen, they can be inverted. Prevent theft.

#### Defense 2: Never Expose Raw Embeddings

```python
class SecureVectorSearch:
    """
    Vector search without exposing raw embeddings.
    """
    def __init__(self, vector_db):
        self.db = vector_db
    
    def search(self, query: str, top_k: int = 10) -> list:
        """
        Search returns document IDs and scores only.
        Raw embeddings are never returned.
        """
        query_embedding = self._embed(query)
        
        # Search returns IDs and scores, NOT embeddings
        results = self.db.search(
            query_embedding,
            top_k=top_k,
            include_vectors=False  # Critical: never return vectors
        )
        
        return [
            {"id": r.id, "score": r.score, "metadata": r.metadata}
            for r in results
        ]
    
    def _embed(self, text: str) -> list:
        """Embedding happens server-side, never exposed."""
        return embedding_model.encode(text)

# Client never sees raw embeddings
# Only search results (IDs, scores, metadata)
```

#### Defense 3: Remove PII Before Embedding

```python
def secure_embedding_pipeline(text: str) -> dict:
    """
    Pipeline that removes PII before creating embeddings.
    """
    # Step 1: Extract and store PII separately
    pii_extracted = pii_extractor.extract(text)
    pii_id = secure_pii_store.store(pii_extracted)
    
    # Step 2: Sanitize text
    sanitized_text = pii_extractor.sanitize(text)
    
    # Step 3: Create embedding from sanitized text
    embedding = embedding_model.encode(sanitized_text)
    
    # Step 4: Store with reference to PII (not PII itself)
    return {
        "embedding": embedding,
        "sanitized_text": sanitized_text,
        "pii_reference": pii_id,  # Separate access control
        "original_hash": hash(text)  # For integrity verification
    }
```

#### Defense 4: Monitor for Exfiltration

```python
class EmbeddingExfiltrationMonitor:
    """
    Detect suspicious patterns that might indicate embedding theft.
    """
    def __init__(self, alert_threshold: int = 1000):
        self.access_counts = defaultdict(lambda: defaultdict(int))
        self.alert_threshold = alert_threshold
    
    def record_access(self, user_id: str, collection: str, count: int = 1):
        """Record embedding access."""
        today = datetime.now().date()
        self.access_counts[user_id][today] += count
        
        # Check for suspicious volume
        daily_total = self.access_counts[user_id][today]
        if daily_total > self.alert_threshold:
            self._alert_exfiltration(user_id, daily_total)
    
    def _alert_exfiltration(self, user_id: str, count: int):
        """Alert on potential exfiltration."""
        security_team.alert({
            "type": "embedding_exfiltration_suspected",
            "user_id": user_id,
            "access_count": count,
            "threshold": self.alert_threshold,
            "action_required": "review_and_revoke_if_necessary"
        })
```

#### Defense 5: Differential Privacy (Trade-off)

```python
import numpy as np

class DifferentiallyPrivateEmbedding:
    """
    Add calibrated noise for differential privacy.
    
    Trade-off: Privacy vs. utility (search quality degrades)
    """
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
    
    def privatize(self, embedding: np.ndarray) -> np.ndarray:
        """Add Laplacian noise for ε-differential privacy."""
        # Sensitivity: how much one record can change output
        sensitivity = 2.0  # Assuming normalized embeddings
        
        # Calibrate noise to privacy budget
        scale = sensitivity / self.epsilon
        
        # Add noise
        noise = np.random.laplace(0, scale, embedding.shape)
        private_embedding = embedding + noise
        
        # Re-normalize
        private_embedding = private_embedding / np.linalg.norm(private_embedding)
        
        return private_embedding
    
    def estimate_utility_loss(self) -> float:
        """Estimate search quality degradation."""
        # Rule of thumb: lower epsilon = more noise = worse search
        if self.epsilon >= 10:
            return 0.02  # ~2% quality loss
        elif self.epsilon >= 1:
            return 0.10  # ~10% quality loss
        else:
            return 0.30  # ~30% quality loss
```

---

## Defense Priority Matrix

| Defense | Effectiveness | Implementation Cost | Recommended For |
|---------|--------------|---------------------|-----------------|
| Access Control | High | Low | Everyone |
| Never Expose Raw Vectors | High | Low | Everyone |
| PII Removal Before Embedding | High | Medium | PII-handling systems |
| Exfiltration Monitoring | Medium | Medium | Enterprise |
| Differential Privacy | Medium | High | Research/Specialized |

---

## Summary

✅ **Inversion attacks can recover 50-70% of words** from embeddings  
✅ **Attribute inference reveals demographics, authorship, sentiment**  
✅ **Access control is the primary defense**—prevent embedding theft  
✅ **Never expose raw embeddings** through APIs or responses  
✅ **Remove PII before embedding** to limit what can be recovered  
✅ **Monitor for exfiltration** of large embedding volumes

---

**Next:** [Secure Storage Practices →](./03-secure-storage-practices.md)

---

<!-- 
Sources Consulted:
- Song & Raghunathan, "Information Leakage in Embedding Models" (2020): https://arxiv.org/abs/2004.00053
- Shokri et al., "Membership Inference Attacks Against Machine Learning Models" (2017)
- Carlini et al., "Extracting Training Data from Large Language Models" (2021)
-->
