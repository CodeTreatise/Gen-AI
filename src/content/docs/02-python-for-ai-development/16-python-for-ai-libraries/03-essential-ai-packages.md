---
title: "Essential AI Packages"
---

# Essential AI Packages

## Introduction

Python's AI ecosystem includes specialized libraries for numerical computing, data manipulation, machine learning, and LLM integration. This lesson covers the must-know packages.

### What We'll Cover

- NumPy for numerical computing
- Pandas for data manipulation
- Matplotlib/Seaborn for visualization
- Scikit-learn for ML algorithms
- Transformers for Hugging Face models
- LangChain and OpenAI for LLMs

### Prerequisites

- Python basics
- Virtual environments

---

## NumPy

### Purpose

Fast numerical operations on arrays and matrices.

### Installation

```bash
pip install numpy
```

### Basic Usage

```python
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Operations
print(arr * 2)        # [2, 4, 6, 8, 10]
print(arr.mean())     # 3.0
print(arr.std())      # 1.414...

# Random numbers
random_arr = np.random.randn(1000)

# Linear algebra
result = np.dot(matrix, matrix.T)
```

### Why It Matters for AI

```python
# Embeddings are numpy arrays
embeddings = np.random.randn(10, 768)  # 10 vectors, 768 dimensions

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## Pandas

### Purpose

Data manipulation and analysis with DataFrames.

### Installation

```bash
pip install pandas
```

### Basic Usage

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'score': [85, 92, 78]
})

# Read from files
df = pd.read_csv('data.csv')
df = pd.read_json('data.json')

# Basic operations
print(df.head())           # First 5 rows
print(df.describe())       # Statistics
print(df['score'].mean())  # Column mean

# Filtering
high_scores = df[df['score'] > 80]

# Grouping
by_age = df.groupby('age').mean()
```

### Why It Matters for AI

```python
# Prepare training data
train_df = pd.read_csv('training_data.csv')
texts = train_df['text'].tolist()
labels = train_df['label'].tolist()

# Analyze model outputs
results_df = pd.DataFrame({
    'input': inputs,
    'prediction': predictions,
    'confidence': confidences
})
results_df.to_csv('results.csv')
```

---

## Matplotlib & Seaborn

### Purpose

Data visualization and plotting.

### Installation

```bash
pip install matplotlib seaborn
```

### Basic Matplotlib

```python
import matplotlib.pyplot as plt

# Line plot
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Simple Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

# Histogram
plt.hist(data, bins=50)
plt.show()

# Save figure
plt.savefig('plot.png', dpi=300)
```

### Seaborn for Better Visuals

```python
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")

# Distribution plot
sns.histplot(data, kde=True)

# Heatmap (for confusion matrices)
sns.heatmap(confusion_matrix, annot=True, fmt='d')

# Scatter with categories
sns.scatterplot(data=df, x='x', y='y', hue='category')
```

---

## Scikit-learn

### Purpose

Machine learning algorithms and utilities.

### Installation

```bash
pip install scikit-learn
```

### Basic Usage

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
predictions = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))
```

### Common Use Cases

```python
# Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(embeddings)

# Dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(high_dim_data)

# Text vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(texts)
```

---

## Hugging Face Transformers

### Purpose

Access pretrained models for NLP, vision, and more.

### Installation

```bash
pip install transformers torch
# Or with TensorFlow
pip install transformers tensorflow
```

### Basic Usage

```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Text generation
generator = pipeline("text-generation", model="gpt2")
output = generator("Once upon a time", max_length=50)
print(output[0]['generated_text'])

# Question answering
qa = pipeline("question-answering")
result = qa(
    question="What is the capital of France?",
    context="France is a country in Europe. Paris is its capital."
)
print(result['answer'])  # Paris
```

### Loading Specific Models

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenize text
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
```

---

## LangChain

### Purpose

Build applications with LLMs.

### Installation

```bash
pip install langchain langchain-openai
```

### Basic Usage

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

# Create chain
chain = prompt | llm

# Run
response = chain.invoke({"question": "What is Python?"})
print(response.content)
```

### Chains and Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

response1 = conversation.predict(input="Hi, I'm Alice")
response2 = conversation.predict(input="What's my name?")
# Remembers: "Your name is Alice"
```

---

## OpenAI Python SDK

### Purpose

Direct access to OpenAI APIs.

### Installation

```bash
pip install openai
```

### Basic Usage

```python
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY env var

# Chat completion
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Python in one sentence."}
    ]
)

print(response.choices[0].message.content)
```

### Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Hello world"
)

embedding = response.data[0].embedding
print(f"Dimension: {len(embedding)}")  # 1536
```

---

## Package Comparison

| Package | Purpose | Install |
|---------|---------|---------|
| NumPy | Arrays, math | `numpy` |
| Pandas | DataFrames | `pandas` |
| Matplotlib | Plotting | `matplotlib` |
| Seaborn | Statistical viz | `seaborn` |
| Scikit-learn | ML algorithms | `scikit-learn` |
| Transformers | HuggingFace models | `transformers` |
| LangChain | LLM apps | `langchain` |
| OpenAI | OpenAI API | `openai` |

---

## Hands-on Exercise

### Your Task

Create a script that:
1. Loads data with pandas
2. Computes statistics with numpy
3. Visualizes with matplotlib
4. Runs sentiment analysis with transformers

<details>
<summary>✅ Solution</summary>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# 1. Create sample data
data = {
    'text': [
        "I love this product, it's amazing!",
        "Terrible experience, never again.",
        "It's okay, nothing special.",
        "Best purchase I've ever made!",
        "Complete waste of money."
    ],
    'rating': [5, 1, 3, 5, 1]
}
df = pd.DataFrame(data)

# 2. Basic statistics with numpy
ratings = np.array(df['rating'])
print(f"Mean rating: {ratings.mean():.2f}")
print(f"Std deviation: {ratings.std():.2f}")

# 3. Sentiment analysis
classifier = pipeline("sentiment-analysis")
df['sentiment'] = df['text'].apply(lambda x: classifier(x)[0]['label'])
df['confidence'] = df['text'].apply(lambda x: classifier(x)[0]['score'])

print("\nResults:")
print(df[['text', 'rating', 'sentiment', 'confidence']])

# 4. Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Rating distribution
axes[0].hist(df['rating'], bins=5, edgecolor='black')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Count')
axes[0].set_title('Rating Distribution')

# Sentiment distribution
sentiment_counts = df['sentiment'].value_counts()
axes[1].bar(sentiment_counts.index, sentiment_counts.values)
axes[1].set_xlabel('Sentiment')
axes[1].set_ylabel('Count')
axes[1].set_title('Sentiment Distribution')

plt.tight_layout()
plt.savefig('analysis.png')
print("\nPlot saved to analysis.png")
```
</details>

---

## Summary

✅ **NumPy** handles numerical arrays efficiently
✅ **Pandas** manages tabular data
✅ **Matplotlib/Seaborn** create visualizations
✅ **Scikit-learn** provides ML algorithms
✅ **Transformers** accesses pretrained models
✅ **LangChain/OpenAI** power LLM applications

**Next:** [Installing AI Packages](./04-installing-ai-packages.md)

---

## Further Reading

- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Hugging Face Docs](https://huggingface.co/docs)
- [LangChain Docs](https://python.langchain.com/docs/)

<!-- 
Sources Consulted:
- Package documentation sites
-->
