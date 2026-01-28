---
title: "Best Practices for AI Code"
---

# Best Practices for AI Code

## Introduction

AI projects require careful attention to reproducibility, organization, and documentation. These practices ensure your experiments are repeatable and your code is maintainable.

### What We'll Cover

- Reproducibility (seeds, versioning)
- Configuration management
- Experiment tracking concepts
- Code organization patterns
- Documentation standards

### Prerequisites

- Python basics
- Virtual environments

---

## Reproducibility

### Random Seeds

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Call at start of training
set_seed(42)
```

### Environment Versioning

```bash
# Save exact versions
pip freeze > requirements.txt

# Or with pip-tools
pip-compile requirements.in
```

```text
# requirements.txt (pinned)
torch==2.2.0
transformers==4.38.0
numpy==1.26.4
```

### Data Versioning

```python
# Hash your datasets
import hashlib

def get_data_hash(filepath: str) -> str:
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

data_hash = get_data_hash('data/train.csv')
print(f"Data version: {data_hash[:8]}")
```

---

## Configuration Management

### Configuration File

```yaml
# config.yaml
model:
  name: "bert-base-uncased"
  max_length: 512
  
training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 3
  seed: 42

data:
  train_path: "data/train.csv"
  val_path: "data/val.csv"
```

### Loading Config

```python
from dataclasses import dataclass
import yaml

@dataclass
class ModelConfig:
    name: str
    max_length: int

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    seed: int

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**data['model']),
            training=TrainingConfig(**data['training'])
        )

config = Config.from_yaml('config.yaml')
print(config.training.learning_rate)  # 2e-5
```

### Using Pydantic

```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class ModelConfig(BaseModel):
    name: str = "bert-base-uncased"
    max_length: int = 512

class TrainingConfig(BaseModel):
    batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: int = 3
    seed: int = 42

class Settings(BaseSettings):
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    openai_api_key: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Experiment Tracking

### Manual Logging

```python
import json
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.timestamp = datetime.now().isoformat()
        self.metrics = {}
        self.config = {}
        
        self.output_dir = Path(f"experiments/{experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def log_config(self, config: dict):
        self.config = config
    
    def log_metric(self, name: str, value: float, step: int = None):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        })
    
    def save(self):
        results = {
            "name": self.name,
            "timestamp": self.timestamp,
            "config": self.config,
            "metrics": self.metrics
        }
        
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

# Usage
logger = ExperimentLogger("bert-v1")
logger.log_config({"model": "bert-base", "lr": 2e-5})
logger.log_metric("loss", 0.5, step=1)
logger.log_metric("accuracy", 0.85, step=1)
logger.save()
```

### Popular Tracking Tools

| Tool | Best For |
|------|----------|
| **Weights & Biases** | Full experiment tracking |
| **MLflow** | Open source, self-hosted |
| **TensorBoard** | PyTorch/TensorFlow visualization |
| **DVC** | Data version control |

```python
# Example: Weights & Biases
import wandb

wandb.init(project="my-project", config=config)
wandb.log({"loss": 0.5, "accuracy": 0.85})
wandb.finish()
```

---

## Code Organization

### Project Structure

```
my-ai-project/
├── config/
│   ├── default.yaml
│   └── experiment1.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── checkpoints/
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   └── test_model.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── .env
├── .gitignore
├── pyproject.toml
└── README.md
```

### Modular Code

```python
# src/models/classifier.py
class TextClassifier:
    def __init__(self, model_name: str, num_labels: int):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    
    def predict(self, texts: list[str]) -> list[dict]:
        # Implementation
        pass

# src/training/trainer.py
class Trainer:
    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config
    
    def train(self, train_data, val_data):
        # Training loop
        pass

# scripts/train.py
from src.models.classifier import TextClassifier
from src.training.trainer import Trainer

def main():
    config = Config.from_yaml("config/default.yaml")
    model = TextClassifier("bert-base-uncased", num_labels=2)
    trainer = Trainer(model, config.training)
    trainer.train(train_data, val_data)
```

---

## Documentation

### Docstrings

```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epochs: int = 10,
    device: str = "cuda"
) -> dict[str, list[float]]:
    """Train a PyTorch model.
    
    Args:
        model: The neural network to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for updating weights.
        epochs: Number of training epochs. Defaults to 10.
        device: Device to train on. Defaults to "cuda".
    
    Returns:
        Dictionary containing training history with keys:
        - 'loss': List of epoch losses
        - 'accuracy': List of epoch accuracies
    
    Raises:
        ValueError: If epochs is less than 1.
    
    Example:
        >>> model = MyModel()
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> optimizer = Adam(model.parameters())
        >>> history = train_model(model, loader, optimizer)
        >>> print(history['loss'][-1])
    """
```

### README Template

```markdown
# Project Name

Brief description of what this project does.

## Installation

```bash
git clone https://github.com/user/project
cd project
pip install -r requirements.txt
```

## Usage

```python
from src.model import Classifier
model = Classifier.load("path/to/model")
result = model.predict("Hello world")
```

## Training

```bash
python scripts/train.py --config config/default.yaml
```

## Project Structure

- `src/` - Source code
- `notebooks/` - Jupyter notebooks
- `config/` - Configuration files
- `data/` - Data directory (not in git)

## License

MIT
```

---

## Best Practices Checklist

### Before Starting

- [ ] Create virtual environment
- [ ] Set up .gitignore
- [ ] Create configuration file
- [ ] Set random seeds
- [ ] Document environment versions

### During Development

- [ ] Use type hints
- [ ] Write docstrings
- [ ] Log experiments
- [ ] Save model checkpoints
- [ ] Test on small data first

### Before Sharing

- [ ] Clear notebook outputs
- [ ] Update requirements.txt
- [ ] Write README
- [ ] Remove hardcoded paths
- [ ] Check for leaked credentials

---

## Hands-on Exercise

### Your Task

Create a reproducible experiment setup:

1. Configuration management
2. Seed setting function
3. Basic experiment logger

<details>
<summary>✅ Solution</summary>

```python
# experiment_utils.py
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Configuration
@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 3
    model_name: str = "bert-base-uncased"

# Reproducibility
def set_seed(seed: int):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    print(f"🎲 Seeds set to {seed}")

# Experiment Logger
class ExperimentLogger:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{config.name}_{self.timestamp}"
        self.metrics = []
        
        self.output_dir = Path(f"experiments/{self.run_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self._save_config()
        print(f"📁 Experiment: {self.run_name}")
    
    def _save_config(self):
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def log(self, step: int, **kwargs):
        entry = {"step": step, **kwargs}
        self.metrics.append(entry)
        print(f"Step {step}: {kwargs}")
    
    def save_metrics(self):
        with open(self.output_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"💾 Metrics saved to {self.output_dir}")
    
    def save_model(self, model, name: str = "model.pt"):
        torch.save(model.state_dict(), self.output_dir / name)
        print(f"💾 Model saved to {self.output_dir / name}")

# Example usage
def main():
    # Setup
    config = ExperimentConfig(
        name="bert_classifier",
        seed=42,
        learning_rate=2e-5
    )
    
    set_seed(config.seed)
    logger = ExperimentLogger(config)
    
    # Simulate training
    for epoch in range(config.epochs):
        loss = 1.0 / (epoch + 1)  # Fake decreasing loss
        accuracy = 0.5 + epoch * 0.1  # Fake increasing accuracy
        logger.log(epoch, loss=loss, accuracy=accuracy)
    
    logger.save_metrics()
    print("\n✅ Experiment complete!")

if __name__ == "__main__":
    main()
```
</details>

---

## Summary

✅ **Set random seeds** for reproducibility
✅ **Pin package versions** in requirements
✅ **Use configuration files** not hardcoded values
✅ **Track experiments** with logging
✅ **Organize code** in modular structure
✅ **Document everything** with docstrings and README

**Back to:** [Python for AI Libraries Overview](./00-python-for-ai-libraries.md)

---

## Further Reading

- [Weights & Biases Guide](https://docs.wandb.ai/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Google's ML Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)

<!-- 
Sources Consulted:
- Various ML best practices guides
-->
