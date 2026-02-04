---
title: "Template Versioning"
---

# Template Versioning

## Introduction

Prompts evolve. You'll tweak instructions, add examples, restructure output formats, and experiment with different approaches. Without proper versioning, you lose the ability to track changes, compare performance, roll back problems, and run controlled experiments. Template versioning applies software engineering practices to prompt management.

> **🔑 Key Insight:** Treat prompts like code. Version control, change tracking, and rollback capabilities aren't optional—they're essential for production systems.

### What We'll Cover

- Version numbering strategies
- Git-based prompt versioning
- Database-backed version storage
- Change tracking and audit logs
- Rollback mechanisms
- A/B testing across versions
- Deployment strategies

### Prerequisites

- [Prompt Libraries](./03-prompt-libraries.md)
- Basic Git knowledge
- Understanding of A/B testing concepts

---

## Version Numbering

### Semantic Versioning for Prompts

Adapt semantic versioning (SemVer) for prompts:

| Version Part | Increment When | Example |
|--------------|----------------|---------|
| **Major** (X.0.0) | Breaking changes to output format, structure, or behavior | 1.0.0 → 2.0.0 |
| **Minor** (0.X.0) | New capabilities, backward-compatible improvements | 1.0.0 → 1.1.0 |
| **Patch** (0.0.X) | Bug fixes, typo corrections, minor wording tweaks | 1.0.0 → 1.0.1 |

```yaml
# prompts/summarizer.yaml
name: summarizer
version: "2.1.3"

changelog:
  - version: "2.1.3"
    date: 2025-06-20
    type: patch
    changes:
      - Fixed typo in output format instruction
      
  - version: "2.1.0"
    date: 2025-06-15
    type: minor
    changes:
      - Added support for bullet-point format
      - Improved handling of technical content
      
  - version: "2.0.0"
    date: 2025-06-01
    type: major
    changes:
      - Changed output from plain text to JSON structure
      - Added confidence score to output
      - Breaking: Consumers must update parsing logic
```

### When to Create New Versions

```
┌─────────────────────────────────────────────────────────────┐
│                 VERSION DECISION TREE                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────┐
              │ Does output format change?  │
              └─────────────────────────────┘
                     │              │
                    Yes             No
                     │              │
                     ▼              ▼
              ┌───────────┐  ┌─────────────────────┐
              │ MAJOR     │  │ Does it add new     │
              │ version   │  │ capabilities?       │
              └───────────┘  └─────────────────────┘
                                   │          │
                                  Yes         No
                                   │          │
                                   ▼          ▼
                            ┌───────────┐  ┌───────────┐
                            │ MINOR     │  │ PATCH     │
                            │ version   │  │ version   │
                            └───────────┘  └───────────┘
```

---

## Git-Based Versioning

### Repository Structure

```
prompt-templates/
├── .github/
│   └── workflows/
│       ├── validate.yaml       # CI validation
│       └── deploy.yaml         # Deployment pipeline
├── prompts/
│   ├── production/             # Production-ready prompts
│   │   ├── summarizer.yaml
│   │   ├── classifier.yaml
│   │   └── translator.yaml
│   ├── staging/                # Testing prompts
│   └── experimental/           # In-development prompts
├── schemas/
│   └── prompt-schema.json      # Validation schema
├── tests/
│   └── prompt_tests.py         # Automated tests
├── CHANGELOG.md
└── README.md
```

### Git Tags for Releases

```bash
# Tag a release
git tag -a prompts-v2.1.0 -m "Release 2.1.0: Added bullet format support"
git push origin prompts-v2.1.0

# List all prompt versions
git tag --list "prompts-v*"

# Checkout specific version
git checkout prompts-v2.0.0
```

### Branch Strategy

```
main (production)
├── develop (integration)
│   ├── feature/summarizer-v2 (new feature)
│   ├── feature/classifier-improvements
│   └── fix/translator-edge-case
```

### Commit Message Convention

```
feat(summarizer): add bullet point output format

- Added new output_format variable: "bullets" | "paragraph"
- Updated examples for bullet format
- Backward compatible with existing usage

BREAKING CHANGE: None
Tested with: gpt-4, gpt-4-turbo, claude-3-5-sonnet
```

---

## Database-Backed Versioning

### Version Storage Schema

```python
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()

class PromptTemplate(Base):
    """Prompt template with version history."""
    __tablename__ = "prompt_templates"
    
    id = Column(String, primary_key=True)  # e.g., "summarizer"
    name = Column(String, nullable=False)
    description = Column(Text)
    current_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    versions = relationship("PromptVersion", back_populates="template")

class PromptVersion(Base):
    """Individual version of a prompt template."""
    __tablename__ = "prompt_versions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(String, ForeignKey("prompt_templates.id"))
    version = Column(String, nullable=False)  # e.g., "2.1.0"
    content = Column(Text, nullable=False)
    variables = Column(JSONB)
    metadata = Column(JSONB)
    
    # Tracking
    created_by = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    change_description = Column(Text)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_deprecated = Column(Boolean, default=False)
    deprecation_reason = Column(Text)
    replacement_version = Column(String)
    
    template = relationship("PromptTemplate", back_populates="versions")

class PromptUsageLog(Base):
    """Track prompt usage for analytics and rollback decisions."""
    __tablename__ = "prompt_usage_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(String)
    version = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Metrics
    latency_ms = Column(Integer)
    token_count = Column(Integer)
    success = Column(Boolean)
    error_message = Column(Text)
    
    # Context
    user_id = Column(String)
    environment = Column(String)  # production, staging, development
```

### Version Manager Class

```python
from typing import Optional, List
from sqlalchemy.orm import Session

class PromptVersionManager:
    """Manage prompt versions with database storage."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get template by ID."""
        return self.session.query(PromptTemplate).filter_by(id=template_id).first()
    
    def get_version(
        self, 
        template_id: str, 
        version: str = None
    ) -> Optional[PromptVersion]:
        """Get specific version or current version."""
        template = self.get_template(template_id)
        if not template:
            return None
        
        target_version = version or template.current_version
        
        return (
            self.session.query(PromptVersion)
            .filter_by(template_id=template_id, version=target_version)
            .first()
        )
    
    def create_version(
        self,
        template_id: str,
        version: str,
        content: str,
        variables: dict = None,
        created_by: str = None,
        change_description: str = None,
        set_as_current: bool = False
    ) -> PromptVersion:
        """Create a new version."""
        prompt_version = PromptVersion(
            template_id=template_id,
            version=version,
            content=content,
            variables=variables or {},
            created_by=created_by,
            change_description=change_description
        )
        
        self.session.add(prompt_version)
        
        if set_as_current:
            template = self.get_template(template_id)
            template.current_version = version
        
        self.session.commit()
        return prompt_version
    
    def set_current_version(
        self, 
        template_id: str, 
        version: str
    ) -> bool:
        """Set which version is current (active)."""
        template = self.get_template(template_id)
        version_exists = self.get_version(template_id, version)
        
        if template and version_exists:
            template.current_version = version
            template.updated_at = datetime.utcnow()
            self.session.commit()
            return True
        return False
    
    def rollback(
        self, 
        template_id: str, 
        target_version: str,
        reason: str
    ) -> bool:
        """Rollback to a previous version."""
        current = self.get_template(template_id)
        if not current:
            return False
        
        # Log the rollback
        rollback_log = {
            "from_version": current.current_version,
            "to_version": target_version,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Perform rollback
        return self.set_current_version(template_id, target_version)
    
    def deprecate_version(
        self,
        template_id: str,
        version: str,
        reason: str,
        replacement: str = None
    ) -> bool:
        """Mark a version as deprecated."""
        prompt_version = self.get_version(template_id, version)
        if prompt_version:
            prompt_version.is_deprecated = True
            prompt_version.deprecation_reason = reason
            prompt_version.replacement_version = replacement
            self.session.commit()
            return True
        return False
    
    def list_versions(
        self, 
        template_id: str,
        include_deprecated: bool = False
    ) -> List[PromptVersion]:
        """List all versions of a template."""
        query = (
            self.session.query(PromptVersion)
            .filter_by(template_id=template_id)
        )
        
        if not include_deprecated:
            query = query.filter_by(is_deprecated=False)
        
        return query.order_by(PromptVersion.created_at.desc()).all()
```

---

## Change Tracking

### Diff Generation

```python
import difflib
from dataclasses import dataclass
from typing import List

@dataclass
class VersionDiff:
    version_from: str
    version_to: str
    added_lines: List[str]
    removed_lines: List[str]
    unified_diff: str

def generate_diff(
    old_content: str, 
    new_content: str,
    old_version: str,
    new_version: str
) -> VersionDiff:
    """Generate diff between two prompt versions."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    # Unified diff
    unified = list(difflib.unified_diff(
        old_lines, 
        new_lines,
        fromfile=f"version-{old_version}",
        tofile=f"version-{new_version}"
    ))
    
    # Categorize changes
    added = [line[1:] for line in unified if line.startswith('+') and not line.startswith('+++')]
    removed = [line[1:] for line in unified if line.startswith('-') and not line.startswith('---')]
    
    return VersionDiff(
        version_from=old_version,
        version_to=new_version,
        added_lines=added,
        removed_lines=removed,
        unified_diff="".join(unified)
    )

# Usage
old = """You are a helpful assistant.
Respond concisely."""

new = """You are a helpful assistant.
Always be polite and professional.
Respond concisely and accurately."""

diff = generate_diff(old, new, "1.0.0", "1.1.0")
print(diff.unified_diff)
```

### Audit Log

```python
from enum import Enum
from datetime import datetime

class AuditAction(Enum):
    CREATED = "created"
    UPDATED = "updated"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"
    DEPRECATED = "deprecated"
    DELETED = "deleted"

class AuditLog:
    """Audit log for prompt changes."""
    
    def __init__(self, storage):
        self.storage = storage
    
    def log(
        self,
        template_id: str,
        version: str,
        action: AuditAction,
        user: str,
        details: dict = None,
        environment: str = "production"
    ):
        """Log an audit event."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "template_id": template_id,
            "version": version,
            "action": action.value,
            "user": user,
            "environment": environment,
            "details": details or {}
        }
        self.storage.append(entry)
        return entry
    
    def get_history(
        self,
        template_id: str,
        limit: int = 50
    ) -> List[dict]:
        """Get audit history for a template."""
        matching = [
            e for e in self.storage 
            if e["template_id"] == template_id
        ]
        return sorted(
            matching, 
            key=lambda x: x["timestamp"], 
            reverse=True
        )[:limit]

# Usage
audit = AuditLog(storage=[])

audit.log(
    template_id="summarizer",
    version="2.1.0",
    action=AuditAction.DEPLOYED,
    user="alice@example.com",
    details={"reason": "Improved accuracy on technical content"}
)

audit.log(
    template_id="summarizer",
    version="2.0.0",
    action=AuditAction.ROLLED_BACK,
    user="bob@example.com",
    details={"reason": "Output format broke downstream parsers"}
)
```

---

## Rollback Mechanisms

### Instant Rollback

```python
class RollbackManager:
    """Manage prompt rollbacks with safety checks."""
    
    def __init__(self, version_manager: PromptVersionManager, audit: AuditLog):
        self.versions = version_manager
        self.audit = audit
    
    def rollback(
        self,
        template_id: str,
        target_version: str,
        user: str,
        reason: str,
        dry_run: bool = False
    ) -> dict:
        """Rollback to a previous version with safety checks."""
        # Get current state
        template = self.versions.get_template(template_id)
        if not template:
            return {"success": False, "error": "Template not found"}
        
        current_version = template.current_version
        
        # Validate target version exists
        target = self.versions.get_version(template_id, target_version)
        if not target:
            return {"success": False, "error": f"Version {target_version} not found"}
        
        # Check if target is deprecated
        if target.is_deprecated:
            return {
                "success": False, 
                "error": f"Version {target_version} is deprecated",
                "replacement": target.replacement_version
            }
        
        # Dry run check
        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "from": current_version,
                "to": target_version,
                "diff": generate_diff(
                    self.versions.get_version(template_id, current_version).content,
                    target.content,
                    current_version,
                    target_version
                )
            }
        
        # Perform rollback
        self.versions.set_current_version(template_id, target_version)
        
        # Log rollback
        self.audit.log(
            template_id=template_id,
            version=target_version,
            action=AuditAction.ROLLED_BACK,
            user=user,
            details={
                "from_version": current_version,
                "reason": reason
            }
        )
        
        return {
            "success": True,
            "from": current_version,
            "to": target_version
        }

# Usage
rollback_manager = RollbackManager(version_manager, audit)

# Preview rollback
preview = rollback_manager.rollback(
    template_id="summarizer",
    target_version="2.0.0",
    user="admin@example.com",
    reason="Production issues",
    dry_run=True
)

# Execute rollback
result = rollback_manager.rollback(
    template_id="summarizer",
    target_version="2.0.0",
    user="admin@example.com",
    reason="Production issues",
    dry_run=False
)
```

---

## A/B Testing Versions

### Experiment Configuration

```python
from dataclasses import dataclass, field
from typing import Dict, Optional
import random
import hashlib

@dataclass
class Experiment:
    name: str
    template_id: str
    variants: Dict[str, float]  # version -> weight (must sum to 1.0)
    start_date: datetime
    end_date: Optional[datetime] = None
    is_active: bool = True
    metrics: Dict[str, list] = field(default_factory=dict)

class ExperimentManager:
    """Manage A/B testing of prompt versions."""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
    
    def create_experiment(
        self,
        name: str,
        template_id: str,
        variants: Dict[str, float]
    ) -> Experiment:
        """Create a new A/B experiment."""
        # Validate weights sum to 1.0
        if abs(sum(variants.values()) - 1.0) > 0.001:
            raise ValueError("Variant weights must sum to 1.0")
        
        experiment = Experiment(
            name=name,
            template_id=template_id,
            variants=variants,
            start_date=datetime.utcnow()
        )
        
        self.experiments[name] = experiment
        return experiment
    
    def assign_variant(
        self,
        experiment_name: str,
        user_id: str
    ) -> Optional[str]:
        """Assign user to a variant (deterministic)."""
        experiment = self.experiments.get(experiment_name)
        if not experiment or not experiment.is_active:
            return None
        
        # Deterministic assignment based on user_id
        hash_input = f"{experiment_name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 1000) / 1000  # 0.0 to 0.999
        
        cumulative = 0.0
        for version, weight in experiment.variants.items():
            cumulative += weight
            if bucket < cumulative:
                return version
        
        return list(experiment.variants.keys())[0]
    
    def record_metric(
        self,
        experiment_name: str,
        variant: str,
        metric_name: str,
        value: float
    ):
        """Record a metric for experiment analysis."""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return
        
        key = f"{variant}:{metric_name}"
        if key not in experiment.metrics:
            experiment.metrics[key] = []
        experiment.metrics[key].append(value)
    
    def get_results(self, experiment_name: str) -> Dict:
        """Get experiment results summary."""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return {}
        
        results = {}
        for key, values in experiment.metrics.items():
            variant, metric = key.split(":", 1)
            if variant not in results:
                results[variant] = {}
            results[variant][metric] = {
                "count": len(values),
                "mean": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0
            }
        
        return results

# Usage
exp_manager = ExperimentManager()

# Create experiment: 70% control, 30% new version
exp_manager.create_experiment(
    name="summarizer-tone-test",
    template_id="summarizer",
    variants={
        "2.0.0": 0.7,  # Control (current)
        "2.1.0": 0.3   # Treatment (new)
    }
)

# In request handler
user_id = "user_123"
version = exp_manager.assign_variant("summarizer-tone-test", user_id)

# After getting response, record metrics
exp_manager.record_metric("summarizer-tone-test", version, "latency_ms", 250)
exp_manager.record_metric("summarizer-tone-test", version, "user_rating", 4.5)

# Get results
results = exp_manager.get_results("summarizer-tone-test")
print(results)
```

---

## Deployment Strategies

### Gradual Rollout

```python
class GradualRollout:
    """Deploy new versions gradually to reduce risk."""
    
    STAGES = [
        {"name": "canary", "percentage": 5, "duration_hours": 2},
        {"name": "early_adopters", "percentage": 25, "duration_hours": 24},
        {"name": "general", "percentage": 50, "duration_hours": 24},
        {"name": "full", "percentage": 100, "duration_hours": None}
    ]
    
    def __init__(self, version_manager: PromptVersionManager):
        self.versions = version_manager
        self.rollouts: Dict[str, dict] = {}
    
    def start_rollout(
        self,
        template_id: str,
        new_version: str,
        old_version: str
    ) -> dict:
        """Start gradual rollout of new version."""
        self.rollouts[template_id] = {
            "new_version": new_version,
            "old_version": old_version,
            "current_stage": 0,
            "started_at": datetime.utcnow(),
            "stage_started_at": datetime.utcnow()
        }
        return self.rollouts[template_id]
    
    def get_version_for_user(
        self,
        template_id: str,
        user_id: str
    ) -> str:
        """Get version for user based on rollout stage."""
        rollout = self.rollouts.get(template_id)
        if not rollout:
            # No active rollout, use current version
            template = self.versions.get_template(template_id)
            return template.current_version if template else None
        
        stage = self.STAGES[rollout["current_stage"]]
        percentage = stage["percentage"]
        
        # Deterministic bucketing
        bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        
        if bucket < percentage:
            return rollout["new_version"]
        else:
            return rollout["old_version"]
    
    def advance_stage(self, template_id: str) -> Optional[dict]:
        """Move to next rollout stage."""
        rollout = self.rollouts.get(template_id)
        if not rollout:
            return None
        
        current = rollout["current_stage"]
        if current >= len(self.STAGES) - 1:
            # Already at full rollout
            return None
        
        rollout["current_stage"] = current + 1
        rollout["stage_started_at"] = datetime.utcnow()
        
        return {
            "template_id": template_id,
            "new_stage": self.STAGES[rollout["current_stage"]]["name"],
            "percentage": self.STAGES[rollout["current_stage"]]["percentage"]
        }
    
    def abort_rollout(self, template_id: str, reason: str) -> bool:
        """Abort rollout and revert to old version."""
        rollout = self.rollouts.get(template_id)
        if rollout:
            del self.rollouts[template_id]
            return True
        return False
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Semantic versioning | Clear expectations about changes |
| Immutable versions | Can always reproduce past behavior |
| Audit all changes | Accountability and debugging |
| Test before deploying | Catch issues before production |
| Gradual rollouts | Limit blast radius of problems |
| Keep rollback simple | Quick recovery from issues |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| No version history | Can't rollback | Store all versions immutably |
| Overwriting versions | Lost history | Create new version instead |
| No deprecation process | Old versions break | Mark deprecated with replacement |
| Manual deployments | Human error | Automate with CI/CD |
| No metrics per version | Can't compare | Instrument from day one |

---

## Summary

- Use semantic versioning: major.minor.patch
- Store versions in Git or database—never overwrite
- Track all changes with audit logs
- Make rollback fast and simple
- Use A/B testing to validate improvements
- Deploy gradually with canary and staged rollouts

**Next:** [Template Testing Frameworks](./07-template-testing-frameworks.md)

---

<!-- Sources: Software versioning best practices, feature flag and A/B testing patterns -->
