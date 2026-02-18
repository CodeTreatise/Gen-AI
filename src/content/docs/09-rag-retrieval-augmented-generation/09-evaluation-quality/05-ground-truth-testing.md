---
title: "Ground Truth Testing"
---

# Ground Truth Testing

## Introduction

Ground truth testing uses curated datasets with known correct answers to measure RAG performance objectively. Unlike LLM-based evaluation, ground truth testing provides deterministic, reproducible metrics.

This lesson covers how to build, maintain, and use golden datasets for RAG evaluation.

### What We'll Cover

- Building golden datasets
- Expected answer formats
- Rubric-based scoring
- Regression testing for RAG

### Prerequisites

- Understanding of RAG evaluation concepts
- Familiarity with test-driven development
- Lessons 01-04 of this module

---

## Golden Datasets

A golden dataset contains questions with verified correct answers.

### Structure of a Golden Dataset

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

class QuestionType(Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    SUMMARIZATION = "summarization"

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class GoldenSample:
    """A single evaluation sample with ground truth."""
    
    id: str
    question: str
    expected_answer: str
    
    # Retrieval ground truth
    relevant_doc_ids: List[str]
    
    # Metadata
    question_type: QuestionType
    difficulty: Difficulty
    category: str
    
    # Optional
    acceptable_variations: List[str] = field(default_factory=list)
    key_facts: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    
    def matches_answer(self, answer: str, threshold: float = 0.8) -> bool:
        """Check if answer matches expected (basic check)."""
        # Normalize
        expected_lower = self.expected_answer.lower()
        answer_lower = answer.lower()
        
        # Exact match
        if expected_lower in answer_lower:
            return True
        
        # Check variations
        for variation in self.acceptable_variations:
            if variation.lower() in answer_lower:
                return True
        
        return False

@dataclass
class GoldenDataset:
    """Collection of golden samples for evaluation."""
    
    name: str
    description: str
    samples: List[GoldenSample]
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_by_type(self, question_type: QuestionType) -> List[GoldenSample]:
        return [s for s in self.samples if s.question_type == question_type]
    
    def get_by_difficulty(self, difficulty: Difficulty) -> List[GoldenSample]:
        return [s for s in self.samples if s.difficulty == difficulty]
    
    def get_by_category(self, category: str) -> List[GoldenSample]:
        return [s for s in self.samples if s.category == category]
    
    @property
    def stats(self) -> Dict:
        return {
            "total": len(self.samples),
            "by_type": {t.value: len(self.get_by_type(t)) for t in QuestionType},
            "by_difficulty": {d.value: len(self.get_by_difficulty(d)) for d in Difficulty}
        }

# Example dataset
dataset = GoldenDataset(
    name="Python Knowledge Base",
    description="Test set for Python documentation RAG",
    samples=[
        GoldenSample(
            id="py_001",
            question="Who created Python?",
            expected_answer="Guido van Rossum",
            relevant_doc_ids=["doc_python_history"],
            question_type=QuestionType.FACTUAL,
            difficulty=Difficulty.EASY,
            category="history",
            acceptable_variations=["Guido", "van Rossum"]
        ),
        GoldenSample(
            id="py_002",
            question="When was Python 3 released?",
            expected_answer="Python 3.0 was released on December 3, 2008",
            relevant_doc_ids=["doc_python_versions"],
            question_type=QuestionType.FACTUAL,
            difficulty=Difficulty.MEDIUM,
            category="history",
            key_facts=["2008", "December", "Python 3"]
        )
    ]
)

print(dataset.stats)
```

---

## Building Golden Datasets

### Strategy 1: Manual Curation

```python
from dataclasses import dataclass
from typing import List
import json

class GoldenDatasetBuilder:
    """Helper for building golden datasets."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.samples: List[GoldenSample] = []
        self._next_id = 1
    
    def add_sample(
        self,
        question: str,
        expected_answer: str,
        relevant_docs: List[str],
        question_type: QuestionType,
        difficulty: Difficulty,
        category: str,
        variations: List[str] = None,
        key_facts: List[str] = None
    ) -> str:
        """Add a sample and return its ID."""
        sample_id = f"{self.name[:3].lower()}_{self._next_id:04d}"
        self._next_id += 1
        
        sample = GoldenSample(
            id=sample_id,
            question=question,
            expected_answer=expected_answer,
            relevant_doc_ids=relevant_docs,
            question_type=question_type,
            difficulty=difficulty,
            category=category,
            acceptable_variations=variations or [],
            key_facts=key_facts or []
        )
        
        self.samples.append(sample)
        return sample_id
    
    def build(self) -> GoldenDataset:
        """Build the final dataset."""
        return GoldenDataset(
            name=self.name,
            description=self.description,
            samples=self.samples
        )
    
    def export_json(self, path: str):
        """Export to JSON file."""
        data = {
            "name": self.name,
            "description": self.description,
            "samples": [
                {
                    "id": s.id,
                    "question": s.question,
                    "expected_answer": s.expected_answer,
                    "relevant_doc_ids": s.relevant_doc_ids,
                    "question_type": s.question_type.value,
                    "difficulty": s.difficulty.value,
                    "category": s.category,
                    "acceptable_variations": s.acceptable_variations,
                    "key_facts": s.key_facts
                }
                for s in self.samples
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

# Usage
builder = GoldenDatasetBuilder(
    name="API Documentation",
    description="Test set for API docs RAG system"
)

builder.add_sample(
    question="What is the rate limit for the API?",
    expected_answer="The API allows 100 requests per minute",
    relevant_docs=["doc_rate_limits"],
    question_type=QuestionType.FACTUAL,
    difficulty=Difficulty.EASY,
    category="limits",
    variations=["100 req/min", "100 requests/minute"],
    key_facts=["100", "per minute"]
)

dataset = builder.build()
builder.export_json("golden_dataset.json")
```

### Strategy 2: LLM-Assisted Generation

```python
from openai import AsyncOpenAI
import json

async def generate_golden_samples(
    documents: List[str],
    samples_per_doc: int,
    client: AsyncOpenAI
) -> List[dict]:
    """
    Use LLM to generate question-answer pairs from documents.
    
    Note: Human review is required after generation!
    """
    all_samples = []
    
    for doc in documents:
        prompt = f"""Generate {samples_per_doc} question-answer pairs from this document.

Document:
{doc}

Generate diverse questions:
- 1-2 factual questions
- 1-2 reasoning questions
- 1 comparison or multi-hop question

Return JSON:
{{
    "samples": [
        {{
            "question": "...",
            "answer": "...",
            "type": "factual|reasoning|comparison|multi_hop",
            "difficulty": "easy|medium|hard",
            "key_facts": ["fact1", "fact2"]
        }}
    ]
}}
"""
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        all_samples.extend(result["samples"])
    
    return all_samples

# Usage
documents = [
    "Python is a high-level programming language created by Guido van Rossum...",
    "FastAPI is a modern web framework for building APIs with Python..."
]

samples = await generate_golden_samples(documents, 3, client)

# IMPORTANT: Review and validate generated samples!
for sample in samples:
    print(f"Q: {sample['question']}")
    print(f"A: {sample['answer']}")
    print(f"Valid? [y/n]: ", end="")
    # ... human review ...
```

---

## Rubric-Based Scoring

Define clear criteria for evaluating answers.

```python
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

class ScoreLevel(Enum):
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    FAILING = 1

@dataclass
class RubricCriterion:
    name: str
    weight: float
    levels: Dict[ScoreLevel, str]  # What each score level means

@dataclass
class Rubric:
    name: str
    criteria: List[RubricCriterion]
    
    def get_max_score(self) -> float:
        return sum(c.weight * ScoreLevel.EXCELLENT.value for c in self.criteria)

# Example rubric for RAG answers
rag_rubric = Rubric(
    name="RAG Answer Quality",
    criteria=[
        RubricCriterion(
            name="Factual Accuracy",
            weight=0.3,
            levels={
                ScoreLevel.EXCELLENT: "All facts correct and verified",
                ScoreLevel.GOOD: "Most facts correct, minor omissions",
                ScoreLevel.ACCEPTABLE: "Core facts correct, some errors",
                ScoreLevel.POOR: "Multiple factual errors",
                ScoreLevel.FAILING: "Mostly incorrect or hallucinated"
            }
        ),
        RubricCriterion(
            name="Completeness",
            weight=0.25,
            levels={
                ScoreLevel.EXCELLENT: "Fully answers all aspects",
                ScoreLevel.GOOD: "Answers main question, minor gaps",
                ScoreLevel.ACCEPTABLE: "Addresses question partially",
                ScoreLevel.POOR: "Misses key aspects",
                ScoreLevel.FAILING: "Does not address question"
            }
        ),
        RubricCriterion(
            name="Groundedness",
            weight=0.3,
            levels={
                ScoreLevel.EXCELLENT: "Fully grounded in sources",
                ScoreLevel.GOOD: "Mostly grounded, minimal extrapolation",
                ScoreLevel.ACCEPTABLE: "Partially grounded",
                ScoreLevel.POOR: "Significant unsupported claims",
                ScoreLevel.FAILING: "Mostly hallucinated"
            }
        ),
        RubricCriterion(
            name="Clarity",
            weight=0.15,
            levels={
                ScoreLevel.EXCELLENT: "Clear, well-organized, easy to understand",
                ScoreLevel.GOOD: "Mostly clear with good structure",
                ScoreLevel.ACCEPTABLE: "Understandable but could be clearer",
                ScoreLevel.POOR: "Confusing or poorly organized",
                ScoreLevel.FAILING: "Incomprehensible"
            }
        )
    ]
)

@dataclass
class RubricScore:
    criterion: str
    level: ScoreLevel
    score: float
    feedback: str

@dataclass
class RubricEvaluation:
    scores: List[RubricScore]
    total_score: float
    max_score: float
    percentage: float
    grade: str
    
    @classmethod
    def calculate_grade(cls, percentage: float) -> str:
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        return "F"

async def evaluate_with_rubric(
    question: str,
    answer: str,
    expected: str,
    rubric: Rubric,
    client: AsyncOpenAI
) -> RubricEvaluation:
    """Evaluate answer against rubric."""
    
    criteria_text = "\n".join([
        f"\n{c.name} (weight: {c.weight}):\n" + "\n".join([
            f"  {level.value}: {desc}"
            for level, desc in c.levels.items()
        ])
        for c in rubric.criteria
    ])
    
    prompt = f"""Evaluate this answer using the rubric.

Question: {question}
Expected Answer: {expected}
Actual Answer: {answer}

Rubric:
{criteria_text}

For each criterion, assign a score (1-5) with brief feedback.

Return JSON:
{{
    "scores": [
        {{
            "criterion": "Factual Accuracy",
            "level": 1-5,
            "feedback": "brief feedback"
        }},
        ...
    ]
}}
"""
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    scores = []
    total = 0.0
    max_total = 0.0
    
    for score_data in result["scores"]:
        criterion_name = score_data["criterion"]
        level_value = score_data["level"]
        
        # Find criterion weight
        criterion = next(
            (c for c in rubric.criteria if c.name == criterion_name),
            None
        )
        
        if criterion:
            weighted_score = criterion.weight * level_value
            total += weighted_score
            max_total += criterion.weight * 5
            
            scores.append(RubricScore(
                criterion=criterion_name,
                level=ScoreLevel(level_value),
                score=weighted_score,
                feedback=score_data["feedback"]
            ))
    
    percentage = (total / max_total * 100) if max_total > 0 else 0
    
    return RubricEvaluation(
        scores=scores,
        total_score=total,
        max_score=max_total,
        percentage=percentage,
        grade=RubricEvaluation.calculate_grade(percentage)
    )

# Usage
evaluation = await evaluate_with_rubric(
    question="What is Python?",
    answer="Python is a programming language.",
    expected="Python is a high-level, interpreted programming language created by Guido van Rossum.",
    rubric=rag_rubric,
    client=client
)

print(f"Grade: {evaluation.grade} ({evaluation.percentage:.1f}%)")
for score in evaluation.scores:
    print(f"  {score.criterion}: {score.level.value}/5 - {score.feedback}")
```

---

## Running Ground Truth Evaluation

```python
from dataclasses import dataclass
from typing import List, Dict, Callable, Awaitable
import asyncio

@dataclass
class EvaluationResult:
    sample_id: str
    question: str
    expected: str
    actual: str
    
    # Retrieval
    retrieved_docs: List[str]
    retrieval_hit: bool
    retrieval_precision: float
    retrieval_recall: float
    
    # Answer quality
    answer_match: bool
    key_facts_covered: int
    key_facts_total: int
    
    # Scores
    rubric_score: float
    rubric_grade: str

class GroundTruthEvaluator:
    """Evaluate RAG system against golden dataset."""
    
    def __init__(
        self,
        dataset: GoldenDataset,
        rag_fn: Callable[[str], Awaitable[tuple[str, List[str]]]],
        client: AsyncOpenAI
    ):
        """
        Args:
            dataset: Golden dataset to evaluate against
            rag_fn: Function that takes query, returns (answer, retrieved_doc_ids)
            client: OpenAI client for rubric evaluation
        """
        self.dataset = dataset
        self.rag_fn = rag_fn
        self.client = client
        self.rubric = rag_rubric  # Use default rubric
    
    async def evaluate_sample(
        self,
        sample: GoldenSample
    ) -> EvaluationResult:
        """Evaluate a single sample."""
        
        # Run RAG
        answer, retrieved_docs = await self.rag_fn(sample.question)
        
        # Retrieval metrics
        retrieved_set = set(retrieved_docs)
        relevant_set = set(sample.relevant_doc_ids)
        
        hit = bool(retrieved_set & relevant_set)
        precision = len(retrieved_set & relevant_set) / len(retrieved_set) if retrieved_set else 0
        recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0
        
        # Answer match
        answer_match = sample.matches_answer(answer)
        
        # Key facts
        facts_covered = sum(
            1 for fact in sample.key_facts
            if fact.lower() in answer.lower()
        )
        
        # Rubric evaluation
        rubric_eval = await evaluate_with_rubric(
            sample.question,
            answer,
            sample.expected_answer,
            self.rubric,
            self.client
        )
        
        return EvaluationResult(
            sample_id=sample.id,
            question=sample.question,
            expected=sample.expected_answer,
            actual=answer,
            retrieved_docs=retrieved_docs,
            retrieval_hit=hit,
            retrieval_precision=precision,
            retrieval_recall=recall,
            answer_match=answer_match,
            key_facts_covered=facts_covered,
            key_facts_total=len(sample.key_facts),
            rubric_score=rubric_eval.percentage,
            rubric_grade=rubric_eval.grade
        )
    
    async def evaluate_all(
        self,
        max_concurrent: int = 5
    ) -> Dict:
        """Evaluate all samples in dataset."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def eval_with_limit(sample):
            async with semaphore:
                return await self.evaluate_sample(sample)
        
        results = await asyncio.gather(*[
            eval_with_limit(sample)
            for sample in self.dataset.samples
        ])
        
        # Aggregate
        return self._aggregate(results)
    
    def _aggregate(self, results: List[EvaluationResult]) -> Dict:
        """Aggregate results into summary metrics."""
        n = len(results)
        
        return {
            "total_samples": n,
            
            # Retrieval
            "retrieval_hit_rate": sum(r.retrieval_hit for r in results) / n,
            "mean_precision": sum(r.retrieval_precision for r in results) / n,
            "mean_recall": sum(r.retrieval_recall for r in results) / n,
            
            # Answer
            "answer_match_rate": sum(r.answer_match for r in results) / n,
            "key_fact_coverage": sum(r.key_facts_covered for r in results) / sum(r.key_facts_total for r in results) if sum(r.key_facts_total for r in results) > 0 else 0,
            
            # Rubric
            "mean_rubric_score": sum(r.rubric_score for r in results) / n,
            "grade_distribution": {
                grade: sum(1 for r in results if r.rubric_grade == grade)
                for grade in ["A", "B", "C", "D", "F"]
            },
            
            # Details
            "results": results
        }

# Usage
async def my_rag_system(query: str) -> tuple[str, List[str]]:
    # Your RAG implementation
    answer = "..."
    doc_ids = ["doc_1", "doc_2"]
    return answer, doc_ids

evaluator = GroundTruthEvaluator(
    dataset=dataset,
    rag_fn=my_rag_system,
    client=client
)

results = await evaluator.evaluate_all()
print(f"Hit Rate: {results['retrieval_hit_rate']:.2%}")
print(f"Answer Match: {results['answer_match_rate']:.2%}")
print(f"Mean Score: {results['mean_rubric_score']:.1f}%")
```

---

## Regression Testing

Detect when changes break existing functionality.

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import json

@dataclass
class RegressionResult:
    sample_id: str
    baseline_score: float
    current_score: float
    delta: float
    is_regression: bool
    is_improvement: bool

@dataclass
class RegressionReport:
    timestamp: datetime
    baseline_version: str
    current_version: str
    
    regressions: List[RegressionResult]
    improvements: List[RegressionResult]
    unchanged: List[RegressionResult]
    
    overall_delta: float
    regression_rate: float
    improvement_rate: float
    
    passed: bool

class RegressionTester:
    """Track RAG performance over time and detect regressions."""
    
    def __init__(
        self,
        baseline_path: str,
        threshold: float = 0.05  # 5% drop = regression
    ):
        self.threshold = threshold
        self.baseline = self._load_baseline(baseline_path)
    
    def _load_baseline(self, path: str) -> Dict[str, float]:
        """Load baseline scores from file."""
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_baseline(self, results: Dict, path: str, version: str):
        """Save current results as new baseline."""
        baseline = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "scores": {
                r.sample_id: r.rubric_score
                for r in results["results"]
            }
        }
        
        with open(path, 'w') as f:
            json.dump(baseline, f, indent=2)
    
    def compare(
        self,
        current_results: Dict,
        current_version: str
    ) -> RegressionReport:
        """Compare current results against baseline."""
        
        regressions = []
        improvements = []
        unchanged = []
        
        baseline_scores = self.baseline.get("scores", {})
        
        for result in current_results["results"]:
            sample_id = result.sample_id
            current_score = result.rubric_score
            
            baseline_score = baseline_scores.get(sample_id)
            
            if baseline_score is None:
                # New sample, skip
                continue
            
            delta = current_score - baseline_score
            
            reg_result = RegressionResult(
                sample_id=sample_id,
                baseline_score=baseline_score,
                current_score=current_score,
                delta=delta,
                is_regression=delta < -self.threshold * 100,
                is_improvement=delta > self.threshold * 100
            )
            
            if reg_result.is_regression:
                regressions.append(reg_result)
            elif reg_result.is_improvement:
                improvements.append(reg_result)
            else:
                unchanged.append(reg_result)
        
        total = len(regressions) + len(improvements) + len(unchanged)
        
        return RegressionReport(
            timestamp=datetime.now(),
            baseline_version=self.baseline.get("version", "unknown"),
            current_version=current_version,
            regressions=regressions,
            improvements=improvements,
            unchanged=unchanged,
            overall_delta=current_results["mean_rubric_score"] - self.baseline.get("mean_score", 0),
            regression_rate=len(regressions) / total if total > 0 else 0,
            improvement_rate=len(improvements) / total if total > 0 else 0,
            passed=len(regressions) == 0
        )
    
    def format_report(self, report: RegressionReport) -> str:
        """Format report as markdown."""
        lines = [
            f"# Regression Report",
            f"",
            f"**Baseline:** {report.baseline_version}",
            f"**Current:** {report.current_version}",
            f"**Status:** {'✅ PASSED' if report.passed else '❌ FAILED'}",
            f"",
            f"## Summary",
            f"- Overall delta: {report.overall_delta:+.1f}%",
            f"- Regressions: {len(report.regressions)} ({report.regression_rate:.1%})",
            f"- Improvements: {len(report.improvements)} ({report.improvement_rate:.1%})",
            f"- Unchanged: {len(report.unchanged)}",
        ]
        
        if report.regressions:
            lines.extend([
                f"",
                f"## ⚠️ Regressions",
                f"",
                f"| Sample | Baseline | Current | Delta |",
                f"|--------|----------|---------|-------|"
            ])
            
            for r in sorted(report.regressions, key=lambda x: x.delta):
                lines.append(
                    f"| {r.sample_id} | {r.baseline_score:.1f}% | "
                    f"{r.current_score:.1f}% | {r.delta:+.1f}% |"
                )
        
        if report.improvements:
            lines.extend([
                f"",
                f"## ✅ Improvements",
                f"",
                f"| Sample | Baseline | Current | Delta |",
                f"|--------|----------|---------|-------|"
            ])
            
            for r in sorted(report.improvements, key=lambda x: -x.delta):
                lines.append(
                    f"| {r.sample_id} | {r.baseline_score:.1f}% | "
                    f"{r.current_score:.1f}% | {r.delta:+.1f}% |"
                )
        
        return "\n".join(lines)

# Usage
tester = RegressionTester(baseline_path="baseline_v1.json")

# Run evaluation
results = await evaluator.evaluate_all()

# Compare
report = tester.compare(results, current_version="v2.0")

print(tester.format_report(report))

if not report.passed:
    print("\n⚠️ Regressions detected! Review before deploying.")
```

---

## Hands-on Exercise

### Your Task

Build a `ContinuousEvaluation` system that:
1. Runs scheduled evaluations
2. Tracks metrics over time
3. Alerts on significant changes
4. Generates trend reports

### Requirements

```python
class ContinuousEvaluation:
    def __init__(self, dataset: GoldenDataset, rag_fn):
        pass
    
    async def run_evaluation(self, version: str) -> dict:
        """Run evaluation and store results."""
        pass
    
    def get_trend(self, metric: str, days: int = 30) -> list:
        """Get metric values over time."""
        pass
    
    def check_alerts(self) -> list[str]:
        """Check for any alert conditions."""
        pass
```

<details>
<summary>💡 Hints</summary>

- Store results in a time-series format
- Calculate moving averages for trend detection
- Alert on standard deviation changes
- Consider persistence (file or database)

</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Awaitable, Optional
from datetime import datetime, timedelta
import json
import statistics

@dataclass
class EvaluationRun:
    version: str
    timestamp: datetime
    metrics: Dict[str, float]

@dataclass
class Alert:
    metric: str
    message: str
    severity: str  # "warning" or "critical"
    current_value: float
    threshold: float

class ContinuousEvaluation:
    def __init__(
        self,
        dataset: GoldenDataset,
        rag_fn: Callable[[str], Awaitable[tuple[str, List[str]]]],
        client: AsyncOpenAI,
        storage_path: str = "evaluation_history.json"
    ):
        self.dataset = dataset
        self.rag_fn = rag_fn
        self.client = client
        self.storage_path = storage_path
        self.history: List[EvaluationRun] = self._load_history()
        
        # Alert thresholds
        self.alert_config = {
            "mean_rubric_score": {
                "warning": -5.0,   # 5% drop
                "critical": -10.0  # 10% drop
            },
            "retrieval_hit_rate": {
                "warning": -0.05,
                "critical": -0.10
            }
        }
    
    def _load_history(self) -> List[EvaluationRun]:
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                return [
                    EvaluationRun(
                        version=r["version"],
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        metrics=r["metrics"]
                    )
                    for r in data
                ]
        except FileNotFoundError:
            return []
    
    def _save_history(self):
        data = [
            {
                "version": r.version,
                "timestamp": r.timestamp.isoformat(),
                "metrics": r.metrics
            }
            for r in self.history
        ]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def run_evaluation(self, version: str) -> dict:
        """Run evaluation and store results."""
        evaluator = GroundTruthEvaluator(
            dataset=self.dataset,
            rag_fn=self.rag_fn,
            client=self.client
        )
        
        results = await evaluator.evaluate_all()
        
        # Extract key metrics
        metrics = {
            "mean_rubric_score": results["mean_rubric_score"],
            "retrieval_hit_rate": results["retrieval_hit_rate"],
            "answer_match_rate": results["answer_match_rate"],
            "mean_precision": results["mean_precision"],
            "mean_recall": results["mean_recall"]
        }
        
        # Store in history
        run = EvaluationRun(
            version=version,
            timestamp=datetime.now(),
            metrics=metrics
        )
        self.history.append(run)
        self._save_history()
        
        return results
    
    def get_trend(self, metric: str, days: int = 30) -> List[Dict]:
        """Get metric values over time."""
        cutoff = datetime.now() - timedelta(days=days)
        
        trend = []
        for run in self.history:
            if run.timestamp >= cutoff and metric in run.metrics:
                trend.append({
                    "timestamp": run.timestamp.isoformat(),
                    "version": run.version,
                    "value": run.metrics[metric]
                })
        
        return trend
    
    def get_moving_average(
        self,
        metric: str,
        window: int = 5
    ) -> Optional[float]:
        """Calculate moving average for metric."""
        values = [
            r.metrics.get(metric)
            for r in self.history[-window:]
            if metric in r.metrics
        ]
        
        if not values:
            return None
        
        return statistics.mean(values)
    
    def check_alerts(self) -> List[Alert]:
        """Check for any alert conditions."""
        alerts = []
        
        if len(self.history) < 2:
            return alerts
        
        current = self.history[-1]
        previous_avg = {}
        
        # Calculate average of last 5 runs (excluding current)
        for metric in self.alert_config.keys():
            values = [
                r.metrics.get(metric)
                for r in self.history[-6:-1]
                if metric in r.metrics
            ]
            if values:
                previous_avg[metric] = statistics.mean(values)
        
        # Check each metric
        for metric, thresholds in self.alert_config.items():
            if metric not in current.metrics or metric not in previous_avg:
                continue
            
            current_value = current.metrics[metric]
            baseline = previous_avg[metric]
            delta = current_value - baseline
            
            if delta <= thresholds["critical"]:
                alerts.append(Alert(
                    metric=metric,
                    message=f"{metric} dropped significantly",
                    severity="critical",
                    current_value=current_value,
                    threshold=thresholds["critical"]
                ))
            elif delta <= thresholds["warning"]:
                alerts.append(Alert(
                    metric=metric,
                    message=f"{metric} is declining",
                    severity="warning",
                    current_value=current_value,
                    threshold=thresholds["warning"]
                ))
        
        return alerts
    
    def generate_report(self, days: int = 30) -> str:
        """Generate trend report."""
        lines = [
            "# Continuous Evaluation Report",
            f"",
            f"**Period:** Last {days} days",
            f"**Total Runs:** {len(self.history)}",
            f""
        ]
        
        # Current status
        if self.history:
            current = self.history[-1]
            lines.extend([
                "## Current Status",
                f"- Version: {current.version}",
                f"- Timestamp: {current.timestamp.isoformat()}"
            ])
            
            for metric, value in current.metrics.items():
                ma = self.get_moving_average(metric)
                trend = "→"
                if ma and value > ma * 1.02:
                    trend = "↑"
                elif ma and value < ma * 0.98:
                    trend = "↓"
                
                lines.append(f"- {metric}: {value:.2f} {trend}")
        
        # Alerts
        alerts = self.check_alerts()
        if alerts:
            lines.extend([
                "",
                "## ⚠️ Alerts"
            ])
            for alert in alerts:
                icon = "🔴" if alert.severity == "critical" else "🟡"
                lines.append(f"- {icon} {alert.message}: {alert.current_value:.2f}")
        
        return "\n".join(lines)

# Usage
continuous = ContinuousEvaluation(
    dataset=dataset,
    rag_fn=my_rag_system,
    client=client
)

# Run evaluation
await continuous.run_evaluation("v2.1.0")

# Check alerts
alerts = continuous.check_alerts()
for alert in alerts:
    print(f"[{alert.severity}] {alert.message}")

# Generate report
print(continuous.generate_report())
```

</details>

---

## Summary

Ground truth testing provides objective RAG evaluation:

✅ **Golden datasets** — Curated Q&A pairs with verified answers
✅ **Rubric scoring** — Consistent evaluation criteria
✅ **Regression testing** — Catch performance degradation
✅ **Continuous monitoring** — Track metrics over time

**Key insight:** Golden datasets are expensive to create but invaluable for reliable evaluation. Start small and grow over time.

**Next:** [User Feedback Loops](./06-user-feedback.md)

---

## Further Reading

- [Building Evaluation Datasets](https://eugeneyan.com/writing/llm-patterns/)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/)

<!--
Sources Consulted:
- RAGAS evaluation approaches
- Industry best practices for ML testing
- Academic benchmarking methodologies
-->
