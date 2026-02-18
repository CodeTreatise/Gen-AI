---
title: "9.15.6 A/B Testing RAG Changes"
---

# 9.15.6 A/B Testing RAG Changes

## Introduction

Every change to a RAG pipeline — different chunk sizes, new embedding models, modified prompts, additional retrieval stages — is a hypothesis. Without controlled experiments, you're guessing whether changes improve or degrade the system. A/B testing lets you compare RAG configurations with statistical rigor, so decisions are based on measured impact rather than intuition.

This lesson covers how to design, implement, and analyze A/B tests for each layer of the RAG pipeline: chunking strategies, retrieval algorithms, reranking approaches, and prompt templates.

## Prerequisites

- Understanding of RAG pipeline components (Lessons 9.1–9.5)
- Monitoring and observability setup (Lesson 9.15.5)
- Basic statistics (mean, variance, p-values)
- Production deployment experience

---

## Why A/B Test RAG Systems?

RAG pipelines have many interacting components, and changes that look good in isolation can hurt end-to-end quality. Common examples:

| Change | Expected Outcome | Actual Outcome (without testing) |
|--------|-----------------|----------------------------------|
| Smaller chunks (256 → 128 tokens) | More precise retrieval | Answers lost context, quality dropped |
| Added reranking stage | Better relevance | Latency increased 300ms, users abandoned |
| Switched to GPT-4.1 from mini | Higher quality answers | 4× cost increase, marginal quality gain |
| New system prompt | More concise answers | Increased hallucination rate by 12% |

### The Cost of Not Testing

```
Scenario: You switch from 512-token chunks to 256-token chunks.

Without A/B test:
  - Deploy to 100% of users
  - Quality drops for 30% of queries (multi-hop questions need more context)
  - You don't notice for 2 weeks because average metrics look fine
  - By then, user trust has eroded

With A/B test:
  - Deploy to 10% of users for 1 week
  - Measure answer quality, retrieval scores, and user satisfaction
  - Discover the 30% degradation on multi-hop queries
  - Keep 512-token chunks, investigate targeted improvements
```

---

## Experiment Framework

```python
import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Variant:
    """A single variant in an A/B test.

    Each variant defines a configuration override and its
    traffic allocation percentage.
    """
    name: str
    config: dict  # Configuration overrides for this variant
    traffic_pct: float  # Percentage of traffic (0-100)
    description: str = ""


@dataclass
class Experiment:
    """An A/B test experiment for a RAG pipeline.

    Defines the variants, traffic allocation, and success metrics
    for a controlled experiment.
    """
    id: str
    name: str
    description: str
    status: ExperimentStatus = ExperimentStatus.DRAFT
    variants: list[Variant] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    min_sample_size: int = 1000  # Queries per variant
    confidence_level: float = 0.95  # Statistical significance threshold
    max_duration_days: int = 14


class ExperimentRouter:
    """Route queries to experiment variants using consistent hashing.

    Uses deterministic assignment so the same user always sees
    the same variant within an experiment. This prevents
    confusion from inconsistent behavior and ensures clean
    measurement.

    Assignment is based on hash(user_id + experiment_id), which
    produces a uniform distribution across the traffic allocation
    buckets.
    """

    def __init__(self):
        self.experiments: dict[str, Experiment] = {}

    def register(self, experiment: Experiment) -> None:
        """Register an experiment for routing."""
        total_traffic = sum(v.traffic_pct for v in experiment.variants)
        if abs(total_traffic - 100) > 0.01:
            raise ValueError(
                f"Variant traffic must sum to 100%, got {total_traffic}%"
            )
        self.experiments[experiment.id] = experiment

    def get_variant(
        self, experiment_id: str, user_id: str
    ) -> Optional[Variant]:
        """Determine which variant a user should see.

        Uses consistent hashing so the same user always gets
        the same variant. This is critical for clean measurement
        and consistent user experience.
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return None

        # Deterministic hash-based assignment
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 10000  # 0-9999 for 0.01% granularity

        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.traffic_pct * 100  # Convert to basis points
            if bucket < cumulative:
                return variant

        return experiment.variants[-1]  # Fallback to last variant
```

### Creating Experiments

```python
# Example 1: Test chunking strategies
chunking_experiment = Experiment(
    id="exp_chunk_size_2025_06",
    name="Chunk Size Comparison",
    description="Compare 256, 512, and 1024 token chunks on answer quality",
    variants=[
        Variant(
            name="control",
            config={"chunk_size": 512, "chunk_overlap": 50},
            traffic_pct=40,
            description="Current production config",
        ),
        Variant(
            name="small_chunks",
            config={"chunk_size": 256, "chunk_overlap": 25},
            traffic_pct=30,
            description="Smaller chunks for precision",
        ),
        Variant(
            name="large_chunks",
            config={"chunk_size": 1024, "chunk_overlap": 100},
            traffic_pct=30,
            description="Larger chunks for more context",
        ),
    ],
    metrics=[
        "retrieval_relevance",
        "answer_faithfulness",
        "answer_completeness",
        "user_thumbs_up_rate",
        "latency_p50",
    ],
    min_sample_size=2000,
)

# Example 2: Test retrieval algorithms
retrieval_experiment = Experiment(
    id="exp_retrieval_algo_2025_06",
    name="Retrieval Algorithm Comparison",
    description="Compare vector-only vs hybrid (vector + BM25) retrieval",
    variants=[
        Variant(
            name="control",
            config={"retrieval_method": "vector", "top_k": 10},
            traffic_pct=50,
            description="Vector search only",
        ),
        Variant(
            name="hybrid",
            config={
                "retrieval_method": "hybrid",
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "top_k": 10,
            },
            traffic_pct=50,
            description="Hybrid vector + BM25",
        ),
    ],
    metrics=[
        "retrieval_mrr_at_10",
        "retrieval_recall_at_10",
        "answer_relevance",
        "latency_p50",
    ],
    min_sample_size=3000,
)
```

---

## Integrating Experiments into the Pipeline

```python
class ABTestableRAGPipeline:
    """RAG pipeline that supports A/B testing of configurations.

    The pipeline checks active experiments, determines the variant
    for the current user, applies configuration overrides, and
    logs which variant was used for later analysis.
    """

    def __init__(self, router: ExperimentRouter, default_config: dict):
        self.router = router
        self.default_config = default_config

    async def query(
        self,
        question: str,
        user_id: str,
        active_experiments: list[str] = None,
    ) -> dict:
        """Execute a RAG query with experiment-specific configuration."""
        # Determine active variants for this user
        config = dict(self.default_config)
        active_variants = {}

        for exp_id in (active_experiments or []):
            variant = self.router.get_variant(exp_id, user_id)
            if variant:
                config.update(variant.config)
                active_variants[exp_id] = variant.name

        # Execute pipeline with experiment config
        result = await self._execute_pipeline(question, config)

        # Tag result with experiment metadata for analysis
        result["experiment_metadata"] = {
            "variants": active_variants,
            "config_snapshot": config,
        }

        return result

    async def _execute_pipeline(self, question: str, config: dict) -> dict:
        """Execute the RAG pipeline with the given configuration."""
        # Chunking is determined by the config
        chunk_size = config.get("chunk_size", 512)
        retrieval_method = config.get("retrieval_method", "vector")
        model = config.get("model", "gpt-4.1-mini")
        top_k = config.get("top_k", 10)

        # Embed
        embedding = await self.embed(question)

        # Retrieve based on experiment config
        if retrieval_method == "hybrid":
            vector_weight = config.get("vector_weight", 0.7)
            bm25_weight = config.get("bm25_weight", 0.3)
            documents = await self.hybrid_search(
                question, embedding, top_k, vector_weight, bm25_weight
            )
        else:
            documents = await self.vector_search(embedding, top_k)

        # Generate with configured model
        answer = await self.generate(question, documents, model=model)

        return {
            "answer": answer,
            "sources": documents,
            "config_used": {
                "chunk_size": chunk_size,
                "retrieval_method": retrieval_method,
                "model": model,
                "top_k": top_k,
            },
        }
```

---

## Collecting Experiment Metrics

```python
import time
from collections import defaultdict


class ExperimentMetricsCollector:
    """Collect per-variant metrics for experiment analysis.

    Stores metric values grouped by experiment and variant,
    so you can compare distributions between control and
    treatment groups.
    """

    def __init__(self):
        # Structure: {experiment_id: {variant_name: {metric: [values]}}}
        self.data: dict[str, dict[str, dict[str, list]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

    def record(
        self,
        experiment_id: str,
        variant_name: str,
        metrics: dict[str, float],
    ) -> None:
        """Record metrics for a single query in an experiment."""
        for metric_name, value in metrics.items():
            self.data[experiment_id][variant_name][metric_name].append(value)

    def get_summary(self, experiment_id: str) -> dict:
        """Get a summary of metrics for all variants in an experiment."""
        if experiment_id not in self.data:
            return {}

        summary = {}
        for variant_name, metrics in self.data[experiment_id].items():
            summary[variant_name] = {}
            for metric_name, values in metrics.items():
                if not values:
                    continue
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                summary[variant_name][metric_name] = {
                    "count": n,
                    "mean": sum(values) / n,
                    "median": sorted_vals[n // 2],
                    "std": (sum((x - sum(values)/n)**2 for x in values) / n) ** 0.5,
                    "p25": sorted_vals[int(n * 0.25)],
                    "p75": sorted_vals[int(n * 0.75)],
                }

        return summary


# Usage in the pipeline
collector = ExperimentMetricsCollector()

async def instrumented_query(question: str, user_id: str):
    """Query with experiment metric collection."""
    result = await pipeline.query(question, user_id, active_experiments=["exp_chunk_size_2025_06"])

    # Collect metrics for analysis
    metadata = result.get("experiment_metadata", {})
    for exp_id, variant_name in metadata.get("variants", {}).items():
        collector.record(exp_id, variant_name, {
            "retrieval_top_score": result["sources"][0]["score"] if result["sources"] else 0,
            "source_count": len(result["sources"]),
            "answer_length": len(result["answer"]),
            "latency_ms": result.get("latency_ms", 0),
        })
```

---

## Statistical Analysis

You need statistical rigor to determine if differences between variants are real or just noise:

```python
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ABTestResult:
    """Result of a statistical comparison between two variants."""
    control_name: str
    treatment_name: str
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_diff: float
    relative_diff_pct: float
    p_value: float
    is_significant: bool
    confidence_level: float
    control_n: int
    treatment_n: int
    recommendation: str


class StatisticalAnalyzer:
    """Analyze A/B test results with statistical significance testing.

    Uses Welch's t-test for comparing means, which does not assume
    equal variances between groups. This is more appropriate than
    Student's t-test for A/B testing because variant distributions
    often have different shapes.

    Also computes required sample sizes for power analysis,
    so you know when you have enough data to make a decision.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def compare_variants(
        self,
        control_values: list[float],
        treatment_values: list[float],
        metric_name: str,
        control_name: str = "control",
        treatment_name: str = "treatment",
        higher_is_better: bool = True,
    ) -> ABTestResult:
        """Compare two variants using Welch's t-test.

        Welch's t-test formula:
        t = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)

        Degrees of freedom (Welch-Satterthwaite):
        df = (var1/n1 + var2/n2)^2 / ((var1/n1)^2/(n1-1) + (var2/n2)^2/(n2-1))
        """
        n1 = len(control_values)
        n2 = len(treatment_values)

        if n1 < 30 or n2 < 30:
            return ABTestResult(
                control_name=control_name,
                treatment_name=treatment_name,
                metric_name=metric_name,
                control_mean=sum(control_values) / n1 if n1 > 0 else 0,
                treatment_mean=sum(treatment_values) / n2 if n2 > 0 else 0,
                absolute_diff=0,
                relative_diff_pct=0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                control_n=n1,
                treatment_n=n2,
                recommendation=f"Insufficient data. Need ≥30 samples per variant, have {n1}/{n2}.",
            )

        mean1 = sum(control_values) / n1
        mean2 = sum(treatment_values) / n2
        var1 = sum((x - mean1) ** 2 for x in control_values) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in treatment_values) / (n2 - 1)

        # Welch's t-statistic
        se = math.sqrt(var1 / n1 + var2 / n2)
        if se == 0:
            t_stat = 0
        else:
            t_stat = (mean2 - mean1) / se

        # Degrees of freedom (Welch-Satterthwaite approximation)
        if var1 / n1 + var2 / n2 == 0:
            df = n1 + n2 - 2
        else:
            numerator = (var1 / n1 + var2 / n2) ** 2
            denominator = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
            df = numerator / denominator

        # Approximate p-value using normal distribution for large df
        # For production, use scipy.stats.t.sf() for exact values
        p_value = self._approximate_p_value(abs(t_stat), df)

        absolute_diff = mean2 - mean1
        relative_diff_pct = (absolute_diff / mean1 * 100) if mean1 != 0 else 0
        is_significant = p_value < self.alpha

        # Generate recommendation
        recommendation = self._make_recommendation(
            is_significant, absolute_diff, relative_diff_pct,
            higher_is_better, treatment_name, control_name, p_value,
        )

        return ABTestResult(
            control_name=control_name,
            treatment_name=treatment_name,
            metric_name=metric_name,
            control_mean=round(mean1, 4),
            treatment_mean=round(mean2, 4),
            absolute_diff=round(absolute_diff, 4),
            relative_diff_pct=round(relative_diff_pct, 2),
            p_value=round(p_value, 4),
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            control_n=n1,
            treatment_n=n2,
            recommendation=recommendation,
        )

    @staticmethod
    def _approximate_p_value(t_stat: float, df: float) -> float:
        """Approximate two-tailed p-value from t-statistic.

        Uses the normal approximation for large degrees of freedom.
        For production use, replace with scipy.stats.t.sf().
        """
        # Normal approximation (good for df > 30)
        z = abs(t_stat)
        # Rational approximation of the normal CDF tail
        p = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        t = 1 / (1 + 0.2316419 * z)
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
        one_tail = p * poly
        return 2 * one_tail  # Two-tailed

    @staticmethod
    def _make_recommendation(
        is_significant: bool,
        absolute_diff: float,
        relative_diff_pct: float,
        higher_is_better: bool,
        treatment_name: str,
        control_name: str,
        p_value: float,
    ) -> str:
        """Generate a human-readable recommendation."""
        if not is_significant:
            return (
                f"No statistically significant difference detected (p={p_value:.3f}). "
                f"Keep {control_name} or collect more data."
            )

        is_improvement = (absolute_diff > 0) == higher_is_better
        direction = "improvement" if is_improvement else "regression"

        if is_improvement:
            if abs(relative_diff_pct) > 10:
                return (
                    f"Strong {direction}: {treatment_name} is {abs(relative_diff_pct):.1f}% "
                    f"better (p={p_value:.4f}). Recommend deploying {treatment_name}."
                )
            else:
                return (
                    f"Modest {direction}: {treatment_name} is {abs(relative_diff_pct):.1f}% "
                    f"better (p={p_value:.4f}). Consider deploying if no latency/cost tradeoff."
                )
        else:
            return (
                f"Significant {direction}: {treatment_name} is {abs(relative_diff_pct):.1f}% "
                f"worse (p={p_value:.4f}). Do NOT deploy. Keep {control_name}."
            )

    @staticmethod
    def required_sample_size(
        baseline_mean: float,
        min_detectable_effect: float,
        baseline_std: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """Calculate minimum sample size per variant for power analysis.

        Parameters:
        - baseline_mean: Current average metric value
        - min_detectable_effect: Smallest relative change you want to detect (e.g., 0.05 = 5%)
        - baseline_std: Standard deviation of the metric
        - alpha: Significance level (Type I error rate)
        - power: Probability of detecting a real effect (1 - Type II error rate)

        Formula (per group):
        n = 2 * (z_alpha/2 + z_beta)^2 * sigma^2 / delta^2
        """
        # Z-scores for common alpha and power values
        z_alpha = {0.01: 2.576, 0.05: 1.96, 0.10: 1.645}.get(alpha, 1.96)
        z_beta = {0.80: 0.842, 0.90: 1.282, 0.95: 1.645}.get(power, 0.842)

        delta = baseline_mean * min_detectable_effect
        if delta == 0:
            return 10000  # Default if we can't compute

        n = 2 * ((z_alpha + z_beta) ** 2) * (baseline_std ** 2) / (delta ** 2)
        return int(math.ceil(n))
```

### Running an Analysis

```python
# Simulated experiment results
analyzer = StatisticalAnalyzer(confidence_level=0.95)

# Compare retrieval relevance between chunk sizes
result = analyzer.compare_variants(
    control_values=[0.82, 0.79, 0.85, 0.81, ...],      # 512-token chunks
    treatment_values=[0.78, 0.76, 0.80, 0.77, ...],    # 256-token chunks
    metric_name="retrieval_relevance",
    control_name="512_tokens",
    treatment_name="256_tokens",
    higher_is_better=True,
)

print(f"Metric: {result.metric_name}")
print(f"Control ({result.control_name}): {result.control_mean:.4f} (n={result.control_n})")
print(f"Treatment ({result.treatment_name}): {result.treatment_mean:.4f} (n={result.treatment_n})")
print(f"Difference: {result.absolute_diff:+.4f} ({result.relative_diff_pct:+.2f}%)")
print(f"P-value: {result.p_value:.4f}")
print(f"Significant: {result.is_significant}")
print(f"Recommendation: {result.recommendation}")

# Output:
# Metric: retrieval_relevance
# Control (512_tokens): 0.8175 (n=2000)
# Treatment (256_tokens): 0.7783 (n=2000)
# Difference: -0.0392 (-4.79%)
# P-value: 0.0003
# Significant: True
# Recommendation: Significant regression: 256_tokens is 4.79% worse (p=0.0003).
#                 Do NOT deploy. Keep 512_tokens.

# Power analysis: How many samples do we need?
n = StatisticalAnalyzer.required_sample_size(
    baseline_mean=0.82,
    min_detectable_effect=0.05,  # Detect 5% change
    baseline_std=0.12,
)
print(f"\nRequired sample size per variant: {n}")
# Output: Required sample size per variant: 672
```

---

## Multi-Metric Decision Making

RAG experiments usually measure multiple metrics. A change might improve relevance but hurt latency. Use a decision matrix:

```python
@dataclass
class MetricWeight:
    """Weight and direction for a metric in multi-metric analysis."""
    name: str
    weight: float  # 0-1, all weights should sum to 1
    higher_is_better: bool


class MultiMetricDecisionMaker:
    """Make deployment decisions when experiments have multiple metrics.

    Uses a weighted scoring system where each metric contributes
    to an overall decision score. A variant must be better on
    weighted average AND not significantly worse on any guardrail
    metric.
    """

    def __init__(
        self,
        metric_weights: list[MetricWeight],
        guardrail_metrics: Optional[list[str]] = None,
        guardrail_max_regression_pct: float = 5.0,
    ):
        self.weights = {m.name: m for m in metric_weights}
        self.guardrails = set(guardrail_metrics or [])
        self.max_regression = guardrail_max_regression_pct

    def decide(self, results: list[ABTestResult]) -> dict:
        """Make a deployment decision from multiple metric results.

        Returns a decision dict with overall recommendation,
        per-metric analysis, and whether guardrails were violated.
        """
        weighted_score = 0.0
        guardrail_violations = []
        metric_details = []

        for result in results:
            weight_info = self.weights.get(result.metric_name)
            if not weight_info:
                continue

            # Normalize the relative difference
            normalized_diff = result.relative_diff_pct
            if not weight_info.higher_is_better:
                normalized_diff = -normalized_diff  # Flip for "lower is better"

            weighted_contribution = normalized_diff * weight_info.weight
            weighted_score += weighted_contribution

            # Check guardrails
            if result.metric_name in self.guardrails:
                is_regression = normalized_diff < -self.max_regression
                if is_regression and result.is_significant:
                    guardrail_violations.append({
                        "metric": result.metric_name,
                        "regression_pct": abs(normalized_diff),
                    })

            metric_details.append({
                "metric": result.metric_name,
                "weight": weight_info.weight,
                "diff_pct": result.relative_diff_pct,
                "significant": result.is_significant,
                "contribution": round(weighted_contribution, 3),
            })

        # Overall decision
        if guardrail_violations:
            decision = "REJECT"
            reason = f"Guardrail violated: {guardrail_violations}"
        elif weighted_score > 0 and any(r.is_significant for r in results):
            decision = "DEPLOY"
            reason = f"Weighted improvement of {weighted_score:.2f}%"
        elif weighted_score < -2:
            decision = "REJECT"
            reason = f"Weighted regression of {abs(weighted_score):.2f}%"
        else:
            decision = "INCONCLUSIVE"
            reason = "No clear winner. Collect more data or re-evaluate."

        return {
            "decision": decision,
            "reason": reason,
            "weighted_score": round(weighted_score, 3),
            "guardrail_violations": guardrail_violations,
            "metric_details": metric_details,
        }


# Usage
decision_maker = MultiMetricDecisionMaker(
    metric_weights=[
        MetricWeight("retrieval_relevance", weight=0.35, higher_is_better=True),
        MetricWeight("answer_faithfulness", weight=0.30, higher_is_better=True),
        MetricWeight("latency_p50", weight=0.20, higher_is_better=False),
        MetricWeight("user_thumbs_up_rate", weight=0.15, higher_is_better=True),
    ],
    guardrail_metrics=["latency_p50", "answer_faithfulness"],
    guardrail_max_regression_pct=10.0,
)

# Results from comparing hybrid vs vector-only retrieval
results = [
    analyzer.compare_variants(control_rel, treatment_rel, "retrieval_relevance",
                              "vector", "hybrid", higher_is_better=True),
    analyzer.compare_variants(control_faith, treatment_faith, "answer_faithfulness",
                              "vector", "hybrid", higher_is_better=True),
    analyzer.compare_variants(control_lat, treatment_lat, "latency_p50",
                              "vector", "hybrid", higher_is_better=False),
    analyzer.compare_variants(control_thumbs, treatment_thumbs, "user_thumbs_up_rate",
                              "vector", "hybrid", higher_is_better=True),
]

decision = decision_maker.decide(results)
print(f"Decision: {decision['decision']}")
print(f"Reason: {decision['reason']}")
# Decision: DEPLOY
# Reason: Weighted improvement of 3.72%
```

---

## Common RAG Experiments

### What to Test at Each Layer

| Layer | What to Test | Key Metrics | Typical Duration |
|-------|-------------|-------------|-----------------|
| **Chunking** | Size (128/256/512/1024), overlap %, strategy (fixed/sentence/semantic) | Retrieval MRR, answer completeness | 1–2 weeks |
| **Embedding** | Model (MiniLM/BGE/E5), dimensions (384/768/1024), quantization | Retrieval recall, latency, cost | 1–2 weeks |
| **Retrieval** | Vector vs hybrid, top_k values, filter strategies | MRR@10, recall@10, latency | 1 week |
| **Reranking** | Model choice, skip vs apply, top_k in/out | Answer relevance, latency | 1 week |
| **Prompt** | Template variations, system instructions, few-shot examples | Answer quality, faithfulness, length | 3–5 days |
| **Model** | GPT-4.1 vs mini vs nano, temperature, max_tokens | Quality, cost, latency | 3–5 days |

### Prompt Variation Testing

```python
# Define prompt variants to test
PROMPT_VARIANTS = {
    "control": {
        "system": (
            "You are a helpful assistant. Answer the question based on "
            "the provided context. If the context doesn't contain the "
            "answer, say so."
        ),
    },
    "structured": {
        "system": (
            "You are a technical documentation assistant.\n\n"
            "Instructions:\n"
            "1. Answer ONLY based on the provided context\n"
            "2. Use bullet points for multiple items\n"
            "3. Include relevant code examples if present in context\n"
            "4. If the context is insufficient, state what information is missing\n"
            "5. Keep answers concise (under 200 words unless code is needed)"
        ),
    },
    "cot": {
        "system": (
            "You are an AI assistant that answers questions using provided context.\n\n"
            "For each question:\n"
            "1. First, identify which parts of the context are relevant\n"
            "2. Then, synthesize an answer from those parts\n"
            "3. Finally, present the answer clearly\n\n"
            "If the context doesn't fully answer the question, explain what's "
            "covered and what's missing."
        ),
    },
}


async def run_prompt_experiment(
    questions: list[str],
    context_per_question: dict[str, str],
    variants: dict[str, dict],
    evaluator,
) -> dict:
    """Run a prompt variation experiment.

    Tests each prompt variant against the same set of questions
    and contexts, then evaluates answer quality.
    """
    results = {}

    for variant_name, variant_config in variants.items():
        variant_answers = []

        for question in questions:
            context = context_per_question[question]
            answer = await generate(
                question=question,
                context=context,
                system_prompt=variant_config["system"],
            )

            # Evaluate answer quality
            evaluation = await evaluator.evaluate(
                question=question,
                context=context,
                answer=answer,
            )

            variant_answers.append({
                "question": question,
                "answer": answer,
                "faithfulness": evaluation["faithfulness"],
                "relevance": evaluation["relevance"],
                "completeness": evaluation["completeness"],
            })

        results[variant_name] = variant_answers

    return results
```

---

## Gradual Rollout Strategy

Once an experiment shows positive results, roll out gradually:

```
Week 1: Experiment at 10% traffic
  ↓ (results look promising)
Week 2: Increase to 25% traffic
  ↓ (no degradation in latency or quality)
Week 3: Increase to 50% traffic
  ↓ (metrics stable, cost acceptable)
Week 4: Full rollout to 100%
  ↓ (old variant becomes the new control for future experiments)
```

```python
class GradualRollout:
    """Manage gradual rollout of experiment winners.

    After an experiment proves a variant is better, this manager
    handles the staged rollout from 10% → 25% → 50% → 100%.
    Automatically rolls back if quality degrades during rollout.
    """

    def __init__(self, stages: list[float] = None):
        self.stages = stages or [10, 25, 50, 100]
        self.current_stage_idx = 0
        self.metrics_per_stage: list[dict] = []

    @property
    def current_percentage(self) -> float:
        return self.stages[self.current_stage_idx]

    def advance(self, stage_metrics: dict) -> dict:
        """Advance to the next rollout stage if metrics are healthy."""
        self.metrics_per_stage.append(stage_metrics)

        # Check for degradation
        if self.current_stage_idx > 0:
            prev = self.metrics_per_stage[-2]
            curr = stage_metrics

            for metric_name in curr:
                if metric_name in prev:
                    change = (curr[metric_name] - prev[metric_name]) / prev[metric_name]
                    if change < -0.10:  # More than 10% degradation
                        return {
                            "action": "rollback",
                            "reason": f"{metric_name} degraded {change*100:.1f}% at {self.current_percentage}%",
                            "stage": self.current_stage_idx,
                        }

        # Advance if possible
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            return {
                "action": "advance",
                "new_percentage": self.current_percentage,
                "stage": self.current_stage_idx,
            }

        return {"action": "complete", "message": "Full rollout achieved"}
```

---

## Summary

| Concept | Key Points |
|---------|-----------|
| **Consistent assignment** | Hash user_id + experiment_id for deterministic routing |
| **Sample size** | Calculate with power analysis before starting (typically 500–3,000 per variant) |
| **Statistical test** | Welch's t-test for comparing means; doesn't assume equal variance |
| **Multi-metric** | Weight metrics and enforce guardrails (latency, faithfulness) |
| **Gradual rollout** | 10% → 25% → 50% → 100% with automatic rollback on degradation |

### Key Takeaways

1. **Test one variable at a time** — changing chunk size AND embedding model simultaneously makes results uninterpretable
2. **Calculate sample size first** — running an experiment for "a few days" isn't rigorous; compute the minimum sample per variant
3. **Use consistent hashing** for user assignment — random assignment per query creates noisy data
4. **Define guardrail metrics** that can veto a deployment even if the primary metric improves
5. **Automate the analysis** — manual analysis doesn't scale and introduces bias
6. **Roll out gradually** after a positive result — don't flip to 100% immediately

## Practice Exercises

1. **Design an experiment** to compare three chunking strategies (fixed 512, sentence-based, semantic). Define metrics, sample size, and guardrails.
2. **Implement the `ExperimentRouter`** and verify that user assignment is deterministic and evenly distributed across variants.
3. **Run a prompt variation test** comparing 3 system prompts on 50 questions. Use the `StatisticalAnalyzer` to determine which performs best.
4. **Build a multi-metric decision matrix** for an experiment comparing GPT-4.1 vs GPT-4.1-mini, weighting quality, latency, and cost.

---

← [Previous: Monitoring & Observability](./05-monitoring-observability.md) | [Back to Overview →](./00-production-rag-patterns.md)
