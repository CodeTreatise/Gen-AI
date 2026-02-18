---
title: "RAG-Specific Metrics"
---

# RAG-Specific Metrics

## Introduction

Standard NLP metrics like BLEU and ROUGE don't capture what matters for RAG: whether retrieved context was relevant and whether the answer faithfully used it. RAG requires specialized metrics that evaluate both retrieval and generation quality.

This lesson covers the core metrics designed specifically for RAG evaluation.

### What We'll Cover

- Context relevance metrics
- Answer faithfulness scoring
- Answer relevancy measurement
- Combined quality scores

### Prerequisites

- RAG architecture understanding
- Basic evaluation concepts
- Python with async support

---

## Context Relevance

Context relevance measures whether retrieved chunks are useful for answering the question.

### Why It Matters

```python
# Scenario: Query about Python's creation

# Good retrieval - high context relevance
contexts_good = [
    "Python was created by Guido van Rossum and first released in 1991.",
    "Python emphasizes code readability and supports multiple paradigms."
]

# Poor retrieval - low context relevance
contexts_poor = [
    "Monty Python's Flying Circus was a British comedy group.",
    "Pythons are large snakes found in Africa and Asia."
]

# Same query, vastly different retrieval quality
```

### Measuring Context Relevance

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ContextRelevanceResult:
    score: float
    relevant_chunks: int
    total_chunks: int
    per_chunk_scores: list[float]

async def evaluate_context_relevance(
    query: str,
    contexts: list[str],
    llm_client,
    model: str = "gpt-4o-mini"
) -> ContextRelevanceResult:
    """
    Evaluate relevance of each retrieved context to the query.
    """
    per_chunk_scores = []
    
    for context in contexts:
        prompt = f"""Rate how relevant this context is for answering the query.

Query: {query}

Context: {context}

Rate from 0.0 (completely irrelevant) to 1.0 (highly relevant).
Consider:
- Does it contain information needed to answer?
- Is it directly related to the query topic?
- Would it help an LLM generate a correct answer?

Return only a number between 0.0 and 1.0."""

        response = await llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = 0.5
        
        per_chunk_scores.append(score)
    
    # Calculate aggregate score
    avg_score = sum(per_chunk_scores) / len(per_chunk_scores) if per_chunk_scores else 0.0
    relevant_count = sum(1 for s in per_chunk_scores if s >= 0.5)
    
    return ContextRelevanceResult(
        score=avg_score,
        relevant_chunks=relevant_count,
        total_chunks=len(contexts),
        per_chunk_scores=per_chunk_scores
    )
```

### Weighted Context Relevance

Position matters—earlier chunks should be more relevant:

```python
def weighted_context_relevance(
    per_chunk_scores: list[float],
    decay_factor: float = 0.9
) -> float:
    """
    Calculate weighted relevance with position decay.
    
    Earlier positions get higher weight.
    """
    if not per_chunk_scores:
        return 0.0
    
    weighted_sum = 0.0
    weight_total = 0.0
    
    for i, score in enumerate(per_chunk_scores):
        weight = decay_factor ** i
        weighted_sum += score * weight
        weight_total += weight
    
    return weighted_sum / weight_total

# Example
scores = [0.9, 0.8, 0.3, 0.2]  # Good chunks first
weighted = weighted_context_relevance(scores)
print(f"Weighted relevance: {weighted:.2f}")  # ~0.77

# Compare to simple average
simple_avg = sum(scores) / len(scores)
print(f"Simple average: {simple_avg:.2f}")  # 0.55
```

---

## Answer Faithfulness

Faithfulness measures whether the answer is grounded in the retrieved context—not hallucinated.

### The Faithfulness Formula

```
Faithfulness = Claims supported by context / Total claims in answer
```

### Claim-Based Faithfulness

```python
from dataclasses import dataclass

@dataclass
class Claim:
    text: str
    supported: bool
    supporting_context: Optional[str] = None

@dataclass
class FaithfulnessResult:
    score: float
    claims: list[Claim]
    total_claims: int
    supported_claims: int

async def evaluate_faithfulness(
    response: str,
    contexts: list[str],
    llm_client,
    model: str = "gpt-4o-mini"
) -> FaithfulnessResult:
    """
    Evaluate faithfulness by extracting and verifying claims.
    """
    # Step 1: Extract claims from response
    claims = await extract_claims(response, llm_client, model)
    
    # Step 2: Verify each claim against context
    context_text = "\n\n".join(contexts)
    verified_claims = []
    
    for claim_text in claims:
        is_supported, supporting = await verify_claim(
            claim_text, context_text, llm_client, model
        )
        verified_claims.append(Claim(
            text=claim_text,
            supported=is_supported,
            supporting_context=supporting
        ))
    
    # Step 3: Calculate score
    supported_count = sum(1 for c in verified_claims if c.supported)
    score = supported_count / len(verified_claims) if verified_claims else 1.0
    
    return FaithfulnessResult(
        score=score,
        claims=verified_claims,
        total_claims=len(verified_claims),
        supported_claims=supported_count
    )

async def extract_claims(
    response: str,
    llm_client,
    model: str
) -> list[str]:
    """Extract atomic claims from response."""
    prompt = f"""Extract all factual claims from this response.
Each claim should be a single, verifiable statement.

Response: {response}

Return claims as a numbered list:
1. [claim]
2. [claim]
..."""

    result = await llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    # Parse numbered list
    lines = result.choices[0].message.content.strip().split("\n")
    claims = []
    for line in lines:
        # Remove numbering
        if ". " in line:
            claim = line.split(". ", 1)[1].strip()
            if claim:
                claims.append(claim)
    
    return claims

async def verify_claim(
    claim: str,
    context: str,
    llm_client,
    model: str
) -> tuple[bool, Optional[str]]:
    """Verify if claim is supported by context."""
    prompt = f"""Determine if this claim is supported by the context.

Claim: {claim}

Context:
{context}

Answer with:
- "SUPPORTED" if the claim can be inferred from the context
- "NOT_SUPPORTED" if the claim cannot be inferred or contradicts the context

Also quote the relevant supporting text if supported.

Format:
VERDICT: [SUPPORTED/NOT_SUPPORTED]
EVIDENCE: [quote or "none"]"""

    result = await llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    content = result.choices[0].message.content
    is_supported = "SUPPORTED" in content.split("\n")[0].upper()
    
    # Extract evidence
    evidence = None
    if "EVIDENCE:" in content:
        evidence_line = content.split("EVIDENCE:")[1].strip()
        if evidence_line.lower() != "none":
            evidence = evidence_line
    
    return is_supported, evidence
```

### Example Faithfulness Evaluation

```python
# Good example - faithful response
response_faithful = "Einstein published his theory of special relativity in 1905."
contexts = ["In 1905, Albert Einstein published four groundbreaking papers, including the special theory of relativity."]

result = await evaluate_faithfulness(response_faithful, contexts, client)
print(f"Faithfulness: {result.score:.2f}")  # 1.0

# Bad example - hallucinated response
response_hallucinated = "Einstein published his theory of special relativity in 1905. He won the Nobel Prize for this work in 1921."
# Context doesn't mention Nobel Prize

result = await evaluate_faithfulness(response_hallucinated, contexts, client)
print(f"Faithfulness: {result.score:.2f}")  # 0.5 (one of two claims unsupported)
```

---

## Answer Relevancy

Answer relevancy measures how well the response addresses the original question.

### The Relevancy Approach

RAGAS uses reverse question generation:
1. Generate questions from the answer
2. Compare generated questions to original
3. High similarity = high relevancy

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class AnswerRelevancyResult:
    score: float
    generated_questions: list[str]
    similarities: list[float]

async def evaluate_answer_relevancy(
    query: str,
    response: str,
    llm_client,
    embedding_client,
    model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    num_questions: int = 3
) -> AnswerRelevancyResult:
    """
    Evaluate answer relevancy using reverse question generation.
    """
    # Step 1: Generate questions from the answer
    prompt = f"""Based on this answer, generate {num_questions} questions that it could be answering.
    
Answer: {response}

Generate {num_questions} diverse questions that this answer addresses.
Format as a numbered list."""

    result = await llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    # Parse questions
    lines = result.choices[0].message.content.strip().split("\n")
    generated_questions = []
    for line in lines:
        if ". " in line:
            q = line.split(". ", 1)[1].strip()
            if q:
                generated_questions.append(q)
    
    # Step 2: Get embeddings
    all_texts = [query] + generated_questions
    embeddings_response = await embedding_client.embeddings.create(
        model=embedding_model,
        input=all_texts
    )
    
    embeddings = [e.embedding for e in embeddings_response.data]
    query_embedding = np.array(embeddings[0])
    question_embeddings = [np.array(e) for e in embeddings[1:]]
    
    # Step 3: Calculate cosine similarities
    similarities = []
    for q_emb in question_embeddings:
        sim = np.dot(query_embedding, q_emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(q_emb)
        )
        similarities.append(float(sim))
    
    # Step 4: Average similarity
    avg_score = sum(similarities) / len(similarities) if similarities else 0.0
    
    return AnswerRelevancyResult(
        score=avg_score,
        generated_questions=generated_questions,
        similarities=similarities
    )
```

### Simple Relevancy Check

For cases where embedding-based comparison is overkill:

```python
async def simple_answer_relevancy(
    query: str,
    response: str,
    llm_client,
    model: str = "gpt-4o-mini"
) -> float:
    """
    Simple LLM-based relevancy check.
    """
    prompt = f"""Rate how well this answer addresses the question.

Question: {query}

Answer: {response}

Consider:
1. Does the answer directly address what was asked?
2. Is the answer complete or does it miss key aspects?
3. Does it include unnecessary information?

Rate from 0.0 (completely off-topic) to 1.0 (perfectly relevant).
Return only a number."""

    result = await llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    try:
        score = float(result.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.5
```

---

## Combined Quality Scores

Combine individual metrics into an overall quality score.

### Weighted Average

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGQualityScore:
    context_relevance: float
    faithfulness: float
    answer_relevancy: float
    
    # Weights for each metric
    context_weight: float = 0.3
    faithfulness_weight: float = 0.4
    relevancy_weight: float = 0.3
    
    @property
    def weighted_average(self) -> float:
        """Calculate weighted average score."""
        total = (
            self.context_relevance * self.context_weight +
            self.faithfulness * self.faithfulness_weight +
            self.answer_relevancy * self.relevancy_weight
        )
        return total
    
    @property
    def harmonic_mean(self) -> float:
        """Calculate harmonic mean (penalizes low scores more)."""
        scores = [self.context_relevance, self.faithfulness, self.answer_relevancy]
        if 0 in scores:
            return 0.0
        return 3 / sum(1/s for s in scores)
    
    @property
    def minimum(self) -> float:
        """Return the minimum score (weakest link)."""
        return min(self.context_relevance, self.faithfulness, self.answer_relevancy)
    
    def get_grade(self) -> str:
        """Convert to letter grade."""
        score = self.weighted_average
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
```

### Complete Evaluation Pipeline

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGEvaluationResult:
    query: str
    response: str
    contexts: list[str]
    
    context_relevance: float
    faithfulness: float
    answer_relevancy: float
    
    quality_score: RAGQualityScore
    
    # Detailed results
    context_details: Optional[ContextRelevanceResult] = None
    faithfulness_details: Optional[FaithfulnessResult] = None
    relevancy_details: Optional[AnswerRelevancyResult] = None

class RAGEvaluator:
    """Complete RAG evaluation pipeline."""
    
    def __init__(self, llm_client, embedding_client):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
    
    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[str],
        include_details: bool = True
    ) -> RAGEvaluationResult:
        """Run full evaluation on a RAG response."""
        
        # Evaluate context relevance
        context_result = await evaluate_context_relevance(
            query, contexts, self.llm_client
        )
        
        # Evaluate faithfulness
        faith_result = await evaluate_faithfulness(
            response, contexts, self.llm_client
        )
        
        # Evaluate answer relevancy
        relevancy_result = await evaluate_answer_relevancy(
            query, response, self.llm_client, self.embedding_client
        )
        
        # Combine scores
        quality = RAGQualityScore(
            context_relevance=context_result.score,
            faithfulness=faith_result.score,
            answer_relevancy=relevancy_result.score
        )
        
        return RAGEvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            context_relevance=context_result.score,
            faithfulness=faith_result.score,
            answer_relevancy=relevancy_result.score,
            quality_score=quality,
            context_details=context_result if include_details else None,
            faithfulness_details=faith_result if include_details else None,
            relevancy_details=relevancy_result if include_details else None
        )
    
    async def evaluate_batch(
        self,
        samples: list[dict]
    ) -> list[RAGEvaluationResult]:
        """Evaluate multiple samples."""
        results = []
        for sample in samples:
            result = await self.evaluate(
                query=sample["query"],
                response=sample["response"],
                contexts=sample["contexts"],
                include_details=False
            )
            results.append(result)
        return results
    
    def aggregate_results(
        self,
        results: list[RAGEvaluationResult]
    ) -> dict:
        """Aggregate evaluation results."""
        if not results:
            return {}
        
        return {
            "count": len(results),
            "avg_context_relevance": sum(r.context_relevance for r in results) / len(results),
            "avg_faithfulness": sum(r.faithfulness for r in results) / len(results),
            "avg_answer_relevancy": sum(r.answer_relevancy for r in results) / len(results),
            "avg_overall": sum(r.quality_score.weighted_average for r in results) / len(results),
            "min_overall": min(r.quality_score.weighted_average for r in results),
            "max_overall": max(r.quality_score.weighted_average for r in results)
        }
```

---

## Hands-on Exercise

### Your Task

Build a `RAGMetricsDashboard` that:
1. Evaluates multiple RAG responses
2. Tracks metrics over time
3. Identifies problematic responses
4. Generates summary reports

### Requirements

```python
class RAGMetricsDashboard:
    def add_evaluation(
        self,
        query: str,
        response: str,
        contexts: list[str],
        scores: dict
    ) -> str:
        """Add evaluation result, return ID."""
        pass
    
    def get_summary(self) -> dict:
        """Get aggregate statistics."""
        pass
    
    def get_problems(self, threshold: float = 0.6) -> list[dict]:
        """Get evaluations below threshold."""
        pass
    
    def export_report(self, format: str = "markdown") -> str:
        """Export summary report."""
        pass
```

<details>
<summary>💡 Hints</summary>

- Store evaluations with timestamps and IDs
- Calculate running averages for each metric
- Use quality score to identify problems
- Include trend data in report

</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

@dataclass
class EvaluationRecord:
    id: str
    query: str
    response: str
    contexts: list[str]
    context_relevance: float
    faithfulness: float
    answer_relevancy: float
    overall: float
    timestamp: datetime = field(default_factory=datetime.now)

class RAGMetricsDashboard:
    def __init__(self):
        self.evaluations: list[EvaluationRecord] = []
    
    def add_evaluation(
        self,
        query: str,
        response: str,
        contexts: list[str],
        scores: dict
    ) -> str:
        """Add evaluation result, return ID."""
        eval_id = str(uuid.uuid4())[:8]
        
        context_rel = scores.get("context_relevance", 0.0)
        faith = scores.get("faithfulness", 0.0)
        relevancy = scores.get("answer_relevancy", 0.0)
        
        # Calculate overall (harmonic mean)
        scores_list = [context_rel, faith, relevancy]
        if 0 in scores_list:
            overall = 0.0
        else:
            overall = 3 / sum(1/s for s in scores_list)
        
        record = EvaluationRecord(
            id=eval_id,
            query=query,
            response=response,
            contexts=contexts,
            context_relevance=context_rel,
            faithfulness=faith,
            answer_relevancy=relevancy,
            overall=overall
        )
        
        self.evaluations.append(record)
        return eval_id
    
    def get_summary(self) -> dict:
        """Get aggregate statistics."""
        if not self.evaluations:
            return {"count": 0}
        
        n = len(self.evaluations)
        
        return {
            "count": n,
            "avg_context_relevance": sum(e.context_relevance for e in self.evaluations) / n,
            "avg_faithfulness": sum(e.faithfulness for e in self.evaluations) / n,
            "avg_answer_relevancy": sum(e.answer_relevancy for e in self.evaluations) / n,
            "avg_overall": sum(e.overall for e in self.evaluations) / n,
            "min_overall": min(e.overall for e in self.evaluations),
            "max_overall": max(e.overall for e in self.evaluations),
            "first_eval": self.evaluations[0].timestamp.isoformat(),
            "last_eval": self.evaluations[-1].timestamp.isoformat()
        }
    
    def get_problems(self, threshold: float = 0.6) -> list[dict]:
        """Get evaluations below threshold."""
        problems = []
        
        for e in self.evaluations:
            issues = []
            
            if e.context_relevance < threshold:
                issues.append(f"Low context relevance: {e.context_relevance:.2f}")
            if e.faithfulness < threshold:
                issues.append(f"Low faithfulness: {e.faithfulness:.2f}")
            if e.answer_relevancy < threshold:
                issues.append(f"Low answer relevancy: {e.answer_relevancy:.2f}")
            
            if issues:
                problems.append({
                    "id": e.id,
                    "query": e.query[:100],
                    "overall": e.overall,
                    "issues": issues,
                    "timestamp": e.timestamp.isoformat()
                })
        
        # Sort by overall score ascending (worst first)
        return sorted(problems, key=lambda p: p["overall"])
    
    def export_report(self, format: str = "markdown") -> str:
        """Export summary report."""
        summary = self.get_summary()
        problems = self.get_problems()
        
        if format == "markdown":
            return self._markdown_report(summary, problems)
        else:
            return str({"summary": summary, "problems": problems})
    
    def _markdown_report(self, summary: dict, problems: list) -> str:
        lines = [
            "# RAG Evaluation Report",
            "",
            f"**Evaluations:** {summary.get('count', 0)}",
            f"**Period:** {summary.get('first_eval', 'N/A')} to {summary.get('last_eval', 'N/A')}",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Average |",
            "|--------|---------|",
            f"| Context Relevance | {summary.get('avg_context_relevance', 0):.2%} |",
            f"| Faithfulness | {summary.get('avg_faithfulness', 0):.2%} |",
            f"| Answer Relevancy | {summary.get('avg_answer_relevancy', 0):.2%} |",
            f"| Overall | {summary.get('avg_overall', 0):.2%} |",
            "",
            "## Problem Areas",
            "",
            f"**Total problems found:** {len(problems)}",
            ""
        ]
        
        if problems:
            for p in problems[:10]:  # Top 10 worst
                lines.append(f"### ID: {p['id']} (Score: {p['overall']:.2%})")
                lines.append(f"**Query:** {p['query']}")
                lines.append("**Issues:**")
                for issue in p['issues']:
                    lines.append(f"- {issue}")
                lines.append("")
        
        return "\n".join(lines)

# Usage
dashboard = RAGMetricsDashboard()

# Add evaluations
dashboard.add_evaluation(
    query="What is machine learning?",
    response="Machine learning is a subset of AI that learns from data.",
    contexts=["ML is a branch of AI focused on learning from data."],
    scores={
        "context_relevance": 0.95,
        "faithfulness": 0.90,
        "answer_relevancy": 0.85
    }
)

dashboard.add_evaluation(
    query="Who invented Python?",
    response="Python was invented by Larry Wall.",  # Wrong!
    contexts=["Python was created by Guido van Rossum."],
    scores={
        "context_relevance": 0.8,
        "faithfulness": 0.2,  # Hallucination
        "answer_relevancy": 0.7
    }
)

# Get report
print(dashboard.export_report())
```

</details>

---

## Summary

RAG-specific metrics capture what matters for retrieval-augmented systems:

✅ **Context Relevance** — Are retrieved chunks useful for the query?
✅ **Faithfulness** — Is the answer grounded in context (not hallucinated)?
✅ **Answer Relevancy** — Does the answer address what was asked?
✅ **Combined Scores** — Harmonic mean or weighted average for overall quality

**Key insight:** Low faithfulness + high relevancy = the answer sounds good but isn't grounded.

**Next:** [Retrieval Quality Evaluation](./02-retrieval-quality.md)

---

## Further Reading

- [RAGAS Metrics Documentation](https://docs.ragas.io/en/stable/concepts/metrics/)
- [TruLens RAG Evaluation](https://www.trulens.org/)
- [RAG Triad Research](https://arxiv.org/abs/2309.15217)

<!--
Sources Consulted:
- RAGAS official documentation (2026-02)
- Faithfulness evaluation papers
- Answer relevancy research
-->
