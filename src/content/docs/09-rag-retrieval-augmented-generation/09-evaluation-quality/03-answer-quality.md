---
title: "Answer Quality Evaluation"
---

# Answer Quality Evaluation

## Introduction

While retrieval metrics tell us if we found the right context, answer quality metrics evaluate what the LLM generates. A RAG system can retrieve perfect context yet still produce poor, incomplete, or incoherent answers.

This lesson covers metrics for evaluating the quality of generated responses.

### What We'll Cover

- Correctness and factual accuracy
- Completeness evaluation
- Coherence and fluency
- Helpfulness assessment

### Prerequisites

- Understanding of RAG generation
- Familiarity with LLM evaluation concepts
- Lesson 01: RAG-Specific Metrics

---

## Correctness

Correctness measures whether the answer contains accurate information that matches expected output.

### Definition

| Aspect | Question |
|--------|----------|
| Factual Accuracy | Are the facts in the answer correct? |
| Reference Match | Does it match the expected/reference answer? |
| No Hallucination | Is everything grounded in source material? |

```python
from dataclasses import dataclass
from openai import AsyncOpenAI
import json

@dataclass
class CorrectnessResult:
    score: float  # 0.0 to 1.0
    is_correct: bool
    matching_facts: list[str]
    incorrect_facts: list[str]
    missing_facts: list[str]
    explanation: str

async def evaluate_correctness(
    question: str,
    answer: str,
    reference: str,
    client: AsyncOpenAI
) -> CorrectnessResult:
    """
    Evaluate answer correctness against a reference answer.
    """
    prompt = f"""Evaluate the correctness of this answer compared to the reference.

Question: {question}

Generated Answer: {answer}

Reference Answer: {reference}

Analyze the answer and return JSON:
{{
    "matching_facts": ["fact1", "fact2"],  // Facts that correctly match reference
    "incorrect_facts": ["fact1"],          // Facts that contradict reference
    "missing_facts": ["fact1"],            // Important facts from reference not covered
    "score": 0.0-1.0,                      // Overall correctness score
    "explanation": "..."                   // Brief explanation
}}
"""
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    return CorrectnessResult(
        score=result["score"],
        is_correct=result["score"] >= 0.8,
        matching_facts=result["matching_facts"],
        incorrect_facts=result["incorrect_facts"],
        missing_facts=result["missing_facts"],
        explanation=result["explanation"]
    )

# Example usage
async def demo():
    client = AsyncOpenAI()
    
    result = await evaluate_correctness(
        question="What is the capital of France?",
        answer="Paris is the capital and largest city of France.",
        reference="Paris is the capital of France.",
        client=client
    )
    
    print(f"Score: {result.score}")
    print(f"Matching: {result.matching_facts}")
    print(f"Missing: {result.missing_facts}")
```

### Semantic Similarity Approach

For cases without explicit fact extraction:

```python
from openai import AsyncOpenAI
import numpy as np

async def correctness_by_similarity(
    answer: str,
    reference: str,
    client: AsyncOpenAI,
    threshold: float = 0.85
) -> tuple[float, bool]:
    """
    Evaluate correctness using semantic similarity.
    """
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=[answer, reference]
    )
    
    emb_answer = response.data[0].embedding
    emb_reference = response.data[1].embedding
    
    # Cosine similarity
    similarity = np.dot(emb_answer, emb_reference) / (
        np.linalg.norm(emb_answer) * np.linalg.norm(emb_reference)
    )
    
    return float(similarity), similarity >= threshold

# Usage
similarity, is_correct = await correctness_by_similarity(
    "Paris is France's capital city.",
    "The capital of France is Paris.",
    client
)
print(f"Similarity: {similarity:.3f}, Correct: {is_correct}")
```

---

## Completeness

Completeness measures whether the answer covers all aspects of the question.

```python
from dataclasses import dataclass
from typing import List

@dataclass
class CompletenessResult:
    score: float
    covered_aspects: List[str]
    missing_aspects: List[str]
    partial_aspects: List[str]

async def evaluate_completeness(
    question: str,
    answer: str,
    required_aspects: List[str],
    client: AsyncOpenAI
) -> CompletenessResult:
    """
    Evaluate if answer covers all required aspects.
    """
    prompt = f"""Evaluate how completely this answer addresses the question.

Question: {question}

Answer: {answer}

Required Aspects to Cover:
{json.dumps(required_aspects, indent=2)}

For each required aspect, determine if it is:
- "covered": Fully addressed in the answer
- "partial": Mentioned but not fully explained
- "missing": Not addressed at all

Return JSON:
{{
    "aspect_coverage": {{"aspect": "covered|partial|missing", ...}},
    "score": 0.0-1.0
}}
"""
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    coverage = result["aspect_coverage"]
    
    return CompletenessResult(
        score=result["score"],
        covered_aspects=[a for a, s in coverage.items() if s == "covered"],
        missing_aspects=[a for a, s in coverage.items() if s == "missing"],
        partial_aspects=[a for a, s in coverage.items() if s == "partial"]
    )

# Example
result = await evaluate_completeness(
    question="Explain how RAG works and its benefits",
    answer="RAG retrieves relevant documents and uses them as context for LLM generation.",
    required_aspects=["retrieval process", "generation process", "benefits", "use cases"],
    client=client
)

print(f"Score: {result.score}")
print(f"Covered: {result.covered_aspects}")
print(f"Missing: {result.missing_aspects}")
```

### Aspect-Based Scoring

```python
from dataclasses import dataclass
from typing import Dict

@dataclass 
class AspectScore:
    aspect: str
    score: float  # 0.0 to 1.0
    weight: float
    feedback: str

class AspectBasedEvaluator:
    """Evaluate answer across multiple weighted aspects."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        aspects: Dict[str, float]  # aspect name -> weight
    ) -> tuple[float, List[AspectScore]]:
        """
        Evaluate answer on multiple aspects with weights.
        
        Args:
            question: The question asked
            answer: The answer to evaluate
            aspects: Dict mapping aspect names to their weights
        
        Returns:
            Overall weighted score and individual aspect scores
        """
        aspect_prompt = "\n".join([
            f"- {aspect}" for aspect in aspects.keys()
        ])
        
        prompt = f"""Evaluate this answer on the following aspects.

Question: {question}
Answer: {answer}

Rate each aspect from 0.0 to 1.0:
{aspect_prompt}

Return JSON:
{{
    "scores": {{"aspect_name": {{"score": 0.0-1.0, "feedback": "..."}}, ...}}
}}
"""
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        aspect_scores = []
        total_weight = sum(aspects.values())
        weighted_sum = 0.0
        
        for aspect, weight in aspects.items():
            score_data = result["scores"].get(aspect, {"score": 0, "feedback": "Not evaluated"})
            
            aspect_score = AspectScore(
                aspect=aspect,
                score=score_data["score"],
                weight=weight,
                feedback=score_data["feedback"]
            )
            aspect_scores.append(aspect_score)
            weighted_sum += aspect_score.score * weight
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return overall_score, aspect_scores

# Usage
evaluator = AspectBasedEvaluator(client)

score, aspects = await evaluator.evaluate(
    question="What is machine learning?",
    answer="Machine learning is a type of AI that learns from data.",
    aspects={
        "accuracy": 1.0,
        "completeness": 1.0,
        "clarity": 0.5,
        "examples": 0.5
    }
)

print(f"Overall: {score:.2f}")
for a in aspects:
    print(f"  {a.aspect}: {a.score:.2f} (weight: {a.weight})")
```

---

## Coherence

Coherence measures how well-structured and logical the answer is.

```python
from dataclasses import dataclass
from enum import Enum

class CoherenceLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class CoherenceResult:
    score: float
    level: CoherenceLevel
    logical_flow: float
    consistency: float
    organization: float
    issues: list[str]

async def evaluate_coherence(
    answer: str,
    client: AsyncOpenAI
) -> CoherenceResult:
    """
    Evaluate the coherence of an answer.
    """
    prompt = f"""Evaluate the coherence of this text.

Text: {answer}

Rate these aspects from 0.0 to 1.0:
1. Logical Flow: Ideas connect logically from one to the next
2. Consistency: No contradictions within the text
3. Organization: Clear structure with introduction, body, conclusion (if applicable)

Also list any coherence issues found.

Return JSON:
{{
    "logical_flow": 0.0-1.0,
    "consistency": 0.0-1.0,
    "organization": 0.0-1.0,
    "issues": ["issue1", "issue2"],
    "overall_score": 0.0-1.0
}}
"""
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    score = result["overall_score"]
    
    # Map score to level
    if score >= 0.9:
        level = CoherenceLevel.EXCELLENT
    elif score >= 0.7:
        level = CoherenceLevel.GOOD
    elif score >= 0.5:
        level = CoherenceLevel.FAIR
    else:
        level = CoherenceLevel.POOR
    
    return CoherenceResult(
        score=score,
        level=level,
        logical_flow=result["logical_flow"],
        consistency=result["consistency"],
        organization=result["organization"],
        issues=result["issues"]
    )
```

### Fluency Evaluation

```python
@dataclass
class FluencyResult:
    score: float
    grammar_score: float
    readability_score: float
    natural_score: float
    issues: list[str]

async def evaluate_fluency(
    answer: str,
    client: AsyncOpenAI
) -> FluencyResult:
    """
    Evaluate grammatical correctness and natural language flow.
    """
    prompt = f"""Evaluate the fluency of this text.

Text: {answer}

Rate from 0.0 to 1.0:
1. Grammar: Grammatically correct with proper punctuation
2. Readability: Easy to read and understand
3. Natural: Sounds natural, not robotic or awkward

Return JSON:
{{
    "grammar": 0.0-1.0,
    "readability": 0.0-1.0,
    "natural": 0.0-1.0,
    "issues": ["issue1"],
    "overall_score": 0.0-1.0
}}
"""
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    return FluencyResult(
        score=result["overall_score"],
        grammar_score=result["grammar"],
        readability_score=result["readability"],
        natural_score=result["natural"],
        issues=result["issues"]
    )
```

---

## Helpfulness

Helpfulness measures whether the answer actually helps the user accomplish their goal.

```python
from dataclasses import dataclass

@dataclass
class HelpfulnessResult:
    score: float
    addresses_intent: bool
    actionable: bool
    appropriate_detail: bool
    feedback: str

async def evaluate_helpfulness(
    question: str,
    answer: str,
    user_context: str,  # What the user is trying to accomplish
    client: AsyncOpenAI
) -> HelpfulnessResult:
    """
    Evaluate how helpful the answer is for the user's actual needs.
    """
    prompt = f"""Evaluate how helpful this answer is for the user.

User's Question: {question}
User's Context/Goal: {user_context}
Answer Provided: {answer}

Evaluate:
1. Addresses Intent: Does it answer what the user actually wants to know?
2. Actionable: Can the user take action based on this answer?
3. Appropriate Detail: Right level of detail (not too brief, not overwhelming)?

Return JSON:
{{
    "addresses_intent": true/false,
    "actionable": true/false,
    "appropriate_detail": true/false,
    "score": 0.0-1.0,
    "feedback": "..."
}}
"""
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    return HelpfulnessResult(
        score=result["score"],
        addresses_intent=result["addresses_intent"],
        actionable=result["actionable"],
        appropriate_detail=result["appropriate_detail"],
        feedback=result["feedback"]
    )

# Example
result = await evaluate_helpfulness(
    question="How do I fix this Python error?",
    answer="Python is a programming language created by Guido van Rossum.",
    user_context="User is debugging a TypeError in their code",
    client=client
)

print(f"Helpfulness: {result.score}")
print(f"Addresses intent: {result.addresses_intent}")  # False
print(f"Feedback: {result.feedback}")
```

---

## Combined Answer Quality Evaluator

```python
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum

class QualityGrade(Enum):
    EXCELLENT = "A"
    GOOD = "B"
    FAIR = "C"
    POOR = "D"
    FAILING = "F"

@dataclass
class AnswerQualityResult:
    """Complete answer quality evaluation."""
    
    # Individual scores
    correctness: float
    completeness: float
    coherence: float
    fluency: float
    helpfulness: float
    
    # Overall
    overall_score: float
    grade: QualityGrade
    
    # Details
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]

class AnswerQualityEvaluator:
    """Comprehensive answer quality evaluation."""
    
    def __init__(
        self,
        client: AsyncOpenAI,
        weights: Optional[Dict[str, float]] = None
    ):
        self.client = client
        self.weights = weights or {
            "correctness": 0.30,
            "completeness": 0.25,
            "coherence": 0.15,
            "fluency": 0.10,
            "helpfulness": 0.20
        }
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        reference: Optional[str] = None,
        context: Optional[str] = None
    ) -> AnswerQualityResult:
        """
        Evaluate answer quality across all dimensions.
        """
        # Build comprehensive prompt
        prompt = f"""Evaluate this answer comprehensively.

Question: {question}
Answer: {answer}
"""
        
        if reference:
            prompt += f"\nReference Answer: {reference}"
        
        if context:
            prompt += f"\nContext/Source: {context}"
        
        prompt += """

Rate each dimension from 0.0 to 1.0:
- Correctness: Factual accuracy, no errors
- Completeness: Covers all aspects of the question
- Coherence: Logical, well-organized
- Fluency: Natural, readable language
- Helpfulness: Actually useful for the user

Also provide:
- 2-3 specific strengths
- 2-3 specific weaknesses
- 2-3 improvement suggestions

Return JSON:
{
    "correctness": 0.0-1.0,
    "completeness": 0.0-1.0,
    "coherence": 0.0-1.0,
    "fluency": 0.0-1.0,
    "helpfulness": 0.0-1.0,
    "strengths": ["...", "..."],
    "weaknesses": ["...", "..."],
    "suggestions": ["...", "..."]
}
"""
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Calculate weighted overall score
        overall = sum(
            result[dim] * weight
            for dim, weight in self.weights.items()
        )
        
        # Determine grade
        if overall >= 0.9:
            grade = QualityGrade.EXCELLENT
        elif overall >= 0.8:
            grade = QualityGrade.GOOD
        elif overall >= 0.7:
            grade = QualityGrade.FAIR
        elif overall >= 0.6:
            grade = QualityGrade.POOR
        else:
            grade = QualityGrade.FAILING
        
        return AnswerQualityResult(
            correctness=result["correctness"],
            completeness=result["completeness"],
            coherence=result["coherence"],
            fluency=result["fluency"],
            helpfulness=result["helpfulness"],
            overall_score=overall,
            grade=grade,
            strengths=result["strengths"],
            weaknesses=result["weaknesses"],
            suggestions=result["suggestions"]
        )
    
    async def evaluate_batch(
        self,
        samples: list[dict]
    ) -> dict:
        """
        Evaluate multiple samples and aggregate.
        
        Each sample: {'question': str, 'answer': str, 'reference': optional str}
        """
        results = []
        
        for sample in samples:
            result = await self.evaluate(
                question=sample['question'],
                answer=sample['answer'],
                reference=sample.get('reference'),
                context=sample.get('context')
            )
            results.append(result)
        
        # Aggregate
        n = len(results)
        return {
            'count': n,
            'mean_correctness': sum(r.correctness for r in results) / n,
            'mean_completeness': sum(r.completeness for r in results) / n,
            'mean_coherence': sum(r.coherence for r in results) / n,
            'mean_fluency': sum(r.fluency for r in results) / n,
            'mean_helpfulness': sum(r.helpfulness for r in results) / n,
            'mean_overall': sum(r.overall_score for r in results) / n,
            'grade_distribution': {
                grade.value: sum(1 for r in results if r.grade == grade)
                for grade in QualityGrade
            }
        }

# Usage
evaluator = AnswerQualityEvaluator(client)

result = await evaluator.evaluate(
    question="What is RAG?",
    answer="RAG (Retrieval Augmented Generation) combines document retrieval with LLM generation to provide accurate, grounded answers.",
    reference="RAG is a technique that retrieves relevant documents and uses them as context for language model generation."
)

print(f"Grade: {result.grade.value}")
print(f"Overall: {result.overall_score:.2f}")
print(f"Strengths: {result.strengths}")
print(f"Suggestions: {result.suggestions}")
```

---

## Hands-on Exercise

### Your Task

Build an `AnswerComparisonEvaluator` that:
1. Takes a question and two candidate answers
2. Evaluates both on quality dimensions
3. Determines which answer is better
4. Explains why with specific reasons

### Requirements

```python
class AnswerComparisonEvaluator:
    async def compare(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        reference: str = None
    ) -> ComparisonResult:
        pass

@dataclass
class ComparisonResult:
    winner: str  # "A", "B", or "tie"
    score_a: float
    score_b: float
    dimension_comparison: dict  # {dimension: "A"|"B"|"tie"}
    explanation: str
```

<details>
<summary>💡 Hints</summary>

- Evaluate both answers on the same dimensions
- Compare dimension by dimension
- Winner is the one with higher weighted score
- Explain differences clearly

</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass
from typing import Dict, Optional
import json
from openai import AsyncOpenAI

@dataclass
class ComparisonResult:
    winner: str  # "A", "B", or "tie"
    score_a: float
    score_b: float
    dimension_comparison: Dict[str, str]
    explanation: str
    
    @property
    def margin(self) -> float:
        return abs(self.score_a - self.score_b)

class AnswerComparisonEvaluator:
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.dimensions = [
            "correctness",
            "completeness", 
            "coherence",
            "helpfulness"
        ]
    
    async def compare(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        reference: Optional[str] = None
    ) -> ComparisonResult:
        """Compare two answers and determine which is better."""
        
        prompt = f"""Compare these two answers to the same question.

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}
"""
        
        if reference:
            prompt += f"\nReference Answer: {reference}"
        
        prompt += """

For each dimension, rate both answers from 0.0 to 1.0 and declare a winner:
- correctness: Factual accuracy
- completeness: Covers the question fully
- coherence: Well-organized and logical
- helpfulness: Useful for the user

Return JSON:
{
    "scores_a": {"correctness": 0.0-1.0, "completeness": 0.0-1.0, ...},
    "scores_b": {"correctness": 0.0-1.0, "completeness": 0.0-1.0, ...},
    "dimension_winners": {"correctness": "A"|"B"|"tie", ...},
    "overall_winner": "A"|"B"|"tie",
    "explanation": "Clear explanation of why the winner is better"
}
"""
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Calculate overall scores
        score_a = sum(result["scores_a"].values()) / len(self.dimensions)
        score_b = sum(result["scores_b"].values()) / len(self.dimensions)
        
        return ComparisonResult(
            winner=result["overall_winner"],
            score_a=score_a,
            score_b=score_b,
            dimension_comparison=result["dimension_winners"],
            explanation=result["explanation"]
        )
    
    async def rank_answers(
        self,
        question: str,
        answers: list[str],
        reference: Optional[str] = None
    ) -> list[tuple[int, float]]:
        """
        Rank multiple answers by quality.
        
        Returns: List of (original_index, score) sorted by score descending
        """
        scores = []
        
        for i, answer in enumerate(answers):
            score = await self._score_single(question, answer, reference)
            scores.append((i, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    async def _score_single(
        self,
        question: str,
        answer: str,
        reference: Optional[str]
    ) -> float:
        """Score a single answer."""
        prompt = f"""Rate this answer on a scale of 0.0 to 1.0.

Question: {question}
Answer: {answer}
"""
        if reference:
            prompt += f"Reference: {reference}"
        
        prompt += "\n\nReturn JSON: {\"score\": 0.0-1.0}"
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result["score"]

# Usage
async def demo():
    client = AsyncOpenAI()
    evaluator = AnswerComparisonEvaluator(client)
    
    result = await evaluator.compare(
        question="What is machine learning?",
        answer_a="Machine learning is AI that learns from data to make predictions.",
        answer_b="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data.",
        reference="Machine learning is a type of AI that learns patterns from data."
    )
    
    print(f"Winner: Answer {result.winner}")
    print(f"Scores: A={result.score_a:.2f}, B={result.score_b:.2f}")
    print(f"Margin: {result.margin:.2f}")
    print(f"\nDimension comparison:")
    for dim, winner in result.dimension_comparison.items():
        print(f"  {dim}: {winner}")
    print(f"\nExplanation: {result.explanation}")

# Run
import asyncio
asyncio.run(demo())
```

</details>

---

## Summary

Answer quality evaluation covers multiple dimensions:

✅ **Correctness** — Factual accuracy and reference matching
✅ **Completeness** — Covers all aspects of the question
✅ **Coherence** — Logical flow and organization
✅ **Helpfulness** — Actually useful for the user's needs

**Key insight:** High correctness with low helpfulness = technically right but misses the user's intent. Balance all dimensions.

**Next:** [Faithfulness Checking](./04-faithfulness-checking.md)

---

## Further Reading

- [G-Eval: NLG Evaluation with GPT-4](https://arxiv.org/abs/2303.16634)
- [BLEURT: Learning Robust Metrics](https://arxiv.org/abs/2004.04696)
- [Semantic Similarity Metrics](https://huggingface.co/tasks/sentence-similarity)

<!--
Sources Consulted:
- G-Eval paper on LLM-based evaluation
- RAGAS answer quality metrics
- Google evaluation guidelines
-->
