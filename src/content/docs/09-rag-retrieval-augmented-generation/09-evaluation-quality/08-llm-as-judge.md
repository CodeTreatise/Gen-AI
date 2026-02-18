---
title: "LLM-as-Judge Patterns"
---

# LLM-as-Judge Patterns

## Introduction

Using LLMs to evaluate LLM outputs is a powerful pattern that enables scalable, nuanced evaluation. Unlike rigid metrics, LLM judges can assess subjective qualities like helpfulness, tone, and contextual appropriateness.

This lesson covers practical LLM-as-judge patterns, calibration techniques, and common pitfalls.

### What We'll Cover

- Pairwise comparison
- Rubric-based scoring
- Aspect-based evaluation
- Calibration and bias mitigation

### Prerequisites

- LLM API usage
- Prompt engineering basics
- Understanding of evaluation challenges

---

## Why LLM-as-Judge?

| Traditional Metrics | LLM-as-Judge |
|---------------------|--------------|
| BLEU, ROUGE, exact match | Semantic understanding |
| Fast but shallow | Slower but nuanced |
| Can't assess subjective qualities | Evaluates helpfulness, tone, safety |
| Brittle to paraphrasing | Robust to variations |
| Deterministic | Probabilistic (needs calibration) |

---

## Pairwise Comparison

Compare two responses and pick the better one.

### Basic Implementation

```python
from dataclasses import dataclass
from typing import Literal
from openai import AsyncOpenAI
import json

@dataclass
class PairwiseResult:
    winner: Literal["A", "B", "tie"]
    confidence: float
    reasoning: str
    aspect_winners: dict  # Which won on each aspect

async def pairwise_compare(
    question: str,
    response_a: str,
    response_b: str,
    client: AsyncOpenAI
) -> PairwiseResult:
    """
    Compare two responses and determine which is better.
    """
    prompt = f"""Compare these two responses to the same question.

Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Evaluate on these aspects:
1. Accuracy: Which is more factually correct?
2. Completeness: Which answers the question more fully?
3. Clarity: Which is easier to understand?
4. Helpfulness: Which would be more useful to the user?

Return JSON:
{{
    "aspect_winners": {{
        "accuracy": "A" | "B" | "tie",
        "completeness": "A" | "B" | "tie",
        "clarity": "A" | "B" | "tie",
        "helpfulness": "A" | "B" | "tie"
    }},
    "overall_winner": "A" | "B" | "tie",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of decision"
}}
"""
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    return PairwiseResult(
        winner=result["overall_winner"],
        confidence=result["confidence"],
        reasoning=result["reasoning"],
        aspect_winners=result["aspect_winners"]
    )

# Usage
result = await pairwise_compare(
    question="What is Python?",
    response_a="Python is a programming language.",
    response_b="Python is a high-level, interpreted programming language known for its clear syntax and versatility.",
    client=client
)

print(f"Winner: {result.winner}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

### Position Bias Mitigation

LLMs tend to prefer responses in certain positions. Mitigate by swapping:

```python
async def pairwise_compare_debiased(
    question: str,
    response_a: str,
    response_b: str,
    client: AsyncOpenAI
) -> PairwiseResult:
    """
    Compare with position bias mitigation.
    """
    # First comparison: A first
    result1 = await pairwise_compare(
        question, response_a, response_b, client
    )
    
    # Second comparison: B first (swapped)
    result2 = await pairwise_compare(
        question, response_b, response_a, client
    )
    
    # Map second result back
    result2_mapped = result2.winner
    if result2_mapped == "A":
        result2_mapped = "B"
    elif result2_mapped == "B":
        result2_mapped = "A"
    
    # Determine final winner
    if result1.winner == result2_mapped:
        # Both agree
        return PairwiseResult(
            winner=result1.winner,
            confidence=(result1.confidence + result2.confidence) / 2,
            reasoning=f"Consistent result: {result1.reasoning}",
            aspect_winners=result1.aspect_winners
        )
    else:
        # Disagreement - likely a tie
        return PairwiseResult(
            winner="tie",
            confidence=0.5,
            reasoning="Inconsistent results between position swaps",
            aspect_winners={}
        )

# Usage
result = await pairwise_compare_debiased(
    question="Explain machine learning",
    response_a="ML is...",
    response_b="Machine learning is...",
    client=client
)
```

### Tournament Ranking

Rank multiple responses using pairwise comparisons:

```python
from typing import List, Dict
import asyncio

async def tournament_rank(
    question: str,
    responses: List[str],
    client: AsyncOpenAI
) -> List[tuple[int, str]]:
    """
    Rank responses using tournament-style pairwise comparison.
    
    Returns: List of (rank, response) tuples
    """
    n = len(responses)
    wins = {i: 0 for i in range(n)}
    
    # Compare all pairs
    comparisons = []
    for i in range(n):
        for j in range(i + 1, n):
            comparisons.append((i, j))
    
    # Run comparisons in parallel
    async def compare_pair(i: int, j: int):
        result = await pairwise_compare_debiased(
            question,
            responses[i],
            responses[j],
            client
        )
        return i, j, result.winner
    
    results = await asyncio.gather(*[
        compare_pair(i, j) for i, j in comparisons
    ])
    
    # Count wins
    for i, j, winner in results:
        if winner == "A":
            wins[i] += 1
        elif winner == "B":
            wins[j] += 1
        else:  # tie
            wins[i] += 0.5
            wins[j] += 0.5
    
    # Sort by wins
    ranked = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    
    return [(rank + 1, responses[idx]) for rank, (idx, _) in enumerate(ranked)]

# Usage
responses = [
    "Python is a language.",
    "Python is a high-level programming language.",
    "Python is a versatile language created by Guido van Rossum."
]

rankings = await tournament_rank("What is Python?", responses, client)
for rank, response in rankings:
    print(f"{rank}. {response[:50]}...")
```

---

## Rubric-Based Scoring

Score responses against predefined criteria.

### Defining Rubrics

```python
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class RubricLevel:
    score: int
    description: str

@dataclass
class RubricDimension:
    name: str
    weight: float
    levels: List[RubricLevel]

@dataclass
class Rubric:
    name: str
    dimensions: List[RubricDimension]
    
    def to_prompt(self) -> str:
        """Convert rubric to prompt format."""
        lines = []
        for dim in self.dimensions:
            lines.append(f"\n{dim.name} (weight: {dim.weight}):")
            for level in dim.levels:
                lines.append(f"  {level.score}: {level.description}")
        return "\n".join(lines)

# Example rubric
rag_quality_rubric = Rubric(
    name="RAG Response Quality",
    dimensions=[
        RubricDimension(
            name="Accuracy",
            weight=0.3,
            levels=[
                RubricLevel(5, "All information is factually correct"),
                RubricLevel(4, "Mostly correct with minor errors"),
                RubricLevel(3, "Some correct information, some errors"),
                RubricLevel(2, "Multiple significant errors"),
                RubricLevel(1, "Mostly incorrect or hallucinated")
            ]
        ),
        RubricDimension(
            name="Groundedness",
            weight=0.3,
            levels=[
                RubricLevel(5, "Fully supported by retrieved context"),
                RubricLevel(4, "Mostly supported, minimal extrapolation"),
                RubricLevel(3, "Partially supported by context"),
                RubricLevel(2, "Significant unsupported claims"),
                RubricLevel(1, "Mostly unsupported/hallucinated")
            ]
        ),
        RubricDimension(
            name="Helpfulness",
            weight=0.25,
            levels=[
                RubricLevel(5, "Directly answers question with actionable info"),
                RubricLevel(4, "Answers question with useful detail"),
                RubricLevel(3, "Partially answers question"),
                RubricLevel(2, "Tangentially related to question"),
                RubricLevel(1, "Does not address the question")
            ]
        ),
        RubricDimension(
            name="Clarity",
            weight=0.15,
            levels=[
                RubricLevel(5, "Crystal clear, well-organized"),
                RubricLevel(4, "Clear with minor issues"),
                RubricLevel(3, "Understandable but could be clearer"),
                RubricLevel(2, "Confusing or poorly structured"),
                RubricLevel(1, "Incomprehensible")
            ]
        )
    ]
)
```

### Rubric Scoring

```python
@dataclass
class RubricScore:
    dimension: str
    score: int
    max_score: int
    feedback: str

@dataclass
class RubricEvaluation:
    scores: List[RubricScore]
    weighted_score: float
    raw_score: float
    max_score: float
    grade: str

async def evaluate_with_rubric(
    question: str,
    response: str,
    context: str,
    rubric: Rubric,
    client: AsyncOpenAI
) -> RubricEvaluation:
    """
    Evaluate response using rubric.
    """
    prompt = f"""Evaluate this response using the rubric below.

Question: {question}
Context Used: {context}
Response: {response}

Rubric:
{rubric.to_prompt()}

For each dimension, provide:
1. A score (1-5)
2. Brief feedback explaining the score

Return JSON:
{{
    "scores": [
        {{"dimension": "Accuracy", "score": 1-5, "feedback": "..."}},
        ...
    ]
}}
"""
    
    response_obj = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response_obj.choices[0].message.content)
    
    scores = []
    weighted_sum = 0.0
    raw_sum = 0
    
    for score_data in result["scores"]:
        dim_name = score_data["dimension"]
        dim = next((d for d in rubric.dimensions if d.name == dim_name), None)
        
        if dim:
            score = RubricScore(
                dimension=dim_name,
                score=score_data["score"],
                max_score=5,
                feedback=score_data["feedback"]
            )
            scores.append(score)
            weighted_sum += score_data["score"] * dim.weight
            raw_sum += score_data["score"]
    
    max_score = len(rubric.dimensions) * 5
    
    # Calculate grade
    percentage = (raw_sum / max_score) * 100
    if percentage >= 90:
        grade = "A"
    elif percentage >= 80:
        grade = "B"
    elif percentage >= 70:
        grade = "C"
    elif percentage >= 60:
        grade = "D"
    else:
        grade = "F"
    
    return RubricEvaluation(
        scores=scores,
        weighted_score=weighted_sum,
        raw_score=raw_sum,
        max_score=max_score,
        grade=grade
    )

# Usage
evaluation = await evaluate_with_rubric(
    question="What is the capital of France?",
    response="Paris is the capital of France, located in the north-central part of the country.",
    context="Paris is the capital and largest city of France.",
    rubric=rag_quality_rubric,
    client=client
)

print(f"Grade: {evaluation.grade}")
print(f"Weighted Score: {evaluation.weighted_score:.2f}")
for score in evaluation.scores:
    print(f"  {score.dimension}: {score.score}/5 - {score.feedback}")
```

---

## Aspect-Based Evaluation

Evaluate specific aspects independently.

```python
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

class EvaluationAspect(Enum):
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    SAFETY = "safety"
    TONE = "tone"
    CONCISENESS = "conciseness"

@dataclass
class AspectResult:
    aspect: EvaluationAspect
    score: float
    passed: bool
    feedback: str

class AspectEvaluator:
    """Evaluate responses on specific aspects."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        
        # Aspect-specific prompts
        self.aspect_prompts = {
            EvaluationAspect.RELEVANCE: """
Rate how relevant this response is to the question.
Score 0.0-1.0 where 1.0 = perfectly on-topic, 0.0 = completely off-topic.
""",
            EvaluationAspect.ACCURACY: """
Rate the factual accuracy of this response.
Score 0.0-1.0 where 1.0 = all facts correct, 0.0 = all facts wrong.
""",
            EvaluationAspect.COMPLETENESS: """
Rate how completely this response answers the question.
Score 0.0-1.0 where 1.0 = fully answers all aspects, 0.0 = doesn't address the question.
""",
            EvaluationAspect.COHERENCE: """
Rate the logical flow and organization of this response.
Score 0.0-1.0 where 1.0 = perfectly coherent, 0.0 = incoherent.
""",
            EvaluationAspect.SAFETY: """
Rate whether this response is safe and appropriate.
Score 0.0-1.0 where 1.0 = completely safe, 0.0 = harmful/unsafe.
""",
            EvaluationAspect.TONE: """
Rate whether the tone is appropriate for a helpful assistant.
Score 0.0-1.0 where 1.0 = perfectly professional, 0.0 = inappropriate tone.
""",
            EvaluationAspect.CONCISENESS: """
Rate whether the response is appropriately concise.
Score 0.0-1.0 where 1.0 = right length, 0.0 = too verbose or too brief.
"""
        }
    
    async def evaluate_aspect(
        self,
        question: str,
        response: str,
        aspect: EvaluationAspect,
        threshold: float = 0.7
    ) -> AspectResult:
        """Evaluate a single aspect."""
        
        prompt = f"""Evaluate this response.

Question: {question}
Response: {response}

{self.aspect_prompts[aspect]}

Return JSON:
{{
    "score": 0.0-1.0,
    "feedback": "Brief explanation"
}}
"""
        
        response_obj = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response_obj.choices[0].message.content)
        
        return AspectResult(
            aspect=aspect,
            score=result["score"],
            passed=result["score"] >= threshold,
            feedback=result["feedback"]
        )
    
    async def evaluate_multiple(
        self,
        question: str,
        response: str,
        aspects: List[EvaluationAspect],
        thresholds: Dict[EvaluationAspect, float] = None
    ) -> Dict[EvaluationAspect, AspectResult]:
        """Evaluate multiple aspects in parallel."""
        
        thresholds = thresholds or {}
        
        tasks = [
            self.evaluate_aspect(
                question,
                response,
                aspect,
                thresholds.get(aspect, 0.7)
            )
            for aspect in aspects
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {result.aspect: result for result in results}

# Usage
evaluator = AspectEvaluator(client)

results = await evaluator.evaluate_multiple(
    question="How do I reset my password?",
    response="To reset your password, go to Settings > Security > Reset Password.",
    aspects=[
        EvaluationAspect.RELEVANCE,
        EvaluationAspect.COMPLETENESS,
        EvaluationAspect.CLARITY
    ]
)

for aspect, result in results.items():
    status = "✅" if result.passed else "❌"
    print(f"{status} {aspect.value}: {result.score:.2f} - {result.feedback}")
```

---

## Calibration Techniques

### 1. Temperature Control

```python
# Use low temperature for consistent evaluation
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0  # Deterministic for evaluation
)
```

### 2. Reference Calibration

```python
async def calibrated_score(
    question: str,
    response: str,
    reference_responses: List[tuple[str, float]],
    client: AsyncOpenAI
) -> float:
    """
    Score response using reference calibration points.
    
    reference_responses: List of (response, known_score) tuples
    """
    # Build calibration context
    calibration = "\n".join([
        f"Example {i+1} (score: {score}):\n{resp}"
        for i, (resp, score) in enumerate(reference_responses)
    ])
    
    prompt = f"""Score this response on a scale of 0.0 to 1.0.

Use these calibrated examples as reference points:
{calibration}

Question: {question}
Response to score: {response}

Based on the calibration examples, what score should this response receive?
Return JSON: {{"score": 0.0-1.0, "reasoning": "..."}}
"""
    
    response_obj = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response_obj.choices[0].message.content)
    return result["score"]

# Usage with calibration points
references = [
    ("Python is a language.", 0.3),  # Low quality
    ("Python is a programming language.", 0.5),  # Medium
    ("Python is a high-level, interpreted programming language created by Guido van Rossum in 1991, known for its readable syntax.", 0.9)  # High quality
]

score = await calibrated_score(
    question="What is Python?",
    response="Python is a versatile programming language.",
    reference_responses=references,
    client=client
)
```

### 3. Multi-Judge Consensus

```python
async def multi_judge_evaluate(
    question: str,
    response: str,
    num_judges: int = 3,
    client: AsyncOpenAI = None
) -> tuple[float, float]:
    """
    Use multiple evaluations and aggregate.
    
    Returns: (mean_score, std_dev)
    """
    import statistics
    
    async def single_judge():
        prompt = f"""Rate this response from 0.0 to 1.0.
Question: {question}
Response: {response}
Return JSON: {{"score": 0.0-1.0}}
"""
        response_obj = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7  # Slight variation for diversity
        )
        result = json.loads(response_obj.choices[0].message.content)
        return result["score"]
    
    scores = await asyncio.gather(*[single_judge() for _ in range(num_judges)])
    
    mean_score = statistics.mean(scores)
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    
    return mean_score, std_dev

# Usage
score, uncertainty = await multi_judge_evaluate(
    question="Explain quantum computing",
    response="Quantum computing uses qubits...",
    num_judges=5,
    client=client
)

print(f"Score: {score:.2f} ± {uncertainty:.2f}")

# High uncertainty → unstable evaluation, review manually
if uncertainty > 0.2:
    print("⚠️ High uncertainty - manual review recommended")
```

---

## Complete LLM Judge System

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class JudgeMode(Enum):
    PAIRWISE = "pairwise"
    RUBRIC = "rubric"
    ASPECT = "aspect"
    CALIBRATED = "calibrated"

@dataclass
class JudgeConfig:
    mode: JudgeMode
    rubric: Optional[Rubric] = None
    aspects: Optional[List[EvaluationAspect]] = None
    calibration_points: Optional[List[tuple]] = None
    num_judges: int = 1
    threshold: float = 0.7

class LLMJudge:
    """Comprehensive LLM-as-judge system."""
    
    def __init__(self, client: AsyncOpenAI, config: JudgeConfig):
        self.client = client
        self.config = config
        self.aspect_evaluator = AspectEvaluator(client)
    
    async def evaluate(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
        comparison_response: Optional[str] = None
    ) -> Dict:
        """Evaluate based on configured mode."""
        
        if self.config.mode == JudgeMode.PAIRWISE:
            if not comparison_response:
                raise ValueError("Pairwise mode requires comparison_response")
            
            return await self._pairwise_evaluate(
                question, response, comparison_response
            )
        
        elif self.config.mode == JudgeMode.RUBRIC:
            if not self.config.rubric:
                raise ValueError("Rubric mode requires rubric config")
            
            return await self._rubric_evaluate(
                question, response, context or ""
            )
        
        elif self.config.mode == JudgeMode.ASPECT:
            if not self.config.aspects:
                raise ValueError("Aspect mode requires aspects config")
            
            return await self._aspect_evaluate(question, response)
        
        elif self.config.mode == JudgeMode.CALIBRATED:
            if not self.config.calibration_points:
                raise ValueError("Calibrated mode requires calibration_points")
            
            return await self._calibrated_evaluate(question, response)
    
    async def _pairwise_evaluate(
        self,
        question: str,
        response_a: str,
        response_b: str
    ) -> Dict:
        result = await pairwise_compare_debiased(
            question, response_a, response_b, self.client
        )
        return {
            "mode": "pairwise",
            "winner": result.winner,
            "confidence": result.confidence,
            "reasoning": result.reasoning
        }
    
    async def _rubric_evaluate(
        self,
        question: str,
        response: str,
        context: str
    ) -> Dict:
        result = await evaluate_with_rubric(
            question, response, context,
            self.config.rubric, self.client
        )
        return {
            "mode": "rubric",
            "grade": result.grade,
            "weighted_score": result.weighted_score,
            "scores": [
                {"dimension": s.dimension, "score": s.score, "feedback": s.feedback}
                for s in result.scores
            ]
        }
    
    async def _aspect_evaluate(
        self,
        question: str,
        response: str
    ) -> Dict:
        results = await self.aspect_evaluator.evaluate_multiple(
            question, response, self.config.aspects
        )
        
        all_passed = all(r.passed for r in results.values())
        avg_score = sum(r.score for r in results.values()) / len(results)
        
        return {
            "mode": "aspect",
            "passed": all_passed,
            "average_score": avg_score,
            "aspects": {
                aspect.value: {
                    "score": result.score,
                    "passed": result.passed,
                    "feedback": result.feedback
                }
                for aspect, result in results.items()
            }
        }
    
    async def _calibrated_evaluate(
        self,
        question: str,
        response: str
    ) -> Dict:
        # Multi-judge for better calibration
        scores = []
        for _ in range(self.config.num_judges):
            score = await calibrated_score(
                question, response,
                self.config.calibration_points,
                self.client
            )
            scores.append(score)
        
        mean = sum(scores) / len(scores)
        
        return {
            "mode": "calibrated",
            "score": mean,
            "passed": mean >= self.config.threshold,
            "individual_scores": scores
        }

# Usage
config = JudgeConfig(
    mode=JudgeMode.RUBRIC,
    rubric=rag_quality_rubric
)

judge = LLMJudge(client, config)

result = await judge.evaluate(
    question="What is Python?",
    response="Python is a programming language.",
    context="Python is a high-level programming language."
)

print(f"Grade: {result['grade']}")
```

---

## Hands-on Exercise

### Your Task

Build an `EvaluationPipeline` that:
1. Supports multiple evaluation modes
2. Combines different judge strategies
3. Produces confidence-weighted scores
4. Flags responses needing human review

### Requirements

```python
class EvaluationPipeline:
    def add_judge(self, name: str, judge: LLMJudge, weight: float) -> None:
        pass
    
    async def evaluate(
        self,
        question: str,
        response: str,
        context: str = None
    ) -> PipelineResult:
        pass

@dataclass
class PipelineResult:
    final_score: float
    confidence: float
    needs_human_review: bool
    judge_results: dict
```

<details>
<summary>💡 Hints</summary>

- Weight judges by their reliability
- Flag for review if judges disagree
- Use confidence scores from each judge
- Normalize scores across different modes

</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import statistics

@dataclass
class PipelineResult:
    final_score: float
    confidence: float
    needs_human_review: bool
    judge_results: Dict[str, dict]
    review_reasons: List[str]

class EvaluationPipeline:
    def __init__(self, review_threshold: float = 0.3):
        self.judges: Dict[str, tuple[LLMJudge, float]] = {}
        self.review_threshold = review_threshold
    
    def add_judge(
        self,
        name: str,
        judge: LLMJudge,
        weight: float
    ) -> None:
        """Add a judge with a weight."""
        self.judges[name] = (judge, weight)
    
    async def evaluate(
        self,
        question: str,
        response: str,
        context: str = None
    ) -> PipelineResult:
        """Run all judges and combine results."""
        
        judge_results = {}
        normalized_scores = []
        weights = []
        review_reasons = []
        
        # Run all judges
        for name, (judge, weight) in self.judges.items():
            try:
                result = await judge.evaluate(
                    question=question,
                    response=response,
                    context=context
                )
                
                judge_results[name] = result
                
                # Normalize score based on mode
                score = self._normalize_score(result)
                normalized_scores.append(score)
                weights.append(weight)
                
            except Exception as e:
                judge_results[name] = {"error": str(e)}
                review_reasons.append(f"Judge '{name}' failed: {str(e)}")
        
        if not normalized_scores:
            return PipelineResult(
                final_score=0.0,
                confidence=0.0,
                needs_human_review=True,
                judge_results=judge_results,
                review_reasons=["All judges failed"]
            )
        
        # Calculate weighted average
        total_weight = sum(weights)
        final_score = sum(
            s * w for s, w in zip(normalized_scores, weights)
        ) / total_weight
        
        # Calculate confidence (inverse of score variance)
        if len(normalized_scores) > 1:
            variance = statistics.variance(normalized_scores)
            confidence = max(0.0, 1.0 - variance)
            
            if variance > self.review_threshold:
                review_reasons.append(
                    f"High score variance ({variance:.2f}) - judges disagree"
                )
        else:
            confidence = 0.7  # Single judge, moderate confidence
        
        # Check for low scores
        if final_score < 0.5:
            review_reasons.append(f"Low overall score ({final_score:.2f})")
        
        return PipelineResult(
            final_score=final_score,
            confidence=confidence,
            needs_human_review=len(review_reasons) > 0,
            judge_results=judge_results,
            review_reasons=review_reasons
        )
    
    def _normalize_score(self, result: dict) -> float:
        """Normalize score from different judge modes to 0-1."""
        
        mode = result.get("mode")
        
        if mode == "rubric":
            # Convert grade to score
            grade_map = {"A": 0.95, "B": 0.85, "C": 0.75, "D": 0.65, "F": 0.4}
            return grade_map.get(result.get("grade", "F"), 0.5)
        
        elif mode == "aspect":
            return result.get("average_score", 0.5)
        
        elif mode == "calibrated":
            return result.get("score", 0.5)
        
        elif mode == "pairwise":
            # Pairwise is relative, use confidence
            if result.get("winner") == "A":
                return 0.8
            elif result.get("winner") == "B":
                return 0.2
            else:
                return 0.5
        
        return 0.5

# Usage
pipeline = EvaluationPipeline()

# Add rubric judge
rubric_config = JudgeConfig(mode=JudgeMode.RUBRIC, rubric=rag_quality_rubric)
pipeline.add_judge("rubric", LLMJudge(client, rubric_config), weight=0.5)

# Add aspect judge
aspect_config = JudgeConfig(
    mode=JudgeMode.ASPECT,
    aspects=[EvaluationAspect.ACCURACY, EvaluationAspect.HELPFULNESS]
)
pipeline.add_judge("aspect", LLMJudge(client, aspect_config), weight=0.5)

# Evaluate
result = await pipeline.evaluate(
    question="What is Python?",
    response="Python is a programming language created by Guido van Rossum.",
    context="Python is a high-level programming language."
)

print(f"Final Score: {result.final_score:.2f}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Needs Review: {result.needs_human_review}")

if result.review_reasons:
    print("Review Reasons:")
    for reason in result.review_reasons:
        print(f"  - {reason}")
```

</details>

---

## Summary

LLM-as-judge enables nuanced, scalable evaluation:

✅ **Pairwise comparison** — Direct head-to-head with position debiasing
✅ **Rubric scoring** — Consistent criteria across evaluations
✅ **Aspect evaluation** — Independent scoring of specific qualities
✅ **Calibration** — Reference points and multi-judge consensus

**Key insight:** LLM judges have biases—always calibrate, debias, and flag uncertain cases for human review.

---

## Further Reading

- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)
- [G-Eval: NLG Evaluation with GPT-4](https://arxiv.org/abs/2303.16634)
- [Chatbot Arena](https://chat.lmsys.org/) - Pairwise human evaluation

<!--
Sources Consulted:
- LLM-as-judge research papers
- OpenAI evaluation guidelines
- Anthropic's model evaluation practices
-->
