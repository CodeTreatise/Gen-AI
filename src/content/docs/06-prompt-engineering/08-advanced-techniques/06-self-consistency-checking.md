---
title: "Self-Consistency Checking"
---

# Self-Consistency Checking

## Introduction

Self-consistency improves reasoning by sampling multiple answers and selecting the most frequent one. Instead of relying on a single generation, we let the model reason multiple times through different paths, then vote on the final answer. This "wisdom of crowds" approach catches errors that slip through any single reasoning chain.

> **🔑 Key Insight:** Greedy decoding picks the single most likely path. Self-consistency explores diverse paths and finds consensus—significantly improving accuracy on reasoning tasks.

### What We'll Cover

- The self-consistency method
- Sampling diverse reasoning paths
- Voting and aggregation strategies
- Confidence estimation
- Inconsistency detection

### Prerequisites

- [Chain-of-Thought Prompting](../07-chain-of-thought-prompting/00-chain-of-thought-overview.md)
- Basic probability concepts

---

## The Self-Consistency Method

### Standard vs Self-Consistent Generation

```
Standard (Greedy) Generation:
┌─────────────┐    ┌──────────────┐    ┌─────────┐
│   Problem   │ →  │ Single CoT   │ →  │ Answer  │
└─────────────┘    │   Path       │    │   X     │
                   └──────────────┘    └─────────┘

Self-Consistency:
┌─────────────┐    ┌──────────────┐    ┌─────────┐
│             │ →  │  CoT Path 1  │ →  │ Answer A│ ─┐
│             │    └──────────────┘    └─────────┘  │
│   Problem   │    ┌──────────────┐    ┌─────────┐  │   ┌────────┐
│             │ →  │  CoT Path 2  │ →  │ Answer A│ ─┼→  │ VOTE:  │
│             │    └──────────────┘    └─────────┘  │   │ A wins │
│             │    ┌──────────────┐    ┌─────────┐  │   └────────┘
│             │ →  │  CoT Path 3  │ →  │ Answer B│ ─┘
└─────────────┘    └──────────────┘    └─────────┘
```

### Why It Works

1. **Different reasoning paths reach same truth** — Correct answers are reachable multiple ways
2. **Errors are path-dependent** — Wrong answers from different errors don't cluster
3. **Majority reveals signal** — Noise cancels, truth emerges

---

## Sampling Diverse Reasoning Paths

### Temperature-Based Sampling

```python
import asyncio
from collections import Counter

async def self_consistent_answer(
    prompt: str, 
    num_samples: int = 5,
    temperature: float = 0.7
) -> dict:
    """Generate multiple answers and find consensus."""
    
    # Sample multiple reasoning paths with temperature > 0
    tasks = [
        call_model_async(
            prompt,
            temperature=temperature,
            # Higher temperature = more diversity
        )
        for _ in range(num_samples)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    # Extract final answers from each response
    answers = [extract_answer(r) for r in responses]
    
    # Vote
    vote_counts = Counter(answers)
    winner, count = vote_counts.most_common(1)[0]
    
    return {
        "answer": winner,
        "confidence": count / num_samples,
        "all_answers": answers,
        "vote_distribution": dict(vote_counts)
    }

def extract_answer(response: str) -> str:
    """Extract the final answer from a CoT response."""
    # Look for common answer patterns
    if "Therefore, the answer is" in response:
        return response.split("Therefore, the answer is")[-1].strip()
    if "Final answer:" in response:
        return response.split("Final answer:")[-1].strip()
    # Return last line as fallback
    return response.strip().split('\n')[-1]
```

### Ensuring Diversity

```python
def create_diverse_prompts(question: str, few_shot_sets: list) -> list:
    """Create prompts with different few-shot examples for diversity."""
    
    base_instruction = """Solve this step by step, then give the final answer.

Question: {question}

Think through this carefully:"""
    
    prompts = []
    for few_shots in few_shot_sets:
        prompt = few_shots + "\n\n" + base_instruction.format(question=question)
        prompts.append(prompt)
    
    return prompts

# Different few-shot example sets
few_shot_set_1 = """Q: What is 15% of 80?
A: 15% means 15/100 = 0.15. 0.15 × 80 = 12.
Final answer: 12"""

few_shot_set_2 = """Q: What is 15% of 80?
A: 10% of 80 is 8. 5% is half of that, so 4. 10% + 5% = 8 + 4 = 12.
Final answer: 12"""

few_shot_set_3 = """Q: What is 15% of 80?
A: 15% of 80 = (15 × 80) / 100 = 1200 / 100 = 12.
Final answer: 12"""
```

---

## Voting and Aggregation

### Majority Voting

```python
def majority_vote(answers: list) -> tuple:
    """Simple majority vote - most common answer wins."""
    counter = Counter(answers)
    winner, count = counter.most_common(1)[0]
    return winner, count / len(answers)

# Example
answers = ["42", "42", "42", "45", "41"]
result, confidence = majority_vote(answers)
# result = "42", confidence = 0.6
```

### Weighted Voting

Weight by reasoning quality:

```python
async def weighted_self_consistency(prompt: str, num_samples: int = 5) -> dict:
    """Self-consistency with quality-weighted voting."""
    
    responses = await generate_samples(prompt, num_samples)
    
    weighted_votes = {}
    for response in responses:
        answer = extract_answer(response)
        # Score the reasoning quality
        quality = await score_reasoning(response)
        
        if answer in weighted_votes:
            weighted_votes[answer] += quality
        else:
            weighted_votes[answer] = quality
    
    # Winner is answer with highest total weight
    winner = max(weighted_votes, key=weighted_votes.get)
    total_weight = sum(weighted_votes.values())
    
    return {
        "answer": winner,
        "confidence": weighted_votes[winner] / total_weight,
        "weights": weighted_votes
    }

async def score_reasoning(response: str) -> float:
    """Score the quality of a reasoning chain."""
    
    scoring_prompt = f"""Rate this reasoning on a scale of 1-10:

{response}

Consider:
- Logical flow
- Correct operations
- Clear final answer

Score (1-10):"""
    
    score_text = await call_model_async(scoring_prompt, max_tokens=5)
    return float(score_text.strip()) / 10  # Normalize to 0-1
```

### Cluster-Based Aggregation

For answers that are close but not identical:

```python
from difflib import SequenceMatcher

def cluster_similar_answers(answers: list, threshold: float = 0.8) -> dict:
    """Group similar answers together before voting."""
    
    clusters = []
    
    for answer in answers:
        matched = False
        for cluster in clusters:
            representative = cluster["representative"]
            similarity = SequenceMatcher(None, answer, representative).ratio()
            
            if similarity >= threshold:
                cluster["members"].append(answer)
                matched = True
                break
        
        if not matched:
            clusters.append({
                "representative": answer,
                "members": [answer]
            })
    
    # Vote by cluster size
    clusters.sort(key=lambda c: len(c["members"]), reverse=True)
    
    return {
        "winner": clusters[0]["representative"],
        "cluster_size": len(clusters[0]["members"]),
        "total_answers": len(answers),
        "clusters": clusters
    }

# Example: These are effectively the same answer
answers = [
    "The answer is 42",
    "Answer: 42",
    "42",
    "The answer is forty-two",
    "24"  # Different answer
]
result = cluster_similar_answers(answers)
# Clusters "42" variants together, giving them higher vote count
```

---

## Confidence Estimation

### Agreement-Based Confidence

```python
def estimate_confidence(answers: list) -> dict:
    """Estimate confidence from answer distribution."""
    
    counter = Counter(answers)
    total = len(answers)
    
    winner, winner_count = counter.most_common(1)[0]
    
    # Metrics
    agreement_ratio = winner_count / total
    entropy = -sum((c/total) * log2(c/total) for c in counter.values())
    max_entropy = log2(len(counter))  # If all answers different
    
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    confidence = 1 - normalized_entropy
    
    return {
        "answer": winner,
        "agreement_ratio": agreement_ratio,
        "entropy": entropy,
        "confidence": confidence,
        "is_high_confidence": agreement_ratio >= 0.6 and len(counter) <= 2
    }
```

### Confidence Thresholds

```python
def decide_with_confidence(answers: list, high_threshold: float = 0.7) -> dict:
    """Make decision based on confidence level."""
    
    confidence_data = estimate_confidence(answers)
    
    if confidence_data["agreement_ratio"] >= high_threshold:
        return {
            "status": "confident",
            "answer": confidence_data["answer"],
            "action": "Use this answer"
        }
    elif confidence_data["agreement_ratio"] >= 0.4:
        return {
            "status": "uncertain",
            "answer": confidence_data["answer"],
            "action": "Consider gathering more samples or human review"
        }
    else:
        return {
            "status": "no_consensus",
            "answers": dict(Counter(answers)),
            "action": "Require human decision or reformulate problem"
        }
```

---

## Inconsistency Detection

### Detecting Reasoning Conflicts

```python
async def detect_inconsistencies(responses: list) -> dict:
    """Identify reasoning conflicts between samples."""
    
    # Pairwise comparison for significant differences
    inconsistencies = []
    
    for i, resp_a in enumerate(responses):
        for j, resp_b in enumerate(responses[i+1:], i+1):
            answer_a = extract_answer(resp_a)
            answer_b = extract_answer(resp_b)
            
            if answer_a != answer_b:
                # Analyze why they differ
                analysis = await call_model_async(f"""
Compare these two reasoning chains that reached different answers:

REASONING A:
{resp_a}

REASONING B:
{resp_b}

Identify:
1. Where do they diverge?
2. Which reasoning has an error?
3. What is the correct answer?

Analysis:""")
                
                inconsistencies.append({
                    "response_a": i,
                    "response_b": j,
                    "answer_a": answer_a,
                    "answer_b": answer_b,
                    "analysis": analysis
                })
    
    return {
        "has_inconsistencies": len(inconsistencies) > 0,
        "count": len(inconsistencies),
        "details": inconsistencies
    }
```

### Flagging Low Agreement

```python
def flag_for_review(answers: list, threshold: float = 0.5) -> dict:
    """Flag cases where self-consistency is low."""
    
    counter = Counter(answers)
    winner_count = counter.most_common(1)[0][1]
    agreement = winner_count / len(answers)
    
    if agreement < threshold:
        return {
            "flag": True,
            "reason": f"Low agreement ({agreement:.0%})",
            "distribution": dict(counter),
            "recommendation": "Human review recommended"
        }
    
    return {"flag": False, "agreement": agreement}
```

---

## Practical Example: Math Problem

```python
async def solve_math_with_consistency(problem: str) -> dict:
    """Solve math problem using self-consistency."""
    
    prompt = f"""Solve this math problem step by step.
Show your work, then provide the final numerical answer.

Problem: {problem}

Solution:"""
    
    # Generate 5 solutions with temperature=0.7 for diversity
    responses = await asyncio.gather(*[
        call_model_async(prompt, temperature=0.7)
        for _ in range(5)
    ])
    
    # Extract numerical answers
    answers = []
    for resp in responses:
        # Look for numbers in final answer
        lines = resp.strip().split('\n')
        last_line = lines[-1]
        numbers = re.findall(r'[-+]?\d*\.?\d+', last_line)
        if numbers:
            answers.append(numbers[-1])
    
    # Vote
    counter = Counter(answers)
    winner, count = counter.most_common(1)[0]
    
    return {
        "answer": winner,
        "confidence": count / len(answers),
        "distribution": dict(counter),
        "sample_reasoning": responses[answers.index(winner)]  # Show a correct one
    }

# Example
result = await solve_math_with_consistency(
    "If a train travels 120 miles in 2 hours, then rests for 30 minutes, "
    "then travels 90 miles in 1.5 hours, what is its average speed for the "
    "entire journey?"
)
```

---

## When to Use Self-Consistency

| Use Self-Consistency | Avoid Self-Consistency |
|---------------------|------------------------|
| Reasoning/math problems | Simple factual lookups |
| Multiple valid reasoning paths | Only one correct path |
| Stakes are high | Cost is a major concern |
| Accuracy > latency | Speed is critical |
| Answers are discrete/comparable | Answers are creative/open-ended |

**Best problem types:**
- Arithmetic and math word problems
- Commonsense reasoning
- Symbolic reasoning
- Multiple-choice questions
- Problems with verifiable answers

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use temperature > 0 | Enables diverse sampling |
| Sample 5-10 responses | Balance accuracy and cost |
| Normalize answers | "42" and "forty-two" should match |
| Track confidence | Know when to trust results |
| Combine with CoT | Self-consistency enhances chain-of-thought |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Temperature = 0 | All samples identical | Use temperature 0.5-0.8 |
| Too few samples | Unstable voting | Use at least 5 samples |
| Answer format varies | Votes split incorrectly | Normalize before voting |
| Ignoring low agreement | False confidence | Flag and review edge cases |
| Open-ended questions | No clear "right" answer | Use for convergent problems only |

---

## Hands-on Exercise

### Your Task

Implement self-consistency for a logic puzzle solver. Given a puzzle, generate multiple reasoning attempts, vote on the answer, and detect if there are fundamental disagreements in the reasoning.

<details>
<summary>💡 Hints</summary>

1. Design a CoT prompt for the puzzle
2. Sample 5+ solutions with temperature
3. Cluster similar answers
4. Calculate confidence from agreement
5. If confidence is low, analyze disagreements

</details>

<details>
<summary>✅ Solution</summary>

```python
async def solve_logic_puzzle(puzzle: str, num_samples: int = 7) -> dict:
    """Solve logic puzzle with self-consistency."""
    
    prompt = f"""Solve this logic puzzle step by step.

Puzzle: {puzzle}

Work through the constraints one by one.
Eliminate impossible options.
State your final answer clearly.

Solution:"""
    
    # Sample multiple solutions
    responses = await asyncio.gather(*[
        call_model_async(prompt, temperature=0.7)
        for _ in range(num_samples)
    ])
    
    # Extract answers
    answers = [extract_final_answer(r) for r in responses]
    
    # Cluster similar answers
    clusters = cluster_similar_answers(answers, threshold=0.85)
    
    # Calculate confidence
    confidence = clusters["cluster_size"] / len(answers)
    
    # Analyze if low confidence
    analysis = None
    if confidence < 0.6:
        # Find where reasoning diverges
        divergent_pair = find_most_different(responses)
        analysis = await call_model_async(f"""
These two solutions to a logic puzzle reached different answers.
Identify the error:

SOLUTION A:
{divergent_pair[0]}

SOLUTION B:
{divergent_pair[1]}

Which is correct and why?""")
    
    return {
        "answer": clusters["winner"],
        "confidence": confidence,
        "distribution": {c["representative"]: len(c["members"]) 
                        for c in clusters["clusters"]},
        "needs_review": confidence < 0.6,
        "divergence_analysis": analysis
    }

def extract_final_answer(response: str) -> str:
    """Extract final answer from logic puzzle solution."""
    patterns = [
        r"[Tt]he answer is:?\s*(.+)",
        r"[Ff]inal answer:?\s*(.+)",
        r"[Tt]herefore,?\s*(.+)",
        r"[Ss]olution:?\s*(.+)$"
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            return match.group(1).strip()
    return response.strip().split('\n')[-1]
```

</details>

---

## Summary

- Self-consistency samples multiple reasoning paths and votes on the answer
- Use temperature > 0 to ensure diverse sampling
- Majority voting reveals consensus; weighted voting incorporates quality
- Agreement ratio estimates confidence in the result
- Low agreement signals need for human review or problem reformulation
- Best for reasoning problems with discrete, verifiable answers

**Next:** [Self-Critique and Reflexion](./07-self-critique-reflexion.md)

---

<!-- Sources: Prompting Guide Self-Consistency, Wang et al. (2022) paper concepts -->
