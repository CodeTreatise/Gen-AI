---
title: "Tree of Thought Prompting"
---

# Tree of Thought Prompting

## Introduction

Tree of Thought (ToT) prompting extends chain-of-thought by exploring multiple reasoning paths simultaneously. Instead of a single linear chain, the model maintains a tree of possibilities, evaluates branches, and navigates toward the best solution. This is especially powerful for problems requiring exploration, backtracking, or multi-step planning.

> **🤖 AI Context:** ToT generalizes over chain-of-thought prompting, enabling deliberate decision-making through search algorithms like breadth-first search (BFS) and depth-first search (DFS).

### What We'll Cover

- ToT framework concepts
- Path generation and evaluation
- Search strategies (BFS, DFS)
- Multi-expert pattern
- Implementation approaches

### Prerequisites

- [Chain-of-Thought Prompting](../07-chain-of-thought-prompting/00-chain-of-thought-overview.md)
- Basic understanding of tree data structures

---

## The Tree of Thoughts Framework

### Conceptual Model

```
                    [Problem]
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      [Thought A]   [Thought B]   [Thought C]
       (maybe)       (sure)        (impossible)
          │             │               ✗
     ┌────┴────┐   ┌────┴────┐
     ▼         ▼   ▼         ▼
  [A.1]     [A.2] [B.1]   [B.2]
  (maybe)   (✗)  (sure)  (maybe)
     │            │         │
     ...         [B.1.1]   ...
                 (SOLUTION!)
```

**Key Components:**
1. **Thoughts** — Intermediate reasoning steps
2. **Evaluation** — Self-assessment of each thought's promise
3. **Search** — Algorithm to navigate the tree (BFS/DFS)
4. **Backtracking** — Ability to abandon dead ends

### Chain-of-Thought vs Tree-of-Thought

| Chain-of-Thought | Tree-of-Thought |
|------------------|-----------------|
| Single reasoning path | Multiple parallel paths |
| Linear progression | Branching exploration |
| No backtracking | Can abandon dead ends |
| Fast, single generation | Slower, multiple generations |
| Good for routine problems | Good for problems requiring search |

---

## Path Generation

### Generating Multiple Thoughts

```python
def generate_thoughts(problem: str, num_thoughts: int = 3) -> list:
    """Generate multiple initial thoughts for a problem."""
    
    prompt = f"""Consider this problem:

{problem}

Generate {num_thoughts} different initial approaches to solve it.
Each approach should be distinct and represent a different strategy.

Format:
APPROACH 1: [Description of first approach]
APPROACH 2: [Description of second approach]
APPROACH 3: [Description of third approach]"""
    
    response = call_model(prompt)
    return parse_approaches(response)

def expand_thought(problem: str, thought: str, depth: int) -> list:
    """Expand a thought into possible next steps."""
    
    prompt = f"""Problem: {problem}

Current thought (depth {depth}):
{thought}

Generate 2-3 possible next steps from this point.
Each should be a distinct logical progression.

Format:
NEXT STEP A: [Description]
NEXT STEP B: [Description]
NEXT STEP C: [Description]"""
    
    response = call_model(prompt)
    return parse_steps(response)
```

### The Multi-Expert Pattern

A powerful ToT technique—imagine experts who collaborate and can drop out when wrong:

```
Imagine three different experts are answering this question.
All experts will write down 1 step of their thinking, 
then share it with the group. 
If any expert realises they're wrong at any point, they leave.

Question: [Your problem here]
```

**Implementation:**

```python
def multi_expert_tot(problem: str, num_experts: int = 3) -> str:
    """Use multi-expert pattern for tree of thought."""
    
    prompt = f"""Imagine {num_experts} different experts are solving this problem.
Each expert will share their thinking step by step.
After each step, all experts consider what was shared.
If an expert realizes their approach won't work, they acknowledge it and stop contributing.
The remaining experts continue until a solution is found.

Problem: {problem}

Begin the expert discussion. Show each step of reasoning, which expert said it,
and when experts realize they should stop. Continue until a solution emerges.

Format each step as:
Expert [N] (Step X): [Their thought]
[Any expert who realizes they're wrong says: "I see my approach won't work because..."]

Final answer should come from remaining expert(s) with explanation."""
    
    return call_model(prompt)
```

---

## Path Evaluation

### Self-Evaluation Prompting

Have the model assess its own reasoning paths:

```python
def evaluate_thought(problem: str, thought: str) -> str:
    """Evaluate a thought's promise for solving the problem."""
    
    prompt = f"""Problem: {problem}

Proposed thought/approach:
{thought}

Evaluate this thought. Does it lead toward a solution?

Respond with exactly one of:
- SURE: This definitely leads to the solution
- MAYBE: This might work, worth exploring further
- IMPOSSIBLE: This approach cannot solve the problem

Then briefly explain your evaluation in one sentence."""
    
    response = call_model(prompt)
    # Parse "SURE", "MAYBE", or "IMPOSSIBLE" from response
    return parse_evaluation(response)
```

### Value Estimation

Assign numerical scores to paths:

```python
def score_thoughts(problem: str, thoughts: list) -> list:
    """Score multiple thoughts for comparison."""
    
    thoughts_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(thoughts)])
    
    prompt = f"""Problem: {problem}

Here are possible approaches:
{thoughts_text}

Score each approach from 1-10 based on:
- Likelihood of reaching solution
- Efficiency of approach
- Clarity of reasoning path

Format:
1. [score]/10 - [brief justification]
2. [score]/10 - [brief justification]
..."""
    
    response = call_model(prompt)
    return parse_scores(response)
```

---

## Search Strategies

### Breadth-First Search (BFS)

Explore all possibilities at current depth before going deeper:

```python
def bfs_tree_of_thought(problem: str, max_depth: int = 3) -> dict:
    """Breadth-first tree of thought search."""
    
    # Initialize with multiple starting thoughts
    queue = generate_thoughts(problem, num_thoughts=3)
    depth = 0
    explored = []
    
    while queue and depth < max_depth:
        current_level = queue.copy()
        queue = []
        
        for thought in current_level:
            # Evaluate this thought
            evaluation = evaluate_thought(problem, thought)
            explored.append({
                "depth": depth,
                "thought": thought,
                "evaluation": evaluation
            })
            
            if evaluation == "SURE":
                # Found solution path
                return {
                    "status": "solved",
                    "path": thought,
                    "explored": explored
                }
            elif evaluation == "MAYBE":
                # Expand this thought
                next_steps = expand_thought(problem, thought, depth + 1)
                queue.extend(next_steps)
            # IMPOSSIBLE thoughts are not expanded
        
        depth += 1
    
    return {
        "status": "exhausted",
        "best_path": find_best(explored),
        "explored": explored
    }
```

### Depth-First Search (DFS)

Explore a path deeply before backtracking:

```python
def dfs_tree_of_thought(problem: str, max_depth: int = 5) -> dict:
    """Depth-first tree of thought search with backtracking."""
    
    def explore(thought: str, depth: int, path: list) -> dict:
        if depth >= max_depth:
            return {"status": "depth_limit", "path": path}
        
        evaluation = evaluate_thought(problem, thought)
        path.append({
            "depth": depth,
            "thought": thought,
            "evaluation": evaluation
        })
        
        if evaluation == "SURE":
            return {"status": "solved", "path": path}
        elif evaluation == "IMPOSSIBLE":
            # Backtrack
            return {"status": "dead_end", "path": path}
        else:  # MAYBE
            # Explore deeper
            next_steps = expand_thought(problem, thought, depth)
            for step in next_steps:
                result = explore(step, depth + 1, path.copy())
                if result["status"] == "solved":
                    return result
            # All branches exhausted
            return {"status": "exhausted", "path": path}
    
    # Start with initial thoughts
    initial_thoughts = generate_thoughts(problem, num_thoughts=3)
    
    for thought in initial_thoughts:
        result = explore(thought, 0, [])
        if result["status"] == "solved":
            return result
    
    return {"status": "no_solution", "explored": []}
```

### Beam Search

Keep top-k most promising paths:

```python
def beam_search_tot(problem: str, beam_width: int = 3, max_depth: int = 4) -> dict:
    """Beam search tree of thought - keep top k paths."""
    
    # Initialize beam with starting thoughts
    beam = generate_thoughts(problem, num_thoughts=beam_width)
    
    for depth in range(max_depth):
        # Expand all current beam members
        candidates = []
        
        for thought in beam:
            evaluation = evaluate_thought(problem, thought)
            
            if evaluation == "SURE":
                return {"status": "solved", "path": thought, "depth": depth}
            elif evaluation == "MAYBE":
                expansions = expand_thought(problem, thought, depth)
                candidates.extend(expansions)
        
        if not candidates:
            break
        
        # Score candidates and keep top k
        scored = score_thoughts(problem, candidates)
        scored_candidates = list(zip(candidates, scored))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Update beam with top k
        beam = [c for c, s in scored_candidates[:beam_width]]
    
    return {"status": "completed", "best_paths": beam}
```

---

## Practical Example: Game of 24

The classic ToT benchmark—use numbers [4, 5, 6, 10] with +, -, *, / to reach 24:

```python
def game_of_24_tot(numbers: list) -> str:
    """Solve Game of 24 using tree of thought."""
    
    prompt = f"""Solve the Game of 24.
    
Numbers available: {numbers}
Goal: Use all four numbers exactly once with +, -, *, / to get 24.

Use Tree of Thought approach:
1. Consider possible first operations (pick 2 numbers, apply operation)
2. Evaluate if the resulting set of 3 numbers can possibly reach 24
3. If "sure" continue, if "maybe" explore, if "impossible" try different first step
4. Continue until you reach 24 or exhaust options

Show your reasoning tree:
Step 1 options → Evaluate each → Best path → Step 2 options → ... → Solution

After finding solution, verify by computing the final expression."""
    
    response = call_model(prompt)
    return response

# Example output structure:
"""
Step 1 - Consider first operations:
A) 4 * 5 = 20, remaining: [20, 6, 10] - MAYBE (20 + 4 = 24, need 4 from 6,10)
B) 6 - 4 = 2, remaining: [2, 5, 10] - MAYBE (need 24 from these)
C) 10 - 4 = 6, remaining: [6, 5, 6] - MAYBE (two 6s available)
D) 5 * 6 = 30, remaining: [30, 4, 10] - MAYBE (30 - 6 = 24, need 6 from 4,10)

Exploring path A (4 * 5 = 20):
- 20 + 6 - 10 = 16 - IMPOSSIBLE
- 20 - 6 + 10 = 24 - SURE! ✓

Solution: (4 * 5) - 6 + 10 = 20 - 6 + 10 = 24

Wait, let me verify: 20 - 6 = 14, 14 + 10 = 24 ✓
But that's 20 - 6 + 10, which equals 24. Correct!

Actually checking: (4 × 5) + 10 - 6 = 20 + 10 - 6 = 24 ✓
"""
```

---

## Single-Prompt ToT Simulation

For simple cases, simulate ToT in one prompt:

```python
def single_prompt_tot(problem: str) -> str:
    """Simulate tree of thought in a single prompt."""
    
    prompt = f"""Solve this problem using Tree of Thought reasoning.

Problem: {problem}

Approach:
1. Generate 3 different initial approaches
2. Evaluate each: "sure" (leads to solution), "maybe" (worth exploring), "impossible" (dead end)
3. Expand the most promising path(s)
4. Continue evaluating and expanding until solved
5. Show when you backtrack from dead ends

Format your response as:

INITIAL APPROACHES:
1. [Approach A] → Evaluation: [sure/maybe/impossible]
2. [Approach B] → Evaluation: [sure/maybe/impossible]
3. [Approach C] → Evaluation: [sure/maybe/impossible]

EXPLORING [most promising]:
├── [Next step 1] → [evaluation]
│   └── [If continued] → [evaluation]
├── [Next step 2] → [evaluation]
...

SOLUTION PATH: [The path that worked]
ANSWER: [Final answer]"""
    
    return call_model(prompt)
```

---

## When to Use Tree of Thought

| Use ToT When | Avoid ToT When |
|--------------|----------------|
| Problem requires search/exploration | Answer is straightforward |
| Multiple valid approaches exist | Single clear path |
| Backtracking may be needed | Linear reasoning suffices |
| Wrong paths are not obvious | Errors are easy to spot |
| Complex multi-step reasoning | Simple transformations |

**Best problem types:**
- Puzzles and games (Sudoku, crosswords, Game of 24)
- Planning with constraints
- Creative problem-solving
- Mathematical proofs
- Code debugging with multiple hypotheses

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Limit branching factor | Too many branches = slow, expensive |
| Prune aggressively | "Impossible" paths waste resources |
| Clear evaluation criteria | Consistent pruning decisions |
| Set depth limits | Prevent infinite exploration |
| Cache evaluations | Don't re-evaluate same thoughts |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Over-branching | Explosion of paths | Limit to 2-3 branches per node |
| Weak evaluation | Bad paths not pruned | Use explicit sure/maybe/impossible |
| No backtracking | Stuck in dead ends | Implement proper search algorithm |
| Token limits | Long trees exceed context | Summarize or truncate paths |
| Cost explosion | Many API calls | Use single-prompt simulation when possible |

---

## Hands-on Exercise

### Your Task

Implement a ToT approach for debugging code. Given buggy code, generate multiple hypotheses about the bug, evaluate each, and find the fix.

<details>
<summary>💡 Hints</summary>

1. Initial thoughts = different bug hypotheses
2. Evaluation = does hypothesis explain the symptoms?
3. Expansion = what would we check to confirm?
4. Solution = confirmed hypothesis + fix

</details>

<details>
<summary>✅ Solution</summary>

```python
def debug_with_tot(code: str, error: str, expected: str, actual: str) -> str:
    prompt = f"""Debug this code using Tree of Thought reasoning.

CODE:
{code}

ERROR/SYMPTOM: {error}
EXPECTED BEHAVIOR: {expected}
ACTUAL BEHAVIOR: {actual}

Use Tree of Thought:

HYPOTHESES (generate 3 possible causes):
1. [Hypothesis A] → Evaluate: Can this explain the symptom? [sure/maybe/impossible]
2. [Hypothesis B] → Evaluate: [sure/maybe/impossible]
3. [Hypothesis C] → Evaluate: [sure/maybe/impossible]

For each "maybe" or "sure" hypothesis, explore:
├── What evidence supports this?
├── What evidence contradicts this?
└── If confirmed, what's the fix?

INVESTIGATION:
[Show your exploration of promising hypotheses]
[Backtrack from hypotheses that don't hold up]

CONFIRMED ROOT CAUSE: [The validated hypothesis]
FIX: [The corrected code]
EXPLANATION: [Why this fixes it]"""
    
    return call_model(prompt)

# Usage
result = debug_with_tot(
    code="def sum_list(lst): return sum(lst[1:])",
    error="Off-by-one results",
    expected="sum_list([1,2,3]) = 6",
    actual="sum_list([1,2,3]) = 5"
)
```

</details>

---

## Summary

- Tree of Thought explores multiple reasoning paths simultaneously
- Thoughts are evaluated as "sure", "maybe", or "impossible"
- BFS explores broadly, DFS explores deeply, beam search balances both
- Multi-expert pattern simulates collaborative reasoning with dropout
- Best for problems requiring search, exploration, or backtracking
- Balance exploration breadth against computational cost

**Next:** [Self-Consistency Checking](./06-self-consistency-checking.md)

---

<!-- Sources: Prompting Guide Tree of Thoughts, Yao et al. (2023) ToT paper concepts -->
