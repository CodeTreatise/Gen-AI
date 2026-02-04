---
title: "Self-Critique and Reflexion"
---

# Self-Critique and Reflexion

## Introduction

Self-critique prompts the model to evaluate and improve its own output before returning it. Reflexion extends this into an iterative loop—generating, evaluating, reflecting on failures, and trying again with accumulated insights. These techniques create an internal feedback cycle that catches errors, improves quality, and builds on experience.

> **🤖 AI Context:** Reflexion implements "verbal reinforcement learning"—the model learns from its mistakes through linguistic feedback rather than gradient updates.

### What We'll Cover

- Basic self-critique patterns
- The Reflexion framework (Actor-Evaluator-Self-Reflection)
- Iterative improvement loops
- Memory and experience accumulation
- When self-critique helps vs hurts

### Prerequisites

- [Chain-of-Thought Prompting](../07-chain-of-thought-prompting/00-chain-of-thought-overview.md)
- [Constraint-Based Prompting](./02-constraint-based-prompting.md)

---

## Basic Self-Critique Patterns

### Critique Before Returning

Ask the model to review its own work:

```python
def generate_with_critique(task: str) -> str:
    """Generate output, then critique and revise."""
    
    prompt = f"""Complete this task, then critique your output.

TASK: {task}

INSTRUCTIONS:
1. First, complete the task fully
2. Then, review your output for:
   - Accuracy: Are all facts correct?
   - Completeness: Did you address all requirements?
   - Clarity: Is it easy to understand?
   - Tone: Does it match what was requested?
3. If you find issues, revise your output
4. Return only the final, revised version

COMPLETE THE TASK:"""
    
    return call_model(prompt)
```

### Explicit Review Checklist

```python
def generate_with_checklist(task: str, checklist: list) -> str:
    """Generate and verify against explicit checklist."""
    
    checklist_text = "\n".join([f"☐ {item}" for item in checklist])
    
    prompt = f"""Complete this task, then verify against the checklist.

TASK: {task}

After completing, verify each item:
{checklist_text}

If any item fails, revise your output before returning.

Process:
1. [Complete task]
2. [Check each item - mark ✓ or ✗]
3. [Revise if any ✗]
4. [Return final output]

BEGIN:"""
    
    return call_model(prompt)

# Usage
checklist = [
    "All code compiles without errors",
    "Includes error handling",
    "Has docstrings/comments",
    "Follows naming conventions",
    "No hardcoded values"
]
result = generate_with_checklist("Write a function to validate email addresses", checklist)
```

### Two-Step Critique

Separate generation from critique for stronger review:

```python
def two_step_critique(task: str) -> dict:
    """Generate, then critique in separate call."""
    
    # Step 1: Generate
    initial = call_model(f"""Complete this task:

{task}

Output:""")
    
    # Step 2: Critique (separate call = fresh perspective)
    critique = call_model(f"""Review this output for a task.

ORIGINAL TASK: {task}

OUTPUT TO REVIEW:
{initial}

Critique:
1. What is done well?
2. What errors or issues exist?
3. What is missing?
4. Specific improvement suggestions?

Provide detailed critique:""")
    
    # Step 3: Revise based on critique
    final = call_model(f"""Revise this output based on the critique.

ORIGINAL TASK: {task}

ORIGINAL OUTPUT:
{initial}

CRITIQUE:
{critique}

Create an improved version that addresses all critique points:""")
    
    return {
        "initial": initial,
        "critique": critique,
        "final": final
    }
```

---

## The Reflexion Framework

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      REFLEXION LOOP                          │
│                                                              │
│  ┌─────────┐    ┌───────────┐    ┌──────────────────┐       │
│  │  Actor  │ →  │ Evaluator │ →  │ Self-Reflection  │       │
│  │         │    │           │    │                  │       │
│  │ Generate│    │ Score     │    │ "What went wrong?│       │
│  │ Action  │    │ Output    │    │  How to improve?"│       │
│  └─────────┘    └───────────┘    └──────────────────┘       │
│       ↑                                    │                 │
│       │           ┌───────────┐            │                 │
│       └───────────│  Memory   │←───────────┘                 │
│                   │           │                              │
│                   │ Store     │                              │
│                   │ Insights  │                              │
│                   └───────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
1. **Actor** — Generates actions/outputs (uses CoT, ReAct, etc.)
2. **Evaluator** — Scores outputs (pass/fail, numeric score, or LLM judgment)
3. **Self-Reflection** — Produces verbal feedback on what went wrong
4. **Memory** — Stores reflections for future attempts

### Implementing Reflexion

```python
class ReflexionAgent:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.memory = []  # Episodic memory of reflections
    
    def solve(self, task: str) -> dict:
        """Solve task with reflexion loop."""
        
        for attempt in range(self.max_attempts):
            # Actor: Generate solution
            solution = self.act(task, attempt)
            
            # Evaluator: Score the solution
            evaluation = self.evaluate(task, solution)
            
            if evaluation["passed"]:
                return {
                    "status": "success",
                    "solution": solution,
                    "attempts": attempt + 1,
                    "reflections": self.memory
                }
            
            # Self-Reflection: Analyze what went wrong
            reflection = self.reflect(task, solution, evaluation)
            self.memory.append(reflection)
        
        return {
            "status": "failed",
            "last_solution": solution,
            "attempts": self.max_attempts,
            "reflections": self.memory
        }
    
    def act(self, task: str, attempt: int) -> str:
        """Generate solution, incorporating past reflections."""
        
        reflections_text = ""
        if self.memory:
            reflections_text = "\n\nLEARNINGS FROM PREVIOUS ATTEMPTS:\n"
            for i, r in enumerate(self.memory, 1):
                reflections_text += f"{i}. {r}\n"
        
        prompt = f"""Solve this task.

TASK: {task}
{reflections_text}
{"Apply these learnings to avoid previous mistakes." if self.memory else ""}

SOLUTION:"""
        
        return call_model(prompt)
    
    def evaluate(self, task: str, solution: str) -> dict:
        """Evaluate if solution is correct."""
        
        prompt = f"""Evaluate this solution.

TASK: {task}

SOLUTION:
{solution}

Evaluation criteria:
1. Does it correctly solve the task?
2. Are there any errors?
3. Is it complete?

Return evaluation as:
PASSED: true/false
SCORE: 1-10
ISSUES: [list any problems]"""
        
        result = call_model(prompt)
        return parse_evaluation(result)
    
    def reflect(self, task: str, solution: str, evaluation: dict) -> str:
        """Generate reflection on failure."""
        
        prompt = f"""Your solution failed. Reflect on what went wrong.

TASK: {task}

YOUR SOLUTION:
{solution}

EVALUATION:
{evaluation}

Reflect:
1. What specific error did you make?
2. Why did you make this error?
3. What should you do differently next time?

Provide a concise reflection (1-2 sentences) that captures the key lesson:"""
        
        return call_model(prompt)
```

### Reflexion for Code Generation

```python
class CodeReflexionAgent(ReflexionAgent):
    """Reflexion specialized for code generation."""
    
    def evaluate(self, task: str, code: str) -> dict:
        """Evaluate code by running tests."""
        
        # Extract test cases from task or generate them
        tests = self.get_tests(task)
        
        results = []
        for test in tests:
            try:
                # Actually execute the code
                exec_result = execute_code(code, test["input"])
                passed = exec_result == test["expected"]
                results.append({
                    "test": test["name"],
                    "passed": passed,
                    "expected": test["expected"],
                    "actual": exec_result
                })
            except Exception as e:
                results.append({
                    "test": test["name"],
                    "passed": False,
                    "error": str(e)
                })
        
        all_passed = all(r["passed"] for r in results)
        
        return {
            "passed": all_passed,
            "score": sum(r["passed"] for r in results) / len(results) * 10,
            "results": results
        }
    
    def reflect(self, task: str, code: str, evaluation: dict) -> str:
        """Reflect on code failures with specific details."""
        
        failed_tests = [r for r in evaluation["results"] if not r["passed"]]
        
        prompt = f"""Your code failed some tests. Analyze the failures.

TASK: {task}

YOUR CODE:
{code}

FAILED TESTS:
{json.dumps(failed_tests, indent=2)}

For each failure:
1. What was expected vs actual?
2. What bug in the code caused this?
3. How should the code be fixed?

Summarize the key fix needed in one sentence:"""
        
        return call_model(prompt)
```

---

## Iterative Improvement Patterns

### Quality Ladder

Progressively improve along quality dimensions:

```python
def quality_ladder(task: str, dimensions: list) -> dict:
    """Iteratively improve output along quality dimensions."""
    
    # Initial generation
    output = call_model(f"Complete this task:\n\n{task}")
    
    history = [{"stage": "initial", "output": output}]
    
    # Improve along each dimension
    for dimension in dimensions:
        prompt = f"""Improve this output specifically for {dimension}.

ORIGINAL TASK: {task}

CURRENT OUTPUT:
{output}

Focus ONLY on improving {dimension}. Other aspects are already good.

Improved version:"""
        
        output = call_model(prompt)
        history.append({"stage": dimension, "output": output})
    
    return {
        "final": output,
        "history": history
    }

# Usage
dimensions = ["accuracy", "clarity", "conciseness", "tone"]
result = quality_ladder(
    "Write an email declining a meeting invitation politely",
    dimensions
)
```

### Critique-Revise Loop

```python
def critique_revise_loop(task: str, max_iterations: int = 3) -> dict:
    """Loop until critique finds no significant issues."""
    
    output = call_model(f"Complete this task:\n\n{task}")
    
    iterations = []
    
    for i in range(max_iterations):
        # Critique
        critique = call_model(f"""Critique this output. List specific issues.

TASK: {task}

OUTPUT:
{output}

Issues found (or "None" if no significant issues):""")
        
        iterations.append({"output": output, "critique": critique})
        
        # Check if done
        if "none" in critique.lower() or "no significant" in critique.lower():
            return {
                "final": output,
                "iterations": len(iterations),
                "history": iterations
            }
        
        # Revise
        output = call_model(f"""Revise the output to address these issues.

TASK: {task}

CURRENT OUTPUT:
{output}

ISSUES TO FIX:
{critique}

Revised output:""")
    
    return {
        "final": output,
        "iterations": max_iterations,
        "history": iterations,
        "note": "Reached max iterations"
    }
```

---

## Memory and Experience Accumulation

### Session Memory

```python
class SessionMemory:
    """Maintain memory across multiple tasks in a session."""
    
    def __init__(self):
        self.lessons = []
        self.patterns = {}
    
    def add_lesson(self, task_type: str, lesson: str):
        """Store a lesson learned."""
        self.lessons.append({
            "type": task_type,
            "lesson": lesson,
            "timestamp": time.time()
        })
        
        # Track patterns
        if task_type not in self.patterns:
            self.patterns[task_type] = []
        self.patterns[task_type].append(lesson)
    
    def get_relevant_lessons(self, task_type: str, limit: int = 3) -> list:
        """Retrieve lessons relevant to current task."""
        
        # Exact type matches
        exact = self.patterns.get(task_type, [])[-limit:]
        
        # General lessons (always applicable)
        general = self.patterns.get("general", [])[-1:]
        
        return exact + general
    
    def format_for_prompt(self, task_type: str) -> str:
        """Format lessons for inclusion in prompt."""
        
        lessons = self.get_relevant_lessons(task_type)
        
        if not lessons:
            return ""
        
        return "\n\nLESSONS FROM EXPERIENCE:\n" + "\n".join(
            f"• {lesson}" for lesson in lessons
        )

# Usage across tasks
memory = SessionMemory()

def solve_with_memory(task: str, task_type: str) -> str:
    lessons = memory.format_for_prompt(task_type)
    
    prompt = f"""Solve this task.
{lessons}

TASK: {task}

SOLUTION:"""
    
    solution = call_model(prompt)
    
    # After evaluation, if there was a lesson learned:
    # memory.add_lesson(task_type, "Always check for edge cases")
    
    return solution
```

### Persistent Reflection Storage

```python
class ReflectionStore:
    """Persist reflections across sessions."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.reflections = self.load()
    
    def load(self) -> dict:
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"reflections": [], "by_category": {}}
    
    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.reflections, f, indent=2)
    
    def add(self, category: str, reflection: str, context: dict = None):
        entry = {
            "category": category,
            "reflection": reflection,
            "context": context,
            "timestamp": time.time()
        }
        
        self.reflections["reflections"].append(entry)
        
        if category not in self.reflections["by_category"]:
            self.reflections["by_category"][category] = []
        self.reflections["by_category"][category].append(reflection)
        
        self.save()
    
    def get_for_category(self, category: str, limit: int = 5) -> list:
        return self.reflections["by_category"].get(category, [])[-limit:]
```

---

## When Self-Critique Helps vs Hurts

### When It Helps ✅

| Scenario | Why It Works |
|----------|--------------|
| Complex multi-step tasks | Catches errors at each step |
| Tasks with clear criteria | Easy to evaluate against |
| High-stakes outputs | Worth the extra latency/cost |
| Ambiguous instructions | Reflection clarifies intent |
| Creative tasks | Iteration improves quality |

### When It Hurts ❌

| Scenario | Why It Fails |
|----------|--------------|
| Simple factual queries | Overcomplicates, may add errors |
| Speed-critical applications | Doubles+ latency |
| Already-confident model | "Critique" becomes praise |
| Vague evaluation criteria | Critique lacks direction |
| Over-iteration | Can degrade quality past optimum |

### Critique Quality Signals

```python
def assess_critique_value(initial: str, critique: str, revised: str) -> dict:
    """Assess whether critique actually helped."""
    
    prompt = f"""Compare these versions. Did the critique help?

INITIAL:
{initial}

CRITIQUE:
{critique}

REVISED:
{revised}

Assess:
1. Did the revision meaningfully improve on initial?
2. Were critique points valid?
3. Did revision introduce new problems?

Verdict: HELPED / NEUTRAL / HURT"""
    
    return call_model(prompt)
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Clear evaluation criteria | Focused, actionable critique |
| Limit iterations | Diminishing returns, drift |
| Separate generation from critique | Fresh perspective |
| Store lessons for reuse | Compound learning |
| Match critique depth to task importance | Efficiency |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Vague critique prompts | "Looks good!" non-answers | Specify what to check |
| No stopping condition | Infinite loops | Set max iterations + "no issues" check |
| Lost original intent | Drift from task | Include original task in each step |
| Self-congratulation | Model says it's perfect | Explicit failure criteria |
| Memory overload | Too much context | Summarize/prune old reflections |

---

## Hands-on Exercise

### Your Task

Build a reflexion agent for writing product descriptions. It should:
1. Generate initial description
2. Evaluate against criteria (engaging, accurate, appropriate length)
3. Reflect on failures
4. Retry with accumulated lessons

<details>
<summary>💡 Hints</summary>

1. Define clear evaluation criteria with pass/fail thresholds
2. Format reflections concisely for memory
3. Include product details in every prompt
4. Stop when evaluation passes or max attempts reached

</details>

<details>
<summary>✅ Solution</summary>

```python
class ProductDescriptionReflexion:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.memory = []
        self.criteria = {
            "engaging_hook": "Opens with attention-grabbing statement",
            "key_features": "Highlights at least 3 product features",
            "benefits_focus": "Emphasizes benefits, not just features",
            "appropriate_length": "Between 50-150 words",
            "call_to_action": "Ends with clear CTA"
        }
    
    def generate(self, product: dict) -> dict:
        for attempt in range(self.max_attempts):
            # Generate with memory
            description = self.write_description(product)
            
            # Evaluate
            evaluation = self.evaluate(product, description)
            
            if evaluation["passed"]:
                return {
                    "status": "success",
                    "description": description,
                    "attempts": attempt + 1
                }
            
            # Reflect
            reflection = self.reflect(product, description, evaluation)
            self.memory.append(reflection)
        
        return {
            "status": "max_attempts",
            "description": description,
            "attempts": self.max_attempts
        }
    
    def write_description(self, product: dict) -> str:
        memory_text = ""
        if self.memory:
            memory_text = "\n\nLEARNINGS:\n" + "\n".join(f"• {m}" for m in self.memory)
        
        prompt = f"""Write a product description.

PRODUCT: {product['name']}
DETAILS: {product['details']}
TARGET AUDIENCE: {product['audience']}
{memory_text}

Requirements:
- Engaging opening hook
- Highlight key features with benefits
- 50-150 words
- End with call-to-action

Description:"""
        
        return call_model(prompt)
    
    def evaluate(self, product: dict, description: str) -> dict:
        word_count = len(description.split())
        results = {}
        
        # Length check (objective)
        results["appropriate_length"] = 50 <= word_count <= 150
        
        # LLM evaluation for subjective criteria
        eval_prompt = f"""Evaluate this product description.

PRODUCT: {product['name']}
DESCRIPTION: {description}

Rate each (PASS/FAIL):
1. Engaging hook: Does it open with attention-grabbing statement?
2. Key features: Does it highlight at least 3 features?
3. Benefits focus: Does it emphasize benefits over specs?
4. Call to action: Does it end with clear CTA?

Format: CRITERION: PASS/FAIL"""
        
        eval_result = call_model(eval_prompt)
        results.update(parse_evaluation_results(eval_result))
        
        passed = all(results.values())
        
        return {
            "passed": passed,
            "results": results,
            "word_count": word_count
        }
    
    def reflect(self, product: dict, description: str, evaluation: dict) -> str:
        failed = [k for k, v in evaluation["results"].items() if not v]
        
        prompt = f"""Your product description failed these criteria: {failed}

DESCRIPTION:
{description}

Why did it fail? What's the one key fix?
(One sentence)"""
        
        return call_model(prompt)
```

</details>

---

## Summary

- Self-critique catches errors before output is returned
- Reflexion loops: generate → evaluate → reflect → retry with memory
- Actor-Evaluator-Self-Reflection architecture enables verbal reinforcement learning
- Store reflections as "lessons learned" for future attempts
- Balance iteration depth against diminishing returns and drift risk
- Best for complex tasks with clear evaluation criteria

**Next:** [Explicit Planning Prompts](./08-explicit-planning-prompts.md)

---

<!-- Sources: Prompting Guide Reflexion, Shinn et al. (2023) Reflexion paper concepts, Google Gemini self-critique patterns -->
