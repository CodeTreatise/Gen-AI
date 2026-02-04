---
title: "Decomposition Strategies"
---

# Decomposition Strategies

## Introduction

Complex tasks overwhelm single prompts. Decomposition breaks them into manageable subtasks, each with focused attention. The model handles one clear objective at a time, then results combine into the final output. This approach mirrors how humans tackle complexity—divide and conquer.

> **🔑 Key Insight:** A model's attention dilutes across a complex prompt. Decomposition gives each subtask full attention.

### What We'll Cover

- Task breakdown approaches
- Sequential vs parallel decomposition
- Result synthesis techniques
- Dependency management
- Implementation patterns

### Prerequisites

- [Chain-of-Thought Prompting](../07-chain-of-thought-prompting/00-chain-of-thought-overview.md)
- Understanding of async programming (for parallel patterns)

---

## Why Decomposition Works

### The Attention Problem

```
Single complex prompt:
┌─────────────────────────────────────────┐
│ Task A + Task B + Task C + Task D       │
│ [Attention spread thin across all]      │
└─────────────────────────────────────────┘
         ↓
   Variable quality, often misses details

Decomposed approach:
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Task A  │  │  Task B  │  │  Task C  │  │  Task D  │
│  [Full   │  │  [Full   │  │  [Full   │  │  [Full   │
│  focus]  │  │  focus]  │  │  focus]  │  │  focus]  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
         ↓           ↓           ↓           ↓
                     ↓
              [Synthesis step]
                     ↓
            High-quality result
```

### When to Decompose

| Signal | Example |
|--------|---------|
| Multiple distinct steps | "Analyze, summarize, and translate" |
| Different expertise needed | "Check grammar AND verify code" |
| Long input with multiple outputs | Document with many sections |
| Quality drops on complex prompts | Model misses requirements |
| Results need to be traceable | Each step needs to be auditable |

---

## Task Breakdown Approaches

### Vertical Decomposition (Sequential)

Tasks depend on previous outputs:

```python
def sequential_decomposition(document: str) -> dict:
    """Process document through sequential steps."""
    
    # Step 1: Extract key information
    extraction = call_model(f"""Extract the following from this document:
- Main topic
- Key arguments (bullet points)
- Data points mentioned
- Conclusions drawn

Document:
{document}""")
    
    # Step 2: Analyze (using Step 1 output)
    analysis = call_model(f"""Based on this extracted information:

{extraction}

Analyze:
1. Strength of the arguments
2. Quality of supporting evidence
3. Logical consistency
4. Potential gaps or weaknesses""")
    
    # Step 3: Summarize (using Step 1 + Step 2)
    summary = call_model(f"""Create an executive summary based on:

EXTRACTED INFO:
{extraction}

ANALYSIS:
{analysis}

Summary should be 3-5 sentences covering the main point, 
key supporting evidence, and any caveats.""")
    
    return {
        "extraction": extraction,
        "analysis": analysis,
        "summary": summary
    }
```

### Horizontal Decomposition (Parallel)

Tasks are independent and can run simultaneously:

```python
import asyncio

async def parallel_decomposition(code: str) -> dict:
    """Analyze code from multiple angles in parallel."""
    
    tasks = [
        analyze_aspect(code, "SECURITY", "Check for security vulnerabilities: SQL injection, XSS, auth issues"),
        analyze_aspect(code, "PERFORMANCE", "Identify performance issues: N+1 queries, memory leaks, inefficient algorithms"),
        analyze_aspect(code, "STYLE", "Review code style: naming, structure, readability"),
        analyze_aspect(code, "TESTING", "Assess testability: dependency injection, modularity, test coverage hints")
    ]
    
    results = await asyncio.gather(*tasks)
    
    return {
        "security": results[0],
        "performance": results[1],
        "style": results[2],
        "testing": results[3]
    }

async def analyze_aspect(code: str, aspect: str, focus: str) -> str:
    return await call_model_async(f"""You are a code reviewer focused on {aspect}.

{focus}

Code to review:
{code}

Provide specific, actionable feedback for this aspect only.""")
```

### Hierarchical Decomposition

Break into levels:

```python
def hierarchical_decomposition(task: str) -> dict:
    """Break task into hierarchy of subtasks."""
    
    # Level 1: High-level breakdown
    high_level = call_model(f"""Break this task into 3-5 major phases:

Task: {task}

For each phase:
- Name
- Objective
- Dependencies on other phases""")
    
    phases = parse_phases(high_level)
    
    # Level 2: Detailed steps for each phase
    detailed = {}
    for phase in phases:
        steps = call_model(f"""Break this phase into specific steps:

Phase: {phase['name']}
Objective: {phase['objective']}

For each step:
- Action to take
- Inputs needed
- Expected output
- Verification method""")
        detailed[phase['name']] = steps
    
    return {
        "high_level": phases,
        "detailed": detailed
    }
```

---

## Result Synthesis

### Simple Aggregation

```python
def synthesize_results(subtask_results: list) -> str:
    """Combine subtask results into coherent output."""
    
    synthesis_prompt = f"""You have results from multiple analysis subtasks.
Synthesize them into a coherent final report.

SUBTASK RESULTS:
{json.dumps(subtask_results, indent=2)}

Create a unified report that:
1. Integrates all findings
2. Resolves any conflicts between results
3. Prioritizes by importance
4. Provides clear recommendations

Do not simply concatenate—create a cohesive narrative."""
    
    return call_model(synthesis_prompt)
```

### Weighted Synthesis

```python
def weighted_synthesis(results: dict, weights: dict) -> str:
    """Synthesize with priority weighting."""
    
    weighted_input = []
    for aspect, result in results.items():
        weight = weights.get(aspect, 1.0)
        weighted_input.append({
            "aspect": aspect,
            "weight": weight,
            "priority": "HIGH" if weight > 0.7 else "MEDIUM" if weight > 0.3 else "LOW",
            "findings": result
        })
    
    synthesis_prompt = f"""Synthesize these findings based on their priority:

WEIGHTED FINDINGS:
{json.dumps(weighted_input, indent=2)}

HIGH priority findings should drive the main conclusions.
MEDIUM priority adds nuance.
LOW priority is mentioned briefly if relevant.

Produce a unified analysis with clear recommendations."""
    
    return call_model(synthesis_prompt)
```

### Conflict Resolution Synthesis

```python
def resolve_and_synthesize(results: dict) -> str:
    """Synthesize results that may conflict."""
    
    synthesis_prompt = f"""These subtasks produced potentially conflicting results.
Analyze and synthesize:

RESULTS FROM DIFFERENT ANALYSES:
{json.dumps(results, indent=2)}

Process:
1. Identify any conflicting findings
2. Analyze why conflicts might exist
3. Determine which finding is more reliable (and why)
4. Create a unified conclusion that acknowledges uncertainty where it exists

Format:
CONFLICTS IDENTIFIED: [list any conflicts]
RESOLUTION: [how you resolved each]
UNIFIED CONCLUSION: [the synthesized result]"""
    
    return call_model(synthesis_prompt)
```

---

## Dependency Management

### Dependency Graph

```python
def execute_with_dependencies(tasks: list) -> dict:
    """Execute tasks respecting dependencies."""
    
    # tasks = [
    #     {"id": "A", "prompt": "...", "depends_on": []},
    #     {"id": "B", "prompt": "...", "depends_on": ["A"]},
    #     {"id": "C", "prompt": "...", "depends_on": ["A"]},
    #     {"id": "D", "prompt": "...", "depends_on": ["B", "C"]}
    # ]
    
    results = {}
    completed = set()
    
    while len(completed) < len(tasks):
        for task in tasks:
            if task["id"] in completed:
                continue
            
            # Check if all dependencies are met
            deps_met = all(d in completed for d in task["depends_on"])
            
            if deps_met:
                # Inject dependency results into prompt
                prompt = task["prompt"]
                for dep_id in task["depends_on"]:
                    prompt = prompt.replace(
                        f"{{{{result_{dep_id}}}}}",
                        results[dep_id]
                    )
                
                results[task["id"]] = call_model(prompt)
                completed.add(task["id"])
    
    return results
```

### Conditional Branching

```python
def conditional_decomposition(input_data: str) -> str:
    """Decompose with conditional paths based on analysis."""
    
    # Step 1: Classify
    classification = call_model(f"""Classify this input:

{input_data}

Categories:
A) Technical documentation
B) Marketing content
C) Legal document
D) General prose

Return only the letter.""")
    
    category = classification.strip().upper()
    
    # Step 2: Branch based on classification
    if category == "A":
        result = process_technical(input_data)
    elif category == "B":
        result = process_marketing(input_data)
    elif category == "C":
        result = process_legal(input_data)
    else:
        result = process_general(input_data)
    
    return result
```

---

## Implementation Patterns

### The Pipeline Pattern

```python
class PromptPipeline:
    """Sequential pipeline of prompt operations."""
    
    def __init__(self):
        self.stages = []
    
    def add_stage(self, name: str, prompt_template: str, processor=None):
        self.stages.append({
            "name": name,
            "template": prompt_template,
            "processor": processor or (lambda x: x)
        })
        return self
    
    def run(self, initial_input: str) -> dict:
        results = {"input": initial_input}
        current = initial_input
        
        for stage in self.stages:
            prompt = stage["template"].replace("{{input}}", current)
            raw_output = call_model(prompt)
            processed = stage["processor"](raw_output)
            
            results[stage["name"]] = processed
            current = processed
        
        return results

# Usage
pipeline = PromptPipeline()
pipeline.add_stage("extract", "Extract key facts: {{input}}")
pipeline.add_stage("analyze", "Analyze these facts: {{input}}")
pipeline.add_stage("summarize", "Summarize this analysis: {{input}}")

results = pipeline.run(document)
```

### The Map-Reduce Pattern

```python
def map_reduce(items: list, map_prompt: str, reduce_prompt: str) -> str:
    """Apply map-reduce pattern to list processing."""
    
    # Map: Process each item
    mapped_results = []
    for item in items:
        result = call_model(map_prompt.replace("{{item}}", str(item)))
        mapped_results.append(result)
    
    # Reduce: Combine results
    combined = "\n\n".join([f"Item {i+1}:\n{r}" for i, r in enumerate(mapped_results)])
    final = call_model(reduce_prompt.replace("{{mapped_results}}", combined))
    
    return final

# Usage: Analyze multiple documents
map_prompt = """Analyze this document for key themes:

{{item}}

Return: 3-5 key themes with brief explanations."""

reduce_prompt = """These themes were extracted from multiple documents:

{{mapped_results}}

Identify:
1. Common themes across documents
2. Unique themes
3. Overall synthesis"""

result = map_reduce(documents, map_prompt, reduce_prompt)
```

### The Scatter-Gather Pattern

```python
async def scatter_gather(task: str, perspectives: list) -> str:
    """Scatter task to multiple perspectives, gather and synthesize."""
    
    # Scatter: Send to multiple "experts"
    async def get_perspective(perspective: str) -> dict:
        prompt = f"""You are an expert in {perspective}.

Analyze this task from your perspective:
{task}

Provide insights specific to {perspective} concerns."""
        
        return {
            "perspective": perspective,
            "analysis": await call_model_async(prompt)
        }
    
    # Run in parallel
    analyses = await asyncio.gather(
        *[get_perspective(p) for p in perspectives]
    )
    
    # Gather and synthesize
    synthesis_prompt = f"""Multiple experts analyzed this task:

TASK: {task}

EXPERT ANALYSES:
{json.dumps(analyses, indent=2)}

Create a comprehensive analysis that:
1. Incorporates all perspectives
2. Identifies consensus and disagreements
3. Provides balanced recommendations"""
    
    return call_model(synthesis_prompt)

# Usage
perspectives = ["security", "performance", "usability", "maintainability"]
result = asyncio.run(scatter_gather(code_review_task, perspectives))
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Clear subtask boundaries | Prevents overlap and confusion |
| Single responsibility per subtask | Focused attention, better results |
| Explicit input/output contracts | Clean handoffs between stages |
| Error handling at each stage | Prevents cascade failures |
| Preserve intermediate results | Debugging and traceability |
| Balance granularity | Too many subtasks = overhead; too few = complexity returns |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Over-decomposition | Too many API calls, high latency | Group related subtasks |
| Under-decomposition | Complexity still too high | Break down further |
| Lost context | Later stages lack needed info | Pass relevant context forward |
| Inconsistent formats | Synthesis struggles | Standardize subtask outputs |
| Ignoring dependencies | Race conditions, missing data | Map dependencies first |

---

## Hands-on Exercise

### Your Task

Create a decomposition pipeline for analyzing a job posting.

**Requirements:**
1. Extract: Role, requirements, company info
2. Analyze: Difficulty level, red flags, opportunities
3. Match: Compare against a candidate profile
4. Recommend: Action items for application

<details>
<summary>Solution</summary>

```python
def analyze_job_posting(posting: str, candidate_profile: str) -> dict:
    # Stage 1: Extract
    extraction = call_model(f"""Extract structured information from this job posting:

{posting}

Return as JSON:
{{
  "role": {{
    "title": "",
    "level": "junior/mid/senior",
    "team_size": "if mentioned"
  }},
  "requirements": {{
    "must_have": [],
    "nice_to_have": [],
    "years_experience": ""
  }},
  "company": {{
    "name": "",
    "industry": "",
    "size": "",
    "culture_hints": []
  }},
  "compensation": {{
    "salary_range": "",
    "benefits_mentioned": []
  }}
}}""")
    
    # Stage 2: Analyze
    analysis = call_model(f"""Analyze this job posting data:

{extraction}

Evaluate:
1. DIFFICULTY LEVEL (1-10): How hard is this role to land?
2. RED FLAGS: Any concerning patterns?
   - Unrealistic requirements
   - Vague responsibilities
   - Signs of poor culture
3. OPPORTUNITIES:
   - Growth potential
   - Learning opportunities
   - Standout positives

Return structured analysis.""")
    
    # Stage 3: Match
    match = call_model(f"""Compare this candidate to the job requirements:

JOB DATA:
{extraction}

CANDIDATE PROFILE:
{candidate_profile}

Evaluate match on:
1. Required skills: Which are met? Which are gaps?
2. Experience level: Over/under qualified?
3. Culture fit signals
4. MATCH SCORE: 1-100

Be specific about gaps and strengths.""")
    
    # Stage 4: Recommend
    recommendation = call_model(f"""Based on this analysis, provide application recommendations:

EXTRACTION: {extraction}
ANALYSIS: {analysis}
MATCH: {match}

Provide:
1. APPLY? (Yes/No/Maybe) with reasoning
2. If applying:
   - Resume highlights to emphasize
   - Skills to showcase
   - Potential concerns to address proactively
   - Cover letter talking points
3. Interview prep priorities
4. Questions to ask the employer""")
    
    return {
        "extraction": extraction,
        "analysis": analysis,
        "match": match,
        "recommendation": recommendation
    }
```

</details>

---

## Summary

- Decomposition gives each subtask the model's full attention
- Sequential for dependent tasks, parallel for independent
- Synthesis combines subtask results into coherent output
- Dependency management prevents race conditions
- Pipeline, Map-Reduce, and Scatter-Gather are common patterns
- Balance granularity to avoid over/under-decomposition

**Next:** [Tree of Thoughts](./05-tree-of-thought-prompting.md)

---

<!-- Sources: Anthropic Prompt Chaining, Google Prompting Strategies -->
