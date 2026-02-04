---
title: "Meta-Prompting"
---

# Meta-Prompting

## Introduction

Meta-prompting uses AI to generate, improve, and debug prompts. Instead of manually crafting every prompt, you create prompts that create prompts. This technique scales prompt engineering and often produces better results than manual efforts—the model knows its own capabilities better than we do.

> **🔑 Key Insight:** The best prompt writer for a model is often the model itself. Meta-prompting leverages this.

### What We'll Cover

- Generating prompts with AI
- Prompt optimization cycles
- Self-improvement patterns
- Prompt debugging assistance
- Automation at scale

### Prerequisites

- [Fundamentals of Effective Prompts](../01-fundamentals-of-effective-prompts/00-fundamentals-overview.md)
- Experience writing and testing prompts

---

## Generating Prompts with AI

### Basic Prompt Generation

```python
meta_prompt = """Create a system prompt for an AI assistant that helps users 
write SQL queries.

The assistant should:
- Support PostgreSQL, MySQL, and SQLite syntax differences
- Explain queries it generates
- Warn about potential performance issues
- Follow security best practices

Generate a comprehensive system prompt that covers:
1. Role and expertise definition
2. Input handling expectations
3. Output format specifications
4. Safety constraints
5. Example interaction patterns"""

generated_prompt = call_model(meta_prompt)
```

### Prompt Generation Template

```python
def generate_prompt(task: str, context: dict) -> str:
    meta_prompt = f"""Generate a system prompt for this task:

TASK: {task}

CONTEXT:
- Target users: {context['users']}
- Expertise level expected: {context['level']}
- Output format needed: {context['format']}
- Key constraints: {context['constraints']}

The generated prompt should include:
1. Clear role definition
2. Behavioral guidelines
3. Output specifications
4. Edge case handling
5. Example interactions (at least 2)

Return ONLY the system prompt, ready to use."""

    return call_model(meta_prompt)

# Usage
context = {
    "users": "junior developers",
    "level": "beginner-friendly explanations",
    "format": "code with inline comments",
    "constraints": "no external dependencies, Python 3.10+"
}

prompt = generate_prompt(
    "Code review assistant for Python projects",
    context
)
```

---

## Prompt Optimization

### The Optimization Loop

```python
def optimize_prompt(initial_prompt: str, test_cases: list, iterations: int = 3) -> str:
    """Iteratively improve a prompt based on test results."""
    
    current_prompt = initial_prompt
    
    for i in range(iterations):
        # Test current prompt
        results = []
        for test in test_cases:
            output = call_model(current_prompt + "\n\n" + test["input"])
            score = evaluate(output, test["expected"])
            results.append({
                "input": test["input"],
                "output": output,
                "expected": test["expected"],
                "score": score
            })
        
        # Generate improvement suggestions
        analysis_prompt = f"""Analyze these prompt test results and suggest improvements.

CURRENT PROMPT:
{current_prompt}

TEST RESULTS:
{json.dumps(results, indent=2)}

Identify:
1. Common failure patterns
2. Missing instructions
3. Unclear specifications
4. Potential improvements

Then generate an improved version of the prompt."""

        current_prompt = call_model(analysis_prompt)
    
    return current_prompt
```

### A/B Testing Pattern

```python
def ab_test_prompts(prompt_a: str, prompt_b: str, test_inputs: list) -> dict:
    """Compare two prompts on the same inputs."""
    
    results = {"a": [], "b": []}
    
    for test_input in test_inputs:
        output_a = call_model(prompt_a + "\n\n" + test_input)
        output_b = call_model(prompt_b + "\n\n" + test_input)
        
        results["a"].append(output_a)
        results["b"].append(output_b)
    
    # Use AI to evaluate which is better
    comparison_prompt = f"""Compare these two sets of outputs and determine 
which prompt performed better overall.

TEST INPUTS:
{json.dumps(test_inputs, indent=2)}

PROMPT A OUTPUTS:
{json.dumps(results['a'], indent=2)}

PROMPT B OUTPUTS:
{json.dumps(results['b'], indent=2)}

Evaluate on:
1. Accuracy
2. Completeness
3. Clarity
4. Consistency

Declare a winner and explain why."""

    return call_model(comparison_prompt)
```

---

## Self-Improvement Cycles

### Prompt Self-Critique

```python
self_improvement_prompt = """You are a prompt engineering expert. 
Review and improve this prompt:

ORIGINAL PROMPT:
{prompt_to_improve}

ANALYSIS FRAMEWORK:
1. CLARITY: Are instructions unambiguous?
2. COMPLETENESS: Are all cases covered?
3. EFFICIENCY: Is there redundancy?
4. ROBUSTNESS: Will it handle edge cases?
5. EFFECTIVENESS: Will it achieve the goal?

For each dimension:
- Rate (1-5)
- Identify specific issues
- Suggest improvements

Then provide an IMPROVED VERSION of the prompt."""
```

### Iterative Refinement

```python
def self_improve_prompt(prompt: str, goal: str, max_iterations: int = 3) -> str:
    """Have the model iteratively improve its own prompt."""
    
    current = prompt
    
    for i in range(max_iterations):
        improve_request = f"""GOAL: {goal}

CURRENT PROMPT (iteration {i+1}):
{current}

As a prompt engineering expert, identify ONE significant improvement 
that would make this prompt more effective for the stated goal.

Return:
1. The issue you identified
2. Why it matters
3. The complete improved prompt

Focus on the highest-impact change."""

        response = call_model(improve_request)
        current = extract_prompt(response)  # Parse out the improved prompt
    
    return current
```

---

## Prompt Debugging

### Debug Analysis Prompt

```python
debug_prompt = """A prompt is not producing expected results. Help debug it.

THE PROMPT:
{failing_prompt}

EXPECTED BEHAVIOR:
{expected}

ACTUAL BEHAVIOR:
{actual}

SPECIFIC ISSUES:
{issues_list}

Analyze:
1. What's causing the gap between expected and actual?
2. Which parts of the prompt are problematic?
3. Are there ambiguous instructions?
4. Are there missing constraints?

Provide:
1. Root cause analysis
2. Specific line-by-line fixes
3. A corrected version of the prompt
4. Test cases to verify the fix"""
```

### Common Issues Detector

```python
def detect_prompt_issues(prompt: str) -> list:
    """Use AI to detect common prompt engineering issues."""
    
    analysis_prompt = f"""Analyze this prompt for common issues:

PROMPT:
{prompt}

Check for:
□ Ambiguous instructions
□ Missing output format specifications
□ Unclear role definition
□ Conflicting constraints
□ Missing edge case handling
□ Overly long or complex sentences
□ Jargon that might confuse the model
□ Missing examples
□ Implicit assumptions
□ Scope creep potential

For each issue found:
- Severity (Critical/Major/Minor)
- Specific location in prompt
- Recommended fix

Return as structured JSON."""

    return json.loads(call_model(analysis_prompt))
```

---

## Prompt Templates Generation

### Template Factory

```python
template_generator = """Generate a reusable prompt template for: {use_case}

Requirements:
- Use {{placeholder}} syntax for variable parts
- Include default values where sensible
- Add validation hints for each placeholder
- Include 2-3 usage examples

Template should be:
1. Immediately usable with minimal customization
2. Flexible enough for variations
3. Self-documenting

Output format:
---
TEMPLATE:
[the template with placeholders]

PLACEHOLDERS:
- {{placeholder_name}}: description, type, default (if any)

EXAMPLES:
Example 1: [filled-in template]
Example 2: [filled-in template]
---"""
```

### Example: Generated Template

```python
# Input: "Generate a template for API documentation"
# Output:

template = """
TEMPLATE:
---
You are a technical writer creating API documentation.

Document this endpoint:
- Method: {{http_method}}
- Path: {{endpoint_path}}
- Description: {{brief_description}}

Code context:
{{code_snippet}}

Generate documentation including:
1. Endpoint description
2. Request parameters (path, query, body)
3. Response format with examples
4. Error codes and meanings
5. {{language}} code example

Format as Markdown suitable for developer docs.
---

PLACEHOLDERS:
- {{http_method}}: HTTP method (GET, POST, etc.), required
- {{endpoint_path}}: The API path, required
- {{brief_description}}: One-line description, optional
- {{code_snippet}}: The implementation code, required
- {{language}}: Programming language for examples, default: "Python"

EXAMPLES:
Example 1 (Python):
  http_method: POST
  endpoint_path: /api/v1/users
  brief_description: Create a new user account
  code_snippet: [function code]
  language: Python

Example 2 (JavaScript):
  http_method: GET
  endpoint_path: /api/v1/products/{id}
  language: JavaScript
"""
```

---

## Automation at Scale

### Batch Prompt Generation

```python
def batch_generate_prompts(specifications: list) -> list:
    """Generate multiple prompts from specifications."""
    
    prompts = []
    
    for spec in specifications:
        meta_prompt = f"""Generate a production-ready system prompt:

SPECIFICATION:
- Purpose: {spec['purpose']}
- Target model: {spec['model']}
- Key constraints: {spec['constraints']}
- Output format: {spec['format']}
- Special requirements: {spec.get('special', 'None')}

Generate a complete, ready-to-use system prompt."""
        
        prompts.append({
            "spec": spec,
            "prompt": call_model(meta_prompt)
        })
    
    return prompts
```

### Prompt Quality Scoring

```python
def score_prompt(prompt: str) -> dict:
    """Use AI to score a prompt on multiple dimensions."""
    
    scoring_prompt = f"""Score this prompt on each dimension (1-10):

PROMPT:
{prompt}

DIMENSIONS:
1. CLARITY: How clear and unambiguous are the instructions?
2. COMPLETENESS: Does it cover all necessary aspects?
3. EFFICIENCY: Is it concise without losing meaning?
4. ROBUSTNESS: Will it handle edge cases well?
5. SPECIFICITY: Is it specific enough to produce consistent results?

For each dimension:
- Score (1-10)
- Brief justification

Also provide:
- OVERALL SCORE (weighted average)
- TOP IMPROVEMENT SUGGESTION

Return as JSON."""

    return json.loads(call_model(scoring_prompt))
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Test generated prompts | AI-generated prompts need validation |
| Include success criteria | Tell meta-prompt what "good" looks like |
| Iterate in small steps | One improvement at a time is more reliable |
| Preserve working versions | Keep history of prompt evolution |
| Use structured outputs | JSON for automation compatibility |
| Validate with real examples | Abstract testing misses real-world issues |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Over-automation | Generating prompts without understanding | Always review generated prompts |
| Infinite loops | Self-improvement cycles that never end | Set iteration limits |
| Overfitting to examples | Prompt works for tests, fails generally | Test with diverse inputs |
| Complexity explosion | Each iteration adds complexity | Include simplification in criteria |
| Lost context | Generated prompts miss original intent | Include purpose in meta-prompts |

---

## Hands-on Exercise

### Your Task

Create a meta-prompt that generates prompts for different customer service scenarios.

**Requirements:**
1. Takes scenario type as input (complaint, inquiry, feedback)
2. Generates appropriate system prompt
3. Includes tone and constraint adjustments per scenario
4. Produces ready-to-use output

<details>
<summary>Solution</summary>

```python
meta_prompt = """Generate a customer service system prompt for this scenario:

SCENARIO TYPE: {{scenario_type}}

Based on the scenario, create a complete system prompt that includes:

1. ROLE DEFINITION
   - Appropriate expertise level
   - Matching personality for the scenario

2. TONE CALIBRATION
   {{#if complaint}}
   - Empathetic and solution-focused
   - Acknowledge frustration first
   - Prioritize resolution over policy
   {{/if}}
   {{#if inquiry}}
   - Helpful and informative
   - Clear and thorough
   - Proactively offer related information
   {{/if}}
   {{#if feedback}}
   - Appreciative and engaged
   - Show genuine interest
   - Explain how feedback is used
   {{/if}}

3. CONSTRAINTS
   - Never blame the customer
   - Always offer next steps
   - Escalation triggers
   - Response time expectations

4. RESPONSE STRUCTURE
   - Opening acknowledgment
   - Main response
   - Clear next steps
   - Closing

5. EXAMPLES (at least 2)
   - Typical interaction
   - Edge case handling

Generate the complete system prompt:"""

# Implementation
def generate_cs_prompt(scenario: str) -> str:
    filled_meta = meta_prompt.replace("{{scenario_type}}", scenario)
    
    # Handle conditional sections
    if scenario == "complaint":
        filled_meta = filled_meta.replace("{{#if complaint}}", "")
        filled_meta = filled_meta.replace("{{/if}}", "")
    # ... similar for other scenarios
    
    return call_model(filled_meta)

# Usage
complaint_prompt = generate_cs_prompt("complaint")
inquiry_prompt = generate_cs_prompt("inquiry")
feedback_prompt = generate_cs_prompt("feedback")
```

</details>

---

## Summary

- Meta-prompting uses AI to generate and improve prompts
- Optimization loops iteratively refine prompts based on test results
- Self-critique enables prompts to improve themselves
- Debugging prompts help identify and fix issues
- Automation scales prompt engineering across use cases
- Always validate generated prompts before production use

**Next:** [Decomposition Strategies](./04-decomposition-strategies.md)

---

<!-- Sources: OpenAI Cookbook Meta Prompting, Prompt Engineering Guide -->
