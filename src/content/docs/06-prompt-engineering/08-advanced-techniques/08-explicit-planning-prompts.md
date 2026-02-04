---
title: "Explicit Planning Prompts"
---

# Explicit Planning Prompts

## Introduction

Explicit planning prompts ask the model to create a structured plan before executing. Instead of diving straight into the task, the model first breaks down the goal, identifies what information is needed, and outlines the steps. This "think before you act" approach is especially powerful with newer reasoning models like Gemini 2.5 and GPT-4, which handle planning instructions exceptionally well.

> **🤖 AI Context:** Gemini 3/2.5 shows particular strength with explicit planning prompts. Adding "Before providing the final answer, please parse the goal into sub-tasks" significantly improves complex task performance.

### What We'll Cover

- Why explicit planning works
- Planning prompt structures
- Goal decomposition techniques
- Input completeness checking
- Execution after planning

### Prerequisites

- [Chain-of-Thought Prompting](../07-chain-of-thought-prompting/00-chain-of-thought-overview.md)
- [Decomposition Strategies](./04-decomposition-strategies.md)

---

## Why Explicit Planning Works

### Without vs With Planning

```
WITHOUT PLANNING:
┌─────────┐
│  Task   │ → [Immediate execution] → [Often misses requirements]
└─────────┘

WITH PLANNING:
┌─────────┐     ┌─────────────────┐     ┌───────────────┐     ┌──────────┐
│  Task   │ →   │ Parse into      │ →   │ Check inputs  │ →   │ Execute  │
│         │     │ sub-tasks       │     │ complete?     │     │ plan     │
└─────────┘     └─────────────────┘     └───────────────┘     └──────────┘
                        │                       │
                  [Structured]            [Request more
                   roadmap]               info if needed]
```

### Benefits

| Benefit | Explanation |
|---------|-------------|
| Catches missing info early | Plan reveals what's needed before wasting effort |
| Improves complex task success | Clear roadmap prevents getting lost |
| Makes reasoning transparent | Plan shows the model's approach |
| Enables course correction | User can adjust plan before execution |
| Reduces hallucination | Model commits to realistic steps |

---

## The Core Planning Template

### Basic Structure

```python
def explicit_planning_prompt(task: str) -> str:
    """Generate prompt with explicit planning phase."""
    
    return f"""Before providing the final answer, please:

1. **Parse the Goal**
   - Break the stated goal into distinct sub-tasks
   - Identify the logical order of sub-tasks
   - Note any dependencies between sub-tasks

2. **Check Inputs**
   - What information is needed to complete each sub-task?
   - Is all required information provided?
   - If not, what assumptions will you make?

3. **Create Execution Plan**
   - List the specific steps you will take
   - Estimate complexity/length of each step
   - Identify potential challenges

4. **Execute Plan**
   - Work through each step
   - Mark steps as complete
   - Adjust if needed

---

TASK: {task}

Begin with your plan:"""
```

### Example: Complex Analysis Task

**Task:** "Analyze our Q3 sales data to identify trends, compare with Q2, and recommend actions for Q4."

**With explicit planning:**

```
PLAN:

1. Parse Goal into Sub-tasks:
   - Sub-task A: Identify Q3 sales trends (patterns, peaks, anomalies)
   - Sub-task B: Compare Q3 vs Q2 performance
   - Sub-task C: Generate actionable Q4 recommendations
   
   Dependencies: A → B → C (each builds on previous)

2. Check Inputs:
   ✓ Q3 data: [Assuming provided or accessible]
   ⚠ Q2 data: Needed for comparison - is it provided?
   ⚠ Context: Industry/product type affects recommendations
   
   ASSUMPTION: Will use provided data; may need clarification on Q2.

3. Execution Plan:
   Step 1: Summarize Q3 metrics (revenue, units, by category)
   Step 2: Identify patterns (growth/decline, seasonality)
   Step 3: Pull Q2 comparison metrics
   Step 4: Calculate deltas and significance
   Step 5: Synthesize findings into recommendations

4. Execute:
   [Proceeds through steps...]
```

---

## Goal Decomposition Techniques

### SMART Goal Parsing

```python
def smart_goal_prompt(goal: str) -> str:
    """Parse goal into SMART components."""
    
    return f"""Parse this goal into SMART components:

GOAL: {goal}

Analyze:
- **Specific**: What exactly needs to be accomplished?
- **Measurable**: How will we know it's done? What's the output?
- **Achievable**: Is this possible with available resources/info?
- **Relevant**: What problem does this solve?
- **Time-bound**: Are there deadlines or time constraints?

Then create a plan that addresses each SMART component.

SMART Analysis and Plan:"""
```

### Hierarchical Decomposition

```python
def hierarchical_plan(task: str, max_depth: int = 3) -> str:
    """Create hierarchical task breakdown."""
    
    return f"""Create a hierarchical plan for this task.

TASK: {task}

Structure your plan as:

1. [Major Phase 1]
   1.1. [Sub-task]
      1.1.1. [Specific action]
      1.1.2. [Specific action]
   1.2. [Sub-task]
      1.2.1. [Specific action]

2. [Major Phase 2]
   2.1. [Sub-task]
   ...

Rules:
- Maximum depth: {max_depth} levels
- Each leaf (lowest level) should be a concrete, executable action
- Include estimated effort for each major phase

After planning, execute the plan:"""
```

### Dependency Mapping

```python
def dependency_aware_plan(task: str) -> str:
    """Plan with explicit dependency identification."""
    
    return f"""Plan this task with attention to dependencies.

TASK: {task}

Planning Steps:

1. **List All Sub-tasks**
   Identify everything that needs to be done.

2. **Map Dependencies**
   For each sub-task, note:
   - What must complete before this can start?
   - What depends on this being complete?
   
   Format: Sub-task A → Sub-task B (A must complete before B)

3. **Identify Critical Path**
   Which sequence determines minimum completion time?

4. **Identify Parallelizable Work**
   Which sub-tasks can run concurrently?

5. **Create Ordered Execution Plan**
   Order tasks respecting dependencies.

Plan:"""
```

---

## Input Completeness Checking

### Pre-Execution Validation

```python
def validate_inputs_prompt(task: str) -> str:
    """Prompt that validates inputs before execution."""
    
    return f"""Before executing this task, validate that you have everything needed.

TASK: {task}

Pre-Execution Checklist:

1. **Required Inputs**
   List what information/data is absolutely required:
   - [ ] Input 1: ___
   - [ ] Input 2: ___
   - [ ] Input 3: ___

2. **Provided vs Missing**
   For each required input, mark:
   ✓ Provided in the task description
   ⚠ Partially provided (note what's missing)
   ✗ Not provided

3. **Decision for Missing Inputs**
   For anything missing:
   - Can reasonable defaults/assumptions be used? If so, state them.
   - Is clarification required? If so, ask before proceeding.

4. **Proceed or Clarify?**
   - If all critical inputs present: Proceed with execution
   - If critical inputs missing: Request clarification

Validation:"""
```

### Assumption Documentation

```python
def document_assumptions(task: str) -> str:
    """Make assumptions explicit before execution."""
    
    return f"""Complete this task, but first document all assumptions.

TASK: {task}

ASSUMPTIONS AUDIT:

1. **Scope Assumptions**
   What am I assuming about the scope?
   - Included: [what's covered]
   - Excluded: [what's not covered]

2. **Data Assumptions**
   What am I assuming about data/inputs?
   - Format: [expected format]
   - Quality: [expected cleanliness]
   - Completeness: [what's provided vs inferred]

3. **Context Assumptions**
   What am I assuming about context?
   - Audience: [who is this for]
   - Purpose: [what it will be used for]
   - Constraints: [unstated limitations]

4. **Default Values**
   Where I need to use defaults:
   - [Default 1]: [value and reasoning]
   - [Default 2]: [value and reasoning]

---

IF any assumption significantly affects the output, flag it.

Now proceed with execution:"""
```

---

## Execution After Planning

### Plan-Then-Execute Pattern

```python
def plan_then_execute(task: str, show_plan: bool = True) -> str:
    """Two-phase prompt: plan, then execute."""
    
    return f"""Complete this task in two phases.

TASK: {task}

═══════════════════════════════════════════════════
PHASE 1: PLANNING
═══════════════════════════════════════════════════

Create a detailed plan:
- What are the sub-tasks?
- What's the order of operations?
- What are the expected outputs at each step?

YOUR PLAN:
[Write your plan here]

═══════════════════════════════════════════════════
PHASE 2: EXECUTION
═══════════════════════════════════════════════════

Now execute your plan step by step.
{"Show your work for each step." if show_plan else "Provide the final output only."}

EXECUTION:
[Execute your plan here]

═══════════════════════════════════════════════════
FINAL OUTPUT
═══════════════════════════════════════════════════

[Your final answer/deliverable]"""
```

### Checkpoint Execution

```python
def checkpoint_execution(task: str) -> str:
    """Execute with checkpoints to verify progress."""
    
    return f"""Complete this task with checkpoints.

TASK: {task}

INSTRUCTIONS:
1. First, create a plan with numbered steps
2. Execute each step
3. After each major step, include a checkpoint:
   
   ─── CHECKPOINT [N] ───
   ✓ Completed: [what was done]
   → Next: [what comes next]
   ⚠ Issues: [any problems] or "None"
   ────────────────────────

4. If a checkpoint reveals issues, adjust before continuing

BEGIN:

PLAN:
[Your plan]

EXECUTION WITH CHECKPOINTS:
[Execute, including checkpoints]"""
```

---

## Planning for Specific Task Types

### Code Generation Planning

```python
def code_planning_prompt(requirements: str) -> str:
    """Planning prompt for code generation."""
    
    return f"""Plan and implement this code.

REQUIREMENTS:
{requirements}

═══ PLANNING PHASE ═══

1. **Understand Requirements**
   - Primary functionality:
   - Inputs expected:
   - Outputs expected:
   - Edge cases to handle:

2. **Design Decisions**
   - Data structures needed:
   - Algorithms to use:
   - Error handling approach:
   - Dependencies/imports:

3. **Implementation Plan**
   - Component 1: [purpose]
   - Component 2: [purpose]
   - Integration approach:

4. **Testing Strategy**
   - Test case 1: [scenario]
   - Test case 2: [scenario]

═══ IMPLEMENTATION ═══

[Write the code following your plan]

═══ VERIFICATION ═══

[Quick check that code meets requirements]"""
```

### Writing/Content Planning

```python
def content_planning_prompt(brief: str) -> str:
    """Planning prompt for content creation."""
    
    return f"""Plan and write this content.

BRIEF:
{brief}

═══ PLANNING ═══

1. **Audience Analysis**
   - Who is reading this?
   - What do they already know?
   - What do they need/want?

2. **Content Structure**
   - Hook/Opening:
   - Main sections:
   - Key points per section:
   - Closing/CTA:

3. **Tone and Style**
   - Voice: [formal/casual/etc.]
   - Reading level:
   - Special requirements:

4. **Outline**
   I. [Section 1]
      - Point A
      - Point B
   II. [Section 2]
      ...

═══ WRITING ═══

[Write following your outline]

═══ SELF-CHECK ═══

- [ ] Addresses audience needs
- [ ] Follows planned structure  
- [ ] Consistent tone
- [ ] Meets length requirements"""
```

### Research/Analysis Planning

```python
def research_planning_prompt(question: str) -> str:
    """Planning prompt for research tasks."""
    
    return f"""Research and answer this question.

QUESTION: {question}

═══ RESEARCH PLAN ═══

1. **Decompose the Question**
   - Core question:
   - Sub-questions to answer:
   - Scope boundaries:

2. **Information Needed**
   - Facts to verify:
   - Data to analyze:
   - Sources to consider:

3. **Analysis Approach**
   - How will I synthesize information?
   - What frameworks apply?
   - How will I handle conflicting info?

4. **Output Structure**
   - Format of final answer:
   - Level of detail:
   - Evidence requirements:

═══ RESEARCH EXECUTION ═══

[Work through each sub-question]

═══ SYNTHESIS ═══

[Combine findings into coherent answer]

═══ FINAL ANSWER ═══

[Clear, well-supported response]"""
```

---

## Model-Specific Considerations

### Gemini 2.5/3 Optimization

Gemini models respond exceptionally well to explicit planning:

```python
def gemini_optimized_planning(task: str) -> str:
    """Planning prompt optimized for Gemini models."""
    
    return f"""You are a precise, methodical assistant. Before providing the final answer, please:

1. Parse the stated goal into distinct sub-tasks
2. Check if the input information is complete
3. Create a structured outline to achieve the goal
4. Execute the plan step by step

Respond in a structured format using clear section headers.

---

TASK: {task}

---

## Sub-task Analysis
[Break down the goal]

## Input Validation  
[Check completeness]

## Execution Plan
[Structured outline]

## Execution
[Step-by-step work]

## Final Result
[Complete output]"""
```

### GPT-4/Claude Adaptation

```python
def general_planning_prompt(task: str) -> str:
    """Planning prompt that works well across models."""
    
    return f"""I need you to complete a task, but please plan before executing.

TASK: {task}

APPROACH:
1. First, take a moment to understand what's being asked
2. Break it down into clear steps
3. Identify any information you need that isn't provided
4. Execute your plan
5. Review your output

Please show your planning, then your execution, then your final answer.

---

YOUR RESPONSE:"""
```

---

## Agentic Planning

For AI agents that take actions:

```python
def agentic_planning_prompt(goal: str, available_tools: list) -> str:
    """Planning prompt for agentic workflows."""
    
    tools_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in available_tools])
    
    return f"""You are an agent that can use tools to accomplish goals.

GOAL: {goal}

AVAILABLE TOOLS:
{tools_desc}

═══ STRATEGIC PLANNING ═══

1. **Goal Analysis**
   - What is the end state we're trying to achieve?
   - How will we know we've succeeded?

2. **Capability Mapping**
   - Which tools are relevant to this goal?
   - What can each tool accomplish?
   - What are the tool limitations?

3. **Action Sequence**
   - Step 1: [Tool to use] → [Expected outcome]
   - Step 2: [Tool to use] → [Expected outcome]
   - ...
   
4. **Contingency Planning**
   - If Step X fails: [alternative approach]
   - If unexpected result: [recovery strategy]

5. **Execution Order**
   - Dependencies: [what must complete first]
   - Can parallelize: [what can run concurrently]

═══ BEGIN EXECUTION ═══

Proceed with your plan, adapting as needed based on tool outputs."""
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Plan before execute | Catches issues early |
| Make assumptions explicit | Reduces misunderstandings |
| Check input completeness | Prevents wasted effort |
| Use structured formatting | Clearer plans |
| Include checkpoints | Enables course correction |
| Match planning depth to task complexity | Efficiency |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Over-planning | More time planning than doing | Keep plans proportional to task |
| Vague plans | Plans don't translate to action | Require concrete steps |
| Ignoring the plan | Model doesn't follow its own plan | Reference plan during execution |
| No input validation | Proceeds with missing info | Explicit completeness check |
| Rigid planning | Can't adapt when issues arise | Build in contingencies |

---

## Hands-on Exercise

### Your Task

Create an explicit planning prompt for a multi-step task: "Create a marketing plan for launching a new mobile app in the fitness category."

<details>
<summary>💡 Hints</summary>

1. Parse goal: What does "marketing plan" include?
2. Check inputs: What info about the app is needed?
3. Structure: What sections should the plan have?
4. Audience/context: Who is this plan for?

</details>

<details>
<summary>✅ Solution</summary>

```python
planning_prompt = """Before creating this marketing plan, let's plan our approach.

TASK: Create a marketing plan for launching a new mobile app in the fitness category.

═══ PHASE 1: UNDERSTANDING THE GOAL ═══

**What does "marketing plan" mean here?**
A comprehensive marketing plan typically includes:
- Target audience analysis
- Competitive positioning
- Channel strategy
- Messaging/creative direction
- Launch timeline
- Budget allocation (if applicable)
- Success metrics

**What do I need to know about the app?**

Required info (checking what's provided):
⚠ App name: Not specified → Will use placeholder "[AppName]"
⚠ Key features: Not specified → Will assume typical fitness app features
⚠ Differentiator: Not specified → Will note this gap
⚠ Target audience: "Fitness" is broad → Will need to narrow
⚠ Budget: Not specified → Will provide tiered options
⚠ Timeline: Not specified → Will assume standard launch timeline

**Assumptions I'll make:**
1. Mobile-first launch (iOS + Android)
2. Consumer app (not B2B)
3. Freemium model (common in fitness)
4. Pre-launch phase exists for building awareness

═══ PHASE 2: PLAN STRUCTURE ═══

My marketing plan will include:

1. Executive Summary
2. Target Audience Definition
   - Primary persona
   - Secondary personas
   - Market size estimate
3. Competitive Analysis
   - Key competitors
   - Positioning opportunity
4. Marketing Channels
   - Digital (social, ASO, ads)
   - Partnerships
   - PR/Earned media
5. Messaging Framework
   - Core value proposition
   - Key messages by audience
6. Launch Timeline
   - Pre-launch (4-6 weeks)
   - Launch week
   - Post-launch (ongoing)
7. Metrics & KPIs
8. Recommendations / Next Steps

═══ PHASE 3: EXECUTION ═══

[Proceeds to write each section following the structure above]

---

NOTE TO USER: This plan uses placeholder assumptions. For a more targeted plan, please provide:
- App name and core features
- Key differentiator from competitors
- Target budget range
- Specific launch date/timeline constraints
"""
```

</details>

---

## Summary

- Explicit planning prompts ask the model to think before acting
- Core pattern: Parse goal → Check inputs → Create plan → Execute
- Catches missing information before wasting effort
- Particularly effective with modern reasoning models (Gemini 2.5/3, GPT-4)
- Use hierarchical decomposition for complex tasks
- Include checkpoints for long executions
- Match planning depth to task complexity

**Next:** [Advanced Techniques Overview](./00-advanced-techniques-overview.md) (review) or proceed to [Lesson 9](../09-prompt-optimization.md)

---

<!-- Sources: Google Gemini Prompting Strategies (explicit planning), Anthropic prompt design patterns -->
