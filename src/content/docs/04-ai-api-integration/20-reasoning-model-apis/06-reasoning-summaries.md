---
title: "Reasoning Summaries"
---

# Reasoning Summaries

## Introduction

Reasoning models can provide summaries of their internal thinking process. While the raw reasoning tokens are not visible, summaries give you insight into how the model approached a problem—useful for debugging, transparency, and building trust with users.

### What We'll Cover

- The summary parameter and its options
- Viewing and interpreting reasoning summaries
- Summary output structure
- Organization verification requirements

### Prerequisites

- Reasoning models overview
- Understanding of reasoning tokens
- API response handling

---

## The Summary Parameter

### Understanding Summary Options

```python
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class SummaryLevel(str, Enum):
    """Levels of reasoning summary detail."""
    
    AUTO = "auto"
    CONCISE = "concise"
    DETAILED = "detailed"


@dataclass
class SummaryOption:
    """Description of a summary option."""
    
    level: SummaryLevel
    description: str
    use_case: str
    typical_length: str
    visibility: str


SUMMARY_OPTIONS = [
    SummaryOption(
        level=SummaryLevel.AUTO,
        description="Let the model decide summary length",
        use_case="General purpose, balanced approach",
        typical_length="Variable based on complexity",
        visibility="Requires organization verification"
    ),
    SummaryOption(
        level=SummaryLevel.CONCISE,
        description="Brief summary of key reasoning steps",
        use_case="Production apps, quick insight",
        typical_length="1-3 sentences per reasoning block",
        visibility="Requires organization verification"
    ),
    SummaryOption(
        level=SummaryLevel.DETAILED,
        description="Comprehensive reasoning explanation",
        use_case="Debugging, education, transparency",
        typical_length="Full paragraph per reasoning block",
        visibility="Requires organization verification"
    )
]


print("Reasoning Summary Options")
print("=" * 60)

for option in SUMMARY_OPTIONS:
    print(f"\n📝 {option.level.value.upper()}")
    print(f"   {option.description}")
    print(f"   Use case: {option.use_case}")
    print(f"   Length: {option.typical_length}")
    print(f"   ⚠️  {option.visibility}")


print("""

📊 Summary vs Raw Reasoning Tokens

┌─────────────────────────────────────────────────────────┐
│                  REASONING PROCESS                      │
├─────────────────────────────────────────────────────────┤
│  Raw Reasoning Tokens                                   │
│  ┌───────────────────────────────────────────────────┐ │
│  │ "Let me think about this step by step..."         │ │
│  │ "First, I need to consider..."                    │ │
│  │ "Actually, that approach won't work because..."   │ │
│  │ "A better approach would be..."                   │ │
│  │ [... potentially thousands of tokens ...]         │ │
│  └───────────────────────────────────────────────────┘ │
│                     ↓                                   │
│  HIDDEN from user (but billed)                         │
│                     ↓                                   │
│  Summary (if requested)                                │
│  ┌───────────────────────────────────────────────────┐ │
│  │ "Analyzed problem using step-by-step approach.    │ │
│  │  Considered multiple strategies and selected      │ │
│  │  optimal solution based on efficiency."           │ │
│  └───────────────────────────────────────────────────┘ │
│                     ↓                                   │
│  VISIBLE in response                                   │
└─────────────────────────────────────────────────────────┘
""")
```

---

## Requesting Summaries

### Configuring Summary in Requests

```python
from typing import Dict, Any


class ReasoningSummaryClient:
    """Client for requesting reasoning summaries."""
    
    def __init__(self, model: str = "gpt-5"):
        self.model = model
    
    def create_request(
        self,
        messages: List[dict],
        summary: SummaryLevel = SummaryLevel.AUTO,
        effort: str = "medium"
    ) -> dict:
        """Create request with summary configuration."""
        
        input_items = []
        
        for msg in messages:
            input_items.append({
                "type": "message",
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return {
            "model": self.model,
            "input": input_items,
            "reasoning": {
                "effort": effort,
                "summary": summary.value
            }
        }
    
    def create_request_with_reasoning_include(
        self,
        messages: List[dict],
        summary: SummaryLevel = SummaryLevel.DETAILED
    ) -> dict:
        """Create request that includes both encrypted and summary."""
        
        input_items = []
        
        for msg in messages:
            input_items.append({
                "type": "message",
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return {
            "model": self.model,
            "input": input_items,
            "reasoning": {
                "effort": "medium",
                "summary": summary.value
            },
            # Can request both encrypted content AND summary
            "include": ["reasoning.encrypted_content"]
        }


print("\nRequest Configuration Examples")
print("=" * 60)

client = ReasoningSummaryClient("gpt-5")

# Basic summary request
request1 = client.create_request(
    [{"role": "user", "content": "Solve this complex equation..."}],
    summary=SummaryLevel.CONCISE
)

print("\n📤 Concise Summary Request:")
print(f"   reasoning.summary: {request1['reasoning']['summary']}")

# Detailed for debugging
request2 = client.create_request(
    [{"role": "user", "content": "Debug this code..."}],
    summary=SummaryLevel.DETAILED
)

print(f"\n📤 Detailed Summary Request:")
print(f"   reasoning.summary: {request2['reasoning']['summary']}")

# Both encrypted and summary
request3 = client.create_request_with_reasoning_include(
    [{"role": "user", "content": "Analyze this problem..."}]
)

print(f"\n📤 Combined Request (encrypted + summary):")
print(f"   reasoning.summary: {request3['reasoning']['summary']}")
print(f"   include: {request3['include']}")
```

---

## Summary Output Structure

### Understanding the Response Format

```python
@dataclass
class ReasoningSummaryItem:
    """A reasoning item with summary."""
    
    item_type: str = "reasoning"
    item_id: str = ""
    summary: List[dict] = None  # List of summary text parts
    encrypted_content: Optional[str] = None
    status: str = "completed"


def parse_reasoning_items(response: dict) -> List[dict]:
    """Parse reasoning items from response."""
    
    reasoning_items = []
    
    for item in response.get("output", []):
        if item.get("type") == "reasoning":
            parsed = {
                "id": item.get("id", ""),
                "status": item.get("status", "completed"),
                "summary": None,
                "encrypted_content": None,
                "summary_text": None
            }
            
            # Extract summary if present
            summary_parts = item.get("summary", [])
            if summary_parts:
                parsed["summary"] = summary_parts
                # Combine text parts
                text_parts = [
                    part.get("text", "") 
                    for part in summary_parts 
                    if part.get("type") == "summary_text"
                ]
                parsed["summary_text"] = " ".join(text_parts)
            
            # Extract encrypted content if present
            if "encrypted_content" in item:
                parsed["encrypted_content"] = item["encrypted_content"]
            
            reasoning_items.append(parsed)
    
    return reasoning_items


# Example response with summary
EXAMPLE_RESPONSE = {
    "id": "resp_001",
    "output": [
        {
            "type": "reasoning",
            "id": "reasoning_001",
            "status": "completed",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "Analyzed the equation by first isolating the variable. "
                },
                {
                    "type": "summary_text",
                    "text": "Applied quadratic formula to find two solutions. "
                },
                {
                    "type": "summary_text",
                    "text": "Verified both solutions satisfy the original equation."
                }
            ]
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "The solutions are x = 3 and x = -2."
                }
            ]
        }
    ]
}


print("\n\nParsing Summary Responses")
print("=" * 60)

items = parse_reasoning_items(EXAMPLE_RESPONSE)

for item in items:
    print(f"\n📝 Reasoning Item: {item['id']}")
    print(f"   Status: {item['status']}")
    
    if item['summary_text']:
        print(f"\n   📖 Summary:")
        print(f"   {item['summary_text']}")
    
    if item['encrypted_content']:
        print(f"   🔒 Encrypted content: Present")


# Full response parser
class SummaryResponseParser:
    """Parse responses with reasoning summaries."""
    
    def parse(self, response: dict) -> dict:
        """Parse complete response."""
        
        result = {
            "output_text": None,
            "reasoning_summaries": [],
            "has_encrypted_reasoning": False,
            "total_reasoning_items": 0
        }
        
        for item in response.get("output", []):
            item_type = item.get("type")
            
            if item_type == "reasoning":
                result["total_reasoning_items"] += 1
                
                # Extract summary
                summary_parts = item.get("summary", [])
                if summary_parts:
                    summary_text = " ".join(
                        p.get("text", "") 
                        for p in summary_parts 
                        if p.get("type") == "summary_text"
                    )
                    result["reasoning_summaries"].append({
                        "id": item.get("id"),
                        "text": summary_text
                    })
                
                if "encrypted_content" in item:
                    result["has_encrypted_reasoning"] = True
            
            elif item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        result["output_text"] = part.get("text")
        
        return result


print("\n\nComplete Response Parsing")
print("=" * 60)

parser = SummaryResponseParser()
parsed = parser.parse(EXAMPLE_RESPONSE)

print(f"\n📊 Parsed Response:")
print(f"   Output: {parsed['output_text']}")
print(f"   Reasoning items: {parsed['total_reasoning_items']}")
print(f"   Has encrypted: {parsed['has_encrypted_reasoning']}")

print(f"\n📖 Summaries:")
for summary in parsed['reasoning_summaries']:
    print(f"   [{summary['id']}]")
    print(f"   {summary['text']}")
```

---

## Interpreting Summaries

### What Summaries Reveal

```python
@dataclass
class SummaryInsight:
    """Types of insights from summaries."""
    
    insight_type: str
    description: str
    example: str
    useful_for: str


SUMMARY_INSIGHTS = [
    SummaryInsight(
        insight_type="Problem decomposition",
        description="How the model broke down the problem",
        example="'First analyzed requirements, then identified constraints...'",
        useful_for="Understanding approach complexity"
    ),
    SummaryInsight(
        insight_type="Strategy selection",
        description="Why the model chose a particular approach",
        example="'Selected iterative approach over recursive due to memory efficiency...'",
        useful_for="Validating solution methodology"
    ),
    SummaryInsight(
        insight_type="Error checking",
        description="Steps taken to verify correctness",
        example="'Verified result by substituting back into original equation...'",
        useful_for="Building confidence in answers"
    ),
    SummaryInsight(
        insight_type="Alternative consideration",
        description="Other approaches that were evaluated",
        example="'Considered three approaches: brute force, dynamic programming, greedy...'",
        useful_for="Understanding solution optimality"
    ),
    SummaryInsight(
        insight_type="Assumption clarification",
        description="Assumptions made during reasoning",
        example="'Assumed input is already sorted as stated in problem...'",
        useful_for="Identifying potential issues"
    )
]


print("What Reasoning Summaries Reveal")
print("=" * 60)

for insight in SUMMARY_INSIGHTS:
    print(f"\n🔍 {insight.insight_type}")
    print(f"   {insight.description}")
    print(f"   Example: {insight.example}")
    print(f"   Useful for: {insight.useful_for}")


class SummaryAnalyzer:
    """Analyze reasoning summaries for insights."""
    
    KEYWORDS = {
        "decomposition": ["first", "then", "next", "finally", "step"],
        "strategy": ["approach", "method", "strategy", "chose", "selected"],
        "verification": ["verified", "checked", "confirmed", "validated"],
        "alternatives": ["considered", "compared", "evaluated", "alternative"],
        "assumptions": ["assumed", "given", "if", "assuming", "suppose"]
    }
    
    def analyze(self, summary_text: str) -> dict:
        """Analyze a summary for insights."""
        
        text_lower = summary_text.lower()
        
        insights = {
            "categories": [],
            "keywords_found": {},
            "complexity_indicators": {
                "multi_step": False,
                "considered_alternatives": False,
                "verified_result": False
            }
        }
        
        for category, keywords in self.KEYWORDS.items():
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                insights["categories"].append(category)
                insights["keywords_found"][category] = found
        
        # Check complexity indicators
        if any(w in text_lower for w in ["first", "then", "next", "finally"]):
            insights["complexity_indicators"]["multi_step"] = True
        
        if any(w in text_lower for w in ["considered", "compared", "alternative"]):
            insights["complexity_indicators"]["considered_alternatives"] = True
        
        if any(w in text_lower for w in ["verified", "checked", "confirmed"]):
            insights["complexity_indicators"]["verified_result"] = True
        
        return insights


print("\n\nSummary Analysis")
print("=" * 60)

analyzer = SummaryAnalyzer()

sample_summary = """
First analyzed the problem requirements and constraints. 
Considered three potential approaches: recursive, iterative, and dynamic programming.
Selected the dynamic programming approach for optimal time complexity.
Verified the solution by tracing through example inputs.
"""

analysis = analyzer.analyze(sample_summary)

print(f"\n📝 Sample Summary:")
print(f"   {sample_summary.strip()}")

print(f"\n📊 Analysis:")
print(f"   Categories: {analysis['categories']}")
print(f"   Multi-step: {analysis['complexity_indicators']['multi_step']}")
print(f"   Considered alternatives: {analysis['complexity_indicators']['considered_alternatives']}")
print(f"   Verified result: {analysis['complexity_indicators']['verified_result']}")
```

---

## Organization Verification

### Understanding Access Requirements

```python
@dataclass
class VerificationRequirement:
    """Requirement for accessing summaries."""
    
    requirement: str
    description: str
    how_to_verify: str


VERIFICATION_REQUIREMENTS = [
    VerificationRequirement(
        requirement="Organization verification",
        description="Organization must be verified with OpenAI",
        how_to_verify="Complete verification in OpenAI dashboard"
    ),
    VerificationRequirement(
        requirement="API access",
        description="API key must belong to verified organization",
        how_to_verify="Use organization API key, not personal"
    ),
    VerificationRequirement(
        requirement="Model support",
        description="Not all models support summaries",
        how_to_verify="Check model documentation for support"
    )
]


print("Summary Access Requirements")
print("=" * 60)

for req in VERIFICATION_REQUIREMENTS:
    print(f"\n⚠️  {req.requirement}")
    print(f"   {req.description}")
    print(f"   Verify: {req.how_to_verify}")


class SummaryAccessChecker:
    """Check if summaries are accessible."""
    
    def __init__(
        self,
        organization_verified: bool = False,
        organization_id: Optional[str] = None
    ):
        self.organization_verified = organization_verified
        self.organization_id = organization_id
    
    def check_access(self) -> dict:
        """Check summary access status."""
        
        issues = []
        
        if not self.organization_id:
            issues.append("No organization ID configured")
        
        if not self.organization_verified:
            issues.append("Organization not verified")
        
        return {
            "can_access_summaries": len(issues) == 0,
            "organization_id": self.organization_id,
            "verified": self.organization_verified,
            "issues": issues,
            "recommendation": self._get_recommendation(issues)
        }
    
    def _get_recommendation(self, issues: List[str]) -> str:
        """Get recommendation based on issues."""
        
        if not issues:
            return "Summaries available. Use reasoning.summary parameter."
        
        if "not verified" in str(issues):
            return "Complete organization verification at platform.openai.com"
        
        if "No organization" in str(issues):
            return "Set organization ID in API configuration"
        
        return "Contact OpenAI support for assistance"


print("\n\nAccess Check Example")
print("=" * 60)

# Unverified organization
checker1 = SummaryAccessChecker(
    organization_verified=False,
    organization_id="org-abc123"
)

result1 = checker1.check_access()
print(f"\n❌ Unverified Organization:")
print(f"   Can access: {result1['can_access_summaries']}")
print(f"   Issues: {result1['issues']}")
print(f"   Recommendation: {result1['recommendation']}")

# Verified organization
checker2 = SummaryAccessChecker(
    organization_verified=True,
    organization_id="org-xyz789"
)

result2 = checker2.check_access()
print(f"\n✅ Verified Organization:")
print(f"   Can access: {result2['can_access_summaries']}")
print(f"   Recommendation: {result2['recommendation']}")
```

---

## Practical Applications

### Using Summaries Effectively

```python
@dataclass
class SummaryUseCase:
    """Use case for reasoning summaries."""
    
    use_case: str
    summary_level: SummaryLevel
    description: str
    implementation: str


SUMMARY_USE_CASES = [
    SummaryUseCase(
        use_case="Debugging AI responses",
        summary_level=SummaryLevel.DETAILED,
        description="Understand why the model gave a specific answer",
        implementation="Log detailed summaries for failed queries"
    ),
    SummaryUseCase(
        use_case="User transparency",
        summary_level=SummaryLevel.CONCISE,
        description="Show users how AI arrived at conclusions",
        implementation="Display concise summary in UI"
    ),
    SummaryUseCase(
        use_case="Quality assurance",
        summary_level=SummaryLevel.DETAILED,
        description="Verify reasoning methodology is sound",
        implementation="Review summaries in QA pipeline"
    ),
    SummaryUseCase(
        use_case="Educational content",
        summary_level=SummaryLevel.DETAILED,
        description="Teach problem-solving approaches",
        implementation="Use summaries as learning examples"
    ),
    SummaryUseCase(
        use_case="Compliance documentation",
        summary_level=SummaryLevel.AUTO,
        description="Record how decisions were made",
        implementation="Store summaries for audit trail"
    )
]


print("Summary Use Cases")
print("=" * 60)

for case in SUMMARY_USE_CASES:
    print(f"\n🎯 {case.use_case}")
    print(f"   Level: {case.summary_level.value}")
    print(f"   {case.description}")
    print(f"   Implementation: {case.implementation}")


class SummaryDisplayFormatter:
    """Format summaries for different contexts."""
    
    def for_ui(self, summary_text: str, max_length: int = 200) -> str:
        """Format summary for UI display."""
        
        if len(summary_text) <= max_length:
            return summary_text
        
        # Truncate at sentence boundary if possible
        truncated = summary_text[:max_length]
        last_period = truncated.rfind(".")
        
        if last_period > max_length * 0.5:
            return truncated[:last_period + 1]
        
        return truncated.rstrip() + "..."
    
    def for_log(self, summary_text: str, request_id: str) -> dict:
        """Format summary for logging."""
        
        return {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "summary": summary_text,
            "length": len(summary_text),
            "word_count": len(summary_text.split())
        }
    
    def for_audit(
        self,
        summary_text: str,
        input_text: str,
        output_text: str
    ) -> dict:
        """Format summary for audit records."""
        
        return {
            "reasoning_summary": summary_text,
            "input_hash": hashlib.sha256(input_text.encode()).hexdigest()[:16],
            "output_hash": hashlib.sha256(output_text.encode()).hexdigest()[:16],
            "recorded_at": datetime.now().isoformat(),
            "summary_length": len(summary_text)
        }


import hashlib
from datetime import datetime

print("\n\nSummary Formatting")
print("=" * 60)

formatter = SummaryDisplayFormatter()

sample = """
Analyzed the code by first examining the error message and stack trace.
Identified the root cause as a null pointer exception in the data processing pipeline.
Traced the issue to an uninitialized variable in the configuration loader.
Verified the fix by running the test suite and checking edge cases.
"""

print(f"\n📱 UI Display (max 200 chars):")
print(f"   {formatter.for_ui(sample, 200)}")

print(f"\n📋 Log Format:")
log_entry = formatter.for_log(sample, "req_123")
for key, value in log_entry.items():
    print(f"   {key}: {value}")
```

---

## Hands-on Exercise

### Your Task

Build a system that requests, parses, analyzes, and displays reasoning summaries effectively.

### Requirements

1. Request summaries at different detail levels
2. Parse summary responses correctly
3. Analyze summaries for insights
4. Format for multiple output contexts

<details>
<summary>💡 Hints</summary>

- Use SummaryLevel enum for configuration
- Extract text from summary parts array
- Look for keywords to categorize insights
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import hashlib


class SummaryLevel(str, Enum):
    AUTO = "auto"
    CONCISE = "concise"
    DETAILED = "detailed"


class InsightCategory(str, Enum):
    METHODOLOGY = "methodology"
    VERIFICATION = "verification"
    ALTERNATIVES = "alternatives"
    ASSUMPTIONS = "assumptions"
    DECOMPOSITION = "decomposition"


@dataclass
class SummaryInsight:
    """An insight extracted from a summary."""
    
    category: InsightCategory
    text: str
    keywords: List[str]
    confidence: float


@dataclass
class AnalyzedSummary:
    """A fully analyzed reasoning summary."""
    
    raw_text: str
    insights: List[SummaryInsight]
    complexity_score: int
    is_multi_step: bool
    verified_result: bool


class ReasoningSummarySystem:
    """Complete reasoning summary handling system."""
    
    CATEGORY_KEYWORDS = {
        InsightCategory.METHODOLOGY: [
            "approach", "method", "strategy", "technique", "algorithm"
        ],
        InsightCategory.VERIFICATION: [
            "verified", "checked", "confirmed", "tested", "validated"
        ],
        InsightCategory.ALTERNATIVES: [
            "considered", "compared", "evaluated", "alternative", "options"
        ],
        InsightCategory.ASSUMPTIONS: [
            "assumed", "given", "if", "assuming", "suppose", "constraint"
        ],
        InsightCategory.DECOMPOSITION: [
            "first", "then", "next", "finally", "step", "break down"
        ]
    }
    
    def __init__(self, model: str = "gpt-5"):
        self.model = model
        self.summaries_received: List[dict] = []
    
    def create_request(
        self,
        messages: List[dict],
        summary_level: SummaryLevel = SummaryLevel.AUTO
    ) -> dict:
        """Create a request with summary configuration."""
        
        input_items = [
            {"type": "message", "role": m["role"], "content": m["content"]}
            for m in messages
        ]
        
        return {
            "model": self.model,
            "input": input_items,
            "reasoning": {
                "effort": "medium",
                "summary": summary_level.value
            }
        }
    
    def parse_response(self, response: dict) -> dict:
        """Parse response and extract summaries."""
        
        result = {
            "output_text": None,
            "summaries": [],
            "raw_summaries": []
        }
        
        for item in response.get("output", []):
            item_type = item.get("type")
            
            if item_type == "reasoning":
                summary_parts = item.get("summary", [])
                if summary_parts:
                    # Combine text parts
                    text = " ".join(
                        p.get("text", "")
                        for p in summary_parts
                        if p.get("type") == "summary_text"
                    )
                    
                    result["raw_summaries"].append(text)
                    
                    # Analyze summary
                    analyzed = self.analyze_summary(text)
                    result["summaries"].append(analyzed)
            
            elif item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        result["output_text"] = part.get("text")
        
        # Store for later analysis
        self.summaries_received.append(result)
        
        return result
    
    def analyze_summary(self, summary_text: str) -> AnalyzedSummary:
        """Analyze a summary for insights."""
        
        text_lower = summary_text.lower()
        insights = []
        
        # Find insights by category
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            
            if found_keywords:
                # Extract relevant sentence
                sentences = summary_text.split(".")
                relevant_text = next(
                    (s for s in sentences if any(kw in s.lower() for kw in found_keywords)),
                    summary_text[:100]
                )
                
                insights.append(SummaryInsight(
                    category=category,
                    text=relevant_text.strip(),
                    keywords=found_keywords,
                    confidence=min(len(found_keywords) * 0.3, 1.0)
                ))
        
        # Calculate complexity score
        complexity = 0
        if "first" in text_lower and "then" in text_lower:
            complexity += 2
        if "considered" in text_lower or "compared" in text_lower:
            complexity += 2
        if "verified" in text_lower or "checked" in text_lower:
            complexity += 1
        complexity += min(len(summary_text.split(".")) - 1, 3)
        
        return AnalyzedSummary(
            raw_text=summary_text,
            insights=insights,
            complexity_score=complexity,
            is_multi_step=any(
                kw in text_lower 
                for kw in ["first", "then", "next", "finally"]
            ),
            verified_result="verified" in text_lower or "checked" in text_lower
        )
    
    def format_for_ui(
        self,
        analyzed: AnalyzedSummary,
        max_length: int = 300
    ) -> dict:
        """Format analyzed summary for UI display."""
        
        text = analyzed.raw_text
        if len(text) > max_length:
            # Truncate at sentence boundary
            truncated = text[:max_length]
            last_period = truncated.rfind(".")
            if last_period > max_length * 0.5:
                text = truncated[:last_period + 1]
            else:
                text = truncated.rstrip() + "..."
        
        # Create insight badges
        badges = [
            f"📊 {insight.category.value.title()}"
            for insight in analyzed.insights
            if insight.confidence > 0.5
        ]
        
        quality_indicators = []
        if analyzed.is_multi_step:
            quality_indicators.append("🔢 Multi-step")
        if analyzed.verified_result:
            quality_indicators.append("✅ Verified")
        if analyzed.complexity_score >= 5:
            quality_indicators.append("🧠 Complex")
        
        return {
            "summary_text": text,
            "insight_badges": badges,
            "quality_indicators": quality_indicators,
            "complexity": analyzed.complexity_score
        }
    
    def format_for_log(
        self,
        analyzed: AnalyzedSummary,
        request_id: str
    ) -> dict:
        """Format for logging."""
        
        return {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "summary_length": len(analyzed.raw_text),
            "insight_categories": [
                i.category.value for i in analyzed.insights
            ],
            "complexity_score": analyzed.complexity_score,
            "is_multi_step": analyzed.is_multi_step,
            "verified_result": analyzed.verified_result,
            "summary_hash": hashlib.sha256(
                analyzed.raw_text.encode()
            ).hexdigest()[:16]
        }
    
    def format_for_audit(
        self,
        analyzed: AnalyzedSummary,
        input_text: str,
        output_text: str
    ) -> dict:
        """Format for audit trail."""
        
        return {
            "recorded_at": datetime.now().isoformat(),
            "reasoning_summary": analyzed.raw_text,
            "insights": [
                {
                    "category": i.category.value,
                    "confidence": i.confidence
                }
                for i in analyzed.insights
            ],
            "input_hash": hashlib.sha256(input_text.encode()).hexdigest()[:16],
            "output_hash": hashlib.sha256(output_text.encode()).hexdigest()[:16],
            "complexity_score": analyzed.complexity_score,
            "verified": analyzed.verified_result
        }
    
    def get_statistics(self) -> dict:
        """Get statistics on received summaries."""
        
        if not self.summaries_received:
            return {"message": "No summaries received yet"}
        
        all_summaries = [
            s for r in self.summaries_received 
            for s in r["summaries"]
        ]
        
        if not all_summaries:
            return {"message": "No summaries parsed"}
        
        return {
            "total_summaries": len(all_summaries),
            "avg_complexity": sum(
                s.complexity_score for s in all_summaries
            ) / len(all_summaries),
            "multi_step_percentage": sum(
                1 for s in all_summaries if s.is_multi_step
            ) / len(all_summaries) * 100,
            "verified_percentage": sum(
                1 for s in all_summaries if s.verified_result
            ) / len(all_summaries) * 100,
            "insight_distribution": self._get_insight_distribution(all_summaries)
        }
    
    def _get_insight_distribution(
        self,
        summaries: List[AnalyzedSummary]
    ) -> Dict[str, int]:
        """Get distribution of insight categories."""
        
        distribution = {cat.value: 0 for cat in InsightCategory}
        
        for summary in summaries:
            for insight in summary.insights:
                distribution[insight.category.value] += 1
        
        return distribution


# Demo
print("\nReasoning Summary System Demo")
print("=" * 60)

system = ReasoningSummarySystem("gpt-5")

# Create request
request = system.create_request(
    [{"role": "user", "content": "Solve this optimization problem..."}],
    summary_level=SummaryLevel.DETAILED
)

print(f"\n📤 Request created:")
print(f"   Summary level: {request['reasoning']['summary']}")

# Simulate response
mock_response = {
    "output": [
        {
            "type": "reasoning",
            "id": "r1",
            "summary": [
                {"type": "summary_text", "text": "First analyzed the problem constraints and objective function. "},
                {"type": "summary_text", "text": "Considered three approaches: gradient descent, genetic algorithms, and linear programming. "},
                {"type": "summary_text", "text": "Selected linear programming due to the linear nature of constraints. "},
                {"type": "summary_text", "text": "Verified the solution satisfies all constraints and is optimal."}
            ]
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "The optimal solution is x=5, y=3."}]
        }
    ]
}

parsed = system.parse_response(mock_response)

print(f"\n📥 Response parsed:")
print(f"   Output: {parsed['output_text']}")
print(f"   Summaries found: {len(parsed['summaries'])}")

if parsed['summaries']:
    analyzed = parsed['summaries'][0]
    
    print(f"\n📊 Analysis:")
    print(f"   Complexity score: {analyzed.complexity_score}")
    print(f"   Multi-step: {analyzed.is_multi_step}")
    print(f"   Verified result: {analyzed.verified_result}")
    
    print(f"\n🔍 Insights:")
    for insight in analyzed.insights:
        print(f"   [{insight.category.value}] {insight.text[:60]}...")
    
    # Format for UI
    ui_format = system.format_for_ui(analyzed)
    print(f"\n📱 UI Format:")
    print(f"   Badges: {ui_format['insight_badges']}")
    print(f"   Quality: {ui_format['quality_indicators']}")
    
    # Format for log
    log_format = system.format_for_log(analyzed, "req_001")
    print(f"\n📋 Log Format:")
    print(f"   Categories: {log_format['insight_categories']}")
    print(f"   Hash: {log_format['summary_hash']}")

# Statistics
stats = system.get_statistics()
print(f"\n📈 Statistics:")
for key, value in stats.items():
    print(f"   {key}: {value}")
```

</details>

---

## Summary

✅ Use `reasoning.summary` parameter with "auto", "concise", or "detailed"  
✅ Summaries are returned in the reasoning item's `summary` array  
✅ Organization verification is required to access summaries  
✅ Summaries reveal methodology, verification steps, and alternatives considered  
✅ Use summaries for debugging, transparency, and compliance documentation

**Next:** [Best Practices](./07-best-practices.md)

---

## Further Reading

- [OpenAI Reasoning Models](https://platform.openai.com/docs/guides/reasoning) — Official guide
- [Organization Settings](https://platform.openai.com/settings/organization) — Verification
- [API Reference](https://platform.openai.com/docs/api-reference/responses) — Response format
