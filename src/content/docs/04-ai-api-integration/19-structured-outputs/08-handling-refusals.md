---
title: "Handling Refusals"
---

# Handling Refusals

## Introduction

When using Structured Outputs, the model may refuse to generate content that violates safety policies. Instead of invalid JSON, the response includes a `refusal` field explaining why the request was declined.

### What We'll Cover

- Understanding the refusal mechanism
- Detecting and handling refusals
- User-friendly refusal messaging
- Designing refusal-resilient applications

### Prerequisites

- Structured Outputs configuration
- Error handling patterns
- Safety policy awareness

---

## The Refusal Mechanism

### How Refusals Work

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class RefusalReason(Enum):
    """Categories of refusal reasons."""
    
    HARMFUL_CONTENT = "harmful_content"
    PRIVACY = "privacy_violation"
    ILLEGAL = "illegal_activity"
    DECEPTIVE = "deceptive_content"
    SAFETY = "general_safety"
    POLICY = "policy_violation"


@dataclass
class RefusalInfo:
    """Information about a refusal."""
    
    reason_category: RefusalReason
    model_message: str
    user_friendly_message: str
    can_retry_modified: bool


REFUSAL_EXAMPLES = [
    RefusalInfo(
        reason_category=RefusalReason.HARMFUL_CONTENT,
        model_message="I can't help with creating harmful content.",
        user_friendly_message="This request involves content that could cause harm. Please modify your request.",
        can_retry_modified=True
    ),
    RefusalInfo(
        reason_category=RefusalReason.PRIVACY,
        model_message="I can't help extract personal information without consent.",
        user_friendly_message="This request may involve private data. Please ensure you have proper authorization.",
        can_retry_modified=True
    ),
    RefusalInfo(
        reason_category=RefusalReason.ILLEGAL,
        model_message="I can't assist with illegal activities.",
        user_friendly_message="This request involves activities that may be illegal. We cannot proceed.",
        can_retry_modified=False
    )
]


print("Refusal Categories")
print("=" * 60)

for info in REFUSAL_EXAMPLES:
    print(f"\n🚫 {info.reason_category.value.replace('_', ' ').title()}")
    print(f"   Model says: \"{info.model_message}\"")
    print(f"   User sees: \"{info.user_friendly_message}\"")
    retry = "Yes" if info.can_retry_modified else "No"
    print(f"   Retry possible: {retry}")
```

### Response Structure with Refusals

```python
@dataclass
class StructuredResponse:
    """Structured output response with potential refusal."""
    
    content: Optional[dict]
    refusal: Optional[str]
    
    @property
    def is_refused(self) -> bool:
        """Check if response was refused."""
        return self.refusal is not None
    
    @property
    def has_content(self) -> bool:
        """Check if response has content."""
        return self.content is not None


# Simulated API responses
NORMAL_RESPONSE = StructuredResponse(
    content={"name": "John Doe", "age": 30},
    refusal=None
)

REFUSED_RESPONSE = StructuredResponse(
    content=None,
    refusal="I can't help with extracting personal information without proper consent."
)


print("\n\nResponse Structure")
print("=" * 60)

print("\n✅ Normal Response:")
print(f"   content: {NORMAL_RESPONSE.content}")
print(f"   refusal: {NORMAL_RESPONSE.refusal}")
print(f"   is_refused: {NORMAL_RESPONSE.is_refused}")

print("\n🚫 Refused Response:")
print(f"   content: {REFUSED_RESPONSE.content}")
print(f"   refusal: {REFUSED_RESPONSE.refusal}")
print(f"   is_refused: {REFUSED_RESPONSE.is_refused}")
```

---

## Detecting Refusals

### SDK Refusal Detection

```python
from pydantic import BaseModel


class ExtractedPerson(BaseModel):
    """Person data being extracted."""
    
    name: str
    age: int
    occupation: str


class MockParsedResponse:
    """Mock SDK parsed response."""
    
    def __init__(
        self,
        parsed: Optional[ExtractedPerson] = None,
        refusal: Optional[str] = None
    ):
        self.parsed = parsed
        self.refusal = refusal


class RefusalDetector:
    """Detect and categorize refusals."""
    
    REFUSAL_PATTERNS = {
        RefusalReason.HARMFUL_CONTENT: ["harmful", "dangerous", "violence"],
        RefusalReason.PRIVACY: ["personal information", "private", "consent"],
        RefusalReason.ILLEGAL: ["illegal", "unlawful", "criminal"],
        RefusalReason.DECEPTIVE: ["deceptive", "misleading", "fake"],
        RefusalReason.POLICY: ["policy", "guidelines", "terms"]
    }
    
    def detect(self, response: MockParsedResponse) -> Optional[RefusalInfo]:
        """Detect if response contains a refusal."""
        
        if response.refusal is None:
            return None
        
        # Categorize the refusal
        category = self._categorize(response.refusal)
        
        return RefusalInfo(
            reason_category=category,
            model_message=response.refusal,
            user_friendly_message=self._get_user_message(category),
            can_retry_modified=category not in [RefusalReason.ILLEGAL]
        )
    
    def _categorize(self, refusal_text: str) -> RefusalReason:
        """Categorize refusal based on content."""
        
        lower_text = refusal_text.lower()
        
        for category, patterns in self.REFUSAL_PATTERNS.items():
            if any(pattern in lower_text for pattern in patterns):
                return category
        
        return RefusalReason.SAFETY
    
    def _get_user_message(self, category: RefusalReason) -> str:
        """Get user-friendly message for category."""
        
        messages = {
            RefusalReason.HARMFUL_CONTENT: "This request may involve harmful content.",
            RefusalReason.PRIVACY: "This request involves private information.",
            RefusalReason.ILLEGAL: "This request cannot be fulfilled.",
            RefusalReason.DECEPTIVE: "This request may create misleading content.",
            RefusalReason.SAFETY: "This request was declined for safety reasons.",
            RefusalReason.POLICY: "This request conflicts with usage policies."
        }
        return messages.get(category, "Request could not be completed.")


# Test detection
print("\n\nRefusal Detection")
print("=" * 60)

detector = RefusalDetector()

# Test cases
responses = [
    MockParsedResponse(
        parsed=ExtractedPerson(name="Alice", age=28, occupation="Engineer"),
        refusal=None
    ),
    MockParsedResponse(
        parsed=None,
        refusal="I can't extract personal information without consent."
    ),
    MockParsedResponse(
        parsed=None,
        refusal="This request involves potentially harmful content."
    )
]

for i, response in enumerate(responses):
    print(f"\nResponse {i + 1}:")
    
    result = detector.detect(response)
    
    if result is None:
        print(f"   ✅ Normal response: {response.parsed}")
    else:
        print(f"   🚫 Refusal detected")
        print(f"   Category: {result.reason_category.value}")
        print(f"   Message: {result.user_friendly_message}")
```

---

## Handling Refusals in Code

### Conditional Logic Pattern

```python
from typing import Union


def process_extraction(
    response: MockParsedResponse
) -> Union[dict, str]:
    """Process extraction with refusal handling."""
    
    # Check for refusal first
    if response.refusal:
        return handle_refusal(response.refusal)
    
    # Process normal response
    if response.parsed:
        return process_success(response.parsed)
    
    # Unexpected state
    return "Error: No content or refusal in response"


def handle_refusal(refusal_message: str) -> str:
    """Handle a refused request."""
    
    # Log for monitoring
    log_refusal(refusal_message)
    
    # Return user-friendly message
    detector = RefusalDetector()
    
    # Create a mock response for categorization
    mock = MockParsedResponse(parsed=None, refusal=refusal_message)
    info = detector.detect(mock)
    
    if info:
        return info.user_friendly_message
    return "Your request could not be completed."


def process_success(data: ExtractedPerson) -> dict:
    """Process successful extraction."""
    
    return {
        "status": "success",
        "data": {
            "name": data.name,
            "age": data.age,
            "occupation": data.occupation
        }
    }


def log_refusal(message: str):
    """Log refusal for monitoring."""
    print(f"[LOG] Refusal: {message}")


# Example usage
print("\n\nConditional Refusal Handling")
print("=" * 60)

# Success case
success_response = MockParsedResponse(
    parsed=ExtractedPerson(name="Bob", age=35, occupation="Designer"),
    refusal=None
)
result = process_extraction(success_response)
print(f"\n✅ Success result: {result}")

# Refusal case
refusal_response = MockParsedResponse(
    parsed=None,
    refusal="I can't help extract private medical records."
)
result = process_extraction(refusal_response)
print(f"\n🚫 Refusal result: {result}")
```

### Exception-Based Pattern

```python
class ExtractionRefusedError(Exception):
    """Raised when extraction is refused."""
    
    def __init__(
        self,
        refusal_message: str,
        category: RefusalReason,
        can_retry: bool
    ):
        self.refusal_message = refusal_message
        self.category = category
        self.can_retry = can_retry
        super().__init__(refusal_message)


class SafeExtractor:
    """Extractor with exception-based refusal handling."""
    
    def __init__(self):
        self.detector = RefusalDetector()
    
    def extract(
        self,
        response: MockParsedResponse
    ) -> ExtractedPerson:
        """Extract data or raise on refusal."""
        
        # Check for refusal
        if response.refusal:
            info = self.detector.detect(response)
            
            raise ExtractionRefusedError(
                refusal_message=response.refusal,
                category=info.reason_category if info else RefusalReason.SAFETY,
                can_retry=info.can_retry_modified if info else True
            )
        
        if response.parsed:
            return response.parsed
        
        raise ValueError("Response has no content")
    
    def extract_safe(
        self,
        response: MockParsedResponse
    ) -> Optional[ExtractedPerson]:
        """Extract with None return on refusal."""
        
        try:
            return self.extract(response)
        except ExtractionRefusedError as e:
            print(f"Extraction refused: {e.refusal_message}")
            return None


# Test exception pattern
print("\n\nException-Based Pattern")
print("=" * 60)

extractor = SafeExtractor()

# Test with refusal
try:
    result = extractor.extract(MockParsedResponse(
        parsed=None,
        refusal="This involves illegal activity."
    ))
except ExtractionRefusedError as e:
    print(f"\n🚫 Caught refusal exception")
    print(f"   Category: {e.category.value}")
    print(f"   Can retry: {e.can_retry}")

# Test safe extraction
result = extractor.extract_safe(MockParsedResponse(
    parsed=None,
    refusal="I can't help with this request."
))
print(f"\n📋 Safe extraction result: {result}")
```

---

## User-Friendly Refusal Display

### Refusal Message Templates

```python
@dataclass
class UserFacingRefusal:
    """User-friendly refusal presentation."""
    
    title: str
    message: str
    suggestions: list[str]
    show_retry: bool
    icon: str = "⚠️"


class RefusalPresenter:
    """Present refusals in user-friendly way."""
    
    TEMPLATES = {
        RefusalReason.HARMFUL_CONTENT: UserFacingRefusal(
            title="Content Issue",
            message="We couldn't process this request because it may involve content that could be harmful.",
            suggestions=[
                "Review and modify the content in your request",
                "Ensure the content doesn't promote harm",
                "Contact support if you believe this is an error"
            ],
            show_retry=True,
            icon="🚫"
        ),
        RefusalReason.PRIVACY: UserFacingRefusal(
            title="Privacy Protection",
            message="This request involves personal or private information that requires proper authorization.",
            suggestions=[
                "Ensure you have consent to process this data",
                "Remove or anonymize personal identifiers",
                "Use aggregate data instead of individual records"
            ],
            show_retry=True,
            icon="🔒"
        ),
        RefusalReason.ILLEGAL: UserFacingRefusal(
            title="Request Unavailable",
            message="We cannot assist with this request.",
            suggestions=[
                "Review our terms of service",
                "Contact support with questions"
            ],
            show_retry=False,
            icon="⛔"
        ),
        RefusalReason.SAFETY: UserFacingRefusal(
            title="Safety Notice",
            message="This request was declined for safety reasons.",
            suggestions=[
                "Try rephrasing your request",
                "Provide more context about your use case",
                "Contact support for assistance"
            ],
            show_retry=True,
            icon="⚠️"
        )
    }
    
    def present(self, refusal_info: RefusalInfo) -> UserFacingRefusal:
        """Get user-facing presentation for refusal."""
        
        template = self.TEMPLATES.get(
            refusal_info.reason_category,
            self.TEMPLATES[RefusalReason.SAFETY]
        )
        return template
    
    def render_html(self, refusal: UserFacingRefusal) -> str:
        """Render refusal as HTML."""
        
        suggestions_html = "".join(
            f"<li>{s}</li>" for s in refusal.suggestions
        )
        
        retry_button = ""
        if refusal.show_retry:
            retry_button = '<button class="retry-btn">Try Again</button>'
        
        return f"""
        <div class="refusal-notice">
            <span class="icon">{refusal.icon}</span>
            <h3>{refusal.title}</h3>
            <p>{refusal.message}</p>
            <h4>What you can do:</h4>
            <ul>{suggestions_html}</ul>
            {retry_button}
        </div>
        """
    
    def render_console(self, refusal: UserFacingRefusal) -> str:
        """Render refusal for console output."""
        
        lines = [
            f"\n{refusal.icon} {refusal.title}",
            "=" * 40,
            refusal.message,
            "",
            "What you can do:"
        ]
        
        for i, suggestion in enumerate(refusal.suggestions, 1):
            lines.append(f"  {i}. {suggestion}")
        
        if refusal.show_retry:
            lines.append("\n[Retry available]")
        
        return "\n".join(lines)


# Test presentation
print("\n\nUser-Friendly Refusal Display")
print("=" * 60)

presenter = RefusalPresenter()

# Privacy refusal
privacy_info = RefusalInfo(
    reason_category=RefusalReason.PRIVACY,
    model_message="I can't extract personal data.",
    user_friendly_message="Privacy issue",
    can_retry_modified=True
)

presentation = presenter.present(privacy_info)
print(presenter.render_console(presentation))
```

---

## Refusal-Resilient Applications

### Application Architecture

```python
from typing import Callable, TypeVar

T = TypeVar('T')


@dataclass
class ExtractionResult:
    """Result of extraction attempt."""
    
    success: bool
    data: Optional[dict] = None
    refusal: Optional[UserFacingRefusal] = None
    error: Optional[str] = None


class ResilientExtractor:
    """Application-level extractor with full refusal handling."""
    
    def __init__(
        self,
        max_retries: int = 2,
        on_refusal: Optional[Callable[[RefusalInfo], None]] = None
    ):
        self.max_retries = max_retries
        self.on_refusal = on_refusal
        self.detector = RefusalDetector()
        self.presenter = RefusalPresenter()
    
    def extract(
        self,
        prompt: str,
        schema: type,
        context: Optional[str] = None
    ) -> ExtractionResult:
        """Extract with comprehensive refusal handling."""
        
        for attempt in range(self.max_retries + 1):
            response = self._call_api(prompt, schema)
            
            if response.parsed:
                return ExtractionResult(
                    success=True,
                    data=response.parsed.model_dump()
                )
            
            if response.refusal:
                info = self.detector.detect(response)
                
                # Notify callback
                if self.on_refusal and info:
                    self.on_refusal(info)
                
                # Check if retry is possible
                if info and info.can_retry_modified and attempt < self.max_retries:
                    prompt = self._modify_prompt(prompt)
                    continue
                
                # Return refusal result
                presentation = self.presenter.present(info) if info else None
                
                return ExtractionResult(
                    success=False,
                    refusal=presentation
                )
        
        return ExtractionResult(
            success=False,
            error="Max retries exceeded"
        )
    
    def _call_api(
        self,
        prompt: str,
        schema: type
    ) -> MockParsedResponse:
        """Simulate API call."""
        
        # Simulate refusal for certain content
        if "harmful" in prompt.lower():
            return MockParsedResponse(
                parsed=None,
                refusal="This request involves potentially harmful content."
            )
        
        # Simulate success
        return MockParsedResponse(
            parsed=schema(name="Test", age=25, occupation="Tester"),
            refusal=None
        )
    
    def _modify_prompt(self, prompt: str) -> str:
        """Modify prompt for retry."""
        
        return prompt + " (Please focus on appropriate content only.)"


# Test resilient extractor
print("\n\nResilient Extractor")
print("=" * 60)

def on_refusal(info: RefusalInfo):
    print(f"[Callback] Refusal detected: {info.reason_category.value}")


extractor = ResilientExtractor(
    max_retries=2,
    on_refusal=on_refusal
)

# Test success
result = extractor.extract(
    prompt="Extract person info",
    schema=ExtractedPerson
)
print(f"\n✅ Success: {result.data}")

# Test refusal
result = extractor.extract(
    prompt="Extract harmful content info",
    schema=ExtractedPerson
)
print(f"\n🚫 Refusal:")
if result.refusal:
    print(f"   Title: {result.refusal.title}")
    print(f"   Can retry: {result.refusal.show_retry}")
```

---

## Hands-on Exercise

### Your Task

Build a complete refusal handling system with detection, categorization, user messaging, and analytics.

### Requirements

1. Detect refusals in responses
2. Categorize by reason type
3. Present user-friendly messages
4. Track refusal statistics

<details>
<summary>💡 Hints</summary>

- Use pattern matching for categorization
- Maintain counters for analytics
- Create templated messages per category
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from datetime import datetime
import json


class RefusalCategory(Enum):
    """Refusal reason categories."""
    
    HARMFUL = "harmful_content"
    PRIVACY = "privacy_violation"
    ILLEGAL = "illegal_request"
    DECEPTIVE = "deceptive_content"
    POLICY = "policy_violation"
    UNKNOWN = "unknown"


@dataclass
class RefusalEvent:
    """A recorded refusal event."""
    
    timestamp: datetime
    category: RefusalCategory
    raw_message: str
    prompt_hash: str
    user_id: Optional[str] = None


@dataclass
class RefusalStats:
    """Statistics about refusals."""
    
    total_requests: int
    total_refusals: int
    refusals_by_category: Dict[str, int]
    refusal_rate: float
    most_common_category: Optional[RefusalCategory]


class RefusalHandlingSystem:
    """Complete refusal handling with analytics."""
    
    CATEGORY_PATTERNS = {
        RefusalCategory.HARMFUL: [
            "harmful", "dangerous", "violence", "weapon",
            "hurt", "damage", "attack"
        ],
        RefusalCategory.PRIVACY: [
            "personal information", "private", "consent",
            "confidential", "sensitive data", "pii"
        ],
        RefusalCategory.ILLEGAL: [
            "illegal", "unlawful", "criminal", "prohibited",
            "against the law"
        ],
        RefusalCategory.DECEPTIVE: [
            "deceptive", "misleading", "fake", "fraud",
            "impersonate", "scam"
        ],
        RefusalCategory.POLICY: [
            "policy", "guidelines", "terms of service",
            "not allowed", "prohibited"
        ]
    }
    
    MESSAGE_TEMPLATES = {
        RefusalCategory.HARMFUL: {
            "title": "Content Safety Notice",
            "message": "Your request may involve content that could be harmful. Please modify your request to focus on safe, constructive content.",
            "suggestions": [
                "Remove any references to harmful activities",
                "Focus on positive, constructive outcomes",
                "Rephrase using neutral language"
            ],
            "icon": "🚫",
            "can_retry": True
        },
        RefusalCategory.PRIVACY: {
            "title": "Privacy Protection",
            "message": "This request involves personal information that requires proper authorization.",
            "suggestions": [
                "Ensure you have consent to process this data",
                "Anonymize personal identifiers",
                "Use aggregate data instead"
            ],
            "icon": "🔒",
            "can_retry": True
        },
        RefusalCategory.ILLEGAL: {
            "title": "Request Unavailable",
            "message": "We cannot assist with this type of request.",
            "suggestions": [
                "Review our terms of service",
                "Contact support if you have questions"
            ],
            "icon": "⛔",
            "can_retry": False
        },
        RefusalCategory.DECEPTIVE: {
            "title": "Content Authenticity",
            "message": "This request may create misleading content.",
            "suggestions": [
                "Ensure transparency about AI-generated content",
                "Avoid impersonation or fraud",
                "Be clear about the purpose"
            ],
            "icon": "⚠️",
            "can_retry": True
        },
        RefusalCategory.POLICY: {
            "title": "Policy Notice",
            "message": "This request conflicts with our usage policies.",
            "suggestions": [
                "Review our acceptable use policy",
                "Modify your request to comply",
                "Contact support for clarification"
            ],
            "icon": "📋",
            "can_retry": True
        },
        RefusalCategory.UNKNOWN: {
            "title": "Request Issue",
            "message": "Your request could not be processed.",
            "suggestions": [
                "Try rephrasing your request",
                "Provide more context",
                "Contact support for help"
            ],
            "icon": "❓",
            "can_retry": True
        }
    }
    
    def __init__(self):
        self.events: List[RefusalEvent] = []
        self.request_count = 0
        self.on_refusal: Optional[Callable[[RefusalEvent], None]] = None
    
    def handle_response(
        self,
        response: MockParsedResponse,
        prompt: str,
        user_id: Optional[str] = None
    ) -> ExtractionResult:
        """Handle a response, detecting and processing refusals."""
        
        self.request_count += 1
        
        if response.parsed:
            return ExtractionResult(
                success=True,
                data=response.parsed.model_dump()
            )
        
        if response.refusal:
            return self._handle_refusal(
                response.refusal,
                prompt,
                user_id
            )
        
        return ExtractionResult(
            success=False,
            error="Empty response"
        )
    
    def _handle_refusal(
        self,
        refusal_message: str,
        prompt: str,
        user_id: Optional[str]
    ) -> ExtractionResult:
        """Process a refusal."""
        
        # Categorize
        category = self._categorize(refusal_message)
        
        # Record event
        event = RefusalEvent(
            timestamp=datetime.now(),
            category=category,
            raw_message=refusal_message,
            prompt_hash=str(hash(prompt)),
            user_id=user_id
        )
        self.events.append(event)
        
        # Notify callback
        if self.on_refusal:
            self.on_refusal(event)
        
        # Get user-facing content
        template = self.MESSAGE_TEMPLATES[category]
        
        presentation = UserFacingRefusal(
            title=template["title"],
            message=template["message"],
            suggestions=template["suggestions"],
            show_retry=template["can_retry"],
            icon=template["icon"]
        )
        
        return ExtractionResult(
            success=False,
            refusal=presentation
        )
    
    def _categorize(self, message: str) -> RefusalCategory:
        """Categorize refusal by pattern matching."""
        
        lower_message = message.lower()
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            if any(pattern in lower_message for pattern in patterns):
                return category
        
        return RefusalCategory.UNKNOWN
    
    def get_stats(self) -> RefusalStats:
        """Get refusal statistics."""
        
        by_category = {}
        for event in self.events:
            key = event.category.value
            by_category[key] = by_category.get(key, 0) + 1
        
        refusal_rate = (
            len(self.events) / self.request_count
            if self.request_count > 0 else 0
        )
        
        most_common = None
        if by_category:
            most_common_key = max(by_category, key=by_category.get)
            most_common = RefusalCategory(most_common_key)
        
        return RefusalStats(
            total_requests=self.request_count,
            total_refusals=len(self.events),
            refusals_by_category=by_category,
            refusal_rate=refusal_rate,
            most_common_category=most_common
        )
    
    def get_user_stats(self, user_id: str) -> dict:
        """Get stats for specific user."""
        
        user_events = [e for e in self.events if e.user_id == user_id]
        
        return {
            "user_id": user_id,
            "refusal_count": len(user_events),
            "categories": [e.category.value for e in user_events]
        }
    
    def render_stats(self) -> str:
        """Render stats as formatted string."""
        
        stats = self.get_stats()
        
        lines = [
            "\n📊 Refusal Statistics",
            "=" * 40,
            f"Total Requests: {stats.total_requests}",
            f"Total Refusals: {stats.total_refusals}",
            f"Refusal Rate: {stats.refusal_rate:.1%}",
            "",
            "By Category:"
        ]
        
        for category, count in stats.refusals_by_category.items():
            lines.append(f"  • {category}: {count}")
        
        if stats.most_common_category:
            lines.append(f"\nMost Common: {stats.most_common_category.value}")
        
        return "\n".join(lines)


# Test the complete system
print("\nComplete Refusal Handling System")
print("=" * 60)

system = RefusalHandlingSystem()

# Set up callback
system.on_refusal = lambda e: print(f"[Event] Refusal: {e.category.value}")

# Simulate various requests
test_cases = [
    ("Extract user info", ExtractedPerson(name="Test", age=30, occupation="Dev")),
    ("harmful content request", None),
    ("Get private data without consent", None),
    ("Normal request", ExtractedPerson(name="User", age=25, occupation="Designer")),
    ("Illegal activity help", None),
]

for prompt, parsed in test_cases:
    refusal = None
    if parsed is None:
        if "harmful" in prompt.lower():
            refusal = "This involves potentially harmful content."
        elif "private" in prompt.lower() or "consent" in prompt.lower():
            refusal = "Can't extract personal information without consent."
        elif "illegal" in prompt.lower():
            refusal = "Can't help with illegal activities."
    
    response = MockParsedResponse(parsed=parsed, refusal=refusal)
    result = system.handle_response(response, prompt, user_id="user123")
    
    status = "✅" if result.success else "🚫"
    print(f"\n{status} {prompt[:30]}...")

# Print stats
print(system.render_stats())

# User-specific stats
user_stats = system.get_user_stats("user123")
print(f"\n👤 User Stats: {user_stats}")
```

</details>

---

## Summary

✅ Refusals include a `refusal` field instead of content when safety policies apply  
✅ Always check `response.refusal` before accessing parsed data  
✅ Categorize refusals to provide appropriate user feedback  
✅ Some refusals can be retried with modified prompts  
✅ Track refusal analytics to improve your application

**Next:** [Use Cases & Patterns](./09-use-cases-patterns.md)

---

## Further Reading

- [OpenAI Refusals](https://platform.openai.com/docs/guides/structured-outputs#refusals) — Official refusal handling
- [Safety Best Practices](https://platform.openai.com/docs/guides/safety-best-practices) — Content safety
- [Error Handling](https://platform.openai.com/docs/guides/error-codes) — API error codes
