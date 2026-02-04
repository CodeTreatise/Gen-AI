---
title: "Custom Output Parsers"
---

# Custom Output Parsers

## Introduction

When built-in parsers don't meet your needs, you can create custom parsers by extending `BaseOutputParser`. This lesson covers building parsers for regex patterns, XML, markdown, and multi-stage processing—giving you complete control over how LLM output becomes structured data.

### What We'll Cover

- Implementing BaseOutputParser
- Regex-based parsing
- XML and markup parsing
- Markdown structure extraction
- Multi-stage parsing pipelines
- Combining parsers

### Prerequisites

- Parser Basics (Lesson 8.3.1)
- Python classes and inheritance
- Regular expressions (basic)

---

## The BaseOutputParser Interface

### Core Methods to Implement

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import TypeVar, Generic

T = TypeVar("T")

class BaseOutputParser(Generic[T]):
    """Base class for output parsers."""
    
    @property
    def OutputType(self) -> type[T]:
        """Return the output type."""
        ...
    
    def parse(self, text: str) -> T:
        """Parse text into the output type."""
        ...
    
    def get_format_instructions(self) -> str:
        """Return format instructions for the prompt."""
        ...
```

### Minimal Custom Parser

```python
from langchain_core.output_parsers import BaseOutputParser

class UppercaseParser(BaseOutputParser[str]):
    """Converts output to uppercase."""
    
    @property
    def _type(self) -> str:
        return "uppercase"
    
    def parse(self, text: str) -> str:
        return text.strip().upper()
    
    def get_format_instructions(self) -> str:
        return "Respond with any text. It will be converted to uppercase."

# Usage
parser = UppercaseParser()
result = parser.parse("Hello, World!")
print(result)  # HELLO, WORLD!
```

---

## Regex-Based Parsers

### Simple Pattern Extraction

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel
import re

class PhoneNumber(BaseModel):
    area_code: str
    number: str
    formatted: str

class PhoneParser(BaseOutputParser[PhoneNumber]):
    """Extract phone numbers from text."""
    
    pattern: str = r'\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})'
    
    @property
    def _type(self) -> str:
        return "phone"
    
    def parse(self, text: str) -> PhoneNumber:
        match = re.search(self.pattern, text)
        if not match:
            raise OutputParserException(
                f"Could not find phone number in: {text}"
            )
        
        area_code = match.group(1)
        rest = f"{match.group(2)}-{match.group(3)}"
        
        return PhoneNumber(
            area_code=area_code,
            number=rest,
            formatted=f"({area_code}) {rest}"
        )
    
    def get_format_instructions(self) -> str:
        return "Include a phone number in format: (XXX) XXX-XXXX"

# Usage
parser = PhoneParser()

result = parser.parse("Call me at (555) 123-4567 tomorrow")
print(result.formatted)  # (555) 123-4567

result = parser.parse("My number is 555.987.6543")
print(result.formatted)  # (555) 987-6543
```

### Multiple Pattern Extraction

```python
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel
import re
from typing import Optional

class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None

class ContactExtractor(BaseOutputParser[ContactInfo]):
    """Extract contact information using regex patterns."""
    
    email_pattern: str = r'[\w.+-]+@[\w-]+\.[\w.-]+'
    phone_pattern: str = r'\+?1?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})'
    url_pattern: str = r'https?://[^\s<>"{}|\\^`\[\]]+'
    
    @property
    def _type(self) -> str:
        return "contact"
    
    def parse(self, text: str) -> ContactInfo:
        # Extract email
        email_match = re.search(self.email_pattern, text)
        email = email_match.group(0) if email_match else None
        
        # Extract phone
        phone_match = re.search(self.phone_pattern, text)
        phone = None
        if phone_match:
            phone = f"({phone_match.group(1)}) {phone_match.group(2)}-{phone_match.group(3)}"
        
        # Extract URL
        url_match = re.search(self.url_pattern, text)
        website = url_match.group(0) if url_match else None
        
        return ContactInfo(email=email, phone=phone, website=website)
    
    def get_format_instructions(self) -> str:
        return """Include contact information:
- Email address
- Phone number (XXX-XXX-XXXX format)
- Website URL"""

# Usage
parser = ContactExtractor()

text = """
Contact us at support@example.com or call 555-123-4567.
Visit https://www.example.com for more info.
"""

result = parser.parse(text)
print(f"Email: {result.email}")    # support@example.com
print(f"Phone: {result.phone}")    # (555) 123-4567
print(f"Website: {result.website}") # https://www.example.com
```

---

## XML Parsers

### Basic XML Extraction

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel
import re
from typing import Optional

class XMLData(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    metadata: dict = {}

class SimpleXMLParser(BaseOutputParser[XMLData]):
    """Parse simple XML structure from LLM output."""
    
    @property
    def _type(self) -> str:
        return "xml"
    
    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        """Extract content from XML tag."""
        pattern = f'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def parse(self, text: str) -> XMLData:
        title = self._extract_tag(text, 'title')
        content = self._extract_tag(text, 'content')
        
        # Extract any metadata tags
        metadata = {}
        meta_pattern = r'<meta\s+name="([^"]+)"[^>]*>([^<]*)</meta>'
        for match in re.finditer(meta_pattern, text):
            metadata[match.group(1)] = match.group(2).strip()
        
        return XMLData(title=title, content=content, metadata=metadata)
    
    def get_format_instructions(self) -> str:
        return """Respond in XML format:
<title>Your title here</title>
<content>Main content here</content>
<meta name="key">value</meta>"""

# Usage
parser = SimpleXMLParser()

xml_text = """
<title>Introduction to AI</title>
<content>
Artificial intelligence is transforming how we work and live.
Machine learning enables computers to learn from data.
</content>
<meta name="author">John Doe</meta>
<meta name="category">Technology</meta>
"""

result = parser.parse(xml_text)
print(f"Title: {result.title}")
print(f"Content: {result.content[:50]}...")
print(f"Metadata: {result.metadata}")
```

### Using LangChain's XMLOutputParser

```python
from langchain_core.output_parsers import XMLOutputParser

# Built-in XML parser
parser = XMLOutputParser(tags=["name", "age", "occupation"])

xml_text = """<?xml version="1.0"?>
<root>
<name>Alice</name>
<age>30</age>
<occupation>Engineer</occupation>
</root>
"""

result = parser.parse(xml_text)
print(result)
# {'name': 'Alice', 'age': '30', 'occupation': 'Engineer'}
```

---

## Markdown Parsers

### Heading and Section Parser

```python
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel
import re

class Section(BaseModel):
    level: int
    title: str
    content: str

class MarkdownDocument(BaseModel):
    title: str
    sections: list[Section]

class MarkdownParser(BaseOutputParser[MarkdownDocument]):
    """Parse markdown into structured sections."""
    
    @property
    def _type(self) -> str:
        return "markdown"
    
    def parse(self, text: str) -> MarkdownDocument:
        lines = text.strip().split('\n')
        
        title = ""
        sections = []
        current_section = None
        content_lines = []
        
        for line in lines:
            # Check for heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(content_lines).strip()
                    sections.append(current_section)
                    content_lines = []
                
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2)
                
                if level == 1 and not title:
                    title = heading_text
                else:
                    current_section = Section(
                        level=level,
                        title=heading_text,
                        content=""
                    )
            else:
                content_lines.append(line)
        
        # Don't forget last section
        if current_section:
            current_section.content = '\n'.join(content_lines).strip()
            sections.append(current_section)
        
        return MarkdownDocument(title=title, sections=sections)
    
    def get_format_instructions(self) -> str:
        return """Format your response as markdown:
# Main Title

## Section 1
Content here...

## Section 2
More content..."""

# Usage
parser = MarkdownParser()

markdown = """# Python Guide

## Introduction
Python is a versatile programming language.

## Installation
Download Python from python.org and install it.

## First Steps
Create a file called hello.py with print("Hello!")
"""

doc = parser.parse(markdown)
print(f"Title: {doc.title}")
for section in doc.sections:
    print(f"  H{section.level}: {section.title}")
    print(f"    {section.content[:30]}...")
```

### Code Block Extractor

```python
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel
import re

class CodeBlock(BaseModel):
    language: str
    code: str

class CodeExtractor(BaseOutputParser[list[CodeBlock]]):
    """Extract code blocks from markdown."""
    
    @property
    def _type(self) -> str:
        return "code_blocks"
    
    def parse(self, text: str) -> list[CodeBlock]:
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        return [
            CodeBlock(
                language=lang or "text",
                code=code.strip()
            )
            for lang, code in matches
        ]
    
    def get_format_instructions(self) -> str:
        return """Include code in fenced blocks:
```python
your_code_here()
```"""

# Usage
parser = CodeExtractor()

text = """Here's how to do it:

```python
def greet(name):
    return f"Hello, {name}!"
```

And in JavaScript:

```javascript
const greet = (name) => `Hello, ${name}!`;
```
"""

blocks = parser.parse(text)
for block in blocks:
    print(f"Language: {block.language}")
    print(f"Code:\n{block.code}\n")
```

---

## Multi-Stage Parsing

### Pipeline Pattern

```python
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel
from typing import Any

class Stage1Result(BaseModel):
    raw_data: str
    extracted: list[str]

class Stage2Result(BaseModel):
    items: list[dict]
    count: int

class MultiStageParser(BaseOutputParser[Stage2Result]):
    """Two-stage parsing pipeline."""
    
    @property
    def _type(self) -> str:
        return "multi_stage"
    
    def _stage1_extract(self, text: str) -> Stage1Result:
        """First stage: extract items."""
        import re
        # Extract items in format: "- item"
        items = re.findall(r'^[-*]\s+(.+)$', text, re.MULTILINE)
        return Stage1Result(raw_data=text, extracted=items)
    
    def _stage2_structure(self, stage1: Stage1Result) -> Stage2Result:
        """Second stage: structure items."""
        items = []
        for idx, item in enumerate(stage1.extracted, 1):
            items.append({
                "id": idx,
                "text": item.strip(),
                "length": len(item)
            })
        return Stage2Result(items=items, count=len(items))
    
    def parse(self, text: str) -> Stage2Result:
        stage1 = self._stage1_extract(text)
        return self._stage2_structure(stage1)
    
    def get_format_instructions(self) -> str:
        return "List items with bullet points (- or *)"

# Usage
parser = MultiStageParser()

text = """Here are the items:
- First item here
- Second item is longer
- Third
- Fourth item in the list
"""

result = parser.parse(text)
print(f"Found {result.count} items:")
for item in result.items:
    print(f"  {item['id']}: {item['text']} ({item['length']} chars)")
```

### Transformer Pipeline

```python
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from pydantic import BaseModel
from typing import Callable

class TransformerParser(BaseOutputParser[str]):
    """Chain multiple transformations."""
    
    transformers: list[Callable[[str], str]] = []
    
    def __init__(self, transformers: list[Callable[[str], str]] = None):
        super().__init__()
        self.transformers = transformers or []
    
    @property
    def _type(self) -> str:
        return "transformer"
    
    def parse(self, text: str) -> str:
        result = text
        for transform in self.transformers:
            result = transform(result)
        return result
    
    def get_format_instructions(self) -> str:
        return "Respond with text."

# Define transformers
def strip_whitespace(text: str) -> str:
    return text.strip()

def remove_markdown(text: str) -> str:
    import re
    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    return text

def normalize_spaces(text: str) -> str:
    import re
    return re.sub(r'\s+', ' ', text)

# Create pipeline
parser = TransformerParser(transformers=[
    strip_whitespace,
    remove_markdown,
    normalize_spaces
])

text = """
# **Important** Announcement

This is *very* important news.
   Multiple   spaces   here.
"""

result = parser.parse(text)
print(result)  # "Important Announcement This is very important news. Multiple spaces here."
```

---

## Combining Parsers

### Parser Wrapper

```python
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel
from typing import Optional
import json
import re

class RobustJsonParser(BaseOutputParser[dict]):
    """Wrapper that handles JSON in various formats."""
    
    inner_parser: JsonOutputParser = None
    
    def __init__(self):
        super().__init__()
        self.inner_parser = JsonOutputParser()
    
    @property
    def _type(self) -> str:
        return "robust_json"
    
    def _extract_json(self, text: str) -> str:
        """Try to extract JSON from text."""
        # Try code block first
        code_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_block:
            return code_block.group(1)
        
        # Try to find JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json_match.group(0)
        
        # Try to find JSON array
        array_match = re.search(r'\[[\s\S]*\]', text)
        if array_match:
            return array_match.group(0)
        
        return text
    
    def parse(self, text: str) -> dict:
        extracted = self._extract_json(text)
        try:
            return self.inner_parser.parse(extracted)
        except OutputParserException:
            # Try to fix common issues
            fixed = extracted.replace("'", '"')
            return self.inner_parser.parse(fixed)
    
    def get_format_instructions(self) -> str:
        return "Respond with JSON (with or without code blocks)."

# Usage
parser = RobustJsonParser()

# Works with code blocks
text1 = """Here's the data:
```json
{"name": "Alice", "age": 30}
```
"""

# Works with raw JSON
text2 = '{"name": "Bob", "age": 25}'

# Works with embedded JSON
text3 = "The result is: {\"name\": \"Charlie\", \"age\": 35} as shown."

for text in [text1, text2, text3]:
    result = parser.parse(text)
    print(f"Parsed: {result}")
```

### Fallback Chain

```python
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException
from typing import Union

class FallbackParser(BaseOutputParser[Union[dict, str]]):
    """Try multiple parsers until one works."""
    
    parsers: list[BaseOutputParser] = []
    
    def __init__(self, parsers: list[BaseOutputParser] = None):
        super().__init__()
        self.parsers = parsers or [JsonOutputParser(), StrOutputParser()]
    
    @property
    def _type(self) -> str:
        return "fallback"
    
    def parse(self, text: str) -> Union[dict, str]:
        errors = []
        for parser in self.parsers:
            try:
                return parser.parse(text)
            except OutputParserException as e:
                errors.append(str(e))
                continue
        
        # All parsers failed - return string as last resort
        return text.strip()
    
    def get_format_instructions(self) -> str:
        return "Respond with JSON if possible, otherwise plain text."

# Usage
parser = FallbackParser()

# JSON works
print(parser.parse('{"key": "value"}'))  # {'key': 'value'}

# Falls back to string
print(parser.parse("Just some text"))  # "Just some text"
```

---

## Integration with LCEL

### Custom Parser in Chain

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
import re

class Rating(BaseModel):
    score: int
    explanation: str

class RatingParser(BaseOutputParser[Rating]):
    """Parse rating responses."""
    
    @property
    def _type(self) -> str:
        return "rating"
    
    def parse(self, text: str) -> Rating:
        # Look for score pattern
        score_match = re.search(r'(\d+)\s*/\s*10', text)
        if not score_match:
            score_match = re.search(r'score[:\s]+(\d+)', text, re.IGNORECASE)
        
        score = int(score_match.group(1)) if score_match else 5
        
        # Get explanation (text after score mention)
        explanation = text
        if score_match:
            explanation = text[score_match.end():].strip()
            if explanation.startswith(':'):
                explanation = explanation[1:].strip()
        
        return Rating(score=min(10, max(1, score)), explanation=explanation)
    
    def get_format_instructions(self) -> str:
        return "Give a rating as X/10 followed by your explanation."

# Use in chain
parser = RatingParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Rate the following. {format_instructions}"),
    ("human", "Rate this movie: {movie}")
])

model = init_chat_model("gpt-4o")

chain = (
    prompt.partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)

result = chain.invoke({"movie": "Inception"})
print(f"Score: {result.score}/10")
print(f"Why: {result.explanation}")
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Inherit from BaseOutputParser | Consistent interface |
| Implement get_format_instructions | Guides LLM output |
| Use OutputParserException | Standard error handling |
| Test with edge cases | Robust parsing |
| Keep parsers focused | Single responsibility |
| Document expected format | Clear contracts |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No error handling | Raise OutputParserException |
| Regex too strict | Allow format variations |
| Missing format instructions | Always implement |
| Overly complex single parser | Split into stages |
| Not testing edge cases | Test malformed input |

---

## Hands-on Exercise

### Your Task

Build a custom parser for extracting structured meeting notes:

1. Parse meeting metadata (date, attendees)
2. Extract action items with assignees
3. Identify decisions made
4. Handle various input formats

### Requirements

```python
result = parser.parse("""
Meeting: Product Review
Date: 2024-07-15
Attendees: Alice, Bob, Charlie

## Action Items
- [ ] Alice: Review PR #123
- [ ] Bob: Update documentation
- [x] Charlie: Deploy to staging (done)

## Decisions
1. Launch date set for August 1st
2. Budget approved for Q3
""")
```

### Expected Result

```python
print(result.title)  # "Product Review"
print(result.attendees)  # ["Alice", "Bob", "Charlie"]
print(result.action_items[0])  # ActionItem(assignee="Alice", task="Review PR #123", done=False)
print(result.decisions)  # ["Launch date set for August 1st", ...]
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use regex to extract date patterns
- Parse attendees as comma-separated list
- Check for `[x]` vs `[ ]` for completion status
- Look for numbered lists for decisions

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel
import re
from typing import Optional

class ActionItem(BaseModel):
    assignee: str
    task: str
    done: bool

class MeetingNotes(BaseModel):
    title: str
    date: Optional[str] = None
    attendees: list[str] = []
    action_items: list[ActionItem] = []
    decisions: list[str] = []

class MeetingNotesParser(BaseOutputParser[MeetingNotes]):
    """Parse meeting notes into structured data."""
    
    @property
    def _type(self) -> str:
        return "meeting_notes"
    
    def _extract_metadata(self, text: str) -> tuple[str, Optional[str], list[str]]:
        """Extract title, date, and attendees."""
        # Title
        title_match = re.search(r'Meeting:\s*(.+)', text)
        title = title_match.group(1).strip() if title_match else "Untitled Meeting"
        
        # Date
        date_match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', text)
        date = date_match.group(1) if date_match else None
        
        # Attendees
        attendees_match = re.search(r'Attendees:\s*(.+)', text)
        attendees = []
        if attendees_match:
            attendees = [a.strip() for a in attendees_match.group(1).split(',')]
        
        return title, date, attendees
    
    def _extract_action_items(self, text: str) -> list[ActionItem]:
        """Extract action items with assignees."""
        items = []
        # Pattern: - [ ] or - [x] followed by Assignee: Task
        pattern = r'-\s*\[([ x])\]\s*([^:]+):\s*(.+)'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            done = match.group(1).lower() == 'x'
            assignee = match.group(2).strip()
            task = match.group(3).strip()
            # Remove "(done)" if present
            task = re.sub(r'\s*\(done\)\s*$', '', task, flags=re.IGNORECASE)
            
            items.append(ActionItem(assignee=assignee, task=task, done=done))
        
        return items
    
    def _extract_decisions(self, text: str) -> list[str]:
        """Extract numbered decisions."""
        decisions = []
        
        # Find decisions section
        decisions_section = re.search(
            r'##\s*Decisions\s*([\s\S]*?)(?=##|$)', 
            text, 
            re.IGNORECASE
        )
        
        if decisions_section:
            section_text = decisions_section.group(1)
            # Extract numbered items
            for match in re.finditer(r'^\d+\.\s*(.+)$', section_text, re.MULTILINE):
                decisions.append(match.group(1).strip())
        
        return decisions
    
    def parse(self, text: str) -> MeetingNotes:
        title, date, attendees = self._extract_metadata(text)
        action_items = self._extract_action_items(text)
        decisions = self._extract_decisions(text)
        
        return MeetingNotes(
            title=title,
            date=date,
            attendees=attendees,
            action_items=action_items,
            decisions=decisions
        )
    
    def get_format_instructions(self) -> str:
        return """Format meeting notes as:
Meeting: [Title]
Date: YYYY-MM-DD
Attendees: Name1, Name2, Name3

## Action Items
- [ ] Assignee: Task description
- [x] Assignee: Completed task (done)

## Decisions
1. First decision
2. Second decision"""

# Test
parser = MeetingNotesParser()

text = """
Meeting: Product Review
Date: 2024-07-15
Attendees: Alice, Bob, Charlie

## Action Items
- [ ] Alice: Review PR #123
- [ ] Bob: Update documentation
- [x] Charlie: Deploy to staging (done)

## Decisions
1. Launch date set for August 1st
2. Budget approved for Q3
3. New feature prioritized for next sprint
"""

result = parser.parse(text)

print(f"Title: {result.title}")
print(f"Date: {result.date}")
print(f"Attendees: {result.attendees}")
print(f"\nAction Items:")
for item in result.action_items:
    status = "✅" if item.done else "⬜"
    print(f"  {status} {item.assignee}: {item.task}")
print(f"\nDecisions:")
for decision in result.decisions:
    print(f"  • {decision}")
```

</details>

### Bonus Challenges

- [ ] Add support for markdown-style headers
- [ ] Parse time ranges (10:00 AM - 11:00 AM)
- [ ] Extract mentioned deadlines
- [ ] Support nested action items

---

## Summary

✅ Extend `BaseOutputParser` for custom parsing logic  
✅ Implement `parse()` and `get_format_instructions()`  
✅ Use regex for pattern-based extraction  
✅ Build multi-stage pipelines for complex parsing  
✅ Combine parsers with wrappers and fallbacks  
✅ Integrate seamlessly with LCEL chains  

**Next:** [Output Fixing](./06-output-fixing.md) — Automatic error correction with OutputFixingParser

---

## Navigation

| Previous | Up | Next |
|----------|-----|------|
| [Structured Output](./04-structured-output.md) | [Output Parsing](./00-output-parsing.md) | [Output Fixing](./06-output-fixing.md) |

<!-- 
Sources Consulted:
- LangChain BaseOutputParser: https://github.com/langchain-ai/langchain/blob/main/libs/core/langchain_core/output_parsers/base.py
- LangChain XMLOutputParser: https://github.com/langchain-ai/langchain/blob/main/libs/core/langchain_core/output_parsers/xml.py
- Python re module: https://docs.python.org/3/library/re.html
-->
