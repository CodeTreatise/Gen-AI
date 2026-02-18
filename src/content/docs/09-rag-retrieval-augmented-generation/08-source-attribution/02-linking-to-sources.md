---
title: "Linking to Source Documents"
---

# Linking to Source Documents

## Introduction

A citation is only as useful as the ability to verify it. Deep linking takes users directly to the relevant section, page, or paragraph—not just to a document's homepage. This transforms citations from references into verifiable proof.

This lesson covers techniques for creating precise, actionable links to source content.

### What We'll Cover

- Deep linking to specific sections
- Page number references
- Text highlighting
- One-click verification
- Link generation strategies

### Prerequisites

- Citation formats
- Understanding of document structure
- URL construction basics

---

## Deep Linking to Sections

### Section-Based Links

```python
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode, quote

@dataclass
class SourceLocation:
    document_id: str
    document_title: str
    url: str
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    paragraph_index: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

def build_deep_link(
    location: SourceLocation
) -> str:
    """
    Build a deep link to specific location in document.
    """
    base_url = location.url
    
    # Add section anchor
    if location.section_id:
        base_url += f"#{location.section_id}"
    
    # Add page reference (for PDF viewers)
    elif location.page_number:
        base_url += f"#page={location.page_number}"
    
    return base_url

def build_highlighted_link(
    location: SourceLocation,
    highlight_text: str
) -> str:
    """
    Build a link that highlights specific text.
    Uses Chrome's Text Fragments feature.
    """
    base_url = build_deep_link(location)
    
    # Text Fragment syntax: #:~:text=highlighted%20text
    encoded_text = quote(highlight_text)
    
    if "#" in base_url:
        return f"{base_url}&:~:text={encoded_text}"
    else:
        return f"{base_url}#:~:text={encoded_text}"

# Examples
location = SourceLocation(
    document_id="doc_123",
    document_title="Python 3.12 Release Notes",
    url="https://docs.python.org/3.12/whatsnew/3.12.html",
    section_id="improved-error-messages"
)

print(build_deep_link(location))
# https://docs.python.org/3.12/whatsnew/3.12.html#improved-error-messages

print(build_highlighted_link(location, "better error messages"))
# https://docs.python.org/3.12/whatsnew/3.12.html#improved-error-messages&:~:text=better%20error%20messages
```

### URL Fragment Types

| Fragment Type | Syntax | Use Case |
|---------------|--------|----------|
| **Section ID** | `#section-name` | HTML headings with IDs |
| **Page Number** | `#page=5` | PDF documents |
| **Named Destination** | `#nameddest=intro` | PDF bookmarks |
| **Text Fragment** | `#:~:text=word` | Highlight specific text |
| **Line Number** | `#L45` | Code files (GitHub) |
| **Time Stamp** | `#t=120` | Video/audio content |

---

## Page Number References

### PDF Deep Linking

```python
def build_pdf_link(
    pdf_url: str,
    page: Optional[int] = None,
    search_text: Optional[str] = None,
    zoom: Optional[int] = None
) -> str:
    """
    Build a deep link to a specific PDF location.
    """
    fragments = []
    
    if page:
        fragments.append(f"page={page}")
    
    if search_text:
        # PDF search parameter
        fragments.append(f"search={quote(search_text)}")
    
    if zoom:
        fragments.append(f"zoom={zoom}")
    
    if fragments:
        return f"{pdf_url}#{','.join(fragments)}"
    
    return pdf_url

# Examples
pdf_url = "https://example.com/docs/manual.pdf"

# Link to page 45
print(build_pdf_link(pdf_url, page=45))
# https://example.com/docs/manual.pdf#page=45

# Link with search highlight
print(build_pdf_link(pdf_url, page=45, search_text="configuration"))
# https://example.com/docs/manual.pdf#page=45,search=configuration
```

### Storing Page References

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class DocumentChunk:
    """
    Store chunk with precise location info.
    """
    content: str
    document_id: str
    document_title: str
    
    # Location info
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    paragraph_index: Optional[int] = None
    
    # Character positions (for highlighting)
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    # Pre-computed link
    deep_link: Optional[str] = None

def create_page_reference(
    chunk: DocumentChunk
) -> str:
    """
    Create human-readable page reference.
    """
    parts = [chunk.document_title]
    
    if chunk.section_title:
        parts.append(f"§ {chunk.section_title}")
    
    if chunk.page_number:
        parts.append(f"p. {chunk.page_number}")
    
    if chunk.paragraph_index:
        parts.append(f"¶{chunk.paragraph_index}")
    
    return ", ".join(parts)

# Example
chunk = DocumentChunk(
    content="Configuration requires setting the API key...",
    document_id="manual_v2",
    document_title="User Manual v2.0",
    page_number=45,
    section_title="Configuration",
    paragraph_index=3
)

print(create_page_reference(chunk))
# User Manual v2.0, § Configuration, p. 45, ¶3
```

---

## Text Highlighting

### Chrome Text Fragments

Modern browsers support the [Text Fragments](https://web.dev/text-fragments/) specification for highlighting specific text.

```python
from urllib.parse import quote

def create_text_fragment_url(
    base_url: str,
    text: str,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None
) -> str:
    """
    Create URL with Text Fragment for highlighting.
    
    Syntax: #:~:text=[prefix-,]textStart[,textEnd][,-suffix]
    """
    fragment_parts = []
    
    # Add prefix for disambiguation
    if prefix:
        fragment_parts.append(f"{quote(prefix)}-")
    
    # Add the main text to highlight
    fragment_parts.append(quote(text))
    
    # Add suffix for disambiguation
    if suffix:
        fragment_parts.append(f"-,{quote(suffix)}")
    
    fragment = "".join(fragment_parts)
    
    # Combine with base URL
    if "#" in base_url:
        return f"{base_url}&:~:text={fragment}"
    else:
        return f"{base_url}#:~:text={fragment}"

def create_multi_highlight_url(
    base_url: str,
    texts: list[str]
) -> str:
    """
    Highlight multiple text fragments.
    """
    fragments = [f"text={quote(t)}" for t in texts]
    combined = "&".join(fragments)
    
    if "#" in base_url:
        return f"{base_url}&:~:{combined}"
    else:
        return f"{base_url}#:~:{combined}"

# Examples
url = "https://docs.python.org/3.12/whatsnew/3.12.html"

# Simple highlight
print(create_text_fragment_url(url, "improved error messages"))

# Highlight with context for disambiguation
print(create_text_fragment_url(
    url,
    "error messages",
    prefix="Python 3.12 introduces",
    suffix="that include"
))

# Multiple highlights
print(create_multi_highlight_url(url, [
    "error messages",
    "performance improvements"
]))
```

### Fallback for Unsupported Browsers

```python
def create_highlighted_link_with_fallback(
    base_url: str,
    highlight_text: str,
    chunk_content: str
) -> dict:
    """
    Create link with fallback for browsers without Text Fragment support.
    """
    # Primary: Text Fragment URL
    primary_url = create_text_fragment_url(base_url, highlight_text)
    
    # Fallback: Store text for manual search
    return {
        "url": primary_url,
        "highlight_text": highlight_text,
        "context": chunk_content[:200],  # For manual finding
        "fallback_instruction": f'Search for: "{highlight_text}"'
    }
```

---

## One-Click Verification

### Verification Link Builder

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class VerificationLink:
    """
    Complete verification link with metadata.
    """
    url: str
    display_text: str
    document_title: str
    location_description: str
    highlight_text: Optional[str] = None
    
    def to_markdown(self) -> str:
        return f"[{self.display_text}]({self.url})"
    
    def to_html(self) -> str:
        return (
            f'<a href="{self.url}" '
            f'target="_blank" '
            f'title="Verify in: {self.document_title}, {self.location_description}">'
            f'{self.display_text}</a>'
        )

class VerificationLinkBuilder:
    """
    Build one-click verification links.
    """
    
    def __init__(self, base_viewer_url: Optional[str] = None):
        self.viewer_url = base_viewer_url
    
    def build_link(
        self,
        source: dict,
        quoted_text: str
    ) -> VerificationLink:
        """
        Build verification link for a citation.
        """
        url = source.get("url", "")
        
        # Determine link type based on URL
        if url.endswith(".pdf"):
            final_url = self._build_pdf_link(source, quoted_text)
        else:
            final_url = self._build_web_link(source, quoted_text)
        
        # Build location description
        location_parts = []
        if source.get("section"):
            location_parts.append(f"Section: {source['section']}")
        if source.get("page"):
            location_parts.append(f"Page {source['page']}")
        
        location = ", ".join(location_parts) or "Document"
        
        return VerificationLink(
            url=final_url,
            display_text=f"[{source.get('citation_num', '?')}]",
            document_title=source.get("title", "Unknown"),
            location_description=location,
            highlight_text=quoted_text
        )
    
    def _build_pdf_link(self, source: dict, text: str) -> str:
        url = source["url"]
        params = []
        
        if source.get("page"):
            params.append(f"page={source['page']}")
        
        if text:
            params.append(f"search={quote(text[:50])}")
        
        if params:
            return f"{url}#{','.join(params)}"
        return url
    
    def _build_web_link(self, source: dict, text: str) -> str:
        url = source["url"]
        
        # Add section anchor if available
        if source.get("section_id"):
            url = f"{url}#{source['section_id']}"
        
        # Add text fragment for highlighting
        if text:
            url = create_text_fragment_url(url, text)
        
        return url

# Usage
builder = VerificationLinkBuilder()

source = {
    "url": "https://docs.python.org/3.12/whatsnew/3.12.html",
    "title": "Python 3.12 Release Notes",
    "section": "Improved Error Messages",
    "section_id": "improved-error-messages",
    "citation_num": 1
}

link = builder.build_link(source, "better tracebacks")
print(link.to_markdown())
print(link.to_html())
```

---

## Complete Source Linking System

```python
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote, urljoin
from enum import Enum

class DocumentType(Enum):
    WEB = "web"
    PDF = "pdf"
    CODE = "code"
    VIDEO = "video"

@dataclass
class SourceReference:
    """
    Complete source reference with linking capability.
    """
    document_id: str
    document_title: str
    document_type: DocumentType
    base_url: str
    content: str
    
    # Location specifics
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    timestamp_seconds: Optional[int] = None
    
    # For highlighting
    highlight_text: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    # Metadata
    author: Optional[str] = None
    date: Optional[str] = None
    score: float = 0.0

class SourceLinkingSystem:
    """
    Generate deep links to source documents.
    """
    
    def generate_link(
        self,
        ref: SourceReference
    ) -> str:
        """
        Generate appropriate deep link based on document type.
        """
        generators = {
            DocumentType.WEB: self._link_web,
            DocumentType.PDF: self._link_pdf,
            DocumentType.CODE: self._link_code,
            DocumentType.VIDEO: self._link_video,
        }
        
        generator = generators.get(ref.document_type, self._link_web)
        return generator(ref)
    
    def generate_citation(
        self,
        ref: SourceReference,
        citation_num: int
    ) -> dict:
        """
        Generate complete citation with link.
        """
        link = self.generate_link(ref)
        location = self._describe_location(ref)
        
        return {
            "number": citation_num,
            "title": ref.document_title,
            "url": link,
            "location": location,
            "display": f"[{citation_num}]",
            "markdown": f"[{citation_num}]({link})",
            "html": (
                f'<a href="{link}" target="_blank" '
                f'title="{ref.document_title} - {location}">'
                f'[{citation_num}]</a>'
            )
        }
    
    def _link_web(self, ref: SourceReference) -> str:
        url = ref.base_url
        
        # Add section anchor
        if ref.section_id:
            url += f"#{ref.section_id}"
        
        # Add text highlight
        if ref.highlight_text:
            url = self._add_text_fragment(url, ref.highlight_text)
        
        return url
    
    def _link_pdf(self, ref: SourceReference) -> str:
        url = ref.base_url
        fragments = []
        
        if ref.page_number:
            fragments.append(f"page={ref.page_number}")
        
        if ref.highlight_text:
            fragments.append(f"search={quote(ref.highlight_text[:50])}")
        
        if fragments:
            url += "#" + ",".join(fragments)
        
        return url
    
    def _link_code(self, ref: SourceReference) -> str:
        url = ref.base_url
        
        # GitHub-style line links
        if ref.line_start:
            if ref.line_end and ref.line_end != ref.line_start:
                url += f"#L{ref.line_start}-L{ref.line_end}"
            else:
                url += f"#L{ref.line_start}"
        
        return url
    
    def _link_video(self, ref: SourceReference) -> str:
        url = ref.base_url
        
        if ref.timestamp_seconds:
            # YouTube-style timestamp
            if "youtube" in url.lower():
                url += f"?t={ref.timestamp_seconds}"
            else:
                url += f"#t={ref.timestamp_seconds}"
        
        return url
    
    def _add_text_fragment(self, url: str, text: str) -> str:
        encoded = quote(text)
        if "#" in url:
            return f"{url}&:~:text={encoded}"
        else:
            return f"{url}#:~:text={encoded}"
    
    def _describe_location(self, ref: SourceReference) -> str:
        parts = []
        
        if ref.section_title:
            parts.append(f"§ {ref.section_title}")
        
        if ref.page_number:
            parts.append(f"p. {ref.page_number}")
        
        if ref.line_start:
            if ref.line_end:
                parts.append(f"lines {ref.line_start}-{ref.line_end}")
            else:
                parts.append(f"line {ref.line_start}")
        
        if ref.timestamp_seconds:
            minutes = ref.timestamp_seconds // 60
            seconds = ref.timestamp_seconds % 60
            parts.append(f"@ {minutes}:{seconds:02d}")
        
        return ", ".join(parts) or "Full document"

# Usage
system = SourceLinkingSystem()

# Web document
web_ref = SourceReference(
    document_id="py312",
    document_title="Python 3.12 Release Notes",
    document_type=DocumentType.WEB,
    base_url="https://docs.python.org/3.12/whatsnew/3.12.html",
    content="Python 3.12 introduces better error messages...",
    section_id="improved-error-messages",
    section_title="Improved Error Messages",
    highlight_text="better error messages"
)

citation = system.generate_citation(web_ref, 1)
print(citation["markdown"])

# PDF document
pdf_ref = SourceReference(
    document_id="manual",
    document_title="User Manual",
    document_type=DocumentType.PDF,
    base_url="https://example.com/manual.pdf",
    content="Configure the system by...",
    page_number=45,
    highlight_text="configuration"
)

citation = system.generate_citation(pdf_ref, 2)
print(citation["markdown"])

# Code file
code_ref = SourceReference(
    document_id="main_py",
    document_title="main.py",
    document_type=DocumentType.CODE,
    base_url="https://github.com/org/repo/blob/main/src/main.py",
    content="def process_data():",
    line_start=45,
    line_end=60
)

citation = system.generate_citation(code_ref, 3)
print(citation["markdown"])
```

---

## Hands-on Exercise

### Your Task

Build a `DeepLinkGenerator` that:
1. Handles multiple document types
2. Creates verification-ready links
3. Provides fallback for unsupported browsers
4. Generates user-friendly location descriptions

### Requirements

```python
class DeepLinkGenerator:
    def generate(
        self,
        source: dict,
        highlight: str = None
    ) -> dict:
        """
        Returns: {
            "url": str,
            "fallback_url": str,
            "location_text": str,
            "verify_instruction": str
        }
        """
        pass
```

<details>
<summary>💡 Hints</summary>

- Check document type from URL extension
- Use different fragment formats for PDF vs HTML
- Text fragments don't work everywhere—provide fallback
- Include search text for manual verification

</details>

<details>
<summary>✅ Solution</summary>

```python
from urllib.parse import quote
from typing import Optional

class DeepLinkGenerator:
    def __init__(self):
        self.pdf_extensions = ['.pdf']
        self.code_extensions = ['.py', '.js', '.ts', '.java', '.cpp']
    
    def generate(
        self,
        source: dict,
        highlight: str = None
    ) -> dict:
        """Generate deep link with fallback."""
        
        url = source.get("url", "")
        doc_type = self._detect_type(url)
        
        # Generate primary link
        if doc_type == "pdf":
            primary = self._pdf_link(source, highlight)
        elif doc_type == "code":
            primary = self._code_link(source)
        else:
            primary = self._web_link(source, highlight)
        
        # Generate fallback (no text fragments)
        fallback = self._fallback_link(source)
        
        # Location description
        location = self._location_text(source)
        
        # Verification instruction
        verify = self._verify_instruction(source, highlight)
        
        return {
            "url": primary,
            "fallback_url": fallback,
            "location_text": location,
            "verify_instruction": verify
        }
    
    def _detect_type(self, url: str) -> str:
        url_lower = url.lower()
        
        for ext in self.pdf_extensions:
            if ext in url_lower:
                return "pdf"
        
        for ext in self.code_extensions:
            if ext in url_lower:
                return "code"
        
        return "web"
    
    def _web_link(self, source: dict, highlight: str) -> str:
        url = source.get("url", "")
        
        # Add section anchor
        if source.get("section_id"):
            url += f"#{source['section_id']}"
        
        # Add text fragment
        if highlight:
            if "#" in url:
                url += f"&:~:text={quote(highlight)}"
            else:
                url += f"#:~:text={quote(highlight)}"
        
        return url
    
    def _pdf_link(self, source: dict, highlight: str) -> str:
        url = source.get("url", "")
        fragments = []
        
        if source.get("page"):
            fragments.append(f"page={source['page']}")
        
        if highlight:
            fragments.append(f"search={quote(highlight[:50])}")
        
        if fragments:
            url += "#" + ",".join(fragments)
        
        return url
    
    def _code_link(self, source: dict) -> str:
        url = source.get("url", "")
        
        if source.get("line_start"):
            if source.get("line_end"):
                url += f"#L{source['line_start']}-L{source['line_end']}"
            else:
                url += f"#L{source['line_start']}"
        
        return url
    
    def _fallback_link(self, source: dict) -> str:
        url = source.get("url", "")
        
        # Just section anchor, no text fragments
        if source.get("section_id"):
            url += f"#{source['section_id']}"
        elif source.get("page"):
            url += f"#page={source['page']}"
        
        return url
    
    def _location_text(self, source: dict) -> str:
        parts = [source.get("title", "Document")]
        
        if source.get("section_title"):
            parts.append(f"→ {source['section_title']}")
        
        if source.get("page"):
            parts.append(f"(p. {source['page']})")
        
        if source.get("line_start"):
            parts.append(f"(line {source['line_start']})")
        
        return " ".join(parts)
    
    def _verify_instruction(
        self,
        source: dict,
        highlight: str
    ) -> str:
        instructions = []
        
        instructions.append(f"Open: {source.get('title', 'the document')}")
        
        if source.get("section_title"):
            instructions.append(f"Go to: {source['section_title']}")
        elif source.get("page"):
            instructions.append(f"Go to page: {source['page']}")
        
        if highlight:
            instructions.append(f'Search for: "{highlight[:50]}"')
        
        return " → ".join(instructions)

# Test
generator = DeepLinkGenerator()

# Web document
web_result = generator.generate(
    {
        "url": "https://docs.python.org/3.12/whatsnew/3.12.html",
        "title": "Python 3.12 What's New",
        "section_id": "improved-error-messages",
        "section_title": "Improved Error Messages"
    },
    highlight="better tracebacks"
)
print(f"URL: {web_result['url']}")
print(f"Location: {web_result['location_text']}")
print(f"Verify: {web_result['verify_instruction']}")

# PDF document
pdf_result = generator.generate(
    {
        "url": "https://example.com/manual.pdf",
        "title": "User Manual",
        "page": 45
    },
    highlight="configuration"
)
print(f"\nPDF URL: {pdf_result['url']}")
print(f"Location: {pdf_result['location_text']}")
```

</details>

---

## Summary

Effective source linking enables verification:

✅ **Deep linking** — Direct links to sections, not just documents
✅ **Page numbers** — PDF navigation with page fragments
✅ **Text highlighting** — Chrome Text Fragments for emphasis
✅ **One-click verification** — Minimal friction to check sources
✅ **Fallbacks** — Graceful degradation for older browsers

**Next:** [Confidence Scoring](./03-confidence-scoring.md)

---

## Further Reading

- [Text Fragments Specification](https://web.dev/text-fragments/) - Chrome highlighting
- [PDF Open Parameters](https://helpx.adobe.com/acrobat/kb/link-html-pdf-page-acrobat.html) - Adobe PDF linking

<!--
Sources Consulted:
- Web.dev Text Fragments documentation
- Adobe PDF URL parameters
- GitHub line linking conventions
-->
