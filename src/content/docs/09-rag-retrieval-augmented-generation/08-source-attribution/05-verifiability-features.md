---
title: "Verifiability Features"
---

# Verifiability Features

## Introduction

Source attribution is only valuable if users can verify the information. Verifiability features let users check claims against original sources, report issues, and build trust in AI-generated content.

This lesson covers building features that make verification easy and accessible.

### What We'll Cover

- Side-by-side source comparison
- Source preview panels
- Fact-check workflow implementation
- User reporting mechanisms

### Prerequisites

- Citation formatting patterns
- Source linking knowledge
- Basic UI/UX concepts

---

## Side-by-Side Source Comparison

Display AI responses alongside the original sources for easy verification.

### Comparison Data Structure

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class MatchType(Enum):
    EXACT = "exact"
    PARAPHRASE = "paraphrase"
    PARTIAL = "partial"
    INFERRED = "inferred"

@dataclass
class SourceSegment:
    """Original source text segment."""
    source_id: str
    source_title: str
    text: str
    location: Optional[str]  # Page, section, etc.
    url: Optional[str]

@dataclass
class ClaimSegment:
    """Claim from AI response."""
    text: str
    start: int
    end: int
    source_segments: list[SourceSegment]
    match_type: MatchType
    confidence: float

@dataclass
class ComparisonView:
    """Side-by-side comparison structure."""
    response_text: str
    claims: list[ClaimSegment]
    unsupported_text: list[str]
```

### Building Comparison Views

```python
def build_comparison_view(
    response: str,
    citations: list[dict],
    sources: list[dict]
) -> ComparisonView:
    """
    Build a side-by-side comparison view.
    """
    claims = []
    covered_ranges = []
    
    # Map source IDs to full source data
    source_map = {s["id"]: s for s in sources}
    
    for citation in citations:
        source = source_map.get(citation["source_id"])
        
        if source:
            segment = SourceSegment(
                source_id=source["id"],
                source_title=source["title"],
                text=source.get("matched_text", source["content"][:200]),
                location=source.get("location"),
                url=source.get("url")
            )
            
            claim = ClaimSegment(
                text=citation.get("cited_text", ""),
                start=citation["start"],
                end=citation["end"],
                source_segments=[segment],
                match_type=_determine_match_type(
                    citation.get("cited_text", ""),
                    segment.text
                ),
                confidence=citation.get("confidence", 0.7)
            )
            
            claims.append(claim)
            covered_ranges.append((citation["start"], citation["end"]))
    
    # Find unsupported text
    unsupported = _find_unsupported_text(response, covered_ranges)
    
    return ComparisonView(
        response_text=response,
        claims=claims,
        unsupported_text=unsupported
    )

def _determine_match_type(claim: str, source: str) -> MatchType:
    """Determine how closely claim matches source."""
    claim_lower = claim.lower().strip()
    source_lower = source.lower()
    
    if claim_lower in source_lower:
        return MatchType.EXACT
    
    # Check word overlap for paraphrase
    claim_words = set(claim_lower.split())
    source_words = set(source_lower.split())
    
    overlap = len(claim_words & source_words) / max(len(claim_words), 1)
    
    if overlap > 0.7:
        return MatchType.PARAPHRASE
    elif overlap > 0.4:
        return MatchType.PARTIAL
    else:
        return MatchType.INFERRED

def _find_unsupported_text(
    text: str,
    covered_ranges: list[tuple[int, int]]
) -> list[str]:
    """Find text not covered by any citation."""
    if not covered_ranges:
        return [text]
    
    # Sort ranges by start
    sorted_ranges = sorted(covered_ranges, key=lambda r: r[0])
    
    unsupported = []
    pos = 0
    
    for start, end in sorted_ranges:
        if start > pos:
            segment = text[pos:start].strip()
            if segment:
                unsupported.append(segment)
        pos = max(pos, end)
    
    # Check remaining text
    if pos < len(text):
        segment = text[pos:].strip()
        if segment:
            unsupported.append(segment)
    
    return unsupported
```

### Rendering Comparison UI

```python
def render_comparison_html(view: ComparisonView) -> str:
    """Render side-by-side comparison as HTML."""
    html = ['<div class="comparison-container">']
    
    # Response column
    html.append('<div class="response-column">')
    html.append('<h3>AI Response</h3>')
    html.append(f'<div class="response-text">')
    
    # Highlight claims in response
    highlighted = _highlight_claims(view.response_text, view.claims)
    html.append(highlighted)
    
    html.append('</div></div>')
    
    # Sources column
    html.append('<div class="sources-column">')
    html.append('<h3>Original Sources</h3>')
    
    for i, claim in enumerate(view.claims):
        html.append(f'<div class="source-card" data-claim-id="{i}">')
        
        for segment in claim.source_segments:
            html.append(f'<h4>{segment.source_title}</h4>')
            html.append(f'<blockquote>{segment.text}</blockquote>')
            
            if segment.url:
                html.append(f'<a href="{segment.url}" target="_blank">')
                html.append('View Source</a>')
            
            if segment.location:
                html.append(f'<span class="location">{segment.location}</span>')
        
        html.append(f'<span class="match-type {claim.match_type.value}">')
        html.append(f'{claim.match_type.value.title()} Match</span>')
        html.append('</div>')
    
    html.append('</div></div>')
    
    return '\n'.join(html)

def _highlight_claims(text: str, claims: list[ClaimSegment]) -> str:
    """Add highlight spans around cited claims."""
    if not claims:
        return text
    
    # Sort by start position descending
    sorted_claims = sorted(claims, key=lambda c: c.start, reverse=True)
    
    result = text
    for i, claim in enumerate(sorted_claims):
        end = claim.end
        start = claim.start
        
        highlight = f'<span class="claim-highlight" data-claim-id="{i}">'
        result = result[:start] + highlight + result[start:end] + '</span>' + result[end:]
    
    return result
```

---

## Source Preview Panels

Show expandable previews of sources inline with the response.

### Preview Component

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SourcePreview:
    """Preview of a source document."""
    source_id: str
    title: str
    url: Optional[str]
    preview_text: str
    full_text: Optional[str]
    metadata: dict

@dataclass
class PreviewPanel:
    """Expandable preview panel."""
    trigger_text: str
    trigger_start: int
    trigger_end: int
    preview: SourcePreview
    expanded: bool = False

class SourcePreviewBuilder:
    """Build preview panels for sources."""
    
    def __init__(self, max_preview_length: int = 200):
        self.max_preview_length = max_preview_length
    
    def create_preview(
        self,
        source: dict,
        matched_text: str = None
    ) -> SourcePreview:
        """Create a source preview."""
        
        full_text = source.get("content", "")
        
        if matched_text:
            # Center preview around matched text
            preview_text = self._extract_context(full_text, matched_text)
        else:
            # Use beginning of document
            preview_text = full_text[:self.max_preview_length]
            if len(full_text) > self.max_preview_length:
                preview_text += "..."
        
        return SourcePreview(
            source_id=source["id"],
            title=source.get("title", "Untitled"),
            url=source.get("url"),
            preview_text=preview_text,
            full_text=full_text,
            metadata={
                "author": source.get("author"),
                "date": source.get("date"),
                "type": source.get("type", "document")
            }
        )
    
    def _extract_context(
        self,
        full_text: str,
        matched_text: str,
        context_chars: int = 100
    ) -> str:
        """Extract text around matched content."""
        
        idx = full_text.lower().find(matched_text.lower())
        
        if idx == -1:
            return full_text[:self.max_preview_length] + "..."
        
        start = max(0, idx - context_chars)
        end = min(len(full_text), idx + len(matched_text) + context_chars)
        
        preview = full_text[start:end]
        
        if start > 0:
            preview = "..." + preview
        if end < len(full_text):
            preview = preview + "..."
        
        return preview

def create_preview_panels(
    response: str,
    citations: list[dict],
    sources: list[dict]
) -> list[PreviewPanel]:
    """Create preview panels for all citations."""
    
    builder = SourcePreviewBuilder()
    source_map = {s["id"]: s for s in sources}
    panels = []
    
    for citation in citations:
        source = source_map.get(citation["source_id"])
        
        if source:
            preview = builder.create_preview(
                source,
                citation.get("matched_text")
            )
            
            panel = PreviewPanel(
                trigger_text=citation.get("cited_text", ""),
                trigger_start=citation["start"],
                trigger_end=citation["end"],
                preview=preview
            )
            panels.append(panel)
    
    return panels
```

### Interactive Preview UI

```python
def render_preview_panels_html(
    response: str,
    panels: list[PreviewPanel]
) -> str:
    """Render response with expandable preview panels."""
    
    html = ['<div class="response-with-previews">']
    
    # Sort panels by position
    sorted_panels = sorted(panels, key=lambda p: p.trigger_start)
    
    current_pos = 0
    
    for i, panel in enumerate(sorted_panels):
        # Add text before panel
        if panel.trigger_start > current_pos:
            html.append(f'<span>{response[current_pos:panel.trigger_start]}</span>')
        
        # Add trigger with preview
        html.append(f'''
        <span class="preview-trigger" data-panel-id="{i}">
            {response[panel.trigger_start:panel.trigger_end]}
            <span class="citation-marker">[{i+1}]</span>
        </span>
        <div class="preview-panel" id="panel-{i}">
            <div class="preview-header">
                <strong>{panel.preview.title}</strong>
                <button class="expand-btn" onclick="togglePanel({i})">
                    Expand
                </button>
            </div>
            <div class="preview-content">
                <p>{panel.preview.preview_text}</p>
            </div>
            <div class="preview-full" style="display: none;">
                <p>{panel.preview.full_text or ''}</p>
            </div>
            <div class="preview-footer">
        ''')
        
        if panel.preview.url:
            html.append(f'''
                <a href="{panel.preview.url}" target="_blank" 
                   class="view-source-btn">View Original</a>
            ''')
        
        # Add metadata
        meta = panel.preview.metadata
        if meta.get("author"):
            html.append(f'<span class="meta">By {meta["author"]}</span>')
        if meta.get("date"):
            html.append(f'<span class="meta">{meta["date"]}</span>')
        
        html.append('</div></div>')
        
        current_pos = panel.trigger_end
    
    # Add remaining text
    if current_pos < len(response):
        html.append(f'<span>{response[current_pos:]}</span>')
    
    html.append('</div>')
    
    # Add JavaScript for interactivity
    html.append('''
    <script>
    function togglePanel(id) {
        const panel = document.getElementById('panel-' + id);
        const preview = panel.querySelector('.preview-content');
        const full = panel.querySelector('.preview-full');
        const btn = panel.querySelector('.expand-btn');
        
        if (full.style.display === 'none') {
            preview.style.display = 'none';
            full.style.display = 'block';
            btn.textContent = 'Collapse';
        } else {
            preview.style.display = 'block';
            full.style.display = 'none';
            btn.textContent = 'Expand';
        }
    }
    </script>
    ''')
    
    return '\n'.join(html)
```

---

## Fact-Check Workflow

Implement a structured workflow for verifying AI claims.

### Fact-Check Data Model

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class VerificationStatus(Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    UNCERTAIN = "uncertain"
    NOT_VERIFIABLE = "not_verifiable"

class ClaimType(Enum):
    FACTUAL = "factual"
    STATISTICAL = "statistical"
    QUOTE = "quote"
    OPINION = "opinion"
    INFERENCE = "inference"

@dataclass
class Claim:
    """A claim extracted from AI response."""
    id: str
    text: str
    claim_type: ClaimType
    source_ids: list[str]
    verification_status: VerificationStatus = VerificationStatus.PENDING
    confidence: float = 0.0
    verified_at: Optional[datetime] = None
    verified_by: Optional[str] = None
    notes: str = ""

@dataclass
class VerificationResult:
    """Result of verification attempt."""
    claim_id: str
    status: VerificationStatus
    evidence: list[str]
    confidence: float
    verifier: str
    timestamp: datetime
    notes: str = ""

@dataclass
class FactCheckReport:
    """Complete fact-check report."""
    response_id: str
    claims: list[Claim]
    verifications: list[VerificationResult]
    overall_confidence: float
    created_at: datetime = field(default_factory=datetime.now)
```

### Fact-Check Workflow Engine

```python
from typing import Callable

class FactCheckWorkflow:
    """Workflow engine for fact-checking."""
    
    def __init__(self):
        self.claims: list[Claim] = []
        self.verifications: list[VerificationResult] = []
        self.verifiers: list[Callable] = []
    
    def add_verifier(self, verifier: Callable):
        """Add a verification function."""
        self.verifiers.append(verifier)
    
    def extract_claims(
        self,
        response: str,
        citations: list[dict]
    ) -> list[Claim]:
        """Extract verifiable claims from response."""
        claims = []
        
        for i, citation in enumerate(citations):
            claim_type = self._detect_claim_type(citation.get("cited_text", ""))
            
            claim = Claim(
                id=f"claim_{i}",
                text=citation.get("cited_text", ""),
                claim_type=claim_type,
                source_ids=[citation["source_id"]],
                confidence=citation.get("confidence", 0.5)
            )
            claims.append(claim)
        
        self.claims = claims
        return claims
    
    def _detect_claim_type(self, text: str) -> ClaimType:
        """Detect the type of claim."""
        text_lower = text.lower()
        
        # Check for quotes
        if '"' in text or "'" in text:
            return ClaimType.QUOTE
        
        # Check for statistics
        if any(char.isdigit() for char in text):
            if any(word in text_lower for word in ["percent", "%", "million", "billion"]):
                return ClaimType.STATISTICAL
        
        # Check for opinion indicators
        opinion_words = ["should", "could", "might", "seems", "appears", "likely"]
        if any(word in text_lower for word in opinion_words):
            return ClaimType.OPINION
        
        # Check for inference
        inference_words = ["therefore", "thus", "suggests", "implies", "indicates"]
        if any(word in text_lower for word in inference_words):
            return ClaimType.INFERENCE
        
        return ClaimType.FACTUAL
    
    def verify_all(
        self,
        sources: list[dict]
    ) -> list[VerificationResult]:
        """Run all verifiers on all claims."""
        results = []
        source_map = {s["id"]: s for s in sources}
        
        for claim in self.claims:
            claim_sources = [
                source_map[sid]
                for sid in claim.source_ids
                if sid in source_map
            ]
            
            result = self._verify_claim(claim, claim_sources)
            results.append(result)
            
            # Update claim status
            claim.verification_status = result.status
            claim.verified_at = result.timestamp
        
        self.verifications = results
        return results
    
    def _verify_claim(
        self,
        claim: Claim,
        sources: list[dict]
    ) -> VerificationResult:
        """Verify a single claim against sources."""
        
        if not sources:
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.NOT_VERIFIABLE,
                evidence=[],
                confidence=0.0,
                verifier="system",
                timestamp=datetime.now(),
                notes="No sources available"
            )
        
        # Run custom verifiers
        for verifier in self.verifiers:
            result = verifier(claim, sources)
            if result:
                return result
        
        # Default verification: text matching
        evidence = []
        total_confidence = 0.0
        
        for source in sources:
            content = source.get("content", "").lower()
            claim_text = claim.text.lower()
            
            if claim_text in content:
                evidence.append(f"Exact match in {source.get('title', 'source')}")
                total_confidence += 0.9
            elif self._fuzzy_match(claim_text, content):
                evidence.append(f"Partial match in {source.get('title', 'source')}")
                total_confidence += 0.6
        
        avg_confidence = total_confidence / len(sources) if sources else 0
        
        if avg_confidence >= 0.8:
            status = VerificationStatus.VERIFIED
        elif avg_confidence >= 0.5:
            status = VerificationStatus.UNCERTAIN
        elif avg_confidence > 0:
            status = VerificationStatus.DISPUTED
        else:
            status = VerificationStatus.NOT_VERIFIABLE
        
        return VerificationResult(
            claim_id=claim.id,
            status=status,
            evidence=evidence,
            confidence=avg_confidence,
            verifier="text_match",
            timestamp=datetime.now()
        )
    
    def _fuzzy_match(self, claim: str, content: str) -> bool:
        """Check for fuzzy text match."""
        words = claim.split()
        matched = sum(1 for word in words if word in content)
        return matched / len(words) > 0.6 if words else False
    
    def generate_report(self, response_id: str) -> FactCheckReport:
        """Generate complete fact-check report."""
        
        # Calculate overall confidence
        if self.verifications:
            total = sum(v.confidence for v in self.verifications)
            overall = total / len(self.verifications)
        else:
            overall = 0.0
        
        return FactCheckReport(
            response_id=response_id,
            claims=self.claims,
            verifications=self.verifications,
            overall_confidence=overall
        )
```

### Rendering Fact-Check Results

```python
def render_fact_check_report(report: FactCheckReport) -> str:
    """Render fact-check report as Markdown."""
    
    lines = [
        f"# Fact-Check Report",
        f"",
        f"**Response ID:** {report.response_id}",
        f"**Overall Confidence:** {report.overall_confidence:.0%}",
        f"**Generated:** {report.created_at.strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"## Claims Summary",
        f""
    ]
    
    # Status summary
    status_counts = {}
    for claim in report.claims:
        status = claim.verification_status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    for status, count in status_counts.items():
        emoji = _status_emoji(status)
        lines.append(f"| {emoji} {status.title()} | {count} |")
    
    lines.extend([
        "",
        "## Detailed Verification",
        ""
    ])
    
    # Verification details for each claim
    verification_map = {v.claim_id: v for v in report.verifications}
    
    for claim in report.claims:
        verification = verification_map.get(claim.id)
        status_emoji = _status_emoji(claim.verification_status.value)
        
        lines.extend([
            f"### {status_emoji} Claim: {claim.text[:100]}...",
            f"",
            f"- **Type:** {claim.claim_type.value.title()}",
            f"- **Status:** {claim.verification_status.value.title()}",
        ])
        
        if verification:
            lines.append(f"- **Confidence:** {verification.confidence:.0%}")
            
            if verification.evidence:
                lines.append("- **Evidence:**")
                for ev in verification.evidence:
                    lines.append(f"  - {ev}")
            
            if verification.notes:
                lines.append(f"- **Notes:** {verification.notes}")
        
        lines.append("")
    
    return "\n".join(lines)

def _status_emoji(status: str) -> str:
    """Get emoji for verification status."""
    emojis = {
        "verified": "✅",
        "disputed": "⚠️",
        "uncertain": "❓",
        "pending": "⏳",
        "not_verifiable": "🚫"
    }
    return emojis.get(status, "•")
```

---

## User Reporting Mechanisms

Let users report issues with sources and citations.

### Report Data Model

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class ReportType(Enum):
    INCORRECT_CITATION = "incorrect_citation"
    OUTDATED_SOURCE = "outdated_source"
    BROKEN_LINK = "broken_link"
    MISATTRIBUTION = "misattribution"
    HALLUCINATION = "hallucination"
    OTHER = "other"

class ReportStatus(Enum):
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"

@dataclass
class UserReport:
    """User-submitted issue report."""
    id: str
    report_type: ReportType
    response_id: str
    claim_id: Optional[str]
    source_id: Optional[str]
    description: str
    user_id: str
    status: ReportStatus = ReportStatus.SUBMITTED
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""

@dataclass
class ReportForm:
    """Form fields for submitting report."""
    report_types: list[dict]
    required_fields: list[str]
    optional_fields: list[str]
```

### Reporting System

```python
from uuid import uuid4
from typing import Callable

class ReportingSystem:
    """System for handling user reports."""
    
    def __init__(self):
        self.reports: list[UserReport] = []
        self.handlers: dict[ReportType, Callable] = {}
    
    def register_handler(
        self,
        report_type: ReportType,
        handler: Callable[[UserReport], None]
    ):
        """Register handler for specific report type."""
        self.handlers[report_type] = handler
    
    def submit_report(
        self,
        report_type: ReportType,
        response_id: str,
        description: str,
        user_id: str,
        claim_id: str = None,
        source_id: str = None
    ) -> UserReport:
        """Submit a new report."""
        
        report = UserReport(
            id=str(uuid4()),
            report_type=report_type,
            response_id=response_id,
            claim_id=claim_id,
            source_id=source_id,
            description=description,
            user_id=user_id
        )
        
        self.reports.append(report)
        
        # Trigger handler if registered
        if report_type in self.handlers:
            self.handlers[report_type](report)
        
        return report
    
    def get_reports(
        self,
        status: ReportStatus = None,
        report_type: ReportType = None
    ) -> list[UserReport]:
        """Get reports with optional filtering."""
        
        filtered = self.reports
        
        if status:
            filtered = [r for r in filtered if r.status == status]
        
        if report_type:
            filtered = [r for r in filtered if r.report_type == report_type]
        
        return filtered
    
    def resolve_report(
        self,
        report_id: str,
        resolution_notes: str,
        dismissed: bool = False
    ) -> Optional[UserReport]:
        """Resolve a report."""
        
        for report in self.reports:
            if report.id == report_id:
                report.status = (
                    ReportStatus.DISMISSED if dismissed 
                    else ReportStatus.RESOLVED
                )
                report.resolved_at = datetime.now()
                report.resolution_notes = resolution_notes
                return report
        
        return None
    
    def get_form(self) -> ReportForm:
        """Get report submission form configuration."""
        
        return ReportForm(
            report_types=[
                {
                    "value": rt.value,
                    "label": rt.value.replace("_", " ").title(),
                    "description": self._get_type_description(rt)
                }
                for rt in ReportType
            ],
            required_fields=["report_type", "description"],
            optional_fields=["claim_id", "source_id"]
        )
    
    def _get_type_description(self, rt: ReportType) -> str:
        """Get description for report type."""
        descriptions = {
            ReportType.INCORRECT_CITATION: 
                "The citation doesn't match the source content",
            ReportType.OUTDATED_SOURCE:
                "The source information is no longer current",
            ReportType.BROKEN_LINK:
                "The source link is broken or inaccessible",
            ReportType.MISATTRIBUTION:
                "The claim is attributed to the wrong source",
            ReportType.HALLUCINATION:
                "The AI made up information not in sources",
            ReportType.OTHER:
                "Other issue with source attribution"
        }
        return descriptions.get(rt, "")
```

### Report UI Components

```python
def render_report_form_html(form: ReportForm, response_id: str) -> str:
    """Render report submission form."""
    
    html = [f'''
    <form class="report-form" id="report-form" data-response="{response_id}">
        <h3>🚩 Report an Issue</h3>
        
        <div class="form-group">
            <label for="report-type">Issue Type *</label>
            <select id="report-type" name="report_type" required>
                <option value="">Select issue type...</option>
    ''']
    
    for rt in form.report_types:
        html.append(f'''
                <option value="{rt['value']}" title="{rt['description']}">
                    {rt['label']}
                </option>
        ''')
    
    html.append('''
            </select>
        </div>
        
        <div class="form-group">
            <label for="description">Description *</label>
            <textarea 
                id="description" 
                name="description" 
                rows="4" 
                placeholder="Describe the issue..."
                required
            ></textarea>
        </div>
        
        <div class="form-group">
            <label for="source-id">Affected Source (optional)</label>
            <input type="text" id="source-id" name="source_id" 
                   placeholder="Source ID or title">
        </div>
        
        <div class="form-actions">
            <button type="submit" class="btn-primary">Submit Report</button>
            <button type="button" class="btn-secondary" onclick="closeForm()">
                Cancel
            </button>
        </div>
    </form>
    ''')
    
    return '\n'.join(html)

def render_report_button_html(response_id: str, claim_id: str = None) -> str:
    """Render report issue button."""
    
    data_attrs = f'data-response="{response_id}"'
    if claim_id:
        data_attrs += f' data-claim="{claim_id}"'
    
    return f'''
    <button class="report-btn" {data_attrs} onclick="openReportForm(this)">
        🚩 Report Issue
    </button>
    '''
```

---

## Complete Verifiability System

Combine all components into a complete verification system.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class VerifiabilityConfig:
    enable_comparison: bool = True
    enable_previews: bool = True
    enable_fact_check: bool = True
    enable_reporting: bool = True
    preview_length: int = 200
    auto_verify: bool = True

class VerifiabilitySystem:
    """Complete verifiability feature system."""
    
    def __init__(self, config: VerifiabilityConfig = None):
        self.config = config or VerifiabilityConfig()
        self.preview_builder = SourcePreviewBuilder(
            self.config.preview_length
        )
        self.fact_checker = FactCheckWorkflow()
        self.reporting = ReportingSystem()
    
    def process_response(
        self,
        response_id: str,
        response_text: str,
        citations: list[dict],
        sources: list[dict]
    ) -> dict:
        """Process response with all verifiability features."""
        
        result = {
            "response_id": response_id,
            "text": response_text,
            "features": {}
        }
        
        # Side-by-side comparison
        if self.config.enable_comparison:
            comparison = build_comparison_view(
                response_text, citations, sources
            )
            result["features"]["comparison"] = comparison
        
        # Source previews
        if self.config.enable_previews:
            panels = create_preview_panels(
                response_text, citations, sources
            )
            result["features"]["previews"] = panels
        
        # Fact-check workflow
        if self.config.enable_fact_check:
            self.fact_checker.extract_claims(response_text, citations)
            
            if self.config.auto_verify:
                self.fact_checker.verify_all(sources)
            
            report = self.fact_checker.generate_report(response_id)
            result["features"]["fact_check"] = report
        
        # Reporting form
        if self.config.enable_reporting:
            form = self.reporting.get_form()
            result["features"]["report_form"] = form
        
        return result
    
    def render_verifiable_response(
        self,
        response_id: str,
        response_text: str,
        citations: list[dict],
        sources: list[dict]
    ) -> str:
        """Render response with all verifiability features."""
        
        processed = self.process_response(
            response_id, response_text, citations, sources
        )
        
        html_parts = ['<div class="verifiable-response">']
        
        # Main response with previews
        if "previews" in processed["features"]:
            html_parts.append(
                render_preview_panels_html(
                    response_text,
                    processed["features"]["previews"]
                )
            )
        else:
            html_parts.append(f'<div class="response">{response_text}</div>')
        
        # Comparison view toggle
        if "comparison" in processed["features"]:
            html_parts.append('''
            <button onclick="toggleComparison()" class="btn-secondary">
                📊 Show Side-by-Side Comparison
            </button>
            <div id="comparison-view" style="display: none;">
            ''')
            html_parts.append(
                render_comparison_html(processed["features"]["comparison"])
            )
            html_parts.append('</div>')
        
        # Fact-check summary
        if "fact_check" in processed["features"]:
            report = processed["features"]["fact_check"]
            verified = sum(
                1 for c in report.claims 
                if c.verification_status == VerificationStatus.VERIFIED
            )
            total = len(report.claims)
            
            html_parts.append(f'''
            <div class="fact-check-summary">
                ✅ {verified}/{total} claims verified
                <button onclick="toggleFactCheck()">View Details</button>
            </div>
            <div id="fact-check-details" style="display: none;">
                <pre>{render_fact_check_report(report)}</pre>
            </div>
            ''')
        
        # Report button
        if "report_form" in processed["features"]:
            html_parts.append(render_report_button_html(response_id))
            html_parts.append(f'''
            <div id="report-modal" style="display: none;">
                {render_report_form_html(
                    processed["features"]["report_form"],
                    response_id
                )}
            </div>
            ''')
        
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)

# Usage
system = VerifiabilitySystem(VerifiabilityConfig(
    enable_comparison=True,
    enable_previews=True,
    enable_fact_check=True,
    enable_reporting=True,
    auto_verify=True
))

html = system.render_verifiable_response(
    response_id="resp_123",
    response_text="Spain won Euro 2024...",
    citations=[...],
    sources=[...]
)
```

---

## Hands-on Exercise

### Your Task

Build a `VerificationDashboard` that:
1. Shows verification status for all claims
2. Allows manual verification
3. Tracks verification history
4. Exports verification reports

### Requirements

```python
class VerificationDashboard:
    def add_response(
        self,
        response_id: str,
        claims: list[dict],
        sources: list[dict]
    ) -> None:
        pass
    
    def verify_claim(
        self,
        response_id: str,
        claim_id: str,
        status: str,
        notes: str
    ) -> dict:
        pass
    
    def get_dashboard_data(self) -> dict:
        """Returns summary stats and pending verifications."""
        pass
    
    def export_report(self, response_id: str, format: str) -> str:
        """Export as 'markdown' or 'json'."""
        pass
```

<details>
<summary>💡 Hints</summary>

- Store responses with their claims in a dictionary
- Track verification history with timestamps
- Calculate stats from claim statuses
- Use the existing report rendering code

</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json

@dataclass
class VerificationEntry:
    claim_id: str
    status: str
    notes: str
    verified_by: str
    timestamp: datetime

@dataclass
class TrackedClaim:
    id: str
    text: str
    source_ids: list[str]
    status: str = "pending"
    history: list[VerificationEntry] = field(default_factory=list)

@dataclass
class TrackedResponse:
    id: str
    claims: list[TrackedClaim]
    sources: list[dict]
    created_at: datetime = field(default_factory=datetime.now)

class VerificationDashboard:
    def __init__(self):
        self.responses: dict[str, TrackedResponse] = {}
    
    def add_response(
        self,
        response_id: str,
        claims: list[dict],
        sources: list[dict]
    ) -> None:
        tracked_claims = [
            TrackedClaim(
                id=claim["id"],
                text=claim["text"],
                source_ids=claim.get("source_ids", [])
            )
            for claim in claims
        ]
        
        self.responses[response_id] = TrackedResponse(
            id=response_id,
            claims=tracked_claims,
            sources=sources
        )
    
    def verify_claim(
        self,
        response_id: str,
        claim_id: str,
        status: str,
        notes: str,
        verified_by: str = "user"
    ) -> dict:
        response = self.responses.get(response_id)
        if not response:
            return {"error": "Response not found"}
        
        for claim in response.claims:
            if claim.id == claim_id:
                # Update status
                old_status = claim.status
                claim.status = status
                
                # Add to history
                entry = VerificationEntry(
                    claim_id=claim_id,
                    status=status,
                    notes=notes,
                    verified_by=verified_by,
                    timestamp=datetime.now()
                )
                claim.history.append(entry)
                
                return {
                    "success": True,
                    "claim_id": claim_id,
                    "old_status": old_status,
                    "new_status": status
                }
        
        return {"error": "Claim not found"}
    
    def get_dashboard_data(self) -> dict:
        stats = {
            "total_responses": len(self.responses),
            "total_claims": 0,
            "verified": 0,
            "disputed": 0,
            "pending": 0
        }
        
        pending_verifications = []
        
        for response in self.responses.values():
            for claim in response.claims:
                stats["total_claims"] += 1
                
                if claim.status == "verified":
                    stats["verified"] += 1
                elif claim.status == "disputed":
                    stats["disputed"] += 1
                else:
                    stats["pending"] += 1
                    pending_verifications.append({
                        "response_id": response.id,
                        "claim_id": claim.id,
                        "text": claim.text[:100],
                        "created_at": response.created_at.isoformat()
                    })
        
        # Sort pending by creation date
        pending_verifications.sort(
            key=lambda x: x["created_at"]
        )
        
        return {
            "stats": stats,
            "pending": pending_verifications[:20],  # Top 20
            "verification_rate": (
                stats["verified"] / stats["total_claims"] 
                if stats["total_claims"] > 0 else 0
            )
        }
    
    def export_report(
        self,
        response_id: str,
        format: str = "markdown"
    ) -> str:
        response = self.responses.get(response_id)
        if not response:
            return "Response not found"
        
        if format == "json":
            return self._export_json(response)
        else:
            return self._export_markdown(response)
    
    def _export_json(self, response: TrackedResponse) -> str:
        data = {
            "response_id": response.id,
            "created_at": response.created_at.isoformat(),
            "claims": [
                {
                    "id": c.id,
                    "text": c.text,
                    "status": c.status,
                    "source_ids": c.source_ids,
                    "history": [
                        {
                            "status": h.status,
                            "notes": h.notes,
                            "verified_by": h.verified_by,
                            "timestamp": h.timestamp.isoformat()
                        }
                        for h in c.history
                    ]
                }
                for c in response.claims
            ],
            "sources": response.sources
        }
        return json.dumps(data, indent=2)
    
    def _export_markdown(self, response: TrackedResponse) -> str:
        lines = [
            f"# Verification Report: {response.id}",
            f"",
            f"**Created:** {response.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"",
            "## Claims",
            ""
        ]
        
        for claim in response.claims:
            emoji = {"verified": "✅", "disputed": "⚠️"}.get(
                claim.status, "⏳"
            )
            
            lines.append(f"### {emoji} {claim.text[:80]}...")
            lines.append(f"- **Status:** {claim.status}")
            lines.append(f"- **Sources:** {', '.join(claim.source_ids)}")
            
            if claim.history:
                lines.append("- **History:**")
                for h in claim.history:
                    lines.append(
                        f"  - {h.timestamp.strftime('%m/%d %H:%M')}: "
                        f"{h.status} by {h.verified_by}"
                    )
                    if h.notes:
                        lines.append(f"    - Note: {h.notes}")
            
            lines.append("")
        
        return "\n".join(lines)

# Usage
dashboard = VerificationDashboard()

# Add response
dashboard.add_response(
    response_id="resp_123",
    claims=[
        {"id": "c1", "text": "Spain won Euro 2024", "source_ids": ["s1"]},
        {"id": "c2", "text": "Score was 2-1", "source_ids": ["s1", "s2"]}
    ],
    sources=[
        {"id": "s1", "title": "Euro 2024 Results"},
        {"id": "s2", "title": "Match Report"}
    ]
)

# Verify a claim
dashboard.verify_claim(
    response_id="resp_123",
    claim_id="c1",
    status="verified",
    notes="Confirmed in official UEFA records"
)

# Get dashboard data
data = dashboard.get_dashboard_data()
print(f"Verified: {data['stats']['verified']}/{data['stats']['total_claims']}")

# Export report
report = dashboard.export_report("resp_123", format="markdown")
print(report)
```

</details>

---

## Summary

Verifiability features build trust in AI-generated content:

✅ **Side-by-side comparison** — Show claims with original sources
✅ **Preview panels** — Expandable source previews inline
✅ **Fact-check workflow** — Structured claim verification
✅ **User reporting** — Crowdsource accuracy improvements

**Key patterns:**
- Match types (exact, paraphrase, inferred)
- Verification statuses (verified, disputed, pending)
- Report types (incorrect citation, hallucination, etc.)

**Next:** [Advanced Retrieval Strategies](../09-advanced-retrieval/00-advanced-retrieval.md)

---

## Further Reading

- [Fact-Checking Methodologies](https://www.poynter.org/ifcn/)
- [Citation Verification Patterns](https://arxiv.org/abs/2305.14627)
- [Building Trust in AI Systems](https://hai.stanford.edu/research/trust-ai)

<!--
Sources Consulted:
- International Fact-Checking Network standards
- Human-AI verification research papers
- UX patterns for transparency
-->
