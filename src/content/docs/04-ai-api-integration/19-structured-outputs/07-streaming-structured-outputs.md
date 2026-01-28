---
title: "Streaming Structured Outputs"
---

# Streaming Structured Outputs

## Introduction

Structured Outputs supports streaming, delivering partial JSON chunks as the model generates them. This enables responsive UIs that display data progressively while maintaining type safety on completion.

### What We'll Cover

- Streaming mechanics for structured outputs
- Partial JSON handling and accumulation
- UI update patterns
- Error handling during streams

### Prerequisites

- Structured Outputs basics
- Streaming API knowledge
- Event handling patterns

---

## Streaming Mechanics

### How Streaming Works

```python
from dataclasses import dataclass
from typing import Optional, List, Generator
import json


@dataclass
class StreamingFeature:
    """Streaming capability description."""
    
    feature: str
    behavior: str
    benefit: str


STREAMING_FEATURES = [
    StreamingFeature(
        feature="Partial JSON chunks",
        behavior="Valid JSON prefixes delivered as they're generated",
        benefit="Early display of completed fields"
    ),
    StreamingFeature(
        feature="Delta accumulation",
        behavior="Each chunk adds to previous content",
        benefit="Build complete response incrementally"
    ),
    StreamingFeature(
        feature="Final validation",
        behavior="Complete JSON validated against schema on completion",
        benefit="Guaranteed schema compliance"
    ),
    StreamingFeature(
        feature="Event-based delivery",
        behavior="Server-sent events with content deltas",
        benefit="Real-time updates without polling"
    )
]


print("Streaming Structured Outputs Features")
print("=" * 60)

for f in STREAMING_FEATURES:
    print(f"\n🔄 {f.feature}")
    print(f"   Behavior: {f.behavior}")
    print(f"   Benefit: {f.benefit}")
```

### Simulated Streaming Response

```python
@dataclass
class StreamChunk:
    """A chunk from a streaming response."""
    
    chunk_index: int
    delta: str
    accumulated: str
    is_complete: bool
    parsed_so_far: Optional[dict] = None


class StreamSimulator:
    """Simulate streaming structured output generation."""
    
    def __init__(self, target_response: dict):
        self.target = json.dumps(target_response, indent=2)
        self.chunk_size = 20
    
    def stream(self) -> Generator[StreamChunk, None, None]:
        """Generate streaming chunks."""
        
        accumulated = ""
        
        for i in range(0, len(self.target), self.chunk_size):
            delta = self.target[i:i + self.chunk_size]
            accumulated += delta
            
            # Try to parse accumulated content
            parsed = self._try_parse(accumulated)
            
            yield StreamChunk(
                chunk_index=i // self.chunk_size,
                delta=delta,
                accumulated=accumulated,
                is_complete=(i + self.chunk_size >= len(self.target)),
                parsed_so_far=parsed
            )
    
    def _try_parse(self, content: str) -> Optional[dict]:
        """Attempt to parse partial JSON."""
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None


# Simulate streaming
target = {
    "name": "John Doe",
    "age": 30,
    "email": "john@example.com"
}

simulator = StreamSimulator(target)

print("\n\nStreaming Simulation")
print("=" * 60)

for chunk in simulator.stream():
    status = "✅ Complete" if chunk.is_complete else "🔄 Streaming"
    parsed_status = "📋 Parseable" if chunk.parsed_so_far else "⏳ Partial"
    
    print(f"\nChunk {chunk.chunk_index}: {status} | {parsed_status}")
    print(f"Delta: {repr(chunk.delta)}")
    
    if chunk.parsed_so_far:
        print(f"Parsed: {chunk.parsed_so_far}")
```

---

## SDK Streaming Patterns

### Python SDK Streaming

```python
from pydantic import BaseModel
from typing import Optional


class ExtractedData(BaseModel):
    """Data being extracted."""
    
    title: str
    summary: str
    key_points: list[str]
    sentiment: str


class MockStreamingResponse:
    """Mock streaming response for demonstration."""
    
    def __init__(self, chunks: List[str]):
        self._chunks = chunks
        self._index = 0
        self._accumulated = ""
        self.parsed: Optional[ExtractedData] = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index >= len(self._chunks):
            raise StopIteration
        
        chunk = self._chunks[self._index]
        self._index += 1
        self._accumulated += chunk
        return self._create_event(chunk)
    
    def _create_event(self, delta: str):
        """Create a streaming event."""
        
        class Event:
            pass
        
        event = Event()
        event.type = "content_block_delta"
        event.delta = type("Delta", (), {"text": delta})()
        event.snapshot = self._accumulated
        
        return event


# SDK-style streaming handler
class StreamingHandler:
    """Handle streaming structured outputs."""
    
    def __init__(self):
        self.content = ""
        self.on_partial: Optional[callable] = None
        self.on_complete: Optional[callable] = None
    
    def process_stream(self, stream):
        """Process a streaming response."""
        
        for event in stream:
            if hasattr(event, "delta"):
                delta = event.delta.text if hasattr(event.delta, "text") else ""
                self.content += delta
                
                if self.on_partial:
                    self.on_partial(delta, self.content)
        
        # Final parse
        try:
            result = json.loads(self.content)
            if self.on_complete:
                self.on_complete(result)
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON after stream: {e}")


# Example usage
print("\n\nSDK Streaming Pattern")
print("=" * 60)

# Create mock stream
chunks = [
    '{"title": "AI A',
    'rticle", "summ',
    'ary": "An over',
    'view of AI", "',
    'key_points": [',
    '"point1", "poi',
    'nt2"], "sentime',
    'nt": "positive"}'
]

stream = MockStreamingResponse(chunks)

handler = StreamingHandler()
handler.on_partial = lambda d, a: print(f"📝 Partial: +{repr(d)}")
handler.on_complete = lambda r: print(f"\n✅ Complete: {r}")

result = handler.process_stream(stream)
```

---

## Partial JSON Parsing

### Graceful Partial Handling

```python
import re


class PartialJSONParser:
    """Parse partial JSON during streaming."""
    
    def __init__(self):
        self.buffer = ""
        self.complete_fields = {}
    
    def feed(self, chunk: str) -> dict:
        """Add chunk and extract what we can."""
        
        self.buffer += chunk
        return self._extract_fields()
    
    def _extract_fields(self) -> dict:
        """Extract completed fields from buffer."""
        
        # Try to find complete key-value pairs
        extracted = {}
        
        # Pattern for complete string fields
        string_pattern = r'"(\w+)":\s*"([^"]*)"'
        for match in re.finditer(string_pattern, self.buffer):
            key, value = match.groups()
            # Check if this field is truly complete (followed by , or })
            end_pos = match.end()
            if end_pos < len(self.buffer):
                next_char = self.buffer[end_pos:end_pos+1].strip()
                if next_char in [',', '}', '']:
                    extracted[key] = value
                    self.complete_fields[key] = value
        
        # Pattern for complete number fields
        number_pattern = r'"(\w+)":\s*(\d+)'
        for match in re.finditer(number_pattern, self.buffer):
            key, value = match.groups()
            end_pos = match.end()
            if end_pos < len(self.buffer):
                next_char = self.buffer[end_pos:end_pos+1].strip()
                if next_char in [',', '}', '']:
                    extracted[key] = int(value)
                    self.complete_fields[key] = int(value)
        
        return self.complete_fields.copy()


# Test partial parsing
print("\n\nPartial JSON Parsing")
print("=" * 60)

parser = PartialJSONParser()

chunks = [
    '{"name": "Joh',
    'n Doe", "age": 30',
    ', "city": "New York"',
    '}'
]

for i, chunk in enumerate(chunks):
    fields = parser.feed(chunk)
    print(f"\nAfter chunk {i + 1}: {repr(chunk)}")
    print(f"Complete fields: {fields}")
```

### Field-by-Field Updates

```python
@dataclass
class FieldUpdate:
    """Update for a single field."""
    
    field_name: str
    value: any
    is_complete: bool
    confidence: float = 1.0


class IncrementalExtractor:
    """Extract fields incrementally during streaming."""
    
    def __init__(self, expected_fields: List[str]):
        self.expected = set(expected_fields)
        self.completed = {}
        self.partial = {}
        self.buffer = ""
    
    def update(self, chunk: str) -> List[FieldUpdate]:
        """Process chunk and return field updates."""
        
        self.buffer += chunk
        updates = []
        
        # Check for newly completed fields
        try:
            # Try parsing as JSON
            data = json.loads(self.buffer + "}")  # Try closing
            
            for field in self.expected:
                if field in data and field not in self.completed:
                    self.completed[field] = data[field]
                    updates.append(FieldUpdate(
                        field_name=field,
                        value=data[field],
                        is_complete=True
                    ))
        except json.JSONDecodeError:
            pass
        
        return updates
    
    def get_progress(self) -> dict:
        """Get extraction progress."""
        
        return {
            "completed": len(self.completed),
            "total": len(self.expected),
            "fields": self.completed,
            "pending": self.expected - set(self.completed.keys())
        }


# Test incremental extraction
print("\n\nIncremental Field Extraction")
print("=" * 60)

extractor = IncrementalExtractor(["name", "age", "email", "role"])

stream_chunks = [
    '{"name": "Alice',
    ' Smith", "age": 28',
    ', "email": "alice@',
    'example.com", "role": "Engineer"}'
]

for chunk in stream_chunks:
    updates = extractor.update(chunk)
    
    if updates:
        for update in updates:
            print(f"✅ {update.field_name}: {update.value}")
    
    progress = extractor.get_progress()
    print(f"   Progress: {progress['completed']}/{progress['total']}")
```

---

## UI Update Patterns

### Progressive Rendering

```python
from typing import Callable


@dataclass
class UIState:
    """UI state during streaming."""
    
    status: str  # "idle", "streaming", "complete", "error"
    fields: dict
    pending_fields: List[str]
    progress_percent: float


class StreamingUIController:
    """Control UI updates during streaming."""
    
    def __init__(
        self,
        expected_fields: List[str],
        on_state_change: Optional[Callable[[UIState], None]] = None
    ):
        self.expected_fields = expected_fields
        self.on_state_change = on_state_change
        self.state = UIState(
            status="idle",
            fields={},
            pending_fields=expected_fields.copy(),
            progress_percent=0.0
        )
    
    def start_streaming(self):
        """Called when streaming begins."""
        
        self.state = UIState(
            status="streaming",
            fields={},
            pending_fields=self.expected_fields.copy(),
            progress_percent=0.0
        )
        self._notify()
    
    def update_field(self, field_name: str, value: any):
        """Called when a field is completed."""
        
        self.state.fields[field_name] = value
        if field_name in self.state.pending_fields:
            self.state.pending_fields.remove(field_name)
        
        self.state.progress_percent = (
            len(self.state.fields) / len(self.expected_fields) * 100
        )
        self._notify()
    
    def complete(self, final_data: dict):
        """Called when streaming completes."""
        
        self.state = UIState(
            status="complete",
            fields=final_data,
            pending_fields=[],
            progress_percent=100.0
        )
        self._notify()
    
    def error(self, message: str):
        """Called on error."""
        
        self.state.status = "error"
        self._notify()
    
    def _notify(self):
        """Notify state change."""
        
        if self.on_state_change:
            self.on_state_change(self.state)


# Example UI update handler
def render_ui(state: UIState):
    """Render UI based on state."""
    
    print(f"\n{'=' * 40}")
    print(f"Status: {state.status.upper()}")
    print(f"Progress: {state.progress_percent:.0f}%")
    
    print("\nFields:")
    for field in ["name", "age", "email"]:  # Fixed order for demo
        if field in state.fields:
            print(f"  ✅ {field}: {state.fields[field]}")
        else:
            print(f"  ⏳ {field}: (waiting...)")


# Simulate streaming with UI updates
print("\n\nUI Update Pattern Demo")
print("=" * 60)

controller = StreamingUIController(
    expected_fields=["name", "age", "email"],
    on_state_change=render_ui
)

# Simulate streaming
controller.start_streaming()

import time
# Simulate field completions
controller.update_field("name", "Bob Smith")
controller.update_field("age", 35)
controller.update_field("email", "bob@example.com")

controller.complete({"name": "Bob Smith", "age": 35, "email": "bob@example.com"})
```

### React Component Pattern

```javascript
// React pattern for streaming structured outputs (conceptual)

/*
import { useState, useEffect } from 'react';

function StreamingExtraction({ prompt, schema }) {
  const [state, setState] = useState({
    status: 'idle',
    fields: {},
    error: null
  });

  async function startExtraction() {
    setState({ status: 'streaming', fields: {}, error: null });

    try {
      const stream = await openai.beta.chat.completions.stream({
        model: 'gpt-4o',
        messages: [{ role: 'user', content: prompt }],
        response_format: {
          type: 'json_schema',
          json_schema: { name: 'extraction', schema, strict: true }
        }
      });

      for await (const chunk of stream) {
        // Parse partial JSON
        const partial = parsePartialJSON(chunk.snapshot);
        setState(prev => ({
          ...prev,
          fields: { ...prev.fields, ...partial }
        }));
      }

      const final = await stream.finalContent();
      setState({ 
        status: 'complete', 
        fields: JSON.parse(final),
        error: null 
      });

    } catch (error) {
      setState({ 
        status: 'error', 
        fields: {}, 
        error: error.message 
      });
    }
  }

  return (
    <div>
      <button onClick={startExtraction}>
        {state.status === 'streaming' ? 'Extracting...' : 'Extract'}
      </button>
      
      <div className="fields">
        {Object.entries(state.fields).map(([key, value]) => (
          <div key={key} className="field">
            <span className="label">{key}:</span>
            <span className="value">{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
*/

console.log("React streaming pattern example (see code comments)");
```

---

## Delta Accumulation

### Robust Accumulator

```python
@dataclass
class AccumulatorState:
    """State of the delta accumulator."""
    
    raw_content: str
    parsed_content: Optional[dict]
    chunk_count: int
    byte_count: int
    last_update_time: float


class DeltaAccumulator:
    """Accumulate streaming deltas into complete response."""
    
    def __init__(self):
        self._content = ""
        self._chunks: List[str] = []
        self._start_time: Optional[float] = None
    
    def start(self):
        """Start accumulation."""
        
        self._content = ""
        self._chunks = []
        import time
        self._start_time = time.time()
    
    def add_delta(self, delta: str) -> AccumulatorState:
        """Add a delta and return current state."""
        
        self._content += delta
        self._chunks.append(delta)
        
        # Try parsing
        parsed = None
        try:
            parsed = json.loads(self._content)
        except json.JSONDecodeError:
            pass
        
        import time
        return AccumulatorState(
            raw_content=self._content,
            parsed_content=parsed,
            chunk_count=len(self._chunks),
            byte_count=len(self._content.encode('utf-8')),
            last_update_time=time.time() - (self._start_time or 0)
        )
    
    def finalize(self) -> dict:
        """Finalize and return parsed result."""
        
        try:
            return json.loads(self._content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse final content: {e}")
    
    def get_stats(self) -> dict:
        """Get accumulation statistics."""
        
        import time
        
        return {
            "total_chunks": len(self._chunks),
            "total_bytes": len(self._content.encode('utf-8')),
            "duration_seconds": time.time() - (self._start_time or 0),
            "avg_chunk_size": (
                len(self._content) / len(self._chunks)
                if self._chunks else 0
            )
        }


# Test accumulator
print("\n\nDelta Accumulation")
print("=" * 60)

accumulator = DeltaAccumulator()
accumulator.start()

deltas = [
    '{"results": [',
    '{"id": 1, "name": "Item 1"},',
    '{"id": 2, "name": "Item 2"}',
    ']}'
]

for delta in deltas:
    state = accumulator.add_delta(delta)
    
    parsed_status = "✅ Valid JSON" if state.parsed_content else "⏳ Incomplete"
    print(f"Chunk {state.chunk_count}: {parsed_status} ({state.byte_count} bytes)")

final = accumulator.finalize()
stats = accumulator.get_stats()

print(f"\nFinal: {final}")
print(f"Stats: {stats}")
```

---

## Error Handling in Streams

### Stream Error Types

```python
from enum import Enum


class StreamErrorType(Enum):
    """Types of streaming errors."""
    
    CONNECTION_LOST = "connection_lost"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    INVALID_JSON = "invalid_json"
    SCHEMA_MISMATCH = "schema_mismatch"
    REFUSAL = "refusal"


@dataclass
class StreamError:
    """Streaming error with recovery info."""
    
    error_type: StreamErrorType
    message: str
    recoverable: bool
    partial_content: Optional[str]
    recovery_action: str


class RobustStreamHandler:
    """Handle streams with error recovery."""
    
    def __init__(
        self,
        max_retries: int = 3,
        timeout_seconds: float = 30.0
    ):
        self.max_retries = max_retries
        self.timeout = timeout_seconds
        self.accumulator = DeltaAccumulator()
    
    def process_stream(
        self,
        stream_generator,
        on_progress: Optional[Callable] = None
    ) -> dict:
        """Process stream with error handling."""
        
        self.accumulator.start()
        retries = 0
        
        while retries < self.max_retries:
            try:
                for chunk in stream_generator():
                    state = self.accumulator.add_delta(chunk)
                    
                    if on_progress:
                        on_progress(state)
                
                # Success
                return self.accumulator.finalize()
                
            except ConnectionError:
                retries += 1
                error = StreamError(
                    error_type=StreamErrorType.CONNECTION_LOST,
                    message="Connection lost during streaming",
                    recoverable=retries < self.max_retries,
                    partial_content=self.accumulator._content,
                    recovery_action="Retrying..."
                )
                print(f"⚠️ {error.message}: {error.recovery_action}")
                
            except TimeoutError:
                retries += 1
                error = StreamError(
                    error_type=StreamErrorType.TIMEOUT,
                    message="Stream timed out",
                    recoverable=retries < self.max_retries,
                    partial_content=self.accumulator._content,
                    recovery_action="Retrying with backoff..."
                )
                print(f"⚠️ {error.message}: {error.recovery_action}")
        
        raise Exception(f"Stream failed after {self.max_retries} retries")
    
    def handle_partial_failure(
        self,
        partial_content: str,
        expected_schema: dict
    ) -> Optional[dict]:
        """Try to salvage partial content."""
        
        # Try to close incomplete JSON
        attempts = [
            partial_content + "}",
            partial_content + "]}",
            partial_content + "\"}"
        ]
        
        for attempt in attempts:
            try:
                result = json.loads(attempt)
                # Validate against schema (simplified)
                if self._validate_partial(result, expected_schema):
                    return result
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _validate_partial(self, data: dict, schema: dict) -> bool:
        """Check if partial data is useful."""
        
        required = schema.get("required", [])
        present = set(data.keys())
        
        # Consider valid if at least half of required fields present
        return len(present & set(required)) >= len(required) / 2


# Example error handling
print("\n\nRobust Stream Handling")
print("=" * 60)

handler = RobustStreamHandler()

# Simulate partial failure recovery
partial = '{"name": "Alice", "age": 30'
schema = {
    "type": "object",
    "required": ["name", "age", "email"]
}

recovered = handler.handle_partial_failure(partial, schema)
if recovered:
    print(f"✅ Recovered partial data: {recovered}")
else:
    print("❌ Could not recover partial data")
```

---

## Hands-on Exercise

### Your Task

Build a streaming structured output processor with UI updates and error recovery.

### Requirements

1. Accumulate streaming deltas
2. Parse and display fields progressively  
3. Handle stream interruptions
4. Provide progress feedback

<details>
<summary>💡 Hints</summary>

- Use a state machine for stream lifecycle
- Buffer deltas until parseable
- Track which fields have been extracted
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Generator
from enum import Enum
import json
import time


class StreamStatus(Enum):
    """Stream processing status."""
    
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    PARSING = "parsing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StreamState:
    """Complete state of stream processor."""
    
    status: StreamStatus
    raw_content: str = ""
    parsed_fields: Dict[str, Any] = field(default_factory=dict)
    pending_fields: List[str] = field(default_factory=list)
    progress: float = 0.0
    chunk_count: int = 0
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    duration: float = 0.0


class StreamingStructuredProcessor:
    """Process streaming structured outputs with UI updates."""
    
    def __init__(
        self,
        expected_fields: List[str],
        on_state_change: Optional[Callable[[StreamState], None]] = None
    ):
        self.expected_fields = expected_fields
        self.on_state_change = on_state_change
        self.state = StreamState(
            status=StreamStatus.IDLE,
            pending_fields=expected_fields.copy()
        )
    
    def process_stream(
        self,
        stream: Generator[str, None, None]
    ) -> Dict[str, Any]:
        """Process a stream of JSON deltas."""
        
        # Start
        self.state = StreamState(
            status=StreamStatus.CONNECTING,
            pending_fields=self.expected_fields.copy(),
            start_time=time.time()
        )
        self._notify()
        
        try:
            self.state.status = StreamStatus.STREAMING
            self._notify()
            
            for chunk in stream:
                self._process_chunk(chunk)
            
            # Parse final result
            self.state.status = StreamStatus.PARSING
            self._notify()
            
            result = self._finalize()
            
            self.state.status = StreamStatus.COMPLETE
            self.state.parsed_fields = result
            self.state.progress = 100.0
            self.state.pending_fields = []
            self.state.duration = time.time() - (self.state.start_time or 0)
            self._notify()
            
            return result
            
        except Exception as e:
            self.state.status = StreamStatus.ERROR
            self.state.error_message = str(e)
            self._notify()
            raise
    
    def _process_chunk(self, chunk: str):
        """Process a single chunk."""
        
        self.state.raw_content += chunk
        self.state.chunk_count += 1
        
        # Try to extract completed fields
        extracted = self._try_extract_fields()
        
        for field_name, value in extracted.items():
            if field_name not in self.state.parsed_fields:
                self.state.parsed_fields[field_name] = value
                if field_name in self.state.pending_fields:
                    self.state.pending_fields.remove(field_name)
        
        # Update progress
        if self.expected_fields:
            self.state.progress = (
                len(self.state.parsed_fields) / 
                len(self.expected_fields) * 100
            )
        
        self._notify()
    
    def _try_extract_fields(self) -> Dict[str, Any]:
        """Try to extract completed fields from buffer."""
        
        extracted = {}
        
        # Try parsing with closing brace
        for suffix in ["}", "]}", "\"}", "\"}]}"]:
            try:
                test_content = self.state.raw_content + suffix
                data = json.loads(test_content)
                
                if isinstance(data, dict):
                    for field in self.expected_fields:
                        if field in data:
                            extracted[field] = data[field]
                
                break
            except json.JSONDecodeError:
                continue
        
        return extracted
    
    def _finalize(self) -> Dict[str, Any]:
        """Parse the final complete JSON."""
        
        try:
            return json.loads(self.state.raw_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid final JSON: {e}")
    
    def _notify(self):
        """Notify of state change."""
        
        if self.on_state_change:
            self.on_state_change(self.state)


# UI Renderer
def render_state(state: StreamState):
    """Render current state to console."""
    
    status_icons = {
        StreamStatus.IDLE: "⚪",
        StreamStatus.CONNECTING: "🔄",
        StreamStatus.STREAMING: "📡",
        StreamStatus.PARSING: "⚙️",
        StreamStatus.COMPLETE: "✅",
        StreamStatus.ERROR: "❌"
    }
    
    print(f"\n{status_icons[state.status]} {state.status.value.upper()}")
    print(f"Progress: {state.progress:.0f}% | Chunks: {state.chunk_count}")
    
    if state.parsed_fields:
        print("Fields:")
        for name, value in state.parsed_fields.items():
            print(f"  ✓ {name}: {value}")
    
    if state.pending_fields:
        print(f"Pending: {', '.join(state.pending_fields)}")
    
    if state.error_message:
        print(f"Error: {state.error_message}")


# Test the processor
def simulate_stream() -> Generator[str, None, None]:
    """Simulate a streaming response."""
    
    chunks = [
        '{"title": "Stream',
        'ing Tutorial", "',
        'author": "AI Cour',
        'se", "sections": ',
        '["intro", "basics',
        '", "advanced"], "',
        'word_count": 1500}'
    ]
    
    for chunk in chunks:
        yield chunk


print("\nStreaming Structured Output Processor Demo")
print("=" * 60)

processor = StreamingStructuredProcessor(
    expected_fields=["title", "author", "sections", "word_count"],
    on_state_change=render_state
)

result = processor.process_stream(simulate_stream())

print(f"\n\n📋 Final Result:")
print(json.dumps(result, indent=2))
```

</details>

---

## Summary

✅ Streaming delivers partial JSON as chunks during generation  
✅ Delta accumulation builds complete response incrementally  
✅ Extract completed fields progressively for responsive UIs  
✅ Handle stream interruptions with retry and recovery logic  
✅ Final validation ensures schema compliance on completion

**Next:** [Handling Refusals](./08-handling-refusals.md)

---

## Further Reading

- [OpenAI Streaming](https://platform.openai.com/docs/api-reference/streaming) — Streaming API reference
- [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) — SSE specification
- [Partial JSON Parsing](https://github.com/prometheuslabs/partial-json-parser-js) — Partial parsing library
