---
title: "Advanced Streaming Patterns"
---

# Advanced Streaming Patterns

## Introduction

Beyond basic text streaming, modern AI platforms support sophisticated patterns like streaming structured outputs, reasoning traces, real-time voice via WebSocket/WebRTC, and telephony integration. This lesson covers these advanced capabilities.

### What We'll Cover

- Streaming structured outputs (JSON)
- Reasoning token streaming
- WebSocket for Realtime API
- WebRTC for browser voice
- SIP for telephony connections

### Prerequisites

- Core streaming concepts
- Semantic event handling
- Function calling streams

---

## Streaming Structured Outputs

### Partial JSON Streaming

When requesting structured output, you can stream partial JSON as it generates:

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class Article(BaseModel):
    title: str
    summary: str
    tags: list[str]

stream = client.responses.parse(
    model="gpt-4.1",
    input="Write an article about streaming APIs",
    text_format=Article,
    stream=True
)

partial_json = ""
for event in stream:
    if event.type == "response.output_text.delta":
        partial_json += event.delta
        print(event.delta, end="", flush=True)
        
        # Try to parse partial JSON for live preview
        try:
            partial = parse_partial_json(partial_json)
            update_preview(partial)
        except:
            pass

# Final parsed object
print(f"\n\nParsed: {stream.output_parsed}")
```

### Partial JSON Parser

```javascript
class PartialJSONParser {
    constructor() {
        this.buffer = "";
    }
    
    addDelta(delta) {
        this.buffer += delta;
        return this.tryParse();
    }
    
    tryParse() {
        let json = this.buffer;
        
        // Try to complete partial JSON for preview
        const openBraces = (json.match(/{/g) || []).length;
        const closeBraces = (json.match(/}/g) || []).length;
        const openBrackets = (json.match(/\[/g) || []).length;
        const closeBrackets = (json.match(/]/g) || []).length;
        
        // Add missing closures
        let completion = "";
        for (let i = 0; i < openBrackets - closeBrackets; i++) {
            completion = "]" + completion;
        }
        for (let i = 0; i < openBraces - closeBraces; i++) {
            completion = "}" + completion;
        }
        
        // Handle incomplete strings
        const lastQuote = json.lastIndexOf('"');
        const quotesBeforeLast = (json.slice(0, lastQuote).match(/"/g) || []).length;
        if (quotesBeforeLast % 2 === 0) {
            // Odd number of quotes, close the string
            json += '"';
        }
        
        try {
            return JSON.parse(json + completion);
        } catch {
            return null;
        }
    }
    
    getComplete() {
        try {
            return JSON.parse(this.buffer);
        } catch {
            return null;
        }
    }
}

// Usage
const parser = new PartialJSONParser();

for await (const event of stream) {
    if (event.type === "response.output_text.delta") {
        const partial = parser.addDelta(event.delta);
        if (partial) {
            // Update UI with partial data
            updatePreview(partial);
        }
    }
}

const final = parser.getComplete();
console.log("Final:", final);
```

---

## Streaming Reasoning Tokens

### O-series Model Reasoning

```python
stream = client.responses.create(
    model="o3",
    input="What is the sum of the first 100 prime numbers?",
    reasoning={"effort": "high", "summary": "detailed"},
    stream=True
)

reasoning = ""
answer = ""

for event in stream:
    match event.type:
        case "response.output_item.added":
            if event.item.type == "reasoning":
                print("🤔 Reasoning started...")
            elif event.item.type == "message":
                print("\n💡 Answer:")
        
        case "response.reasoning_summary_text.delta":
            reasoning += event.delta
            print(f"  {event.delta}", end="")
        
        case "response.output_text.delta":
            answer += event.delta
            print(event.delta, end="")
        
        case "response.completed":
            print(f"\n\n📊 Stats:")
            print(f"  Reasoning tokens: {event.response.usage.reasoning_tokens}")
            print(f"  Output tokens: {event.response.usage.output_tokens}")
```

### Dual-Panel Reasoning UI

```javascript
class ReasoningStreamUI {
    constructor(elements) {
        this.reasoningPanel = elements.reasoning;
        this.answerPanel = elements.answer;
        this.currentPhase = "idle";
    }
    
    handleEvent(event) {
        switch (event.type) {
            case "response.output_item.added":
                if (event.item.type === "reasoning") {
                    this.currentPhase = "reasoning";
                    this.reasoningPanel.classList.add("active");
                } else if (event.item.type === "message") {
                    this.currentPhase = "answering";
                    this.answerPanel.classList.add("active");
                    this.reasoningPanel.classList.remove("active");
                }
                break;
            
            case "response.reasoning_summary_text.delta":
                this.reasoningPanel.textContent += event.delta;
                this.scrollToBottom(this.reasoningPanel);
                break;
            
            case "response.output_text.delta":
                this.answerPanel.textContent += event.delta;
                this.scrollToBottom(this.answerPanel);
                break;
            
            case "response.completed":
                this.answerPanel.classList.remove("active");
                this.showComplete(event.response.usage);
                break;
        }
    }
    
    scrollToBottom(element) {
        element.scrollTop = element.scrollHeight;
    }
    
    showComplete(usage) {
        const stats = document.createElement("div");
        stats.className = "stats";
        stats.innerHTML = `
            <span>Reasoning: ${usage.reasoning_tokens} tokens</span>
            <span>Output: ${usage.output_tokens} tokens</span>
        `;
        this.answerPanel.appendChild(stats);
    }
}
```

---

## WebSocket for Realtime API

The Realtime API uses WebSocket for bidirectional voice streaming:

### Connection Setup

```javascript
const ws = new WebSocket(
    "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview",
    {
        headers: {
            "Authorization": `Bearer ${API_KEY}`,
            "OpenAI-Beta": "realtime=v1"
        }
    }
);

ws.onopen = () => {
    console.log("Connected to Realtime API");
    
    // Configure session
    ws.send(JSON.stringify({
        type: "session.update",
        session: {
            modalities: ["audio", "text"],
            voice: "alloy",
            input_audio_format: "pcm16",
            output_audio_format: "pcm16",
            turn_detection: {
                type: "server_vad"
            }
        }
    }));
};
```

### Handling Realtime Events

```javascript
ws.onmessage = (message) => {
    const event = JSON.parse(message.data);
    
    switch (event.type) {
        case "session.created":
            console.log("Session ready:", event.session.id);
            break;
        
        case "response.audio.delta":
            // Play audio chunk
            const audioData = base64ToArrayBuffer(event.delta);
            playAudioChunk(audioData);
            break;
        
        case "response.audio_transcript.delta":
            // Show what's being spoken
            transcriptDiv.textContent += event.delta;
            break;
        
        case "response.text.delta":
            // Text response if requested
            textDiv.textContent += event.delta;
            break;
        
        case "input_audio_buffer.speech_started":
            console.log("User started speaking");
            break;
        
        case "input_audio_buffer.speech_stopped":
            console.log("User stopped speaking");
            break;
        
        case "response.done":
            console.log("Response complete");
            break;
        
        case "error":
            console.error("Realtime error:", event.error);
            break;
    }
};
```

### Sending Audio Input

```javascript
class RealtimeAudioClient {
    constructor(ws) {
        this.ws = ws;
        this.mediaRecorder = null;
    }
    
    async startListening() {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 24000
            }
        });
        
        const audioContext = new AudioContext({ sampleRate: 24000 });
        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        processor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            const pcm16 = this.float32ToPCM16(inputData);
            const base64 = this.arrayBufferToBase64(pcm16.buffer);
            
            this.ws.send(JSON.stringify({
                type: "input_audio_buffer.append",
                audio: base64
            }));
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
    }
    
    float32ToPCM16(float32Array) {
        const pcm16 = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return pcm16;
    }
    
    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        return btoa(String.fromCharCode(...bytes));
    }
    
    sendTextMessage(text) {
        this.ws.send(JSON.stringify({
            type: "conversation.item.create",
            item: {
                type: "message",
                role: "user",
                content: [{ type: "input_text", text }]
            }
        }));
        
        this.ws.send(JSON.stringify({
            type: "response.create"
        }));
    }
}
```

---

## WebRTC for Browser Voice

WebRTC provides lower latency for browser-based voice:

```javascript
class WebRTCVoiceClient {
    constructor(options = {}) {
        this.pc = null;
        this.dc = null;  // Data channel
        this.audioElement = options.audioElement;
        this.onTranscript = options.onTranscript || (() => {});
    }
    
    async connect() {
        // Get ephemeral token from your backend
        const { client_secret } = await fetch("/api/realtime/session", {
            method: "POST"
        }).then(r => r.json());
        
        // Create peer connection
        this.pc = new RTCPeerConnection();
        
        // Handle incoming audio
        this.pc.ontrack = (event) => {
            this.audioElement.srcObject = event.streams[0];
        };
        
        // Get user's microphone
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.pc.addTrack(stream.getTracks()[0]);
        
        // Create data channel for events
        this.dc = this.pc.createDataChannel("oai-events");
        this.dc.onmessage = (e) => this.handleEvent(JSON.parse(e.data));
        
        // Create and set local offer
        const offer = await this.pc.createOffer();
        await this.pc.setLocalDescription(offer);
        
        // Exchange SDP with OpenAI
        const response = await fetch("https://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${client_secret.value}`,
                "Content-Type": "application/sdp"
            },
            body: offer.sdp
        });
        
        const answer = await response.text();
        await this.pc.setRemoteDescription({ type: "answer", sdp: answer });
    }
    
    handleEvent(event) {
        switch (event.type) {
            case "response.audio_transcript.delta":
                this.onTranscript(event.delta);
                break;
            
            case "response.done":
                console.log("Response complete");
                break;
        }
    }
    
    sendEvent(event) {
        if (this.dc && this.dc.readyState === "open") {
            this.dc.send(JSON.stringify(event));
        }
    }
    
    disconnect() {
        if (this.pc) {
            this.pc.close();
            this.pc = null;
        }
    }
}

// Usage
const voiceClient = new WebRTCVoiceClient({
    audioElement: document.getElementById("audio-output"),
    onTranscript: (delta) => {
        transcriptDiv.textContent += delta;
    }
});

await voiceClient.connect();
```

---

## SIP for Telephony

Connect AI to phone systems via SIP (Session Initiation Protocol):

```python
# Backend setup for SIP integration
from openai import OpenAI

client = OpenAI()

# Create a phone number capable session
response = client.realtime.sessions.create(
    model="gpt-4o-realtime-preview",
    modalities=["audio"],
    voice="alloy"
)

# Get the SIP URI for phone integration
sip_uri = f"sip:{response.client_secret.id}@realtime.openai.com"

# Configure your SIP provider (Twilio, etc.) to connect to this URI
```

### Twilio Integration Example

```python
from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse, Connect

app = Flask(__name__)

@app.route("/incoming-call", methods=["POST"])
def handle_incoming():
    """Handle incoming phone call and connect to OpenAI Realtime."""
    
    # Create OpenAI session
    session = client.realtime.sessions.create(
        model="gpt-4o-realtime-preview",
        modalities=["audio"],
        voice="alloy",
        instructions="You are a helpful phone assistant for Acme Corp."
    )
    
    response = VoiceResponse()
    
    # Connect call to OpenAI via SIP
    connect = Connect()
    connect.stream(
        url=f"wss://realtime.openai.com/v1/stream/{session.id}",
        name="OpenAI"
    )
    response.append(connect)
    
    return str(response)
```

---

## Streaming with Image Input

### Multimodal Streaming

```python
import base64

# Encode image
with open("chart.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

stream = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_data}"
                },
                {
                    "type": "input_text",
                    "text": "Describe this chart in detail"
                }
            ]
        }
    ],
    stream=True
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

---

## Combined Pattern: Multi-Modal Agent

```javascript
class MultiModalStreamingAgent {
    constructor(options = {}) {
        this.client = options.client;
        this.tools = options.tools || [];
        this.onText = options.onText || (() => {});
        this.onToolCall = options.onToolCall || (() => {});
        this.onAudio = options.onAudio || (() => {});
    }
    
    async *process(input) {
        const stream = await this.client.responses.create({
            model: "gpt-4.1",
            input: input,
            tools: this.tools,
            stream: true
        });
        
        const toolHandler = new StreamingToolCallHandler();
        
        for await (const event of stream) {
            // Text content
            if (event.type === "response.output_text.delta") {
                this.onText(event.delta);
                yield { type: "text", delta: event.delta };
            }
            
            // Tool calls
            toolHandler.handleEvent(event);
            if (event.type === "response.output_item.done" &&
                event.item.type === "function_call") {
                
                const call = toolHandler.getCompletedCalls().pop();
                this.onToolCall(call);
                yield { type: "tool_call", call };
            }
            
            // Audio (if present)
            if (event.type === "response.audio.delta") {
                this.onAudio(event.delta);
                yield { type: "audio", delta: event.delta };
            }
        }
    }
}
```

---

## Hands-on Exercise

### Your Task

Build a streaming structured output handler that provides live preview as JSON generates.

### Requirements

1. Stream JSON structured output
2. Show partial parsed data in real-time
3. Handle incomplete JSON gracefully
4. Display final validated result

### Expected Result

```javascript
// As stream progresses:
// { title: "API Stre..." }       // Partial
// { title: "API Streaming" }      // Field complete
// { title: "API Streaming", summary: "Learn how..." }  // More fields
// { title: "...", summary: "...", tags: ["api", "streaming"] }  // Complete
```

<details>
<summary>💡 Hints</summary>

- Balance unclosed braces/brackets
- Handle strings that are mid-generation
- Use try/catch for parse attempts
</details>

<details>
<summary>✅ Solution</summary>

```javascript
class StructuredStreamHandler {
    constructor(options = {}) {
        this.onPartial = options.onPartial || (() => {});
        this.onComplete = options.onComplete || (() => {});
        
        this.buffer = "";
        this.lastValidPartial = null;
    }
    
    handleDelta(delta) {
        this.buffer += delta;
        
        // Try to parse as complete JSON first
        try {
            const complete = JSON.parse(this.buffer);
            this.lastValidPartial = complete;
            return { status: "partial", data: complete };
        } catch {
            // Try partial parsing
            const partial = this.parsePartial();
            if (partial && this.hasChanged(partial)) {
                this.lastValidPartial = partial;
                this.onPartial(partial);
                return { status: "partial", data: partial };
            }
        }
        
        return null;
    }
    
    parsePartial() {
        let json = this.buffer;
        
        // Handle incomplete string at end
        json = this.closeIncompleteString(json);
        
        // Count and balance brackets/braces
        json = this.balanceDelimiters(json);
        
        try {
            return JSON.parse(json);
        } catch {
            return null;
        }
    }
    
    closeIncompleteString(json) {
        // Count quotes
        const quotes = (json.match(/(?<!\\)"/g) || []).length;
        
        if (quotes % 2 === 1) {
            // Odd number of quotes - close the string
            // But first, remove any incomplete escape sequence
            json = json.replace(/\\$/, "");
            json += '"';
        }
        
        return json;
    }
    
    balanceDelimiters(json) {
        const openBraces = (json.match(/{/g) || []).length;
        const closeBraces = (json.match(/}/g) || []).length;
        const openBrackets = (json.match(/\[/g) || []).length;
        const closeBrackets = (json.match(/]/g) || []).length;
        
        // Remove trailing comma if present
        json = json.replace(/,\s*$/, "");
        
        // Add missing closures
        for (let i = 0; i < openBrackets - closeBrackets; i++) {
            json += "]";
        }
        for (let i = 0; i < openBraces - closeBraces; i++) {
            json += "}";
        }
        
        return json;
    }
    
    hasChanged(newPartial) {
        return JSON.stringify(newPartial) !== JSON.stringify(this.lastValidPartial);
    }
    
    finalize() {
        try {
            const complete = JSON.parse(this.buffer);
            this.onComplete(complete);
            return { status: "complete", data: complete };
        } catch (e) {
            return { status: "error", error: e.message, partial: this.lastValidPartial };
        }
    }
}

// Usage
async function streamStructured() {
    const handler = new StructuredStreamHandler({
        onPartial: (data) => {
            console.log("Preview:", JSON.stringify(data, null, 2));
        },
        onComplete: (data) => {
            console.log("✓ Complete:", data);
        }
    });
    
    const stream = await openai.responses.parse({
        model: "gpt-4.1",
        input: "Write an article about streaming",
        text_format: {
            type: "json_schema",
            json_schema: {
                name: "article",
                schema: {
                    type: "object",
                    properties: {
                        title: { type: "string" },
                        summary: { type: "string" },
                        tags: { type: "array", items: { type: "string" } }
                    },
                    required: ["title", "summary", "tags"]
                }
            }
        },
        stream: true
    });
    
    for await (const event of stream) {
        if (event.type === "response.output_text.delta") {
            handler.handleDelta(event.delta);
        }
    }
    
    const result = handler.finalize();
    console.log("Final result:", result);
}

streamStructured();
```

</details>

---

## Summary

✅ Structured outputs can stream with partial JSON parsing  
✅ Reasoning models stream thinking traces separately from answers  
✅ WebSocket enables bidirectional real-time voice communication  
✅ WebRTC provides low-latency browser voice with peer connections  
✅ SIP integration connects AI to phone systems  
✅ Multi-modal streaming combines text, tools, and audio

---

## Further Reading

- [Realtime API](https://platform.openai.com/docs/guides/realtime) — OpenAI WebSocket docs
- [WebRTC Guide](https://platform.openai.com/docs/guides/realtime-webrtc) — Browser voice setup
- [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — JSON mode
- [Twilio + OpenAI](https://www.twilio.com/docs/voice/tutorials/consume-real-time-media-streams-using-websockets) — Telephony integration

<!-- 
Sources Consulted:
- OpenAI Realtime API: https://platform.openai.com/docs/guides/realtime
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
-->
