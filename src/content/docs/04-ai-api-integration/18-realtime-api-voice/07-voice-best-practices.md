---
title: "Voice Best Practices"
---

# Voice Best Practices

## Introduction

Building effective voice agents requires attention to latency, natural conversation flow, error handling, and cost optimization. These practices help create responsive and engaging voice experiences.

### What We'll Cover

- Latency optimization techniques
- Natural conversation flow design
- Fallback and error handling
- Cost management strategies
- Audio quality optimization

### Prerequisites

- Understanding of Realtime API basics
- Experience with voice agent implementation
- Session management knowledge

---

## Latency Optimization

### Understanding Voice Latency

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import asyncio


class LatencyComponent(Enum):
    """Components contributing to voice latency."""
    
    NETWORK_UPLOAD = "network_upload"
    VAD_DETECTION = "vad_detection"
    TRANSCRIPTION = "transcription"
    LLM_PROCESSING = "llm_processing"
    TTS_GENERATION = "tts_generation"
    NETWORK_DOWNLOAD = "network_download"
    AUDIO_PLAYBACK = "audio_playback"


@dataclass
class LatencyMetrics:
    """Track latency metrics."""
    
    component_times: Dict[str, float] = field(default_factory=dict)
    total_response_time_ms: float = 0
    time_to_first_audio_ms: float = 0
    perceived_latency_ms: float = 0
    
    def add_component(self, component: LatencyComponent, time_ms: float):
        """Add component timing."""
        self.component_times[component.value] = time_ms
    
    def calculate_totals(self):
        """Calculate total metrics."""
        self.total_response_time_ms = sum(self.component_times.values())
        
        # Time to first audio excludes playback
        first_audio_components = [
            LatencyComponent.NETWORK_UPLOAD,
            LatencyComponent.VAD_DETECTION,
            LatencyComponent.TRANSCRIPTION,
            LatencyComponent.LLM_PROCESSING
        ]
        self.time_to_first_audio_ms = sum(
            self.component_times.get(c.value, 0)
            for c in first_audio_components
        )
        
        # Perceived latency = time until user hears response
        self.perceived_latency_ms = (
            self.time_to_first_audio_ms +
            self.component_times.get(LatencyComponent.NETWORK_DOWNLOAD.value, 0)
        )
    
    def get_summary(self) -> dict:
        """Get latency summary."""
        self.calculate_totals()
        return {
            "components": self.component_times,
            "total_ms": self.total_response_time_ms,
            "time_to_first_audio_ms": self.time_to_first_audio_ms,
            "perceived_latency_ms": self.perceived_latency_ms
        }


# Target latency thresholds
LATENCY_TARGETS = {
    "excellent": 300,    # < 300ms perceived latency
    "good": 500,         # < 500ms
    "acceptable": 800,   # < 800ms
    "poor": 1200         # < 1200ms
}


def evaluate_latency(perceived_ms: float) -> str:
    """Evaluate latency quality."""
    for quality, threshold in LATENCY_TARGETS.items():
        if perceived_ms < threshold:
            return quality
    return "unacceptable"


# Example metrics
metrics = LatencyMetrics()
metrics.add_component(LatencyComponent.NETWORK_UPLOAD, 50)
metrics.add_component(LatencyComponent.VAD_DETECTION, 100)
metrics.add_component(LatencyComponent.TRANSCRIPTION, 0)  # Speech-to-speech
metrics.add_component(LatencyComponent.LLM_PROCESSING, 150)
metrics.add_component(LatencyComponent.TTS_GENERATION, 0)  # Native audio
metrics.add_component(LatencyComponent.NETWORK_DOWNLOAD, 50)

summary = metrics.get_summary()
quality = evaluate_latency(summary["perceived_latency_ms"])
print(f"Latency: {summary['perceived_latency_ms']}ms ({quality})")
```

### Latency Optimization Techniques

```python
@dataclass
class OptimizationConfig:
    """Configuration for latency optimization."""
    
    # Connection
    use_websocket: bool = True  # Lower latency than HTTP
    enable_compression: bool = False  # Trade-off: CPU vs bandwidth
    connection_pooling: bool = True
    
    # Audio
    audio_chunk_size_ms: int = 100  # Smaller = lower latency
    sample_rate: int = 24000
    use_pcm16: bool = True  # Native format, no conversion
    
    # VAD
    vad_threshold: float = 0.5
    silence_duration_ms: int = 500  # Lower = faster response
    prefix_padding_ms: int = 300
    
    # Response
    stream_audio: bool = True
    buffer_audio_chunks: int = 2  # Trade-off: smoothness vs latency


class LatencyOptimizer:
    """Apply latency optimization strategies."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.measurements: List[LatencyMetrics] = []
    
    def get_session_config(self) -> dict:
        """Get optimized session configuration."""
        
        return {
            "input_audio_format": "pcm16" if self.config.use_pcm16 else "g711_ulaw",
            "output_audio_format": "pcm16" if self.config.use_pcm16 else "g711_ulaw",
            "turn_detection": {
                "type": "server_vad",
                "threshold": self.config.vad_threshold,
                "silence_duration_ms": self.config.silence_duration_ms,
                "prefix_padding_ms": self.config.prefix_padding_ms
            }
        }
    
    def get_audio_config(self) -> dict:
        """Get optimized audio configuration."""
        
        return {
            "chunk_size_ms": self.config.audio_chunk_size_ms,
            "sample_rate": self.config.sample_rate,
            "channels": 1,
            "bytes_per_chunk": (
                self.config.sample_rate *
                2 *  # 16-bit
                self.config.audio_chunk_size_ms // 1000
            )
        }
    
    def record_measurement(self, metrics: LatencyMetrics):
        """Record latency measurement."""
        self.measurements.append(metrics)
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations based on measurements."""
        
        if not self.measurements:
            return ["No measurements yet"]
        
        recommendations = []
        avg_metrics = self._calculate_averages()
        
        # Check each component
        if avg_metrics.get("network_upload", 0) > 100:
            recommendations.append(
                "High upload latency: Check network connection, "
                "reduce audio chunk size"
            )
        
        if avg_metrics.get("vad_detection", 0) > 200:
            recommendations.append(
                "High VAD latency: Reduce silence_duration_ms, "
                "adjust threshold"
            )
        
        if avg_metrics.get("llm_processing", 0) > 300:
            recommendations.append(
                "High LLM latency: Use shorter instructions, "
                "reduce max_response_output_tokens"
            )
        
        if not recommendations:
            recommendations.append("Latency is within acceptable bounds")
        
        return recommendations
    
    def _calculate_averages(self) -> dict:
        """Calculate average metrics."""
        
        averages = {}
        for component in LatencyComponent:
            values = [
                m.component_times.get(component.value, 0)
                for m in self.measurements
                if component.value in m.component_times
            ]
            if values:
                averages[component.value] = sum(values) / len(values)
        
        return averages


# Usage
optimizer = LatencyOptimizer(OptimizationConfig(
    audio_chunk_size_ms=80,
    silence_duration_ms=400,
    vad_threshold=0.5
))

session_config = optimizer.get_session_config()
print(f"Session config: {session_config}")

audio_config = optimizer.get_audio_config()
print(f"Audio config: {audio_config}")
```

---

## Natural Conversation Flow

### Conversation Design Principles

```python
class ConversationStyle(Enum):
    """Voice conversation styles."""
    
    FORMAL = "formal"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    EMPATHETIC = "empathetic"


@dataclass
class ConversationDesign:
    """Design parameters for natural conversation."""
    
    style: ConversationStyle = ConversationStyle.FRIENDLY
    
    # Timing
    response_delay_ms: int = 200  # Natural pause before responding
    max_response_length: int = 150  # Words per response
    speaking_rate: float = 1.0  # 0.9-1.1 feels natural
    
    # Interaction
    allow_interruptions: bool = True
    use_backchannels: bool = True  # "mm-hmm", "I see"
    acknowledge_silence: bool = True  # "Are you still there?"
    
    # Content
    use_filler_phrases: bool = True  # "Let me think..."
    personalize_responses: bool = True
    remember_context: bool = True


STYLE_INSTRUCTIONS = {
    ConversationStyle.FORMAL: """
        Use formal language and complete sentences.
        Address the user respectfully.
        Avoid contractions and slang.
    """,
    ConversationStyle.FRIENDLY: """
        Be warm and approachable.
        Use natural, conversational language.
        Show genuine interest in helping.
    """,
    ConversationStyle.PROFESSIONAL: """
        Be efficient and clear.
        Focus on providing accurate information.
        Maintain a helpful but businesslike tone.
    """,
    ConversationStyle.CASUAL: """
        Keep it relaxed and easygoing.
        Use everyday language.
        Feel free to use humor when appropriate.
    """,
    ConversationStyle.EMPATHETIC: """
        Show understanding and compassion.
        Acknowledge emotions and concerns.
        Be supportive and reassuring.
    """
}


class ConversationFlowManager:
    """Manage natural conversation flow."""
    
    def __init__(self, design: ConversationDesign = None):
        self.design = design or ConversationDesign()
        
        # State
        self.turn_count = 0
        self.silence_count = 0
        self.last_user_input: Optional[str] = None
        self.context_history: List[str] = []
    
    def get_system_instructions(self) -> str:
        """Get instructions for natural conversation."""
        
        base_instructions = STYLE_INSTRUCTIONS[self.design.style].strip()
        
        additional = []
        
        if self.design.use_backchannels:
            additional.append(
                "Use brief acknowledgments like 'I see' or 'Got it' "
                "to show you're following along."
            )
        
        if self.design.use_filler_phrases:
            additional.append(
                "When thinking, use natural phrases like 'Let me check that' "
                "or 'That's a good question'."
            )
        
        if self.design.max_response_length:
            additional.append(
                f"Keep responses concise, around {self.design.max_response_length} "
                "words or less for voice delivery."
            )
        
        return f"{base_instructions}\n\n" + "\n".join(additional)
    
    def should_acknowledge_silence(self, silence_duration_ms: int) -> bool:
        """Check if we should acknowledge user silence."""
        
        if not self.design.acknowledge_silence:
            return False
        
        # After 10 seconds of silence
        if silence_duration_ms > 10000:
            self.silence_count += 1
            return self.silence_count <= 2  # Only ask twice
        
        return False
    
    def get_silence_prompt(self) -> str:
        """Get prompt for acknowledging silence."""
        
        prompts = [
            "Are you still there?",
            "Take your time, I'm here when you're ready.",
            "Let me know if you need anything else."
        ]
        
        return prompts[min(self.silence_count - 1, len(prompts) - 1)]
    
    def format_for_speech(self, text: str) -> str:
        """Format text for natural speech delivery."""
        
        # Break long sentences
        if len(text) > 200:
            # Add pauses at natural break points
            text = text.replace(". ", "... ")
            text = text.replace(", ", ", ... ")
        
        # Convert numbers for speech
        # e.g., "3" -> "three" for small numbers
        
        return text
    
    def update_context(self, user_input: str, assistant_response: str):
        """Update conversation context."""
        
        self.turn_count += 1
        self.last_user_input = user_input
        self.silence_count = 0
        
        if self.design.remember_context:
            self.context_history.append(f"User: {user_input}")
            self.context_history.append(f"Assistant: {assistant_response}")
            
            # Keep last 10 exchanges
            if len(self.context_history) > 20:
                self.context_history = self.context_history[-20:]


# Usage
design = ConversationDesign(
    style=ConversationStyle.FRIENDLY,
    max_response_length=100,
    allow_interruptions=True
)

flow_manager = ConversationFlowManager(design)
instructions = flow_manager.get_system_instructions()
print(f"Instructions:\n{instructions}")
```

### Voice Prompting Best Practices

```python
@dataclass
class VoicePromptTemplate:
    """Template for voice agent prompts."""
    
    role: str
    greeting: str
    capabilities: List[str]
    constraints: List[str]
    voice_style: str
    
    def to_system_prompt(self) -> str:
        """Generate system prompt."""
        
        capabilities_text = "\n".join(f"- {c}" for c in self.capabilities)
        constraints_text = "\n".join(f"- {c}" for c in self.constraints)
        
        return f"""You are {self.role}.

## Greeting
When the conversation starts, say: "{self.greeting}"

## Capabilities
You can help with:
{capabilities_text}

## Guidelines
{constraints_text}

## Voice Style
{self.voice_style}

## Important
- Keep responses concise for voice delivery
- Use natural, conversational language
- Confirm understanding before taking actions
- Ask clarifying questions when needed
"""


# Example templates
CUSTOMER_SERVICE_TEMPLATE = VoicePromptTemplate(
    role="a friendly customer service representative",
    greeting="Hi there! Thanks for calling. How can I help you today?",
    capabilities=[
        "Answer questions about products and services",
        "Help with order status and tracking",
        "Process returns and exchanges",
        "Connect you with a specialist if needed"
    ],
    constraints=[
        "Never share customer personal information",
        "Always verify identity for account changes",
        "Escalate billing disputes to human agents",
        "Keep responses under 30 seconds of speech"
    ],
    voice_style="Warm, professional, and patient. "
                "Speak clearly at a moderate pace."
)

VIRTUAL_ASSISTANT_TEMPLATE = VoicePromptTemplate(
    role="a helpful virtual assistant",
    greeting="Hello! I'm your virtual assistant. What would you like help with?",
    capabilities=[
        "Set reminders and calendar events",
        "Answer general knowledge questions",
        "Provide weather and news updates",
        "Help with calculations and conversions"
    ],
    constraints=[
        "Don't make promises about future capabilities",
        "Recommend consulting professionals for medical or legal advice",
        "Acknowledge when you don't know something"
    ],
    voice_style="Friendly and efficient. "
                "Get to the point while remaining personable."
)


print("Customer Service Prompt:")
print("=" * 50)
print(CUSTOMER_SERVICE_TEMPLATE.to_system_prompt()[:500] + "...")
```

---

## Error Handling and Fallbacks

### Voice-Specific Error Handling

```python
class VoiceErrorType(Enum):
    """Types of voice-specific errors."""
    
    AUDIO_QUALITY = "audio_quality"
    SPEECH_NOT_DETECTED = "speech_not_detected"
    TRANSCRIPTION_FAILED = "transcription_failed"
    UNDERSTANDING_FAILED = "understanding_failed"
    CONNECTION_LOST = "connection_lost"
    TIMEOUT = "timeout"


@dataclass
class VoiceError:
    """Voice-specific error with recovery info."""
    
    type: VoiceErrorType
    message: str
    recoverable: bool
    user_prompt: str  # What to say to user
    retry_action: Optional[str] = None


VOICE_ERRORS = {
    VoiceErrorType.AUDIO_QUALITY: VoiceError(
        type=VoiceErrorType.AUDIO_QUALITY,
        message="Poor audio quality detected",
        recoverable=True,
        user_prompt="I'm having trouble hearing you clearly. "
                    "Could you speak a bit louder or move closer to your microphone?",
        retry_action="request_repeat"
    ),
    VoiceErrorType.SPEECH_NOT_DETECTED: VoiceError(
        type=VoiceErrorType.SPEECH_NOT_DETECTED,
        message="No speech detected",
        recoverable=True,
        user_prompt="I didn't catch that. Could you please repeat what you said?",
        retry_action="request_repeat"
    ),
    VoiceErrorType.TRANSCRIPTION_FAILED: VoiceError(
        type=VoiceErrorType.TRANSCRIPTION_FAILED,
        message="Failed to transcribe speech",
        recoverable=True,
        user_prompt="I'm sorry, I couldn't understand that. "
                    "Could you try saying it differently?",
        retry_action="request_rephrase"
    ),
    VoiceErrorType.UNDERSTANDING_FAILED: VoiceError(
        type=VoiceErrorType.UNDERSTANDING_FAILED,
        message="Failed to understand intent",
        recoverable=True,
        user_prompt="I'm not sure I understood what you're looking for. "
                    "Could you give me a bit more detail?",
        retry_action="request_clarification"
    ),
    VoiceErrorType.CONNECTION_LOST: VoiceError(
        type=VoiceErrorType.CONNECTION_LOST,
        message="Connection interrupted",
        recoverable=True,
        user_prompt="We got disconnected for a moment. I'm back now. "
                    "Where were we?",
        retry_action="reconnect"
    ),
    VoiceErrorType.TIMEOUT: VoiceError(
        type=VoiceErrorType.TIMEOUT,
        message="Response timeout",
        recoverable=True,
        user_prompt="Sorry for the delay. Let me try that again.",
        retry_action="retry_request"
    )
}


class VoiceErrorHandler:
    """Handle voice-specific errors with graceful recovery."""
    
    def __init__(self):
        self.error_history: List[VoiceError] = []
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
    
    def handle_error(
        self,
        error_type: VoiceErrorType
    ) -> tuple[str, Optional[str]]:
        """Handle error and return user prompt and retry action."""
        
        error = VOICE_ERRORS.get(error_type)
        if not error:
            error = VoiceError(
                type=error_type,
                message="Unknown error",
                recoverable=False,
                user_prompt="I'm experiencing a technical issue. "
                            "Please try again in a moment."
            )
        
        self.error_history.append(error)
        self.consecutive_errors += 1
        
        # Check for repeated errors
        if self.consecutive_errors >= self.max_consecutive_errors:
            return self._get_escalation_prompt(), "escalate"
        
        return error.user_prompt, error.retry_action
    
    def _get_escalation_prompt(self) -> str:
        """Get prompt when too many errors occur."""
        return (
            "I'm sorry, we're having some technical difficulties. "
            "Let me connect you with a team member who can help."
        )
    
    def reset_consecutive(self):
        """Reset consecutive error count on success."""
        self.consecutive_errors = 0
    
    def get_error_rate(self) -> float:
        """Get error rate over last 10 interactions."""
        
        if len(self.error_history) < 10:
            return len(self.error_history) / 10
        
        return len(self.error_history[-10:]) / 10


class FallbackManager:
    """Manage fallback strategies."""
    
    def __init__(self):
        self.fallback_responses = {
            "default": "I'm not sure how to help with that. "
                       "Could you try asking in a different way?",
            "out_of_scope": "That's outside what I can help with. "
                            "Is there something else I can assist you with?",
            "complex_query": "That's a complex question. "
                             "Let me connect you with someone who can help better.",
            "sensitive_topic": "I'd recommend speaking with a professional about that. "
                               "Is there anything else I can help with?"
        }
        
        self.fallback_count = 0
    
    def get_fallback(
        self,
        reason: str = "default"
    ) -> str:
        """Get appropriate fallback response."""
        
        self.fallback_count += 1
        return self.fallback_responses.get(
            reason,
            self.fallback_responses["default"]
        )
    
    def should_escalate(self) -> bool:
        """Check if we should escalate to human."""
        return self.fallback_count >= 3


# Usage
error_handler = VoiceErrorHandler()

# Handle an error
prompt, action = error_handler.handle_error(VoiceErrorType.SPEECH_NOT_DETECTED)
print(f"User prompt: {prompt}")
print(f"Action: {action}")

# Check error rate
print(f"Error rate: {error_handler.get_error_rate():.1%}")
```

---

## Cost Management

### Audio Token Cost Tracking

```python
@dataclass
class AudioCostConfig:
    """Audio pricing configuration."""
    
    # Per million tokens (as of 2024)
    input_audio_per_million: float = 100.00
    output_audio_per_million: float = 200.00
    input_text_per_million: float = 5.00
    output_text_per_million: float = 15.00
    
    # Audio token estimation
    tokens_per_second_audio: int = 166  # Approximate


class CostTracker:
    """Track and optimize API costs."""
    
    def __init__(self, config: AudioCostConfig = None):
        self.config = config or AudioCostConfig()
        
        # Tracking
        self.session_costs: List[dict] = []
        self.total_input_audio_tokens = 0
        self.total_output_audio_tokens = 0
        self.total_input_text_tokens = 0
        self.total_output_text_tokens = 0
    
    def record_usage(
        self,
        input_audio_tokens: int = 0,
        output_audio_tokens: int = 0,
        input_text_tokens: int = 0,
        output_text_tokens: int = 0
    ):
        """Record token usage."""
        
        self.total_input_audio_tokens += input_audio_tokens
        self.total_output_audio_tokens += output_audio_tokens
        self.total_input_text_tokens += input_text_tokens
        self.total_output_text_tokens += output_text_tokens
        
        cost = self._calculate_cost(
            input_audio_tokens,
            output_audio_tokens,
            input_text_tokens,
            output_text_tokens
        )
        
        self.session_costs.append({
            "timestamp": datetime.now().isoformat(),
            "cost_usd": cost,
            "tokens": {
                "input_audio": input_audio_tokens,
                "output_audio": output_audio_tokens,
                "input_text": input_text_tokens,
                "output_text": output_text_tokens
            }
        })
    
    def _calculate_cost(
        self,
        input_audio: int,
        output_audio: int,
        input_text: int,
        output_text: int
    ) -> float:
        """Calculate cost in USD."""
        
        return (
            (input_audio / 1_000_000) * self.config.input_audio_per_million +
            (output_audio / 1_000_000) * self.config.output_audio_per_million +
            (input_text / 1_000_000) * self.config.input_text_per_million +
            (output_text / 1_000_000) * self.config.output_text_per_million
        )
    
    def estimate_audio_cost(self, duration_seconds: float, is_input: bool) -> float:
        """Estimate cost for audio duration."""
        
        tokens = int(duration_seconds * self.config.tokens_per_second_audio)
        
        if is_input:
            return (tokens / 1_000_000) * self.config.input_audio_per_million
        else:
            return (tokens / 1_000_000) * self.config.output_audio_per_million
    
    def get_total_cost(self) -> float:
        """Get total session cost."""
        
        return self._calculate_cost(
            self.total_input_audio_tokens,
            self.total_output_audio_tokens,
            self.total_input_text_tokens,
            self.total_output_text_tokens
        )
    
    def get_cost_breakdown(self) -> dict:
        """Get cost breakdown by category."""
        
        return {
            "input_audio": {
                "tokens": self.total_input_audio_tokens,
                "cost_usd": (self.total_input_audio_tokens / 1_000_000) *
                            self.config.input_audio_per_million
            },
            "output_audio": {
                "tokens": self.total_output_audio_tokens,
                "cost_usd": (self.total_output_audio_tokens / 1_000_000) *
                            self.config.output_audio_per_million
            },
            "input_text": {
                "tokens": self.total_input_text_tokens,
                "cost_usd": (self.total_input_text_tokens / 1_000_000) *
                            self.config.input_text_per_million
            },
            "output_text": {
                "tokens": self.total_output_text_tokens,
                "cost_usd": (self.total_output_text_tokens / 1_000_000) *
                            self.config.output_text_per_million
            },
            "total_cost_usd": self.get_total_cost()
        }


class CostOptimizer:
    """Strategies for cost optimization."""
    
    def __init__(self, tracker: CostTracker):
        self.tracker = tracker
        self.budget_limit_usd: Optional[float] = None
    
    def set_budget(self, limit_usd: float):
        """Set session budget limit."""
        self.budget_limit_usd = limit_usd
    
    def is_within_budget(self) -> bool:
        """Check if within budget."""
        
        if self.budget_limit_usd is None:
            return True
        
        return self.tracker.get_total_cost() < self.budget_limit_usd
    
    def get_recommendations(self) -> List[str]:
        """Get cost optimization recommendations."""
        
        recommendations = []
        breakdown = self.tracker.get_cost_breakdown()
        
        # Check audio vs text ratio
        audio_cost = (
            breakdown["input_audio"]["cost_usd"] +
            breakdown["output_audio"]["cost_usd"]
        )
        text_cost = (
            breakdown["input_text"]["cost_usd"] +
            breakdown["output_text"]["cost_usd"]
        )
        
        if audio_cost > text_cost * 10:
            recommendations.append(
                "Consider using text input when possible to reduce audio costs"
            )
        
        # Check output audio ratio
        if breakdown["output_audio"]["tokens"] > breakdown["input_audio"]["tokens"] * 2:
            recommendations.append(
                "Reduce response verbosity to lower output audio costs"
            )
        
        # Suggest max tokens if not set
        if self.tracker.total_output_audio_tokens > 50000:
            recommendations.append(
                "Set max_response_output_tokens to limit long responses"
            )
        
        return recommendations or ["Cost profile looks optimized"]


# Usage
tracker = CostTracker()

# Simulate usage
tracker.record_usage(
    input_audio_tokens=5000,   # ~30 seconds of input
    output_audio_tokens=10000, # ~60 seconds of output
    input_text_tokens=100,
    output_text_tokens=500
)

breakdown = tracker.get_cost_breakdown()
print(f"Cost Breakdown:")
for category, data in breakdown.items():
    if isinstance(data, dict):
        print(f"  {category}: ${data['cost_usd']:.4f}")

print(f"\nTotal: ${breakdown['total_cost_usd']:.4f}")

# Get optimization recommendations
optimizer = CostOptimizer(tracker)
optimizer.set_budget(1.00)

for rec in optimizer.get_recommendations():
    print(f"  - {rec}")
```

---

## Audio Quality Optimization

### Audio Quality Management

```python
@dataclass
class AudioQualityConfig:
    """Audio quality configuration."""
    
    sample_rate: int = 24000
    bit_depth: int = 16
    channels: int = 1
    
    # Processing
    noise_reduction: bool = True
    echo_cancellation: bool = True
    automatic_gain_control: bool = True
    
    # Thresholds
    min_volume_db: float = -40.0
    max_volume_db: float = -3.0
    target_volume_db: float = -14.0


class AudioQualityAnalyzer:
    """Analyze and optimize audio quality."""
    
    def __init__(self, config: AudioQualityConfig = None):
        self.config = config or AudioQualityConfig()
        self.quality_history: List[dict] = []
    
    def analyze_audio_chunk(self, audio_data: bytes) -> dict:
        """Analyze audio chunk quality."""
        
        # Calculate basic metrics
        # In practice, use proper audio analysis libraries
        
        # Estimate volume from sample amplitude
        if len(audio_data) < 2:
            return {"error": "Audio too short"}
        
        # Simple RMS calculation for 16-bit audio
        samples = []
        for i in range(0, len(audio_data) - 1, 2):
            sample = int.from_bytes(
                audio_data[i:i+2],
                byteorder='little',
                signed=True
            )
            samples.append(sample)
        
        if not samples:
            return {"error": "No samples"}
        
        # RMS volume
        rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
        
        # Convert to dB (with small offset to avoid log(0))
        import math
        volume_db = 20 * math.log10(max(rms, 1) / 32768)
        
        # Determine quality
        quality = "good"
        issues = []
        
        if volume_db < self.config.min_volume_db:
            quality = "poor"
            issues.append("Volume too low")
        elif volume_db > self.config.max_volume_db:
            quality = "poor"
            issues.append("Possible clipping")
        
        result = {
            "volume_db": volume_db,
            "sample_count": len(samples),
            "quality": quality,
            "issues": issues
        }
        
        self.quality_history.append(result)
        return result
    
    def get_quality_summary(self) -> dict:
        """Get quality summary over session."""
        
        if not self.quality_history:
            return {"status": "no_data"}
        
        valid = [q for q in self.quality_history if "error" not in q]
        
        if not valid:
            return {"status": "all_errors"}
        
        volumes = [q["volume_db"] for q in valid]
        quality_counts = {}
        for q in valid:
            qual = q["quality"]
            quality_counts[qual] = quality_counts.get(qual, 0) + 1
        
        return {
            "total_chunks": len(valid),
            "avg_volume_db": sum(volumes) / len(volumes),
            "min_volume_db": min(volumes),
            "max_volume_db": max(volumes),
            "quality_distribution": quality_counts
        }
    
    def get_improvement_suggestions(self) -> List[str]:
        """Get audio quality improvement suggestions."""
        
        summary = self.get_quality_summary()
        
        if summary.get("status") in ["no_data", "all_errors"]:
            return ["No audio data to analyze"]
        
        suggestions = []
        
        if summary["avg_volume_db"] < -30:
            suggestions.append(
                "Average volume is low. Ask user to speak louder or "
                "move closer to microphone."
            )
        
        if summary["max_volume_db"] > -6:
            suggestions.append(
                "Possible audio clipping detected. Reduce input gain or "
                "ask user to speak softer."
            )
        
        poor_ratio = summary["quality_distribution"].get("poor", 0) / summary["total_chunks"]
        if poor_ratio > 0.3:
            suggestions.append(
                f"{poor_ratio:.0%} of audio had quality issues. "
                "Check microphone and environment."
            )
        
        return suggestions or ["Audio quality is acceptable"]


# Usage
analyzer = AudioQualityAnalyzer()

# Analyze sample audio
sample_audio = bytes([0x00, 0x10] * 1000)  # Simulated audio
result = analyzer.analyze_audio_chunk(sample_audio)
print(f"Analysis: {result}")

# Get suggestions
for suggestion in analyzer.get_improvement_suggestions():
    print(f"  - {suggestion}")
```

---

## Hands-on Exercise

### Your Task

Build a voice agent quality monitor that tracks and optimizes all aspects.

### Requirements

1. Monitor latency metrics
2. Track conversation quality
3. Handle errors gracefully
4. Manage costs
5. Analyze audio quality

<details>
<summary>💡 Hints</summary>

- Aggregate metrics across turns
- Calculate running averages
- Set alerting thresholds
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import json


class VoiceAgentQualityMonitor:
    """Complete quality monitoring for voice agents."""
    
    def __init__(self):
        # Components
        self.latency_optimizer = LatencyOptimizer()
        self.flow_manager = ConversationFlowManager()
        self.error_handler = VoiceErrorHandler()
        self.fallback_manager = FallbackManager()
        self.cost_tracker = CostTracker()
        self.audio_analyzer = AudioQualityAnalyzer()
        
        # Thresholds
        self.thresholds = {
            "max_latency_ms": 800,
            "max_error_rate": 0.2,
            "max_cost_per_turn_usd": 0.10,
            "min_audio_quality_ratio": 0.8
        }
        
        # Tracking
        self.turn_metrics: List[dict] = []
        self.alerts: List[dict] = []
    
    def start_turn(self) -> str:
        """Start tracking a new turn."""
        turn_id = f"turn_{len(self.turn_metrics) + 1}"
        self.turn_metrics.append({
            "turn_id": turn_id,
            "started_at": datetime.now().isoformat(),
            "latency": {},
            "cost": {},
            "errors": [],
            "audio_quality": None
        })
        return turn_id
    
    def record_latency(
        self,
        turn_id: str,
        component: LatencyComponent,
        time_ms: float
    ):
        """Record latency for a turn."""
        
        for turn in self.turn_metrics:
            if turn["turn_id"] == turn_id:
                turn["latency"][component.value] = time_ms
                
                # Check threshold
                total = sum(turn["latency"].values())
                if total > self.thresholds["max_latency_ms"]:
                    self._add_alert(
                        "latency",
                        f"Turn {turn_id} latency {total}ms exceeds threshold"
                    )
                break
    
    def record_audio_quality(
        self,
        turn_id: str,
        audio_data: bytes
    ):
        """Record audio quality for a turn."""
        
        result = self.audio_analyzer.analyze_audio_chunk(audio_data)
        
        for turn in self.turn_metrics:
            if turn["turn_id"] == turn_id:
                turn["audio_quality"] = result
                
                if result.get("quality") == "poor":
                    self._add_alert(
                        "audio_quality",
                        f"Turn {turn_id} has poor audio quality"
                    )
                break
    
    def record_error(
        self,
        turn_id: str,
        error_type: VoiceErrorType
    ) -> str:
        """Record error and get user prompt."""
        
        prompt, action = self.error_handler.handle_error(error_type)
        
        for turn in self.turn_metrics:
            if turn["turn_id"] == turn_id:
                turn["errors"].append({
                    "type": error_type.value,
                    "action": action
                })
                break
        
        # Check error rate
        if self.error_handler.get_error_rate() > self.thresholds["max_error_rate"]:
            self._add_alert(
                "error_rate",
                f"Error rate {self.error_handler.get_error_rate():.1%} exceeds threshold"
            )
        
        return prompt
    
    def record_cost(
        self,
        turn_id: str,
        input_audio_tokens: int,
        output_audio_tokens: int
    ):
        """Record cost for a turn."""
        
        self.cost_tracker.record_usage(
            input_audio_tokens=input_audio_tokens,
            output_audio_tokens=output_audio_tokens
        )
        
        # Calculate turn cost
        turn_cost = (
            (input_audio_tokens / 1_000_000) * 100.00 +
            (output_audio_tokens / 1_000_000) * 200.00
        )
        
        for turn in self.turn_metrics:
            if turn["turn_id"] == turn_id:
                turn["cost"] = {
                    "input_tokens": input_audio_tokens,
                    "output_tokens": output_audio_tokens,
                    "cost_usd": turn_cost
                }
                break
        
        if turn_cost > self.thresholds["max_cost_per_turn_usd"]:
            self._add_alert(
                "cost",
                f"Turn {turn_id} cost ${turn_cost:.4f} exceeds threshold"
            )
    
    def complete_turn(self, turn_id: str, success: bool):
        """Complete a turn."""
        
        for turn in self.turn_metrics:
            if turn["turn_id"] == turn_id:
                turn["completed_at"] = datetime.now().isoformat()
                turn["success"] = success
                break
        
        if success:
            self.error_handler.reset_consecutive()
    
    def _add_alert(self, category: str, message: str):
        """Add an alert."""
        self.alerts.append({
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "message": message
        })
    
    def get_session_health(self) -> dict:
        """Get overall session health."""
        
        total_turns = len(self.turn_metrics)
        successful_turns = sum(
            1 for t in self.turn_metrics
            if t.get("success", False)
        )
        
        # Calculate average latency
        latencies = []
        for turn in self.turn_metrics:
            if turn["latency"]:
                latencies.append(sum(turn["latency"].values()))
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        # Calculate audio quality ratio
        quality_summary = self.audio_analyzer.get_quality_summary()
        good_ratio = quality_summary.get(
            "quality_distribution", {}
        ).get("good", 0)
        total_analyzed = quality_summary.get("total_chunks", 1)
        audio_quality_ratio = good_ratio / total_analyzed
        
        # Determine health status
        health_score = 100
        issues = []
        
        if avg_latency > self.thresholds["max_latency_ms"]:
            health_score -= 20
            issues.append("High latency")
        
        if self.error_handler.get_error_rate() > self.thresholds["max_error_rate"]:
            health_score -= 30
            issues.append("High error rate")
        
        if audio_quality_ratio < self.thresholds["min_audio_quality_ratio"]:
            health_score -= 20
            issues.append("Poor audio quality")
        
        return {
            "health_score": max(0, health_score),
            "status": "healthy" if health_score >= 70 else "degraded" if health_score >= 40 else "unhealthy",
            "metrics": {
                "total_turns": total_turns,
                "success_rate": successful_turns / total_turns if total_turns > 0 else 0,
                "avg_latency_ms": avg_latency,
                "error_rate": self.error_handler.get_error_rate(),
                "audio_quality_ratio": audio_quality_ratio,
                "total_cost_usd": self.cost_tracker.get_total_cost()
            },
            "issues": issues,
            "alerts_count": len(self.alerts)
        }
    
    def get_recommendations(self) -> List[str]:
        """Get all optimization recommendations."""
        
        recommendations = []
        
        # Latency recommendations
        recommendations.extend(
            self.latency_optimizer.get_optimization_recommendations()
        )
        
        # Cost recommendations
        cost_optimizer = CostOptimizer(self.cost_tracker)
        recommendations.extend(cost_optimizer.get_recommendations())
        
        # Audio recommendations
        recommendations.extend(
            self.audio_analyzer.get_improvement_suggestions()
        )
        
        # Deduplicate
        return list(set(recommendations))
    
    def export_report(self) -> str:
        """Export full session report."""
        
        return json.dumps({
            "timestamp": datetime.now().isoformat(),
            "health": self.get_session_health(),
            "turn_count": len(self.turn_metrics),
            "cost_breakdown": self.cost_tracker.get_cost_breakdown(),
            "audio_quality": self.audio_analyzer.get_quality_summary(),
            "error_summary": self.error_handler.get_error_rate(),
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "recommendations": self.get_recommendations()
        }, indent=2)


# Usage
monitor = VoiceAgentQualityMonitor()

# Simulate a turn
turn_id = monitor.start_turn()
print(f"Started turn: {turn_id}")

# Record metrics
monitor.record_latency(turn_id, LatencyComponent.NETWORK_UPLOAD, 50)
monitor.record_latency(turn_id, LatencyComponent.VAD_DETECTION, 100)
monitor.record_latency(turn_id, LatencyComponent.LLM_PROCESSING, 200)

# Record audio
sample_audio = bytes([0x00, 0x20] * 500)
monitor.record_audio_quality(turn_id, sample_audio)

# Record cost
monitor.record_cost(turn_id, input_audio_tokens=3000, output_audio_tokens=6000)

# Complete turn
monitor.complete_turn(turn_id, success=True)

# Get health
health = monitor.get_session_health()
print(f"\nSession Health:")
print(f"  Score: {health['health_score']}")
print(f"  Status: {health['status']}")
print(f"  Metrics: {health['metrics']}")

# Get recommendations
print(f"\nRecommendations:")
for rec in monitor.get_recommendations():
    print(f"  - {rec}")
```

</details>

---

## Summary

✅ Optimize latency with smaller chunks and tuned VAD  
✅ Design natural conversation flow with appropriate pauses  
✅ Handle errors gracefully with user-friendly prompts  
✅ Track and optimize costs with token monitoring  
✅ Ensure audio quality with analysis and feedback

**Next:** [Gemini Live API](./08-gemini-live-api.md)

---

## Further Reading

- [OpenAI Realtime Best Practices](https://platform.openai.com/docs/guides/realtime) — Official guidance
- [Voice UX Design](https://design.google/library/conversation-design) — Google's voice design principles
- [Audio Engineering Basics](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) — Web audio concepts
