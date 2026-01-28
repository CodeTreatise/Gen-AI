---
title: "User-Friendly Error Messages"
---

# User-Friendly Error Messages

## Introduction

Technical API errors confuse users and erode trust. Translating cryptic error codes into helpful, actionable messages improves user experience and reduces support burden.

### What We'll Cover

- Translating error codes to plain language
- Contextual error messages
- Actionable guidance
- Internationalization considerations
- Logging technical details separately

### Prerequisites

- Error response parsing
- Common API errors

---

## The Problem with Technical Errors

```python
# ❌ What users see without translation
{
    "error": "RateLimitError: Rate limit reached for gpt-4.1 in organization org-abc123 on tokens per min (TPM): Limit 40000, Used 39500, Requested 2000."
}

# ✅ What users should see
{
    "message": "Our AI service is experiencing high demand. Please wait a moment and try again.",
    "retry_in": "30 seconds"
}
```

---

## Error Translation Map

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class UserMessage:
    title: str
    message: str
    action: Optional[str] = None
    retry_seconds: Optional[int] = None

ERROR_TRANSLATIONS = {
    # Rate limiting
    "rate_limit_error": UserMessage(
        title="Service Busy",
        message="Our AI assistant is helping many users right now.",
        action="Please wait a moment and try again.",
        retry_seconds=30
    ),
    
    # Authentication
    "authentication_error": UserMessage(
        title="Session Expired",
        message="Your session has expired for security reasons.",
        action="Please log in again to continue."
    ),
    
    # Invalid request
    "invalid_request_error": UserMessage(
        title="Request Issue",
        message="We couldn't process your request.",
        action="Please try rephrasing your message."
    ),
    
    # Content policy
    "content_policy_violation": UserMessage(
        title="Content Guidelines",
        message="Your request may not meet our content guidelines.",
        action="Please modify your request and try again."
    ),
    
    # Server errors
    "server_error": UserMessage(
        title="Temporary Issue",
        message="We're experiencing a temporary technical issue.",
        action="Please try again in a few minutes."
    ),
    
    # Context length
    "context_length_exceeded": UserMessage(
        title="Message Too Long",
        message="Your conversation has become too long to process.",
        action="Please start a new conversation or summarize your request."
    ),
    
    # Timeout
    "timeout": UserMessage(
        title="Response Delayed",
        message="The response is taking longer than expected.",
        action="Please try a shorter or simpler request."
    ),
    
    # Default fallback
    "unknown": UserMessage(
        title="Something Went Wrong",
        message="We encountered an unexpected issue.",
        action="Please try again. If the problem persists, contact support."
    )
}
```

---

## Error Translator Class

```python
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ErrorTranslator:
    """Translate technical errors to user-friendly messages."""
    
    def __init__(self, translations: Dict[str, UserMessage] = None):
        self.translations = translations or ERROR_TRANSLATIONS
    
    def _classify_error(self, error) -> str:
        """Classify error into a translation key."""
        
        error_type = type(error).__name__.lower()
        
        # Check for specific error types
        if "ratelimit" in error_type or "rate_limit" in error_type:
            return "rate_limit_error"
        
        if "authentication" in error_type or "auth" in error_type:
            return "authentication_error"
        
        if "permission" in error_type or "forbidden" in error_type:
            return "authentication_error"
        
        if "invalid" in error_type or "bad_request" in error_type:
            return "invalid_request_error"
        
        if "timeout" in error_type:
            return "timeout"
        
        if "server" in error_type or "internal" in error_type:
            return "server_error"
        
        # Check error message content
        error_msg = str(error).lower()
        
        if "content_policy" in error_msg or "content policy" in error_msg:
            return "content_policy_violation"
        
        if "context_length" in error_msg or "maximum context" in error_msg:
            return "context_length_exceeded"
        
        if "rate limit" in error_msg:
            return "rate_limit_error"
        
        return "unknown"
    
    def translate(self, error) -> Dict[str, Any]:
        """Translate error to user-friendly format."""
        
        error_key = self._classify_error(error)
        translation = self.translations.get(error_key, self.translations["unknown"])
        
        # Log technical details
        logger.error(
            f"API Error [{error_key}]: {type(error).__name__}: {error}",
            extra={"error_class": type(error).__name__, "error_key": error_key}
        )
        
        result = {
            "title": translation.title,
            "message": translation.message,
            "action": translation.action,
            "error_code": error_key
        }
        
        if translation.retry_seconds:
            result["retry_in"] = f"{translation.retry_seconds} seconds"
            result["retry_seconds"] = translation.retry_seconds
        
        return result
    
    def get_display_message(self, error) -> str:
        """Get a single display-ready message."""
        
        translated = self.translate(error)
        parts = [translated["message"]]
        
        if translated.get("action"):
            parts.append(translated["action"])
        
        return " ".join(parts)


# Global translator instance
translator = ErrorTranslator()
```

---

## Contextual Error Messages

### Adding Context to Errors

```python
from enum import Enum

class UserContext(Enum):
    CHAT = "chat"
    DOCUMENT_ANALYSIS = "document"
    CODE_GENERATION = "code"
    IMAGE_GENERATION = "image"
    VOICE = "voice"

CONTEXTUAL_MESSAGES = {
    "rate_limit_error": {
        UserContext.CHAT: "The chat is busy. Your message will be processed shortly.",
        UserContext.DOCUMENT_ANALYSIS: "Document processing is delayed. Please wait.",
        UserContext.CODE_GENERATION: "Code generation is temporarily limited. Please wait.",
        UserContext.IMAGE_GENERATION: "Image generation queue is full. Try again soon.",
        UserContext.VOICE: "Voice processing is busy. Please wait a moment.",
    },
    "context_length_exceeded": {
        UserContext.CHAT: "This conversation is too long. Please start a new chat.",
        UserContext.DOCUMENT_ANALYSIS: "The document is too large. Try a smaller file.",
        UserContext.CODE_GENERATION: "The code context is too large. Reduce scope.",
    }
}

def get_contextual_message(
    error_key: str,
    context: UserContext = UserContext.CHAT
) -> str:
    """Get error message appropriate for the user context."""
    
    context_messages = CONTEXTUAL_MESSAGES.get(error_key, {})
    
    if context in context_messages:
        return context_messages[context]
    
    # Fall back to default translation
    return ERROR_TRANSLATIONS.get(error_key, ERROR_TRANSLATIONS["unknown"]).message
```

### Context-Aware Translator

```python
class ContextualErrorTranslator(ErrorTranslator):
    """Error translator with context awareness."""
    
    def translate_with_context(
        self,
        error,
        context: UserContext = UserContext.CHAT
    ) -> Dict[str, Any]:
        """Translate error with context-specific messaging."""
        
        error_key = self._classify_error(error)
        base_translation = self.translations.get(error_key, self.translations["unknown"])
        
        # Get contextual message if available
        contextual_msg = get_contextual_message(error_key, context)
        
        logger.error(
            f"API Error [{error_key}] in {context.value}: {error}",
            extra={"context": context.value}
        )
        
        return {
            "title": base_translation.title,
            "message": contextual_msg,
            "action": base_translation.action,
            "error_code": error_key,
            "context": context.value
        }
```

---

## Response Wrapper

```python
from typing import Union, TypedDict

class SuccessResponse(TypedDict):
    success: bool
    data: dict

class ErrorResponse(TypedDict):
    success: bool
    error: dict

def safe_api_call(
    client,
    messages: list,
    context: UserContext = UserContext.CHAT,
    **kwargs
) -> Union[SuccessResponse, ErrorResponse]:
    """Execute API call with user-friendly error handling."""
    
    translator = ContextualErrorTranslator()
    
    try:
        response = client.chat.completions.create(
            model=kwargs.get("model", "gpt-4.1"),
            messages=messages,
            **{k: v for k, v in kwargs.items() if k != "model"}
        )
        
        return {
            "success": True,
            "data": {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                }
            }
        }
    
    except Exception as e:
        error_info = translator.translate_with_context(e, context)
        
        return {
            "success": False,
            "error": error_info
        }


# Usage
result = safe_api_call(
    client,
    [{"role": "user", "content": "Analyze this code..."}],
    context=UserContext.CODE_GENERATION
)

if not result["success"]:
    print(f"Error: {result['error']['message']}")
    if result['error'].get('action'):
        print(f"Action: {result['error']['action']}")
```

---

## Toast/Notification Messages

```python
from enum import Enum

class ToastType(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"

def get_toast_notification(error) -> dict:
    """Get notification-style error for UI toast/snackbar."""
    
    translator = ErrorTranslator()
    translated = translator.translate(error)
    
    # Determine toast type based on error severity
    error_key = translated["error_code"]
    
    if error_key in ["rate_limit_error", "timeout"]:
        toast_type = ToastType.WARNING
        duration = 5000  # 5 seconds
    elif error_key in ["server_error", "unknown"]:
        toast_type = ToastType.ERROR
        duration = 8000  # 8 seconds
    else:
        toast_type = ToastType.INFO
        duration = 4000  # 4 seconds
    
    return {
        "type": toast_type.value,
        "title": translated["title"],
        "message": translated["message"],
        "duration": duration,
        "action": translated.get("action"),
        "dismissible": True
    }
```

---

## JavaScript Implementation

```javascript
const ERROR_TRANSLATIONS = {
    rate_limit_error: {
        title: 'Service Busy',
        message: 'Our AI assistant is helping many users right now.',
        action: 'Please wait a moment and try again.',
        retrySeconds: 30
    },
    authentication_error: {
        title: 'Session Expired',
        message: 'Your session has expired for security reasons.',
        action: 'Please log in again to continue.'
    },
    invalid_request_error: {
        title: 'Request Issue',
        message: "We couldn't process your request.",
        action: 'Please try rephrasing your message.'
    },
    content_policy_violation: {
        title: 'Content Guidelines',
        message: 'Your request may not meet our content guidelines.',
        action: 'Please modify your request and try again.'
    },
    server_error: {
        title: 'Temporary Issue',
        message: "We're experiencing a temporary technical issue.",
        action: 'Please try again in a few minutes.'
    },
    unknown: {
        title: 'Something Went Wrong',
        message: 'We encountered an unexpected issue.',
        action: 'Please try again. If the problem persists, contact support.'
    }
};

class ErrorTranslator {
    classifyError(error) {
        const errorType = error.constructor.name.toLowerCase();
        const status = error.status;
        
        if (status === 429) return 'rate_limit_error';
        if (status === 401 || status === 403) return 'authentication_error';
        if (status === 400) return 'invalid_request_error';
        if (status >= 500) return 'server_error';
        
        const message = (error.message || '').toLowerCase();
        if (message.includes('content_policy')) return 'content_policy_violation';
        if (message.includes('rate limit')) return 'rate_limit_error';
        
        return 'unknown';
    }
    
    translate(error) {
        const errorKey = this.classifyError(error);
        const translation = ERROR_TRANSLATIONS[errorKey] || ERROR_TRANSLATIONS.unknown;
        
        // Log technical details
        console.error(`API Error [${errorKey}]:`, error);
        
        const result = {
            title: translation.title,
            message: translation.message,
            action: translation.action,
            errorCode: errorKey
        };
        
        if (translation.retrySeconds) {
            result.retryIn = `${translation.retrySeconds} seconds`;
            result.retrySeconds = translation.retrySeconds;
        }
        
        return result;
    }
    
    getDisplayMessage(error) {
        const translated = this.translate(error);
        return `${translated.message} ${translated.action || ''}`.trim();
    }
}

// React hook example
function useApiWithFriendlyErrors() {
    const translator = new ErrorTranslator();
    
    async function callApi(messages) {
        try {
            const response = await openai.chat.completions.create({
                model: 'gpt-4.1',
                messages
            });
            
            return {
                success: true,
                data: response.choices[0].message.content
            };
        } catch (error) {
            return {
                success: false,
                error: translator.translate(error)
            };
        }
    }
    
    return { callApi };
}
```

---

## Internationalization (i18n)

```python
from typing import Dict

# Translation dictionaries by locale
TRANSLATIONS: Dict[str, Dict[str, UserMessage]] = {
    "en": ERROR_TRANSLATIONS,
    "es": {
        "rate_limit_error": UserMessage(
            title="Servicio Ocupado",
            message="Nuestro asistente de IA está ayudando a muchos usuarios.",
            action="Por favor espere un momento e intente de nuevo.",
            retry_seconds=30
        ),
        "server_error": UserMessage(
            title="Problema Temporal",
            message="Estamos experimentando un problema técnico temporal.",
            action="Por favor intente de nuevo en unos minutos."
        ),
        "unknown": UserMessage(
            title="Algo Salió Mal",
            message="Encontramos un problema inesperado.",
            action="Por favor intente de nuevo."
        )
    },
    "fr": {
        "rate_limit_error": UserMessage(
            title="Service Occupé",
            message="Notre assistant IA aide beaucoup d'utilisateurs en ce moment.",
            action="Veuillez patienter un moment et réessayer.",
            retry_seconds=30
        ),
        # ... more translations
    }
}

class I18nErrorTranslator(ErrorTranslator):
    """Error translator with internationalization support."""
    
    def __init__(self, locale: str = "en"):
        self.locale = locale
        self.translations = TRANSLATIONS.get(locale, TRANSLATIONS["en"])
        self.fallback = TRANSLATIONS["en"]
    
    def translate(self, error) -> Dict[str, Any]:
        """Translate with locale fallback."""
        
        error_key = self._classify_error(error)
        
        # Try locale-specific translation
        translation = self.translations.get(error_key)
        
        # Fall back to English
        if not translation:
            translation = self.fallback.get(error_key, self.fallback["unknown"])
        
        logger.error(f"API Error [{error_key}]: {error}")
        
        return {
            "title": translation.title,
            "message": translation.message,
            "action": translation.action,
            "error_code": error_key,
            "locale": self.locale
        }


# Usage
translator_es = I18nErrorTranslator(locale="es")
result = translator_es.translate(some_error)
```

---

## Hands-on Exercise

### Your Task

Create a custom error translator with contextual messages.

### Requirements

1. Translate at least 4 error types
2. Add context-specific messages for 2 contexts
3. Include retry timing for rate limits
4. Log technical details while showing friendly messages

### Expected Result

```python
translator = CustomErrorTranslator()
result = translator.translate(rate_limit_error, context="image_generation")

print(result["message"])  # "Image generation is busy. Please try again shortly."
print(result["retry_seconds"])  # 30
```

<details>
<summary>💡 Hints</summary>

- Use a nested dict for context-specific messages
- Check context first, then fall back to default
- Return consistent structure with optional fields
</details>

<details>
<summary>✅ Solution</summary>

```python
import logging

logger = logging.getLogger(__name__)

class CustomErrorTranslator:
    def __init__(self):
        self.base_messages = {
            "rate_limit": {
                "title": "Service Busy",
                "message": "Too many requests. Please wait.",
                "retry_seconds": 30
            },
            "auth": {
                "title": "Authentication Required",
                "message": "Please log in to continue."
            },
            "server": {
                "title": "Server Error",
                "message": "Something went wrong on our end."
            },
            "invalid": {
                "title": "Invalid Request",
                "message": "We couldn't process that request."
            }
        }
        
        self.context_messages = {
            "rate_limit": {
                "chat": "Chat is experiencing high volume. Please wait.",
                "image_generation": "Image generation is busy. Please try again shortly.",
                "code": "Code assistance is temporarily limited."
            },
            "server": {
                "image_generation": "Image service is temporarily unavailable.",
                "code": "Code service encountered an error."
            }
        }
    
    def _classify(self, error) -> str:
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        if "rate" in error_str or "ratelimit" in error_type:
            return "rate_limit"
        if "auth" in error_type or "401" in error_str:
            return "auth"
        if "server" in error_type or "500" in error_str:
            return "server"
        return "invalid"
    
    def translate(self, error, context: str = "chat") -> dict:
        # Log technical details
        logger.error(f"Technical error: {type(error).__name__}: {error}")
        
        error_key = self._classify(error)
        base = self.base_messages.get(error_key, self.base_messages["invalid"])
        
        # Get contextual message
        context_msgs = self.context_messages.get(error_key, {})
        message = context_msgs.get(context, base["message"])
        
        result = {
            "title": base["title"],
            "message": message,
            "error_code": error_key,
            "context": context
        }
        
        if "retry_seconds" in base:
            result["retry_seconds"] = base["retry_seconds"]
            result["action"] = f"Try again in {base['retry_seconds']} seconds."
        
        return result


# Test
from openai import RateLimitError

try:
    raise RateLimitError("Rate limit exceeded")
except Exception as e:
    translator = CustomErrorTranslator()
    
    # Test different contexts
    chat_result = translator.translate(e, context="chat")
    print(f"Chat: {chat_result['message']}")
    
    image_result = translator.translate(e, context="image_generation")
    print(f"Image: {image_result['message']}")
    print(f"Retry in: {image_result['retry_seconds']}s")
```

**Output:**
```
Chat: Chat is experiencing high volume. Please wait.
Image: Image generation is busy. Please try again shortly.
Retry in: 30s
```

</details>

---

## Summary

✅ Technical errors should never be shown directly to users  
✅ Map error types to clear, actionable messages  
✅ Use context to tailor messages to user activity  
✅ Include retry timing for rate limits and timeouts  
✅ Log technical details separately for debugging

**Next:** [Circuit Breaker Pattern](./06-circuit-breaker.md)

---

## Further Reading

- [Error Message Guidelines](https://www.nngroup.com/articles/error-message-guidelines/) — Nielsen Norman Group
- [Friendly Error Messages](https://uxwritinghub.com/error-messages/) — UX Writing Hub
- [Microcopy Best Practices](https://uxdesign.cc/microcopy-best-practices-for-error-messages-9e7c7a3f9f9b) — UX Collective

<!-- 
Sources Consulted:
- NN/g Error Message Guidelines: https://www.nngroup.com/articles/error-message-guidelines/
- UX Writing Hub: https://uxwritinghub.com/error-messages/
-->
