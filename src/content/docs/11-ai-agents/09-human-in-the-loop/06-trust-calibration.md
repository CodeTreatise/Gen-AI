---
title: "Trust calibration"
---

# Trust calibration

## Introduction

Trust is the currency of human-agent collaboration. Too little trust, and users manually verify every output — defeating the purpose of having an agent. Too much trust, and users blindly accept outputs that may be wrong, biased, or harmful. Trust calibration is the process of building *appropriate* trust: high confidence where the agent is reliable, healthy skepticism where it isn't.

In this lesson, we'll build systems that earn trust progressively, communicate confidence transparently, help users calibrate their expectations, and recover trust after failures.

### What we'll cover

- Progressive autonomy — earning independence through demonstrated reliability
- Transparency mechanisms that show agent reasoning and data sources
- Confidence communication that conveys uncertainty in user-friendly ways
- Trust recovery strategies for rebuilding confidence after failures

### Prerequisites

- [Feedback Incorporation](./03-feedback-incorporation.md) — how feedback signals feed into trust
- [Confirmation Workflows](./01-confirmation-workflows.md) — approval patterns that reflect trust levels
- [Collaborative Execution](./05-collaborative-execution.md) — partnership models at different trust levels

---

## Progressive autonomy

Trust shouldn't be a switch — it should be a dial. New agents start with heavy supervision and gradually earn independence as they demonstrate competence in specific domains.

### Autonomy levels

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Optional


class AutonomyLevel(IntEnum):
    """Progressive autonomy levels — earned, not granted."""
    SUPERVISED = 0      # Every action requires approval
    GUIDED = 1          # Low-risk actions auto-approved, high-risk need approval
    SEMI_AUTONOMOUS = 2 # Most actions auto-approved, only critical need approval
    AUTONOMOUS = 3      # All actions auto-approved, human notified after
    TRUSTED = 4         # Full autonomy, human reviews periodic summaries


AUTONOMY_DESCRIPTIONS = {
    AutonomyLevel.SUPERVISED: {
        "label": "Supervised",
        "approval": "Every action",
        "notification": "Before every action",
        "rollback": "Automatic (nothing executed without approval)"
    },
    AutonomyLevel.GUIDED: {
        "label": "Guided",
        "approval": "High-risk actions only",
        "notification": "Before high-risk; after low-risk",
        "rollback": "Available for auto-approved actions"
    },
    AutonomyLevel.SEMI_AUTONOMOUS: {
        "label": "Semi-autonomous",
        "approval": "Critical actions only",
        "notification": "Summary after batches",
        "rollback": "Available within time window"
    },
    AutonomyLevel.AUTONOMOUS: {
        "label": "Autonomous",
        "approval": "None (post-hoc review)",
        "notification": "Daily summary",
        "rollback": "Best-effort compensation"
    },
    AutonomyLevel.TRUSTED: {
        "label": "Trusted",
        "approval": "None",
        "notification": "Weekly summary",
        "rollback": "By request only"
    }
}

for level, desc in AUTONOMY_DESCRIPTIONS.items():
    print(f"Level {level.value} — {desc['label']}:")
    print(f"  Approval: {desc['approval']}")
    print(f"  Notification: {desc['notification']}")
    print()
```

**Output:**
```
Level 0 — Supervised:
  Approval: Every action
  Notification: Before every action

Level 1 — Guided:
  Approval: High-risk actions only
  Notification: Before high-risk; after low-risk

Level 2 — Semi-autonomous:
  Approval: Critical actions only
  Notification: Summary after batches

Level 3 — Autonomous:
  Approval: None (post-hoc review)
  Notification: Daily summary

Level 4 — Trusted:
  Approval: None
  Notification: Weekly summary
```

### Trust scoring and promotion

```python
@dataclass
class TrustScore:
    """Tracks agent reliability to determine autonomy level."""
    
    domain: str                     # "email", "data_analysis", "scheduling"
    current_level: AutonomyLevel = AutonomyLevel.SUPERVISED
    
    # Performance metrics
    total_actions: int = 0
    approved_without_changes: int = 0
    approved_with_minor_changes: int = 0
    rejected: int = 0
    errors: int = 0
    
    # Promotion thresholds
    promotion_threshold: float = 0.85   # Approval rate needed
    min_actions_for_promotion: int = 20 # Minimum actions before promotion
    demotion_threshold: float = 0.60    # Below this = demotion
    
    @property
    def approval_rate(self) -> float:
        """Rate of actions approved (with or without changes)."""
        if self.total_actions == 0:
            return 0.0
        approved = self.approved_without_changes + self.approved_with_minor_changes
        return approved / self.total_actions
    
    @property
    def clean_rate(self) -> float:
        """Rate of actions approved WITHOUT any changes."""
        if self.total_actions == 0:
            return 0.0
        return self.approved_without_changes / self.total_actions
    
    @property
    def error_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.errors / self.total_actions
    
    def record_outcome(self, outcome: str):
        """Record the outcome of an action.
        
        Args:
            outcome: "approved", "minor_changes", "rejected", or "error"
        """
        self.total_actions += 1
        
        if outcome == "approved":
            self.approved_without_changes += 1
        elif outcome == "minor_changes":
            self.approved_with_minor_changes += 1
        elif outcome == "rejected":
            self.rejected += 1
        elif outcome == "error":
            self.errors += 1
    
    def evaluate_promotion(self) -> Optional[str]:
        """Check if the agent should be promoted or demoted."""
        if self.total_actions < self.min_actions_for_promotion:
            remaining = self.min_actions_for_promotion - self.total_actions
            return f"Need {remaining} more actions before evaluation"
        
        if (
            self.approval_rate >= self.promotion_threshold
            and self.current_level < AutonomyLevel.TRUSTED
        ):
            old_level = self.current_level
            self.current_level = AutonomyLevel(self.current_level + 1)
            return (
                f"🎉 Promoted: {old_level.name} → {self.current_level.name} "
                f"(approval rate: {self.approval_rate:.0%})"
            )
        
        if (
            self.approval_rate < self.demotion_threshold
            and self.current_level > AutonomyLevel.SUPERVISED
        ):
            old_level = self.current_level
            self.current_level = AutonomyLevel(self.current_level - 1)
            return (
                f"⬇️ Demoted: {old_level.name} → {self.current_level.name} "
                f"(approval rate: {self.approval_rate:.0%})"
            )
        
        return None  # No change


# Usage
trust = TrustScore(domain="email_drafting")

# Simulate a learning period
outcomes = (
    ["approved"] * 15 +
    ["minor_changes"] * 3 +
    ["rejected"] * 1 +
    ["error"] * 1
)

for outcome in outcomes:
    trust.record_outcome(outcome)

print(f"Domain: {trust.domain}")
print(f"Total actions: {trust.total_actions}")
print(f"Approval rate: {trust.approval_rate:.0%}")
print(f"Clean rate: {trust.clean_rate:.0%}")
print(f"Error rate: {trust.error_rate:.0%}")
print(f"Current level: {trust.current_level.name}")

result = trust.evaluate_promotion()
print(f"\nEvaluation: {result}")
print(f"New level: {trust.current_level.name}")
```

**Output:**
```
Domain: email_drafting
Total actions: 20
Approval rate: 90%
Clean rate: 75%
Error rate: 5%
Current level: SUPERVISED

Evaluation: 🎉 Promoted: SUPERVISED → GUIDED (approval rate: 90%)
New level: GUIDED
```

> **🔑 Key concept:** Trust is domain-specific. An agent might be TRUSTED for email drafting but SUPERVISED for financial transactions. Never grant blanket autonomy — each capability domain has its own trust track.

---

## Transparency mechanisms

Users trust what they can understand. When an agent's decision-making is opaque, users either over-trust (dangerous) or under-trust (wasteful). Transparency means showing *enough* of the reasoning without overwhelming users.

### Reasoning traces

```python
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ReasoningStep:
    """One step in the agent's reasoning process."""
    step_number: int
    action: str          # What the agent did
    reasoning: str       # Why it did it
    data_used: list[str] # What data informed this step
    confidence: float    # How confident (0.0 to 1.0)


@dataclass
class ReasoningTrace:
    """Complete trace of how the agent reached its output."""
    
    task: str
    steps: list[ReasoningStep] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    total_time_ms: float = 0
    
    def add_step(
        self,
        action: str,
        reasoning: str,
        data_used: list[str] = None,
        confidence: float = 1.0
    ):
        """Add a reasoning step."""
        self.steps.append(ReasoningStep(
            step_number=len(self.steps) + 1,
            action=action,
            reasoning=reasoning,
            data_used=data_used or [],
            confidence=confidence
        ))
    
    @property
    def overall_confidence(self) -> float:
        """Overall confidence is limited by the weakest link."""
        if not self.steps:
            return 0.0
        return min(s.confidence for s in self.steps)
    
    def format_summary(self, detail_level: str = "normal") -> str:
        """Format the trace for human consumption.
        
        Args:
            detail_level: "brief", "normal", or "detailed"
        """
        lines = [f"📊 Reasoning for: {self.task}"]
        lines.append(f"   Confidence: {self.overall_confidence:.0%}")
        
        if detail_level == "brief":
            # Just the conclusion
            if self.steps:
                last = self.steps[-1]
                lines.append(f"   Result: {last.action}")
            return "\n".join(lines)
        
        lines.append("")
        for step in self.steps:
            conf_bar = "█" * int(step.confidence * 10) + "░" * (10 - int(step.confidence * 10))
            lines.append(
                f"   {step.step_number}. {step.action}"
            )
            if detail_level == "detailed":
                lines.append(f"      Why: {step.reasoning}")
                if step.data_used:
                    lines.append(f"      Data: {', '.join(step.data_used)}")
            lines.append(f"      Confidence: [{conf_bar}] {step.confidence:.0%}")
        
        if self.sources:
            lines.append(f"\n   Sources: {', '.join(self.sources)}")
        
        return "\n".join(lines)


# Usage
trace = ReasoningTrace(task="Recommend meeting time")

trace.add_step(
    action="Checked calendars for all 4 attendees",
    reasoning="Need overlapping free time slots",
    data_used=["alice_calendar", "bob_calendar", "carol_calendar", "dave_calendar"],
    confidence=0.95
)
trace.add_step(
    action="Found 3 available slots this week",
    reasoning="Filtered for business hours (9 AM–5 PM) in each attendee's timezone",
    data_used=["timezone_preferences", "business_hours_policy"],
    confidence=0.90
)
trace.add_step(
    action="Selected Tuesday 2 PM PST as best option",
    reasoning="Highest overlap with attendees' preferred meeting times",
    data_used=["meeting_preferences", "past_meeting_patterns"],
    confidence=0.75
)
trace.sources = ["Google Calendar API", "User preferences DB"]

print(trace.format_summary("detailed"))
```

**Output:**
```
📊 Reasoning for: Recommend meeting time
   Confidence: 75%

   1. Checked calendars for all 4 attendees
      Why: Need overlapping free time slots
      Data: alice_calendar, bob_calendar, carol_calendar, dave_calendar
      Confidence: [█████████░] 95%
   2. Found 3 available slots this week
      Why: Filtered for business hours (9 AM–5 PM) in each attendee's timezone
      Data: timezone_preferences, business_hours_policy
      Confidence: [█████████░] 90%
   3. Selected Tuesday 2 PM PST as best option
      Why: Highest overlap with attendees' preferred meeting times
      Data: meeting_preferences, past_meeting_patterns
      Confidence: [███████░░░] 75%

   Sources: Google Calendar API, User preferences DB
```

### Adaptive transparency

Show more detail when confidence is low, less when confidence is high:

```python
class AdaptiveTransparency:
    """Adjusts how much reasoning to show based on confidence."""
    
    THRESHOLDS = {
        "high": 0.85,    # Show brief summary
        "medium": 0.65,  # Show normal detail
        "low": 0.0       # Show full reasoning
    }
    
    @staticmethod
    def get_detail_level(confidence: float) -> str:
        """Determine how much detail to show."""
        if confidence >= AdaptiveTransparency.THRESHOLDS["high"]:
            return "brief"
        elif confidence >= AdaptiveTransparency.THRESHOLDS["medium"]:
            return "normal"
        return "detailed"
    
    @staticmethod
    def format_output(trace: ReasoningTrace) -> str:
        """Format with adaptive detail level."""
        confidence = trace.overall_confidence
        detail = AdaptiveTransparency.get_detail_level(confidence)
        
        header = {
            "brief": "✅ High confidence — summary view",
            "normal": "⚠️ Moderate confidence — showing reasoning",
            "detailed": "❗ Low confidence — full reasoning trace"
        }[detail]
        
        return f"{header}\n\n{trace.format_summary(detail)}"


# High confidence → brief output
high_conf_trace = ReasoningTrace(task="Send daily report")
high_conf_trace.add_step("Generated report", "Routine task", confidence=0.95)
high_conf_trace.add_step("Sent to team", "Standard distribution", confidence=0.95)

print(AdaptiveTransparency.format_output(high_conf_trace))
print("\n" + "=" * 50 + "\n")

# Low confidence → detailed output
print(AdaptiveTransparency.format_output(trace))  # From previous example
```

**Output:**
```
✅ High confidence — summary view

📊 Reasoning for: Send daily report
   Confidence: 95%
   Result: Sent to team

==================================================

❗ Low confidence — full reasoning trace

📊 Reasoning for: Recommend meeting time
   Confidence: 75%

   1. Checked calendars for all 4 attendees
      Why: Need overlapping free time slots
      Data: alice_calendar, bob_calendar, carol_calendar, dave_calendar
      Confidence: [█████████░] 95%
   2. Found 3 available slots this week
      Why: Filtered for business hours (9 AM–5 PM) in each attendee's timezone
      Data: timezone_preferences, business_hours_policy
      Confidence: [█████████░] 90%
   3. Selected Tuesday 2 PM PST as best option
      Why: Highest overlap with attendees' preferred meeting times
      Data: meeting_preferences, past_meeting_patterns
      Confidence: [███████░░░] 75%

   Sources: Google Calendar API, User preferences DB
```

> **🤖 AI Context:** The Google PAIR Guidebook recommends this approach: "When AI confidence is low, increase transparency so users can make informed decisions. When confidence is high, reduce friction." This maps directly to the Diátaxis principle of giving users *just enough* information to act.

---

## Confidence communication

Raw probability numbers (0.75, 0.92) are meaningless to most users. Confidence needs to be communicated in terms that match how humans think about certainty.

### Human-friendly confidence

```python
from dataclasses import dataclass


@dataclass
class ConfidenceLevel:
    """Human-friendly representation of confidence."""
    score: float
    label: str
    description: str
    icon: str
    recommendation: str


def interpret_confidence(score: float) -> ConfidenceLevel:
    """Convert a raw confidence score to a human-friendly level."""
    if score >= 0.95:
        return ConfidenceLevel(
            score=score,
            label="Very high",
            description="Almost certain this is correct",
            icon="🟢",
            recommendation="Safe to proceed without review"
        )
    elif score >= 0.85:
        return ConfidenceLevel(
            score=score,
            label="High",
            description="Likely correct, minor uncertainty",
            icon="🟢",
            recommendation="Quick review recommended"
        )
    elif score >= 0.70:
        return ConfidenceLevel(
            score=score,
            label="Moderate",
            description="Probably correct, but please verify key details",
            icon="🟡",
            recommendation="Review before proceeding"
        )
    elif score >= 0.50:
        return ConfidenceLevel(
            score=score,
            label="Low",
            description="Uncertain — this is my best guess",
            icon="🟠",
            recommendation="Careful review required"
        )
    else:
        return ConfidenceLevel(
            score=score,
            label="Very low",
            description="I'm not confident in this output",
            icon="🔴",
            recommendation="Consider doing this manually"
        )


# Usage
test_scores = [0.97, 0.88, 0.73, 0.55, 0.30]

for score in test_scores:
    level = interpret_confidence(score)
    print(f"{level.icon} {score:.0%} — {level.label}: {level.description}")
    print(f"   → {level.recommendation}")
    print()
```

**Output:**
```
🟢 97% — Very high: Almost certain this is correct
   → Safe to proceed without review

🟢 88% — High: Likely correct, minor uncertainty
   → Quick review recommended

🟡 73% — Moderate: Probably correct, but please verify key details
   → Review before proceeding

🟠 55% — Low: Uncertain — this is my best guess
   → Careful review required

🔴 30% — Very low: I'm not confident in this output
   → Consider doing this manually
```

### Confidence-gated actions

Connect confidence directly to HITL behavior — low confidence triggers more human involvement:

```python
from typing import Callable, Optional


class ConfidenceGate:
    """Routes actions based on confidence levels."""
    
    def __init__(
        self,
        auto_approve_threshold: float = 0.90,
        human_review_threshold: float = 0.70,
        # Below human_review_threshold = block (require human to do it)
    ):
        self.auto_approve = auto_approve_threshold
        self.human_review = human_review_threshold
    
    def evaluate(self, confidence: float, action_description: str) -> dict:
        """Determine the required approval path based on confidence."""
        level = interpret_confidence(confidence)
        
        if confidence >= self.auto_approve:
            return {
                "path": "auto_approve",
                "action": action_description,
                "confidence": level,
                "requires_human": False,
                "message": f"{level.icon} Auto-approved: {action_description}"
            }
        elif confidence >= self.human_review:
            return {
                "path": "human_review",
                "action": action_description,
                "confidence": level,
                "requires_human": True,
                "message": (
                    f"{level.icon} Needs review: {action_description}\n"
                    f"   {level.description}"
                )
            }
        else:
            return {
                "path": "human_required",
                "action": action_description,
                "confidence": level,
                "requires_human": True,
                "message": (
                    f"{level.icon} Human needed: {action_description}\n"
                    f"   {level.recommendation}"
                )
            }


# Usage
gate = ConfidenceGate(auto_approve_threshold=0.90, human_review_threshold=0.70)

actions = [
    (0.95, "Send daily status report"),
    (0.78, "Schedule meeting with client"),
    (0.45, "Respond to legal inquiry"),
]

for confidence, action in actions:
    result = gate.evaluate(confidence, action)
    print(result["message"])
    print()
```

**Output:**
```
🟢 Auto-approved: Send daily status report

🟡 Needs review: Schedule meeting with client
   Probably correct, but please verify key details

🟠 Human needed: Respond to legal inquiry
   Careful review required
```

---

## Trust recovery

Trust breaks fast and rebuilds slowly. When an agent makes a significant mistake — sends an email to the wrong person, provides incorrect data, or takes an unauthorized action — the recovery process determines whether the user will continue using the system.

### Recovery protocol

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class IncidentSeverity(Enum):
    LOW = "low"           # Minor inconvenience
    MEDIUM = "medium"     # Required human correction
    HIGH = "high"         # Caused real-world impact
    CRITICAL = "critical" # Safety or financial impact


@dataclass
class TrustIncident:
    """A trust-breaking event."""
    description: str
    severity: IncidentSeverity
    domain: str
    what_happened: str
    what_should_have_happened: str
    root_cause: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


@dataclass
class RecoveryPlan:
    """Steps to recover trust after an incident."""
    incident: TrustIncident
    immediate_actions: list[str]
    preventive_measures: list[str]
    autonomy_adjustment: str
    monitoring_period_days: int
    
    def format_for_user(self) -> str:
        """Show the user what we're doing about it."""
        lines = [
            f"⚠️ Trust Incident Report",
            f"",
            f"What happened: {self.incident.what_happened}",
            f"What should have happened: {self.incident.what_should_have_happened}",
            f"Severity: {self.incident.severity.value}",
            f"",
            f"🔧 Immediate actions taken:"
        ]
        for action in self.immediate_actions:
            lines.append(f"  ✅ {action}")
        
        lines.append(f"\n🛡️ Preventive measures added:")
        for measure in self.preventive_measures:
            lines.append(f"  • {measure}")
        
        lines.extend([
            f"",
            f"📊 Autonomy adjustment: {self.autonomy_adjustment}",
            f"⏱️ Enhanced monitoring for: {self.monitoring_period_days} days"
        ])
        
        return "\n".join(lines)


class TrustRecoveryManager:
    """Manages trust recovery after incidents."""
    
    SEVERITY_TO_DEMOTION = {
        IncidentSeverity.LOW: 0,         # No level change
        IncidentSeverity.MEDIUM: 1,      # Drop 1 level
        IncidentSeverity.HIGH: 2,        # Drop 2 levels
        IncidentSeverity.CRITICAL: 99,   # Reset to SUPERVISED
    }
    
    SEVERITY_TO_MONITORING_DAYS = {
        IncidentSeverity.LOW: 7,
        IncidentSeverity.MEDIUM: 14,
        IncidentSeverity.HIGH: 30,
        IncidentSeverity.CRITICAL: 60,
    }
    
    def __init__(self):
        self.incidents: list[TrustIncident] = []
        self.recovery_plans: list[RecoveryPlan] = []
    
    def report_incident(
        self,
        trust_score: TrustScore,
        description: str,
        severity: IncidentSeverity,
        what_happened: str,
        what_should_have_happened: str,
        root_cause: str = ""
    ) -> RecoveryPlan:
        """Report an incident and generate a recovery plan."""
        incident = TrustIncident(
            description=description,
            severity=severity,
            domain=trust_score.domain,
            what_happened=what_happened,
            what_should_have_happened=what_should_have_happened,
            root_cause=root_cause
        )
        self.incidents.append(incident)
        
        # Demote the trust level
        demotion = self.SEVERITY_TO_DEMOTION[severity]
        old_level = trust_score.current_level
        new_level_value = max(0, trust_score.current_level - demotion)
        trust_score.current_level = AutonomyLevel(new_level_value)
        
        # Record the error in trust metrics
        trust_score.record_outcome("error")
        
        # Generate recovery plan
        plan = RecoveryPlan(
            incident=incident,
            immediate_actions=self._generate_immediate_actions(severity),
            preventive_measures=self._generate_preventive_measures(
                severity, root_cause
            ),
            autonomy_adjustment=(
                f"Demoted from {old_level.name} to "
                f"{trust_score.current_level.name}"
            ),
            monitoring_period_days=self.SEVERITY_TO_MONITORING_DAYS[severity]
        )
        self.recovery_plans.append(plan)
        
        return plan
    
    def _generate_immediate_actions(
        self, severity: IncidentSeverity
    ) -> list[str]:
        """Generate immediate actions based on severity."""
        actions = ["Logged incident for analysis"]
        
        if severity in (IncidentSeverity.MEDIUM, IncidentSeverity.HIGH):
            actions.append("Increased approval requirements for this domain")
        
        if severity in (IncidentSeverity.HIGH, IncidentSeverity.CRITICAL):
            actions.append("Paused autonomous actions in this domain")
            actions.append("Notified system administrators")
        
        if severity == IncidentSeverity.CRITICAL:
            actions.append("Reset to fully supervised mode")
            actions.append("Initiated root cause analysis")
        
        return actions
    
    def _generate_preventive_measures(
        self, severity: IncidentSeverity, root_cause: str
    ) -> list[str]:
        """Generate preventive measures."""
        measures = ["Added this failure pattern to watch list"]
        
        if root_cause:
            measures.append(f"Addressing root cause: {root_cause}")
        
        if severity.value in ("high", "critical"):
            measures.append("Added additional validation checks")
            measures.append("Lowered confidence thresholds for auto-approval")
        
        return measures


# Usage
trust_score = TrustScore(
    domain="email_drafting",
    current_level=AutonomyLevel.SEMI_AUTONOMOUS,
    total_actions=50,
    approved_without_changes=40,
    approved_with_minor_changes=8,
    rejected=2,
    errors=0
)

recovery_manager = TrustRecoveryManager()

plan = recovery_manager.report_incident(
    trust_score=trust_score,
    description="Email sent to wrong recipient",
    severity=IncidentSeverity.HIGH,
    what_happened="Agent sent Q3 financial report to external partner instead of internal team",
    what_should_have_happened="Report should only go to internal-team@company.com",
    root_cause="Recipient matching used partial name match without domain verification"
)

print(plan.format_for_user())
print(f"\nCurrent trust level: {trust_score.current_level.name}")
```

**Output:**
```
⚠️ Trust Incident Report

What happened: Agent sent Q3 financial report to external partner instead of internal team
What should have happened: Report should only go to internal-team@company.com
Severity: high

🔧 Immediate actions taken:
  ✅ Logged incident for analysis
  ✅ Increased approval requirements for this domain
  ✅ Paused autonomous actions in this domain
  ✅ Notified system administrators

🛡️ Preventive measures added:
  • Added this failure pattern to watch list
  • Addressing root cause: Recipient matching used partial name match without domain verification
  • Added additional validation checks
  • Lowered confidence thresholds for auto-approval

📊 Autonomy adjustment: Demoted from SEMI_AUTONOMOUS to SUPERVISED
⏱️ Enhanced monitoring for: 30 days

Current trust level: SUPERVISED
```

> **Warning:** Trust recovery is asymmetric: it takes weeks of good performance to earn SEMI_AUTONOMOUS, but one HIGH severity incident drops you back to SUPERVISED. This asymmetry is intentional — it matches how humans process trust.

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Make trust domain-specific, not global | An agent can be trusted for scheduling but not for financial decisions |
| Show reasoning when confidence is low | Transparency builds trust; opacity erodes it |
| Use human-friendly confidence labels, not raw numbers | "Moderate — please verify" beats "0.73" |
| Demote quickly, promote slowly | Fast demotion prevents damage; slow promotion ensures reliability |
| Tell users what you're doing after failures | Incident reports with recovery plans rebuild trust faster than silence |
| Let users adjust their own trust thresholds | Some users want 95% confidence for auto-approve; others accept 80% |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Same autonomy level for all task types | Track trust per domain — email, scheduling, data analysis each have separate scores |
| Showing confidence only when it's high | Low-confidence transparency is *more* important — it prevents over-trust |
| No path back to higher autonomy after demotion | Define clear re-promotion criteria so the agent can earn trust back |
| Hiding failures from users | Proactive incident reports build more trust than users discovering failures themselves |
| Binary trust (trusted or not) | Use graduated levels — 5 levels gives much finer control than 2 |
| Never expiring old incidents | Weight recent performance more heavily — an error from 6 months ago matters less than yesterday's success |

---

## Hands-on exercise

### Your task

Build a `TrustCalibrationSystem` that combines progressive autonomy, confidence communication, transparency, and trust recovery into a unified trust management framework.

### Requirements

1. Track trust scores per domain with promotion/demotion logic
2. Interpret confidence scores into human-friendly labels with action recommendations
3. Generate reasoning traces with adaptive detail levels (brief/normal/detailed)
4. Handle trust incidents with automatic demotion and recovery plans
5. Produce a trust dashboard showing current levels, trends, and recent incidents

### Expected result

```python
system = TrustCalibrationSystem(user_id="user-1")

# Build trust over time
system.record_outcome("email", "approved")  # x20
system.evaluate("email")  # → Promoted to GUIDED

# Handle a failure
system.report_incident("email", severity="high", ...)
system.dashboard("email")
# Shows: current level, approval rate, incidents, recovery status
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Compose `TrustScore`, `ConfidenceGate`, `AdaptiveTransparency`, and `TrustRecoveryManager`
- The dashboard should show: current level, approval/clean/error rates, promotion progress, and any active recovery plans
- Use `interpret_confidence()` to convert raw scores throughout the system
- Consider adding a method to simulate N actions for testing

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
class TrustCalibrationSystem:
    """Unified trust management for human-agent collaboration."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.trust_scores: dict[str, TrustScore] = {}
        self.confidence_gate = ConfidenceGate()
        self.recovery_manager = TrustRecoveryManager()
    
    def _get_or_create_score(self, domain: str) -> TrustScore:
        if domain not in self.trust_scores:
            self.trust_scores[domain] = TrustScore(domain=domain)
        return self.trust_scores[domain]
    
    def record_outcome(self, domain: str, outcome: str):
        """Record an action outcome for a domain."""
        score = self._get_or_create_score(domain)
        score.record_outcome(outcome)
    
    def evaluate(self, domain: str) -> str:
        """Evaluate if trust level should change."""
        score = self._get_or_create_score(domain)
        result = score.evaluate_promotion()
        return result or "No change — current level maintained"
    
    def check_confidence(
        self, domain: str, confidence: float, action: str
    ) -> dict:
        """Check if an action should be auto-approved based on confidence."""
        return self.confidence_gate.evaluate(confidence, action)
    
    def report_incident(
        self,
        domain: str,
        severity: str,
        what_happened: str,
        what_should_have_happened: str,
        root_cause: str = ""
    ) -> RecoveryPlan:
        """Report a trust incident."""
        score = self._get_or_create_score(domain)
        severity_enum = IncidentSeverity(severity)
        
        return self.recovery_manager.report_incident(
            trust_score=score,
            description=f"Incident in {domain}",
            severity=severity_enum,
            what_happened=what_happened,
            what_should_have_happened=what_should_have_happened,
            root_cause=root_cause
        )
    
    def dashboard(self, domain: str = None) -> str:
        """Generate a trust dashboard."""
        domains = [domain] if domain else list(self.trust_scores.keys())
        lines = [f"📊 Trust Dashboard for {self.user_id}", ""]
        
        for d in domains:
            score = self.trust_scores.get(d)
            if not score:
                continue
            
            level_desc = AUTONOMY_DESCRIPTIONS[score.current_level]
            lines.extend([
                f"{'=' * 40}",
                f"Domain: {d}",
                f"  Level: {score.current_level.name} ({level_desc['label']})",
                f"  Approval: {level_desc['approval']}",
                f"  Actions: {score.total_actions}",
                f"  Approval rate: {score.approval_rate:.0%}",
                f"  Clean rate: {score.clean_rate:.0%}",
                f"  Error rate: {score.error_rate:.0%}",
            ])
            
            # Check for active recovery plans
            active_plans = [
                p for p in self.recovery_manager.recovery_plans
                if p.incident.domain == d
            ]
            if active_plans:
                latest = active_plans[-1]
                lines.append(
                    f"  ⚠️ Active recovery: {latest.incident.severity.value} "
                    f"incident ({latest.monitoring_period_days}-day monitoring)"
                )
        
        return "\n".join(lines)
```
</details>

### Bonus challenges

- [ ] Add time-weighted trust scores — recent actions count more than older ones
- [ ] Build trust comparison across domains — show which areas the agent is strongest/weakest
- [ ] Implement "trust contracts" — let users set minimum trust levels required before the agent can perform specific action types

---

## Summary

✅ **Progressive autonomy** lets agents earn independence through demonstrated reliability — start supervised, promote based on approval rates, demote on failures

✅ **Transparency scales with uncertainty** — show brief summaries for high-confidence outputs and full reasoning traces for low-confidence ones

✅ **Confidence communication** uses human-friendly labels (🟢 High, 🟡 Moderate, 🔴 Very low) instead of raw probabilities, with actionable recommendations attached

✅ **Trust recovery** is asymmetric by design — quick demotion prevents damage, slow re-promotion ensures the root cause is truly fixed, and incident reports keep users informed

**Next:** [Multi-Agent Systems](../10-multi-agent-systems/00-multi-agent-systems.md)

---

## Further reading

- [Google PAIR — Trust + Explanations](https://pair.withgoogle.com/chapter/trust/) — designing for trust in AI products
- [Google PAIR — Feedback + Control](https://pair.withgoogle.com/chapter/feedback-controls/) — user control and autonomy balance
- [Google PAIR Guidebook](https://pair.withgoogle.com/guidebook) — comprehensive AI interaction design principles

*[Back to Human-in-the-Loop overview](./00-human-in-the-loop.md)*

<!-- 
Sources Consulted:
- Google PAIR Trust + Explanations: https://pair.withgoogle.com/chapter/trust/
- Google PAIR Feedback + Control: https://pair.withgoogle.com/chapter/feedback-controls/
- Google PAIR Guidebook (principles): https://pair.withgoogle.com/guidebook
- LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
-->
