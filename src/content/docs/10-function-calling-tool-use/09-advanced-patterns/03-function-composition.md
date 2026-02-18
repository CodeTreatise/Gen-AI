---
title: "Function Composition"
---

# Function Composition

## Introduction

Individual tools do one thing well — get weather, look up a customer, calculate a price. But real workflows require combining multiple tools into higher-level operations. **Function composition** is the pattern of building complex functionality from simple, reusable building blocks.

Rather than creating monolithic tools that do everything, we compose small tools into macros (fixed sequences), workflows (conditional sequences), and abstraction layers (that hide complexity from the model).

### What we'll cover

- Why composition beats monolithic tools
- Macro functions: fixed multi-step sequences
- Workflow functions: conditional branching
- Abstraction layers: hiding implementation complexity
- Gemini's compositional function calling

### Prerequisites

- [Lesson 04: Executing Functions](../04-executing-functions/00-executing-functions.md) — Dispatch and results
- [Lesson 07: Multi-Turn Function Calling](../07-multi-turn-function-calling/00-multi-turn-function-calling.md) — Multi-step loops
- [Dynamic Registration](./02-dynamic-registration.md) — Runtime tool management

---

## Monolithic vs. composed tools

Consider a "create order" workflow. The monolithic approach puts everything in one tool:

```python
# ❌ Monolithic — too many responsibilities
def create_order(
    customer_id: str, items: list, 
    discount_code: str, payment_method: str,
    shipping_address: str
) -> dict:
    customer = lookup_customer(customer_id)
    total = calculate_total(items)
    total = apply_discount(total, discount_code)
    validate_address(shipping_address)
    charge = process_payment(total, payment_method)
    return create_shipment(customer, items, shipping_address)
```

The composed approach keeps tools small and lets the model (or your application logic) orchestrate them:

```python
# ✅ Composed — each tool does one thing
def lookup_customer(customer_id: str) -> dict: ...
def calculate_total(items: list) -> dict: ...
def apply_discount(total: float, code: str) -> dict: ...
def validate_address(address: str) -> dict: ...
def process_payment(amount: float, method: str) -> dict: ...
def create_shipment(order: dict) -> dict: ...
```

### Trade-offs

| Aspect | Monolithic | Composed |
|--------|-----------|----------|
| API calls | 1 round trip | Multiple round trips |
| Model complexity | Low (one decision) | Higher (multiple decisions) |
| Reusability | None | High — each tool works standalone |
| Error handling | All or nothing | Granular — fail at specific step |
| Testability | Hard | Easy — unit test each piece |
| Flexibility | Rigid | Mix and match as needed |

> **🔑 Key concept:** OpenAI recommends keeping tools small and combining ones that are always called in sequence. The sweet spot is somewhere between fully monolithic and fully atomic.

---

## Macro functions

A **macro** is a fixed sequence of tool calls packaged as a single higher-level operation. The model calls the macro; your application executes the steps internally.

```python
from typing import Callable


class MacroFunction:
    """Execute a fixed sequence of functions as one operation."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._steps: list[dict] = []
    
    def add_step(
        self,
        name: str,
        handler: Callable,
        arg_mapper: Callable[[dict, dict], dict]
    ) -> "MacroFunction":
        """Add a step to the macro.
        
        Args:
            name: Step identifier for logging
            handler: The function to execute
            arg_mapper: Maps (macro_args, previous_results) 
                        to this step's arguments
        """
        self._steps.append({
            "name": name,
            "handler": handler,
            "arg_mapper": arg_mapper
        })
        return self  # Enable chaining
    
    def execute(self, **macro_args) -> dict:
        """Run all steps in sequence, piping results forward."""
        results = {}
        
        for step in self._steps:
            # Map macro arguments + previous results to step args
            step_args = step["arg_mapper"](macro_args, results)
            
            try:
                result = step["handler"](**step_args)
                results[step["name"]] = result
                print(f"  ✅ {step['name']}: {result}")
            except Exception as e:
                return {
                    "success": False,
                    "failed_step": step["name"],
                    "error": str(e),
                    "completed_steps": list(results.keys())
                }
        
        return {"success": True, "results": results}


# Define simple tool handlers
def lookup_customer(customer_id: str) -> dict:
    return {"name": "Alice", "tier": "gold", "customer_id": customer_id}

def calculate_total(items: list[str]) -> dict:
    prices = {"widget": 29.99, "gadget": 49.99, "gizmo": 19.99}
    total = sum(prices.get(item, 0) for item in items)
    return {"total": total, "item_count": len(items)}

def apply_tier_discount(total: float, tier: str) -> dict:
    discounts = {"gold": 0.15, "silver": 0.10, "bronze": 0.05}
    discount = discounts.get(tier, 0)
    final = total * (1 - discount)
    return {"original": total, "discount_pct": discount, "final": final}


# Build the macro
checkout_macro = (
    MacroFunction(
        name="quick_checkout",
        description="Look up customer, calculate total, apply tier discount."
    )
    .add_step(
        "lookup",
        lookup_customer,
        lambda args, _: {"customer_id": args["customer_id"]}
    )
    .add_step(
        "calculate",
        calculate_total,
        lambda args, _: {"items": args["items"]}
    )
    .add_step(
        "discount",
        apply_tier_discount,
        lambda args, prev: {
            "total": prev["calculate"]["total"],
            "tier": prev["lookup"]["tier"]
        }
    )
)

# Execute
result = checkout_macro.execute(
    customer_id="C-42",
    items=["widget", "gadget", "gizmo"]
)
print(f"\nFinal: ${result['results']['discount']['final']:.2f}")
```

**Output:**
```
  ✅ lookup: {'name': 'Alice', 'tier': 'gold', 'customer_id': 'C-42'}
  ✅ calculate: {'total': 99.97, 'item_count': 3}
  ✅ discount: {'original': 99.97, 'discount_pct': 0.15, 'final': 84.9745}

Final: $84.97
```

### Exposing macros as tools

The model sees the macro as a single tool. Your application handles the internal orchestration:

```python
# The model only sees this schema:
macro_tool_schema = {
    "type": "function",
    "name": "quick_checkout",
    "description": (
        "Run a complete checkout: look up customer, "
        "calculate total, and apply their tier discount."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Customer ID"
            },
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of item names"
            }
        },
        "required": ["customer_id", "items"],
        "additionalProperties": False
    },
    "strict": True
}

# When the model calls quick_checkout, execute the macro
def dispatch(name: str, args: dict) -> dict:
    if name == "quick_checkout":
        return checkout_macro.execute(**args)
    # ... other tools ...
```

---

## Workflow functions

Unlike macros (fixed sequence), **workflows** include branching logic — steps that execute conditionally based on intermediate results:

```python
class WorkflowFunction:
    """Execute steps with conditional branching."""
    
    def __init__(self, name: str):
        self.name = name
        self._steps: list[dict] = []
    
    def add_step(
        self,
        name: str,
        handler: Callable,
        arg_mapper: Callable,
        condition: Callable[[dict], bool] = None
    ) -> "WorkflowFunction":
        """Add a step with optional condition."""
        self._steps.append({
            "name": name,
            "handler": handler,
            "arg_mapper": arg_mapper,
            "condition": condition or (lambda _: True)
        })
        return self
    
    def execute(self, **kwargs) -> dict:
        """Run steps, skipping those whose conditions aren't met."""
        results = {}
        skipped = []
        
        for step in self._steps:
            if not step["condition"](results):
                skipped.append(step["name"])
                print(f"  ⏭️  {step['name']}: skipped")
                continue
            
            step_args = step["arg_mapper"](kwargs, results)
            
            try:
                result = step["handler"](**step_args)
                results[step["name"]] = result
                print(f"  ✅ {step['name']}: {result}")
            except Exception as e:
                return {
                    "success": False,
                    "failed_step": step["name"],
                    "error": str(e)
                }
        
        return {
            "success": True,
            "results": results,
            "skipped": skipped
        }


# Handlers
def check_inventory(item_id: str) -> dict:
    stock = {"W-1": 50, "G-1": 0, "Z-1": 3}
    qty = stock.get(item_id, 0)
    return {"item_id": item_id, "in_stock": qty > 0, "quantity": qty}

def reserve_item(item_id: str) -> dict:
    return {"reserved": True, "item_id": item_id}

def notify_backorder(item_id: str, email: str) -> dict:
    return {"notified": True, "item_id": item_id, "email": email}

def calculate_shipping(item_id: str) -> dict:
    return {"shipping_cost": 5.99, "estimated_days": 3}


# Build workflow with branching
order_workflow = (
    WorkflowFunction("process_order_item")
    .add_step(
        "check_stock",
        check_inventory,
        lambda args, _: {"item_id": args["item_id"]}
    )
    .add_step(
        "reserve",
        reserve_item,
        lambda args, _: {"item_id": args["item_id"]},
        condition=lambda r: r.get("check_stock", {}).get("in_stock", False)
    )
    .add_step(
        "backorder",
        notify_backorder,
        lambda args, _: {
            "item_id": args["item_id"],
            "email": args.get("email", "")
        },
        condition=lambda r: not r.get("check_stock", {}).get("in_stock", True)
    )
    .add_step(
        "shipping",
        calculate_shipping,
        lambda args, _: {"item_id": args["item_id"]},
        condition=lambda r: "reserve" in r  # Only if reserved
    )
)

# Test: in-stock item
print("Order W-1 (in stock):")
result = order_workflow.execute(item_id="W-1", email="alice@example.com")

print("\nOrder G-1 (out of stock):")
result = order_workflow.execute(item_id="G-1", email="alice@example.com")
```

**Output:**
```
Order W-1 (in stock):
  ✅ check_stock: {'item_id': 'W-1', 'in_stock': True, 'quantity': 50}
  ✅ reserve: {'reserved': True, 'item_id': 'W-1'}
  ⏭️  backorder: skipped
  ✅ shipping: {'shipping_cost': 5.99, 'estimated_days': 3}

Order G-1 (out of stock):
  ✅ check_stock: {'item_id': 'G-1', 'in_stock': False, 'quantity': 0}
  ⏭️  reserve: skipped
  ✅ backorder: {'notified': True, 'item_id': 'G-1', 'email': 'alice@example.com'}
  ⏭️  shipping: skipped
```

---

## Abstraction layers

An **abstraction layer** sits between the model and your actual services. The model interacts with a simplified, high-level API while your abstraction handles the messy details:

```python
class DataAbstractionLayer:
    """Provide the model with a clean API over messy backends."""
    
    def __init__(self):
        self._sources = {}
    
    def register_source(
        self, name: str, handler: Callable, 
        transform: Callable = None
    ) -> None:
        """Register a data source with optional transform."""
        self._sources[name] = {
            "handler": handler,
            "transform": transform or (lambda x: x)
        }
    
    def query(self, source: str, **kwargs) -> dict:
        """Unified query interface — model only knows this."""
        if source not in self._sources:
            return {"error": f"Unknown source: {source}"}
        
        src = self._sources[source]
        raw = src["handler"](**kwargs)
        return src["transform"](raw)


# Backend services (messy, different formats)
def legacy_crm_lookup(id: str) -> dict:
    """Returns weird legacy format."""
    return {
        "CUST_REC": {
            "F_NAME": "Alice", "L_NAME": "Smith",
            "E_MAIL": "alice@example.com",
            "ACCT_TIER": "GOLD_V2",
            "CREATED_TS": "20230115T120000Z"
        }
    }

def rest_api_orders(customer_id: str) -> dict:
    """Returns nested REST response."""
    return {
        "data": {
            "orders": [
                {"id": "ORD-1", "total": 99.99, "status": "delivered"},
                {"id": "ORD-2", "total": 49.99, "status": "shipped"}
            ],
            "pagination": {"page": 1, "total": 2}
        }
    }


# Clean transforms
def clean_customer(raw: dict) -> dict:
    rec = raw["CUST_REC"]
    return {
        "name": f"{rec['F_NAME']} {rec['L_NAME']}",
        "email": rec["E_MAIL"],
        "tier": rec["ACCT_TIER"].replace("_V2", "").lower()
    }

def clean_orders(raw: dict) -> dict:
    orders = raw["data"]["orders"]
    return {
        "orders": [
            {"id": o["id"], "total": o["total"], "status": o["status"]}
            for o in orders
        ],
        "count": len(orders)
    }


# Build abstraction
dal = DataAbstractionLayer()
dal.register_source("customer", legacy_crm_lookup, clean_customer)
dal.register_source("orders", rest_api_orders, clean_orders)

# Model sees clean data
print(dal.query("customer", id="C-42"))
print(dal.query("orders", customer_id="C-42"))
```

**Output:**
```
{'name': 'Alice Smith', 'email': 'alice@example.com', 'tier': 'gold'}
{'orders': [{'id': 'ORD-1', 'total': 99.99, 'status': 'delivered'}, {'id': 'ORD-2', 'total': 49.99, 'status': 'shipped'}], 'count': 2}
```

The model only needs one tool schema:

```python
unified_query_schema = {
    "type": "function",
    "name": "query_data",
    "description": (
        "Query business data. Sources: 'customer' (by id), "
        "'orders' (by customer_id)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["customer", "orders"],
                "description": "Data source to query"
            },
            "params": {
                "type": "object",
                "description": "Query parameters for the source",
                "additionalProperties": True
            }
        },
        "required": ["source", "params"]
    }
}
```

---

## Gemini compositional function calling

Google Gemini natively supports **compositional (sequential) function calling** — the model automatically chains tool calls where the output of one feeds into the next:

```python
from google import genai
from google.genai import types


def get_current_location() -> dict:
    """Get the user's current location."""
    print("Tool Call: get_current_location()")
    return {"city": "London", "country": "UK"}

def get_weather(location: str) -> dict:
    """Get weather for a location."""
    print(f"Tool Call: get_weather(location={location})")
    return {"temperature": 18, "condition": "cloudy"}

def suggest_activity(weather: str, temperature: int) -> dict:
    """Suggest an activity based on weather."""
    print(f"Tool Call: suggest_activity(weather={weather}, temp={temperature})")
    if temperature > 20 and weather == "sunny":
        return {"activity": "Go to the park"}
    return {"activity": "Visit a museum"}


# With automatic function calling, Gemini chains these:
client = genai.Client()
config = types.GenerateContentConfig(
    tools=[get_current_location, get_weather, suggest_activity]
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What should I do today based on the weather here?",
    config=config
)

print(response.text)
```

**Output:**
```
Tool Call: get_current_location()
Tool Call: get_weather(location=London)
Tool Call: suggest_activity(weather=cloudy, temp=18)
Based on the current cloudy weather at 18°C in London, I'd suggest visiting a museum!
```

> **🤖 AI Context:** Gemini's automatic function calling with the Python SDK handles the entire composition loop — calling each function, passing results back, and continuing until the model has enough information. OpenAI and Anthropic require you to implement this loop manually.

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Keep individual tools atomic (one responsibility) | Reusable, testable, debuggable |
| Use macros for always-sequential operations | Reduces round trips without losing modularity |
| Use workflows for branching logic | Keeps the model out of complex conditionals |
| Build abstraction layers over messy backends | Model gets clean, consistent data |
| Document the composition in tool descriptions | Model understands what the macro actually does |
| Return intermediate results for transparency | Debugging is easier when each step is visible |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Creating one giant tool that does everything | Break into composable pieces with clear boundaries |
| Letting the model orchestrate every step | Use macros/workflows for predictable sequences |
| Hiding errors from intermediate steps | Bubble up which step failed and why |
| Inconsistent return formats between tools | Standardize on `{"success": bool, "data": ...}` |
| Not testing individual tools before composing | Unit test each tool, then integration test the composition |

---

## Hands-on exercise

### Your task

Build a `travel_booking` macro that composes three tools:

1. `search_flights` — Find flights between two cities
2. `check_hotel_availability` — Check hotels at the destination
3. `calculate_trip_cost` — Sum flight + hotel costs

### Requirements

1. Create the three handler functions with mock data
2. Build a `MacroFunction` that chains them (flight results feed into cost calculation)
3. Expose the macro as a single tool schema
4. Execute with sample inputs and print each step

### Expected result

```
  ✅ flights: {'cheapest': 450.00, 'airline': 'AirMock'}
  ✅ hotels: {'cheapest': 120.00, 'hotel': 'Mock Inn', 'per_night': 120.00}
  ✅ total_cost: {'flight': 450.00, 'hotel': 360.00, 'total': 810.00}
```

<details>
<summary>💡 Hints (click to expand)</summary>

- The `calculate_trip_cost` step needs results from both previous steps
- Use `arg_mapper` to extract the cheapest flight and hotel from previous results
- The hotel cost should multiply `per_night` by the number of nights

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
def search_flights(origin: str, destination: str) -> dict:
    return {"cheapest": 450.00, "airline": "AirMock", 
            "origin": origin, "destination": destination}

def check_hotel_availability(city: str, nights: int) -> dict:
    return {"cheapest": 120.00, "hotel": "Mock Inn", 
            "per_night": 120.00, "nights": nights}

def calculate_trip_cost(
    flight_cost: float, hotel_per_night: float, nights: int
) -> dict:
    hotel_total = hotel_per_night * nights
    return {
        "flight": flight_cost,
        "hotel": hotel_total,
        "total": flight_cost + hotel_total
    }

booking_macro = (
    MacroFunction("travel_booking", "Search flights, hotels, and calculate total cost.")
    .add_step(
        "flights",
        search_flights,
        lambda args, _: {
            "origin": args["origin"],
            "destination": args["destination"]
        }
    )
    .add_step(
        "hotels",
        check_hotel_availability,
        lambda args, _: {
            "city": args["destination"],
            "nights": args["nights"]
        }
    )
    .add_step(
        "total_cost",
        calculate_trip_cost,
        lambda args, prev: {
            "flight_cost": prev["flights"]["cheapest"],
            "hotel_per_night": prev["hotels"]["per_night"],
            "nights": args["nights"]
        }
    )
)

result = booking_macro.execute(
    origin="New York",
    destination="London",
    nights=3
)
```

</details>

### Bonus challenges

- [ ] Add a conditional step: if `flight_cost > 500`, search for alternative airports
- [ ] Convert the macro into a `WorkflowFunction` with branching
- [ ] Implement the same composition using Gemini's automatic function calling

---

## Summary

✅ **Function composition** builds complex operations from simple, reusable tools — matching real software engineering principles

✅ **Macros** execute a fixed sequence of tools as a single operation — reducing round trips for predictable pipelines

✅ **Workflows** add conditional branching — steps execute only when their conditions are met

✅ **Abstraction layers** clean up messy backend data before the model sees it — one unified query interface over multiple services

✅ **Gemini compositional calling** automates multi-step chains natively in the SDK — the model calls tools sequentially, piping outputs forward

**Next:** [Nested Function Calling →](./04-nested-function-calling.md)

---

[← Previous: Dynamic Registration](./02-dynamic-registration.md) | [Back to Lesson Overview](./00-advanced-patterns.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling Guide (best practices): https://platform.openai.com/docs/guides/function-calling
- Google Gemini Compositional Function Calling: https://ai.google.dev/gemini-api/docs/function-calling#compositional-function-calling
- Google Gemini Automatic Function Calling: https://ai.google.dev/gemini-api/docs/function-calling#automatic-function-calling
-->
