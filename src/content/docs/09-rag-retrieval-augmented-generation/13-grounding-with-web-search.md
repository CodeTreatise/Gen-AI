---
title: "Grounding with Web Search"
---

# Grounding with Web Search

- Gemini Google Search grounding
  - Real-time web search integration
  - Enable with `google_search` tool:
    - Import genai and types from google.genai library
    - Create genai.Client() instance for API access
    - Create grounding tool: `types.Tool(google_search=types.GoogleSearch())`
    - Create config: `types.GenerateContentConfig(tools=[grounding_tool])`
    - Call `client.models.generate_content()` with model name, contents, and config
    - Use model like "gemini-3-flash-preview" for search-grounded responses
    - Response includes grounding_metadata with sources and citations
  - Model decides when to search
  - Returns grounding metadata with sources
  - Automatic citation generation
  - Billing: per search query executed (Gemini 3)
- How grounding works
  1. User prompt sent with search tool enabled
  2. Model analyzes if search would help
  3. Model generates and executes search queries
  4. Model processes search results
  5. Returns grounded response with citations
- OpenAI web search connector
  - Web search as a tool in Responses API
  - Automatic source attribution
  - Integration with function calling
- Combining web + private knowledge
  - Route queries to appropriate source
  - Merge web results with internal docs
  - Priority and freshness weighting
  - Conflict resolution strategies
- Real-time RAG use cases
  - Current events questions
  - Stock prices and market data
  - Weather and live information
  - News and recent developments
