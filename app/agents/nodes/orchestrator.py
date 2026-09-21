"""Orchestrator node — decides which agent path to take.

The planner decides CONVERSATIONAL-vs-technical; the orchestrator refines the
technical path into tool / research / code using lightweight keyword heuristics
(no extra LLM cost):
  - tool     → a single informational tool (e.g. list_sources)
  - research → researcher sub-agent (multi-query retrieval) → analyst → responder
  - code     → researcher (retrieval) → coder → responder
"""
import logfire

from app.agents.state import AgentState

TOOL_KEYWORDS = (
    "list", "listing", "how many", "which documents", "which files",
    "what documents", "document list", "available sources", "what sources",
    "sources in the", "categories of",
)
CODE_KEYWORDS = (
    "code", "function that", "function to", "script", "implement",
    "implementation", "write a", "regex", "yaml for", "manifest for",
    "dockerfile", "helm", "terraform", "python", "api endpoint", "snippet",
    "debug", "example of", "example for",
)


def _classify_intent(state: AgentState) -> str:
    if state.get("current_query") == "CONVERSATIONAL":
        return "chat"
    user_msg = state["messages"][-1]["content"] if state["messages"] else ""
    text = (user_msg + " " + state.get("current_query", "")).lower()

    if any(k in text for k in TOOL_KEYWORDS):
        return "tool"
    if any(k in text for k in CODE_KEYWORDS):
        return "code"
    return "research"


def orchestrator_node(state: AgentState):
    """Set the intent and route. Returns the intent for conditional edges."""
    intent = _classify_intent(state)
    logfire.info(f"🧭 Orchestration intent: {intent}")
    return {
        "intent": intent,
        "status": {
            "chat": "Handling conversationally (using memory)...",
            "tool": "Running an informational tool...",
            "research": "Running researcher sub-agent...",
            "code": "Running coder sub-agent...",
        }.get(intent, "Agentic research..."),
        "plan": state["plan"] + [f"Orchestration: {intent}"],
    }