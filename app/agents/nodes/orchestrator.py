"""Orchestrator node — decides which agent path to take.

The planner decides CONVERSATIONAL-vs-technical; the orchestrator refines the
technical path into tool / research / code using lightweight keyword heuristics
(no extra LLM cost):
  - tool     → a single informational tool (e.g. list_sources)
  - research → researcher sub-agent (multi-query retrieval) → analyst → responder
  - code     → researcher (retrieval) → coder → responder

Planner is an LLM and can be too quick to say "CONVERSATIONAL" even when the
user is really asking about the site's content ("tell me about the projects").
The RESEARCH_OVERRIDE_KEYWORDS below force research for such casual-but-topical
questions so retrieval (and the Supabase-fed corpus) is actually used.
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
# Topic words that should always trigger retrieval even in casual phrasing.
RESEARCH_OVERRIDE_KEYWORDS = (
    "projects", "project", "skills", "skill", "experience", "about",
    "portfolio", "stack", "technologies", "technology", "education",
    "resume", "achievements", "achievement", "what can you tell",
)


def _classify_intent(state: AgentState, user_msg: str) -> str:
    """Keyword intent for the non-conversational path: tool | code | research."""
    text = (user_msg + " " + state.get("current_query", "")).lower()

    if any(k in text for k in TOOL_KEYWORDS):
        return "tool"
    if any(k in text for k in CODE_KEYWORDS):
        return "code"
    return "research"


def orchestrator_node(state: AgentState):
    """Set the intent and route. Returns the intent for conditional edges."""
    user_msg = state["messages"][-1]["content"] if state["messages"] else ""
    raw = state.get("current_query", "")

    forced_research = False
    if raw == "CONVERSATIONAL":
        if any(k in user_msg.lower() for k in RESEARCH_OVERRIDE_KEYWORDS):
            forced_research = True
            intent = "research"
            logfire.info("🧭 Conversational but topic terms present — forcing research.")
        else:
            intent = "chat"
    else:
        intent = _classify_intent(state, user_msg)

    updates: dict = {
        "intent": intent,
        "status": {
            "chat": "Handling conversationally (using memory)...",
            "tool": "Running an informational tool...",
            "research": "Running researcher sub-agent...",
            "code": "Running coder sub-agent...",
        }.get(intent, "Agentic research..."),
    }

    if forced_research:
        # Rewrite the query to the user's actual words so the researcher
        # retrieves for "tell me about the projects", not "CONVERSATIONAL".
        updates["current_query"] = user_msg
        updates["plan"] = [
            "Intent: Technical",
            f"Search Term: {user_msg}",
            "Orchestration: research",
        ]
    else:
        updates["plan"] = state["plan"] + [f"Orchestration: {intent}"]

    logfire.info(f"🧭 Orchestration intent: {intent}")
    return updates