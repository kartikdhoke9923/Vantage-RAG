"""Orchestrator node — decides which agent path to take.

Routing policy (no extra LLM cost — pure heuristics on top of the planner):

  - small talk / greetings             → chat (no retrieval)
  - content questions (everything else) → research by default, so the retrieval
                                          + guardrails + Supabase-fed corpus is
                                          actually used instead of the responder
                                          free-forming an answer
  - explicit "list the sources" asks   → tool (list_sources) — keyword list is
                                          deliberately narrow: bare "list" and
                                          "how many" misfire on factual count
                                          questions and must retrieve instead
  - explicit "write a script/function" → code

The planner LLM is too quick to say "CONVERSATIONAL" even for real content
questions ("how can I contact Kartik"). Research is the default for anything
that is not small talk.
"""
import re

import logfire

from app.agents.state import AgentState

# Explicit source-listing asks only. Bare "list" / "how many" were removed —
# they fire on factual questions (e.g. "how many emails per day?") and skip
# retrieval, which produced false "documentation does not cover this" answers.
TOOL_KEYWORDS = (
    "list the sources", "list of sources", "available sources", "what sources",
    "which documents", "which files", "what documents", "document list",
    "sources in the", "categories of",
)
CODE_KEYWORDS = (
    "code", "function that", "function to", "script", "implement",
    "implementation", "write a", "regex", "yaml for", "manifest for",
    "dockerfile", "helm", "terraform", "api endpoint", "snippet", "debug",
    "example of", "example for",
    # Note: "python" / "sql" deliberately absent — they are content topics in
    # Kartik's stack, not code-execution requests.
)

_GREETING_RE = re.compile(
    r"^(hello|hi|hey|yo|howdy|namaste|good\s?(morning|afternoon|evening)"
    r"|how are you|how's it going|what's up|gm|sup|thanks?|thank you"
    r"|bye|goodbye|ok|okay|cool|nice|great|sure|got it)\b.*$",
    re.IGNORECASE,
)
_GREETING_MAX_CHARS = 80

_SMALL_TALK_WORDS = {
    "ok", "okay", "cool", "nice", "great", "sure", "got it", "thanks",
    "thank you", "bye", "goodbye", "hi", "hello", "hey", "yo", "hi there",
    "hello there", "morning", "good morning", "no", "yes", "sure thing",
}


def _is_small_talk(message: str) -> bool:
    """Greetings / short acknowledgements → chat; everything else → research."""
    m = (message or "").strip().lower()
    if not m:
        return True
    if len(m) <= _GREETING_MAX_CHARS and _GREETING_RE.match(m):
        return True
    return m in _SMALL_TALK_WORDS


def _classify_intent(state: AgentState, user_msg: str) -> str:
    """Keyword intent for the non-small-talk path: tool | code | research."""
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

    if _is_small_talk(user_msg):
        intent = "chat"
        forced_research = False
    elif raw == "CONVERSATIONAL":
        # Planner gave up on the search; the message is substantive (not small
        # talk), so run retrieval on the user's actual words.
        intent = "research"
        forced_research = True
        logfire.info("🧭 Non-conversational query despite CONVERSATIONAL planner — forcing research.")
    else:
        intent = _classify_intent(state, user_msg)
        forced_research = False

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
        # retrieves for "how can I contact Kartik", not "CONVERSATIONAL".
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