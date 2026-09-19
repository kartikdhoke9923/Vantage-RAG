import logfire

from app.agents.state import AgentState
from app.gateway import get_langchain_llm
from app.services.prompts import render_prompt

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatOpenAI
# Kept for module-level compatibility; planner_node rebuilds it per request so the
# UI gateway switcher (slug/model overrides) takes effect.
llm = get_langchain_llm(feature="planner")


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = render_prompt(
        "planner",
        history=history,
        question=user_message,
    )

    # Honour per-request slug/model overrides from the UI; fall back to settings.
    active_llm = get_langchain_llm(
        feature="planner",
        slug=state.get("slug"),
        model=state.get("model"),
    )

    with logfire.span("🧠 Planner Decision"):
        decision = active_llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"],
    }