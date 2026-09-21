import logfire

from app.agents.state import AgentState
from app.gateway import invoke_llm_with_fallback
from app.services.prompts import render_prompt


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

    with logfire.span("🧠 Planner Decision"):
        decision = invoke_llm_with_fallback(
            prompt,
            feature="planner",
            slug=state.get("slug"),
            model=state.get("model"),
        ).strip()
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