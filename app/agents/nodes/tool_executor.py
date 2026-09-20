"""Tool executor node — runs a registry tool for the 'tool' intent.

Currently the informational path (e.g. list_sources). Retrieval is handled by
the researcher node; tool results are stored in `tool_results` for the
responder.
"""
import logfire

from app.agents.state import AgentState
from app.tools.registry import TOOL_REGISTRY

# Orchestrator's 'tool' intent currently maps to a single informational tool.
# Add more mappings here as tools are registered.
TOOL_LOOKUP = {
    "list_sources": "list_sources",
}


def tool_executor_node(state: AgentState):
    """Execute the tool chosen by the orchestrator and store its output."""
    tool_name = TOOL_LOOKUP.get(state.get("tool_name") or "list_sources", "list_sources")
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        logfire.warning(f"🧰 Tool '{tool_name}' not found in registry.")
        return {
            "tool_results": f"Tool '{tool_name}' is not available.",
            "plan": state["plan"] + [f"Tool: {tool_name} (missing)"],
        }

    with logfire.span(f"🧰 Executing tool: {tool_name}"):
        result = tool.handler()
        logfire.info(f"🧰 Tool '{tool_name}' executed.")

    return {
        "tool_results": str(result),
        "plan": state["plan"] + [f"Tool: {tool_name}"],
        "status": f"Tool '{tool_name}' executed.",
    }