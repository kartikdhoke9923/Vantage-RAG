import logfire
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.nodes.analyst import analyst_node
from app.agents.nodes.coder import coder_node
from app.agents.nodes.fact_checker import fact_check_node
from app.agents.nodes.orchestrator import orchestrator_node
from app.agents.nodes.planner import planner_node
from app.agents.nodes.researcher import researcher_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.tool_executor import tool_executor_node
from app.agents.state import AgentState
from app.config import settings


def create_checkpointer() -> BaseCheckpointSaver:
    """
    Create a durable Postgres checkpointer for production.
    Falls back to in-memory MemorySaver only if Postgres is unreachable.

    If POSTGRES_URI is empty (local/dev), skip Postgres entirely and go
    straight to MemorySaver — no connection attempt, no startup delay.

    Note: we run setup() through a single autocommit connection because
    LangGraph's migrations include CREATE INDEX CONCURRENTLY, which Neon
    rejects when run inside a transaction (the default ConnectionPool mode).
    """
    if not settings.postgres_uri:
        logfire.info("🚦 POSTGRES_URI not set — using in-memory MemorySaver checkpointer.")
        return MemorySaver()

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            conninfo=settings.postgres_uri,
            max_size=20,
            open=False,
            # Generous enough for Neon serverless cold starts (first connect can
            # take several seconds) without hanging boot for ages.
            timeout=10,
            num_workers=3,
            check=ConnectionPool.check_connection,
            max_idle=240,
        )
        # Verify connectivity before committing to Postgres; otherwise the first
        # graph invocation will hang on connection retries.
        pool.open()
        conn = pool.getconn()
        pool.putconn(conn)

        # Run migrations on a separate autocommit connection so that
        # CREATE INDEX CONCURRENTLY succeeds on Neon.
        try:
            with PostgresSaver.from_conn_string(settings.postgres_uri) as setup_saver:
                setup_saver.setup()
        except Exception as e:
            logfire.warning(f"⚠️ Postgres checkpointer setup failed ({e}); falling back to MemorySaver.")
            pool.close()
            return MemorySaver()

        checkpointer = PostgresSaver(pool)
        logfire.info("🗄️ Postgres checkpointer configured.")
        return checkpointer
    except Exception as e:
        logfire.warning(
            f"⚠️ Postgres checkpointer unavailable ({e}); falling back to MemorySaver. "
            "Do not use MemorySaver in production — state is lost on restart."
        )
        return MemorySaver()


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """
    Build and compile the LangGraph RAG agent.

    Args:
        checkpointer: Optional checkpointer. If None, a Postgres-backed
            checkpointer is created. Pass a MemorySaver in tests.
    """
    if checkpointer is None:
        checkpointer = create_checkpointer()

    # 1. Initialize the State Graph
    workflow = StateGraph(AgentState)

    # 2. Define the Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("responder", generate_node)
    workflow.add_node("fact_checker", fact_check_node)

    # 3. Define the Edges & Routing Logic
    def route_orchestrator(state: AgentState):
        """Route by the orchestrator's intent classification (map key, not node)."""
        return state.get("intent", "research")

    def route_after_research(state: AgentState):
        """Researchers gather evidence; analysts/coder digest it."""
        return "coder" if state.get("intent") == "code" else "analyst"

    workflow.set_entry_point("planner")

    # Conditional Edge: Planner -> Orchestrator -> Agent path -> Responder.
    workflow.add_edge("planner", "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        route_orchestrator,
        {
            "chat": "responder",
            "tool": "tool_executor",
            "research": "researcher",
            "code": "researcher",
        },
    )
    workflow.add_conditional_edges(
        "researcher",
        route_after_research,
        {"analyst": "analyst", "coder": "coder"},
    )

    workflow.add_edge("tool_executor", "responder")
    workflow.add_edge("analyst", "responder")
    workflow.add_edge("coder", "responder")
    workflow.add_edge("responder", "fact_checker")
    workflow.add_edge("fact_checker", END)

    # 4. Compile the Graph with Memory
    return workflow.compile(checkpointer=checkpointer)


# Production code should call build_graph() explicitly (see app.main.startup_event).
# The old module-level `rag_agent` has been removed to avoid constructing two
# checkpointers and to make dependency injection/testability cleaner.
