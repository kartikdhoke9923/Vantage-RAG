import operator
from typing import Annotated, NotRequired, TypedDict


class AgentState(TypedDict):
    # Using Annotated with operator.add ensures that messages
    # are appended to the history rather than replaced.
    messages: Annotated[list[dict], operator.add]
    current_query: str
    # Structured retrieval chunks: {content, source, source_type}. The
    # responder numbers them so citations like [1] can be enforced.
    documents: list[dict[str, str]]
    plan: list[str]
    status: str
    final_answer: str
    # Citation guard output: any problems found in the generated answer.
    citation_warning: str | None
    # Fact-checker verdict: GROUNDED / PARTIAL / UNGROUNDED / UNKNOWN / SKIPPED.
    fact_check: str | None
    # Orchestration intent: chat | tool | research | code (set by orchestrator).
    intent: str
    # Sub-agent outputs.
    sub_queries: list[str]          # researcher's decomposed retrieval queries
    analysis: str | None            # analyst / coder evidence digest for the responder
    tool_results: str | None        # tool_executor output (e.g. list_sources)
    # Per-request Portkey gateway overrides from the UI. When absent, the
    # settings defaults (PORTKEY_PRIMARY_SLUG / PORTKEY_PRIMARY_MODEL) apply.
    slug: NotRequired[str | None]
    model: NotRequired[str | None]