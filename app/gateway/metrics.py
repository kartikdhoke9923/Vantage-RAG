"""Prometheus metrics for the LLM gateway: calls, tokens, cache and cost."""
from prometheus_client import Counter

# One label set per feature (planner, responder, fact_checker, researcher,
# analyst, coder, rag) + status (cache-hit, portkey, groq-fallback, error).
GATEWAY_CALLS_TOTAL = Counter(
    "gateway_calls_total",
    "LLM gateway call attempts",
    ["feature", "status"],
)
GATEWAY_PROMPT_TOKENS = Counter(
    "gateway_prompt_tokens_total",
    "Prompt tokens sent to the gateway",
    ["feature"],
)
GATEWAY_COMPLETION_TOKENS = Counter(
    "gateway_completion_tokens_total",
    "Completion tokens received from the gateway",
    ["feature"],
)
GATEWAY_CACHE_HITS_TOTAL = Counter(
    "gateway_cache_hits_total",
    "Responses served from the Redis prompt/response cache",
    ["feature"],
)
GATEWAY_COST_USD_TOTAL = Counter(
    "gateway_cost_usd_total",
    "Approximate LLM spend in USD",
    ["feature"],
)

# Approx USD per 1M tokens (input, output). Free-tier models default to 0;
# add entries as new models are adopted. Overridable via GATEWAY_MODEL_COSTS.
COST_PER_1M: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.15, 0.60),
    "openai/gpt-oss-120b": (0.30, 1.20),
    "qwen/qwen3.8-27b": (0.10, 0.40),
    "gpt-5-mini": (0.15, 0.60),
}


def _model_key(model: str | None) -> str:
    """Normalize a routing/model string to a COST_PER_1M key or ''."""
    if not model:
        return ""
    return model.split("/")[-1]


def record_usage(feature: str, model: str | None, prompt_tokens: int, completion_tokens: int) -> None:
    """Increment token/cost counters for a completed gateway call."""
    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    GATEWAY_PROMPT_TOKENS.labels(feature=feature).inc(prompt_tokens)
    GATEWAY_COMPLETION_TOKENS.labels(feature=feature).inc(completion_tokens)

    key = _model_key(model)
    in_price, out_price = COST_PER_1M.get(key, (0.0, 0.0))
    cost = (prompt_tokens / 1_000_000 * in_price) + (completion_tokens / 1_000_000 * out_price)
    if cost:
        GATEWAY_COST_USD_TOTAL.labels(feature=feature).inc(cost)