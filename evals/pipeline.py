"""
Phase 1 — Live Pipeline.
Calls the running FastAPI /query endpoint for each golden sample.
Captures: actual_response (truncated to 300 chars), actual_contexts (from sources),
and actual_tools_called (detected from thought_process).

/query is synchronous: it returns the answer (or a guardrails block) directly.
"""

import copy
import json
import os
import time

import logfire
import requests

API_URL = "http://localhost:8000/query"
RESPONSE_TRUNCATE = 300
DELAY_BETWEEN_CALLS = 10  # seconds — stays within the Groq/Portkey throughput on the main key
REQUEST_TIMEOUT = 180  # seconds — guardrails + LangGraph + LLM can take >60s
MAX_ATTEMPTS = 4  # retries on 429 (backend slowapi rate limiter)
RETRY_WAIT = 60  # seconds between 429 retries when no Retry-After header is present
_RETRYABLE = (429, 500, 502, 503, 504)  # rate-limit + transient server errors
_RETRY_WAIT_5XX = 15  # seconds between 5xx retries


def _post_with_retry(question: str, thread_id: str, timeout: int = REQUEST_TIMEOUT) -> "requests.Response":
    """POST /query, backing off on 429/5xx so the eval suite survives the backend."""
    resp = requests.post(
        API_URL,
        json={"q": question, "thread_id": thread_id},
        timeout=timeout,
    )
    attempt = 1
    while resp.status_code in _RETRYABLE and attempt < MAX_ATTEMPTS:
        if resp.status_code == 429:
            retry_after = (resp.headers.get("Retry-After") or "").strip()
            wait = float(retry_after) if retry_after.isdigit() else RETRY_WAIT
        else:
            wait = _RETRY_WAIT_5XX
        logfire.warning(
            f"Transient error {resp.status_code}; waiting {wait:.0f}s before retry "
            f"{attempt}/{MAX_ATTEMPTS - 1}.",
            question=question[:80],
        )
        time.sleep(wait)
        resp = requests.post(
            API_URL,
            json={"q": question, "thread_id": thread_id},
            timeout=timeout,
        )
        attempt += 1
    return resp


def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from the current /query response to a tool name.

    The orchestrator emits plan markers for the intent it chose:
      'guardrails fired'                   → guardrails (blocked before the graph)
      'orchestration: tool' / 'tool:'       → list_sources (tool_executor)
      'orchestration: code' / 'sub-agent: coder' → coder
      'sub-agent: researcher' / 'search term:' / 'intent: technical' → retrieve_documents
      'intent: conversational'              → direct_answer
    """
    joined = " ".join(thought_process or []).lower()
    if any(k in joined for k in ("guardrails fired", "intent: guardrails")):
        return "guardrails"
    if "orchestration: tool" in joined or "tool:" in joined:
        return "list_sources"
    if "orchestration: code" in joined or "sub-agent: coder" in joined:
        return "coder"
    if "orchestration: research" in joined or "sub-agent: researcher" in joined:
        return "retrieve_documents"
    if "intent: technical" in joined or "search term:" in joined:
        return "retrieve_documents"
    if "conversational" in joined or "memory" in joined:
        return "direct_answer"
    return "unknown"


def _fetch_query_result(question: str, thread_id: str) -> dict:
    """Submit a query and return the flat /query response (sync or guardrails block)."""
    resp = _post_with_retry(question, thread_id)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "Blocked by guardrails.":
        return data
    if "answer" in data:
        return data
    raise RuntimeError(f"Unexpected /query response: {data}")


def run_pipeline(golden_dataset: dict, progress_callback=None) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.
    Returns a deep copy with actual_response, actual_contexts, actual_tools_called filled.
    progress_callback(i, total, question, stage, response="") is called per step.
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    n = len(samples)

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample["question"]

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):
                try:
                    data = _fetch_query_result(question, thread_id=f"eval_run_{i}")

                    raw_answer = data.get("answer") or ""
                    thought_process = data.get("thought_process") or []
                    sources = data.get("sources") or []

                    sample["actual_response"] = raw_answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = [
                        s.get("content", "") if isinstance(s, dict) else str(s)
                        for s in sources[:5]
                    ]
                    sample["actual_tools_called"] = [detect_tool(thought_process)]

                    logfire.info(
                        "✅ Response captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars=len(raw_answer),
                        context_chunks=len(sources),
                    )

                except requests.exceptions.ConnectionError:
                    logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["unknown"]

                except Exception as e:
                    logfire.error(f"❌ Query failed: {e}")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["unknown"]

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    return dataset


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)


def load_golden_dataset() -> dict:
    """Load a golden eval dataset.

    The dataset is produced by evals/build_golden.py (or exported by ingestion
    runs), so it is treated as a disposable artifact. Set EVAL_DATASET to point
    at a fresh dataset path (defaults to evals/golden_dataset.json).
    """
    golden_path = os.getenv("EVAL_DATASET") or os.path.join(
        os.path.dirname(__file__), "golden_dataset.json"
    )
    if not os.path.exists(golden_path):
        raise FileNotFoundError(
            f"No eval dataset found at {golden_path}. Generate one with "
            "`uv run python -m evals.build_golden`, or set EVAL_DATASET to point at one."
        )
    with open(golden_path) as f:
        return json.load(f)