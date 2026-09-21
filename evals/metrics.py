"""
Phase 2 — RAGAS + Tool Correctness metrics.

The judge LLM reuses the production Portkey gateway (openai/gpt-oss-20b via the
@slug/model routing), so no separate OpenAI key is required. All LLM calls are
serialized (max_workers=1) as a conservative provider-throughput buffer.
Contexts are truncated to 300 chars (2 chunks max) so no single request blows up.

Uses the installed ragas `evaluate()` / Dataset API.
"""

import sys
import types

# ragas imports `langchain_community.chat_models.vertexai`, which moved to the
# standalone `langchain_google_vertexai` package in langchain-community 0.4.x.
# Shim the name in before ragas loads so the import succeeds (VertexAI is never
# instantiated by our judge → ChatOpenAI through Portkey).
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
    try:
        from langchain_google_vertexai import ChatVertexAI
    except Exception:  # noqa: BLE001 - dummy class is enough for the isinstance table
        ChatVertexAI = type("ChatVertexAI", (), {})
    _vertexai.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai

import logfire
import pandas as pd
from datasets import Dataset
from langchain_community.embeddings import (
    HuggingFaceEmbeddings as _LCHuggingFaceEmbeddings,
)
from ragas import evaluate
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from app.gateway.client import get_langchain_llm

CONTEXT_TRUNCATE = 300  # chars per context chunk — reduces single request token count
CONTEXT_LIMIT = 2  # number of context chunks passed to RAGAS per sample

_RUN_CONFIG = RunConfig(timeout=240, max_retries=4, max_wait=60, max_workers=1)


def _build_judge():
    """RAGAS judge via the production Portkey gateway (OpenAI-compatible base URL).

    Uses get_langchain_llm directly instead of ragas' llm_factory because the
    latter builds ChatOpenAI without an api_key and fails on machines that do
    not export OPENAI_API_KEY. The gateway accepts a placeholder key — Portkey
    auth travels in x-portkey-* headers.
    """
    llm = LangchainLLMWrapper(get_langchain_llm(feature="eval-judge"), run_config=_RUN_CONFIG)
    embeddings = LangchainEmbeddingsWrapper(
        _LCHuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    return llm, embeddings


def _prep_dataset(golden_dataset: dict) -> Dataset:
    """
    Returns only samples with actual_response populated, as a ragas Dataset
    with the single-turn columns ragas expects.
    """
    rows = []
    for s in golden_dataset["rag_samples"]:
        response = (s.get("actual_response") or "").strip()
        if not response:
            continue
        contexts = [
            c[:CONTEXT_TRUNCATE]
            for c in ((s.get("actual_contexts") or s.get("relevant_contexts") or [])[:CONTEXT_LIMIT])
        ]
        rows.append(
            {
                "user_input": s["question"],
                "response": response,
                "retrieved_contexts": contexts,
                "reference": s.get("reference", ""),
            }
        )

    if not rows:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")
    return Dataset.from_list(rows)


_CORE_COLUMNS = {"user_input", "response", "retrieved_contexts", "reference"}


def _extract_scores(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Pull the metric's score column out of the ragas result table."""
    score_col = next((c for c in df.columns if c not in _CORE_COLUMNS), key)
    return pd.DataFrame(
        {
            "question": df["user_input"] if "user_input" in df.columns else df.index,
            key: df[score_col],
        }
    ).reset_index(drop=True)


async def run_all_metrics(golden_dataset: dict, status_cb=None) -> dict:
    """
    Runs the 6 metric experiments. Returns dict keyed by metric name → DataFrame.
    status_cb(message: str) is called for live UI updates.
    """
    judge_llm, ragas_embeddings = _build_judge()
    dataset = _prep_dataset(golden_dataset)

    def _notify(msg: str):
        if status_cb:
            status_cb(msg)

    experiments: list[tuple[str, object]] = [
        ("faithfulness", Faithfulness()),
        ("answer_relevancy", AnswerRelevancy()),
        ("context_precision", ContextPrecision()),
        ("context_recall", ContextRecall()),
        ("answer_correctness", AnswerCorrectness()),
    ]

    results: dict[str, pd.DataFrame] = {}
    with logfire.span("🧪 Eval Phase 2 — RAGAS Metrics", total_samples=len(dataset)):
        for key, metric in experiments:
            _notify(f"🧪 Scoring {key} ({len(dataset)} samples)...")
            with logfire.span(f"🧪 {key}") as span:
                result = evaluate(
                    dataset,
                    metrics=[metric],
                    llm=judge_llm,
                    embeddings=ragas_embeddings,
                    run_config=_RUN_CONFIG,
                    show_progress=False,
                    batch_size=1,
                )
                df = result.to_pandas()
                results[key] = _extract_scores(df, key)
                avg = round(float(results[key][key].mean()), 3)
                span.set_attributes({f"{key}_avg": avg})
                logfire.info(f"🧪 {key} done", avg=avg)
                _notify(f"✅ {key}: avg {avg}")

        # ── Exp 6: Tool Correctness (no LLM — Jaccard) ───────────────────────
        _notify("⚡ Exp 6/6 — Tool Correctness (zero LLM calls)...")
        tool_rows = []
        for s in golden_dataset["rag_samples"]:
            if not (s.get("actual_response") or "").strip():
                continue
            called = set(s.get("actual_tools_called") or [])
            expected = set(s.get("expected_tools") or [])
            union = len(called | expected)
            score = len(called & expected) / union if union > 0 else 0.0
            tool_rows.append({"question": s["question"][:65], "tool_correctness": round(score, 3)})
        results["tool_correctness"] = pd.DataFrame(tool_rows)

        _notify("✅ All 6 experiments complete!")

    return results