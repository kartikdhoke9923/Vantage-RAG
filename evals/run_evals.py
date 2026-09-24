"""
Headless evaluation runner.

Loads the golden dataset, calls the live FastAPI /query endpoint for each sample,
runs guardrails tests, and writes a JSON report to evals/report.json.

With --metrics, also runs the full 6-metric RAGAS suite (gateway judge) and merges
the scores into the report. That pass is slow (~40-50 min) due to per-sample cooldowns.

Phases are decoupled so a rerun never burns tokens on data you already have:
    --metrics-only     re-score the cached Phase-1 responses already saved in
                       report.json (no live calls, no /query traffic).
    --guardrails-only  rerun just the guardrail tests against the live backend
                       (no Phase-1 pipeline replay).

Usage:
    python -m evals.run_evals
    python -m evals.run_evals --metrics
    EVAL_JUDGE_PROVIDER=gemini python -m evals.run_evals --metrics-only
    python -m evals.run_evals --guardrails-only

Requires the FastAPI backend to be running on localhost:8000 (or BACKEND_URL)
UNLESS you only use --metrics-only (which replays saved responses).
"""

import argparse
import asyncio
import json
import os
import sys

import logfire

# Allow running from repo root without package installation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="evals")

# Keep console output (emoji banners) from crashing on redirected/piped
# Windows consoles that default to cp1252; no-op elsewhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from evals.guardrails_eval import compute_guardrails_metrics, run_guardrails_eval
from evals.metrics import run_all_metrics
from evals.pipeline import load_golden_dataset, run_pipeline


def _print_progress(i: int, total: int, label: str, item: str):
    pct = int((i / total) * 100)
    print(f"[{pct:3d}%] {label}: {item[:80]}")


def _metric_summary(metric_results: dict) -> dict:
    import pandas as pd

    summary = {}
    for key, df in metric_results.items():
        if isinstance(df, pd.DataFrame) and key in df.columns:
            summary[key] = round(float(df[key].mean()), 3)
    return summary


def _load_previous_report(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _run_metrics(enriched: dict) -> tuple[dict, dict]:
    """Phase 2 only — RAGAS + tool-correctness over the given samples."""
    print("🧪 Running 6-metric RAGAS suite (~40-50 min)...")
    metric_results = asyncio.run(run_all_metrics(enriched))
    metric_scores = _metric_summary(metric_results)
    metric_details = {
        key: df.to_dict(orient="records") for key, df in metric_results.items()
    }
    return metric_scores, metric_details


def main(include_metrics: bool = False, guardrails_only: bool = False, metrics_only: bool = False) -> dict:
    """Run the live eval suite (optionally per-phase) and return the report."""
    golden = load_golden_dataset()
    report_path = os.path.join(os.path.dirname(__file__), "report.json")
    prev = _load_previous_report(report_path)

    rag_samples = None
    guardrails_results = None
    guardrails_metrics = None
    metric_scores = None
    metric_details = None

    if metrics_only:
        # Phase 2 only: re-score the responses already captured in the last run.
        rag_samples = prev.get("rag_samples")
        if not rag_samples or not any(
            (s.get("actual_response") or "").strip() for s in rag_samples
        ):
            raise RuntimeError(
                "No captured responses found in evals/report.json — "
                "run a full eval first (python -m evals.run_evals) so Phase 1 data exists."
            )
        print("🧪 Scoring cached Phase-1 results (--metrics-only; no live calls)...")
        metric_scores, metric_details = _run_metrics({"rag_samples": rag_samples})
        guardrails_results = prev.get("guardrails_results")
        guardrails_metrics = prev.get("guardrails_metrics")

    elif guardrails_only:
        # Guardrails only: rerun the live tests, keep the cached rag_samples.
        print("🛡️ Running guardrails tests only (--guardrails-only; Phase 1 skipped)...")
        guardrails_results = run_guardrails_eval(
            golden["guardrails_samples"],
            progress_callback=lambda i, total, input_text: _print_progress(i, total, "guardrails", input_text),
        )
        guardrails_metrics = compute_guardrails_metrics(guardrails_results)
        rag_samples = prev.get("rag_samples")

    else:
        print("🚀 Running live RAG pipeline...")
        enriched = run_pipeline(
            golden,
            progress_callback=lambda i, total, question, stage, response="": _print_progress(i, total, stage, question),
        )
        rag_samples = enriched["rag_samples"]

        print("🛡️ Running guardrails tests...")
        guardrails_results = run_guardrails_eval(
            enriched["guardrails_samples"],
            progress_callback=lambda i, total, input_text: _print_progress(i, total, "guardrails", input_text),
        )
        guardrails_metrics = compute_guardrails_metrics(guardrails_results)

        if include_metrics:
            metric_scores, metric_details = _run_metrics(enriched)

    report = {
        "rag_samples": rag_samples,
        "guardrails_results": guardrails_results,
        "guardrails_metrics": guardrails_metrics,
    }
    if metric_scores is not None:
        report["metric_scores"] = metric_scores
        report["metric_details"] = metric_details

    if report["rag_samples"] is None and report["guardrails_results"] is None:
        raise RuntimeError("Nothing to write — no phases ran.")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Report saved to {report_path}")
    if guardrails_metrics:
        print(
            f"🛡️ Guardrails — correct: {guardrails_metrics['correct']}/{guardrails_metrics['valid']} "
            f"valid (skipped/invalid: {guardrails_metrics['invalid']}), "
            f"precision: {guardrails_metrics['precision']}, recall: {guardrails_metrics['recall']}, "
            f"accuracy: {guardrails_metrics['accuracy']}"
        )
    if metric_scores is not None:
        print("📊 RAGAS metric averages:")
        for key, avg in metric_scores.items():
            print(f"   {key}: {avg}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the live eval suite (optionally per-phase with RAGAS metrics).")
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Also run the full 6-metric RAGAS suite (~40-50 min) and save scores to report.json.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Re-use the cached Phase-1 responses already in report.json and only (re)compute Phase-2 RAGAS metrics. No live calls.",
    )
    parser.add_argument(
        "--guardrails-only",
        action="store_true",
        help="Run only the guardrail test suite against the live backend (skip the Phase-1 pipeline replay).",
    )
    args = parser.parse_args()
    main(
        include_metrics=args.metrics,
        guardrails_only=args.guardrails_only,
        metrics_only=args.metrics_only,
    )