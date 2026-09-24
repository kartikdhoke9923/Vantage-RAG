"""
Guardrails binary evaluation.
Sends each test input to the live /query API and checks if the guardrail fired.
Classifies each result as TP / TN / FP / FN and computes precision + recall.
"""

import copy
import time

import logfire
import requests

from evals.pipeline import REQUEST_TIMEOUT, _post_with_retry

API_URL = "http://localhost:8000/query"
DELAY_BETWEEN_CALLS = 10  # seconds — keeps us inside the backend rate limiter window


def _is_blocked(response_json: dict) -> bool:
    tp = response_json.get("thought_process") or []
    return any("guardrails fired" in step.lower() for step in tp)


def _classify_gate(response_json: dict) -> str:
    """
    Reads the explicit `gate` field (/query now reports it) to distinguish
    "gate ran and blocked/safe" from "gate did not run at all".

    Returns:
      "blocked" → guardrail fired (marker or state=="blocked")
      "safe"    → gate ran and allowed the message
      "invalid" → gate skipped / fail-open / provider error — NOT a valid
                  detection test (cannot be counted as FN/FP)
    """
    if _is_blocked(response_json):
        return "blocked"
    gate = response_json.get("gate")
    if gate in ("safe", None):
        return "safe"
    return "invalid"  # skipped-not-ready | skipped-disabled | fail-open-error | error


def run_guardrails_eval(guardrails_samples: list, progress_callback=None) -> list:
    """
    Runs each guardrails test case against the live API.
    Adds actual_blocked, gate (blocked/safe/invalid), and result (TP/TN/FP/FN/INVALID)
    to each sample in place. Returns the enriched list.
    """
    samples = copy.deepcopy(guardrails_samples)
    n = len(samples)

    with logfire.span("🛡️ Eval — Guardrails Tests", total=n):
        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i, n, sample["input"])

            with logfire.span(
                f"🛡️ Test {sample['id']}",
                input_text=sample["input"][:80],
                expected_blocked=sample["expected_blocked"],
            ):
                try:
                    resp = _post_with_retry(sample["input"], thread_id=f"guardrail_eval_{i}", timeout=REQUEST_TIMEOUT)
                    resp.raise_for_status()
                    gate_class = _classify_gate(resp.json())

                except requests.exceptions.ConnectionError:
                    logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                    gate_class = "invalid"

                except Exception as e:
                    logfire.error(f"❌ Guardrails test error: {e}")
                    gate_class = "invalid"

                expected = sample["expected_blocked"]
                sample["gate"] = gate_class
                sample["actual_blocked"] = gate_class == "blocked"

                if gate_class == "invalid":
                    sample["result"] = "INVALID"
                elif gate_class == "blocked":
                    sample["result"] = "TP" if expected else "FP"
                else:  # safe
                    sample["result"] = "TN" if not expected else "FN"

                logfire.info(
                    f"🛡️ {sample['result']} (gate={gate_class})",
                    expected_blocked=expected,
                    actual_blocked=gate_class == "blocked",
                    input_preview=sample["input"][:60],
                )

            time.sleep(DELAY_BETWEEN_CALLS)

    return samples


def compute_guardrails_metrics(results: list) -> dict:
    tp = sum(1 for r in results if r["result"] == "TP")
    tn = sum(1 for r in results if r["result"] == "TN")
    fp = sum(1 for r in results if r["result"] == "FP")
    fn = sum(1 for r in results if r["result"] == "FN")
    invalid = sum(1 for r in results if r["result"] == "INVALID")
    valid = len(results) - invalid

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / valid if valid > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid": invalid,
        "valid": valid,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "total": len(results),
        "correct": tp + tn,
    }
