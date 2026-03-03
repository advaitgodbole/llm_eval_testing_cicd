"""
Component-level evaluation: triage + explanation pipeline.

╔══════════════════════════════════════════════════════════════════╗
║  DeepEval equivalent                                            ║
║  ────────────────────────────────────────────────────────────    ║
║  Component-level evals with @observe decorator:                 ║
║    assert_test(golden=golden, observed_callback=your_llm_app)   ║
║  Metrics are defined at the span level within @observe.         ║
║                                                                  ║
║  MLflow equivalent                                               ║
║  ────────────────────────────────────────────────────────────    ║
║  Custom scorers that inspect the MLflow Trace object to check   ║
║  individual component behavior (triage accuracy, retriever      ║
║  usage, latency per stage, explanation quality, etc.).           ║
╚══════════════════════════════════════════════════════════════════╝

The pipeline under test: triage_risk → retrieve_drug_safety_docs → explain.
Each component gets its own trace span, and custom scorers inspect them.
"""

import mlflow
import pytest
from mlflow.entities import Feedback, SpanType, Trace
from mlflow.genai.scorers import RelevanceToQuery, Safety, scorer

from conftest import assert_quality_gates
from my_agent.agent import triage_then_explain


# ---------------------------------------------------------------------------
# Custom component-level scorers
# ---------------------------------------------------------------------------


@scorer
def triage_accuracy(
    inputs: dict,
    outputs: dict,
    expectations: dict,
    trace: Trace,
) -> Feedback:
    """Check if the triage component assigned the correct risk level.

    Inspects the triage span in the trace — the MLflow equivalent of
    DeepEval's @observe with span-level metrics on the classifier.
    """
    expected_level = expectations.get("expected_risk_level")
    if expected_level is None:
        return Feedback(
            name="triage_accuracy",
            value="yes",
            rationale="No expected_risk_level — skipping",
        )

    triage_spans = [
        s for s in trace.search_spans() if "triage" in s.name.lower()
    ]

    if not triage_spans:
        return Feedback(
            name="triage_accuracy",
            value="no",
            rationale="No triage span found in trace",
        )

    span_outputs = triage_spans[0].outputs or {}
    actual_level = (
        span_outputs.get("risk_level", "").strip().lower()
        if isinstance(span_outputs, dict)
        else ""
    )

    is_correct = actual_level == expected_level.strip().lower()
    return Feedback(
        name="triage_accuracy",
        value="yes" if is_correct else "no",
        rationale=f"Expected '{expected_level}', got '{actual_level}'",
    )


@scorer
def high_risk_not_undertriaged(
    inputs: dict,
    outputs: dict,
    expectations: dict,
    trace: Trace,
) -> Feedback:
    """Critical safety check: high-risk cases must never be classified lower.

    This scorer only evaluates rows where expected_risk_level == "high".
    For other rows it passes automatically.
    """
    expected = expectations.get("expected_risk_level", "").strip().lower()
    if expected != "high":
        return Feedback(
            name="high_risk_not_undertriaged",
            value="yes",
            rationale="Not a high-risk case — skipping",
        )

    triage_spans = [
        s for s in trace.search_spans() if "triage" in s.name.lower()
    ]
    actual = ""
    if triage_spans:
        span_out = triage_spans[0].outputs or {}
        actual = span_out.get("risk_level", "").strip().lower() if isinstance(span_out, dict) else ""

    if not actual:
        actual = outputs.get("risk_level", "").strip().lower()

    is_safe = actual == "high"
    return Feedback(
        name="high_risk_not_undertriaged",
        value="yes" if is_safe else "no",
        rationale=(
            f"High-risk case correctly triaged as '{actual}'"
            if is_safe
            else f"DANGER: High-risk case under-triaged as '{actual}'"
        ),
    )


@scorer
def retriever_called(trace: Trace) -> Feedback:
    """Verify that the RAG pipeline actually called the drug-safety retriever."""
    retriever_spans = trace.search_spans(span_type=SpanType.RETRIEVER)
    return Feedback(
        name="retriever_called",
        value="yes" if retriever_spans else "no",
        rationale=(
            f"Found {len(retriever_spans)} retriever span(s)"
            if retriever_spans
            else "No RETRIEVER span found in trace"
        ),
    )


@scorer
def latency_budget(trace: Trace) -> Feedback:
    """Enforce a per-request latency budget (10 seconds).

    In a pharmacovigilance context, timely classification matters.
    """
    root_spans = [s for s in trace.search_spans() if s.parent_id is None]
    if not root_spans:
        return Feedback(name="latency_budget", value="no", rationale="No root span")

    root = root_spans[0]
    duration_s = (root.end_time_ns - root.start_time_ns) / 1e9
    budget_s = 10.0

    return Feedback(
        name="latency_budget",
        value="yes" if duration_s <= budget_s else "no",
        rationale=f"Duration {duration_s:.2f}s vs budget {budget_s}s",
    )


@scorer
def response_not_empty(outputs: dict) -> Feedback:
    """Sanity check: the classifier must always produce a response."""
    response = str(outputs.get("response", ""))
    return Feedback(
        name="response_not_empty",
        value="yes" if len(response.strip()) > 0 else "no",
        rationale=f"Response length: {len(response)} chars",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.llm_eval
@pytest.mark.component
def test_triage_accuracy(component_dataset):
    """Verify the triage component assigns the correct risk level."""
    with mlflow.start_run(run_name="ci_triage_accuracy"):
        results = mlflow.genai.evaluate(
            data=component_dataset,
            predict_fn=triage_then_explain,
            scorers=[triage_accuracy],
        )

    assert_quality_gates(results, gates={"triage_accuracy": 0.75})


@pytest.mark.llm_eval
@pytest.mark.component
def test_high_risk_never_undertriaged(component_dataset):
    """Critical: high-risk cases must NEVER be classified as medium or low.

    This is the single most important quality gate in pharmacovigilance.
    A missed high-risk case could endanger patient safety.
    """
    with mlflow.start_run(run_name="ci_high_risk_safety"):
        results = mlflow.genai.evaluate(
            data=component_dataset,
            predict_fn=triage_then_explain,
            scorers=[high_risk_not_undertriaged],
        )

    assert_quality_gates(results, gates={"high_risk_not_undertriaged": 1.0})


@pytest.mark.llm_eval
@pytest.mark.component
def test_retriever_is_called(component_dataset):
    """Verify the drug-safety retriever fires for every request."""
    with mlflow.start_run(run_name="ci_retriever_called"):
        results = mlflow.genai.evaluate(
            data=component_dataset,
            predict_fn=triage_then_explain,
            scorers=[retriever_called],
        )

    assert_quality_gates(results, gates={"retriever_called": 1.0})


@pytest.mark.llm_eval
@pytest.mark.component
def test_latency_budget(component_dataset):
    """Every classification must complete within 10 seconds."""
    with mlflow.start_run(run_name="ci_latency_budget"):
        results = mlflow.genai.evaluate(
            data=component_dataset,
            predict_fn=triage_then_explain,
            scorers=[latency_budget],
        )

    assert_quality_gates(results, gates={"latency_budget": 1.0})


@pytest.mark.llm_eval
@pytest.mark.component
@pytest.mark.quality_gate
def test_full_component_pipeline(component_dataset):
    """Run all component-level scorers together.

    Comprehensive component-level evaluation — equivalent to DeepEval's
    full observed_callback test with all span-level metrics.
    """
    with mlflow.start_run(run_name="ci_full_components"):
        results = mlflow.genai.evaluate(
            data=component_dataset,
            predict_fn=triage_then_explain,
            scorers=[
                triage_accuracy,
                high_risk_not_undertriaged,
                retriever_called,
                latency_budget,
                response_not_empty,
                RelevanceToQuery(),
                Safety(),
            ],
        )

    assert_quality_gates(
        results,
        gates={
            "triage_accuracy": 0.75,
            "high_risk_not_undertriaged": 1.0,
            "retriever_called": 1.0,
            "latency_budget": 1.0,
            "response_not_empty": 1.0,
            "relevance_to_query": 0.8,
            "safety": 1.0,
        },
    )
