"""
End-to-end single-turn evaluation tests for side-effect risk classification.

╔══════════════════════════════════════════════════════════════════╗
║  DeepEval equivalent                                            ║
║  ────────────────────────────────────────────────────────────    ║
║  @pytest.mark.parametrize("golden", dataset.goldens)            ║
║  def test_llm_app(golden):                                      ║
║      res = your_llm_app(golden.input)                           ║
║      test_case = LLMTestCase(...)                                ║
║      assert_test(test_case, metrics=[AnswerRelevancyMetric()])  ║
║                                                                  ║
║  MLflow equivalent                                               ║
║  ────────────────────────────────────────────────────────────    ║
║  mlflow.genai.evaluate(data=dataset, predict_fn=agent,          ║
║                        scorers=[RelevanceToQuery(), ...])       ║
║  assert_quality_gates(results)                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import mlflow
import pytest
from mlflow.genai.scorers import (
    Correctness,
    ExpectationsGuidelines,
    Guidelines,
    RelevanceToQuery,
    Safety,
)
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

from conftest import assert_quality_gates
from my_agent.agent import side_effect_classifier, rag_side_effect_agent  # noqa: E402


# ---------------------------------------------------------------------------
# Domain-specific scorer: risk-level accuracy
# ---------------------------------------------------------------------------

@scorer
def risk_level_accuracy(inputs: dict, outputs: dict, expectations: dict) -> Feedback:
    """Check if the classified risk level matches the expected ground truth."""
    expected = expectations.get("expected_risk_level")
    if expected is None:
        return Feedback(
            name="risk_level_accuracy",
            value="yes",
            rationale="No expected_risk_level provided — skipping",
        )

    actual = outputs.get("risk_level", "").strip().lower()
    expected = expected.strip().lower()
    is_correct = actual == expected

    return Feedback(
        name="risk_level_accuracy",
        value="yes" if is_correct else "no",
        rationale=f"Expected '{expected}', got '{actual}'",
    )


# ── Test 1: Full golden-dataset evaluation ────────────────────────

@pytest.mark.llm_eval
@pytest.mark.quality_gate
def test_classifier_golden_dataset(golden_dataset, e2e_scorers):
    """Evaluate the side-effect classifier over the full golden dataset."""
    with mlflow.start_run(run_name="ci_classifier_golden"):
        results = mlflow.genai.evaluate(
            data=golden_dataset,
            predict_fn=side_effect_classifier,
            scorers=e2e_scorers + [risk_level_accuracy],
        )

    assert_quality_gates(
        results,
        gates={
            "safety": 1.0,
            "relevance_to_query": 0.8,
            "clinical_appropriateness": 0.85,
            "risk_level_accuracy": 0.85,
        },
    )


# ── Test 2: Correctness with clinical ground truth ────────────────

@pytest.mark.llm_eval
@pytest.mark.quality_gate
def test_clinical_correctness(golden_dataset):
    """Verify the classifier's rationale includes expected clinical facts."""
    data_with_facts = [
        row for row in golden_dataset
        if row.get("expectations", {}).get("expected_facts")
    ]

    with mlflow.start_run(run_name="ci_clinical_correctness"):
        results = mlflow.genai.evaluate(
            data=data_with_facts,
            predict_fn=side_effect_classifier,
            scorers=[Correctness()],
        )

    assert_quality_gates(results, gates={"correctness": 0.7})


# ── Test 3: RAG-based classifier with drug safety KB ──────────────

@pytest.mark.llm_eval
@pytest.mark.quality_gate
def test_rag_classifier(golden_dataset, e2e_scorers):
    """Evaluate the RAG-based classifier (retrieves drug safety docs first)."""
    with mlflow.start_run(run_name="ci_rag_classifier"):
        results = mlflow.genai.evaluate(
            data=golden_dataset,
            predict_fn=rag_side_effect_agent,
            scorers=e2e_scorers + [risk_level_accuracy],
        )

    assert_quality_gates(
        results,
        gates={
            "safety": 1.0,
            "clinical_appropriateness": 0.85,
            "risk_level_accuracy": 0.85,
        },
    )


# ── Test 4: Per-row clinical guidelines ───────────────────────────

@pytest.mark.llm_eval
def test_per_row_guidelines(golden_dataset):
    """Evaluate per-report clinical guidelines (e.g. 'must recommend ER')."""
    data_with_guidelines = [
        row for row in golden_dataset
        if row.get("expectations", {}).get("guidelines")
    ]

    with mlflow.start_run(run_name="ci_clinical_guidelines"):
        results = mlflow.genai.evaluate(
            data=data_with_guidelines,
            predict_fn=side_effect_classifier,
            scorers=[ExpectationsGuidelines()],
        )

    assert_quality_gates(results, gates={"expectations_guidelines": 0.7})


# ── Test 5: Safety gate — zero tolerance ──────────────────────────

@pytest.mark.llm_eval
@pytest.mark.quality_gate
def test_safety_gate(golden_dataset):
    """Safety must pass at 100% — the classifier must never produce harmful content."""
    with mlflow.start_run(run_name="ci_safety_gate"):
        results = mlflow.genai.evaluate(
            data=golden_dataset,
            predict_fn=side_effect_classifier,
            scorers=[Safety()],
        )

    assert_quality_gates(results, gates={"safety": 1.0})


# ── Test 6: High-risk recall — never under-triage critical cases ──

@pytest.mark.llm_eval
@pytest.mark.quality_gate
def test_high_risk_recall(golden_dataset):
    """Every truly high-risk case MUST be classified as high.

    Under-triaging a high-risk case is the worst failure mode in
    pharmacovigilance. This test isolates high-risk ground-truth
    cases and demands 100% recall.
    """
    high_risk_cases = [
        row for row in golden_dataset
        if row.get("expectations", {}).get("expected_risk_level") == "high"
    ]

    with mlflow.start_run(run_name="ci_high_risk_recall"):
        results = mlflow.genai.evaluate(
            data=high_risk_cases,
            predict_fn=side_effect_classifier,
            scorers=[risk_level_accuracy],
        )

    assert_quality_gates(results, gates={"risk_level_accuracy": 1.0})


# ── Test 7: A/B comparison — plain vs RAG classifier ─────────────

@pytest.mark.llm_eval
def test_ab_comparison(golden_dataset):
    """Compare plain classifier vs RAG-based classifier on accuracy."""
    with mlflow.start_run(run_name="ci_ab_plain_classifier"):
        results_plain = mlflow.genai.evaluate(
            data=golden_dataset,
            predict_fn=side_effect_classifier,
            scorers=[risk_level_accuracy, Safety()],
        )

    with mlflow.start_run(run_name="ci_ab_rag_classifier"):
        results_rag = mlflow.genai.evaluate(
            data=golden_dataset,
            predict_fn=rag_side_effect_agent,
            scorers=[risk_level_accuracy, Safety()],
        )

    plain_acc = results_plain.metrics.get("risk_level_accuracy/mean", 0)
    rag_acc = results_rag.metrics.get("risk_level_accuracy/mean", 0)

    print(f"Plain classifier accuracy: {plain_acc:.2%}")
    print(f"RAG classifier accuracy:   {rag_acc:.2%}")
