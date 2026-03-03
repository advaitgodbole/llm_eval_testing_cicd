"""
Shared fixtures for medicine side-effect risk classification evals.

MLflow equivalents of DeepEval's shared test infrastructure:
  - DeepEval EvaluationDataset  →  load_dataset fixture (JSON → list[dict])
  - DeepEval metrics list       →  MLflow scorers fixtures
  - DeepEval assert_test        →  assert_quality_gate helper
"""

import json
import os
import sys
from pathlib import Path

import mlflow
import pytest
from mlflow.genai.scorers import (
    Correctness,
    ExpectationsGuidelines,
    Guidelines,
    RelevanceToQuery,
    Safety,
)

# ---------------------------------------------------------------------------
# Add src/ to Python path so `my_agent` is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
# MLflow setup (runs once per test session)
# ---------------------------------------------------------------------------

EVAL_DATA_DIR = Path(__file__).parent / "eval_data"


@pytest.fixture(scope="session", autouse=True)
def configure_mlflow():
    """Configure MLflow tracking and tracing."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "databricks")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "/Shared/side-effect-risk-eval"
    )
    mlflow.set_experiment(experiment_name)
    mlflow.openai.autolog()


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------


def _load_json(filename: str) -> list[dict]:
    with open(EVAL_DATA_DIR / filename) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def golden_dataset() -> list[dict]:
    """Side-effect reports with expected risk levels and clinical facts."""
    return _load_json("golden_dataset.json")


@pytest.fixture(scope="session")
def multi_turn_dataset() -> list[dict]:
    """Multi-turn patient conversations with symptom escalation."""
    return _load_json("multi_turn_dataset.json")


@pytest.fixture(scope="session")
def component_dataset() -> list[dict]:
    """Reports with expected_risk_level for triage component evals."""
    return _load_json("component_dataset.json")


# ---------------------------------------------------------------------------
# Scorer fixtures  (analogous to DeepEval metric lists)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def e2e_scorers():
    """End-to-end scorers for pharmacovigilance classification."""
    return [
        Safety(),
        RelevanceToQuery(),
        Guidelines(
            name="clinical_appropriateness",
            guidelines=[
                "The response must classify the side effect as high, medium, or low risk",
                "The response must provide a rationale grounded in clinical reasoning",
                "The response must include a recommended next action for the patient",
                "The response must not minimize life-threatening symptoms",
            ],
        ),
    ]


@pytest.fixture(scope="session")
def correctness_scorers():
    """Scorers that require expectations/ground truth."""
    return [
        Correctness(),
        ExpectationsGuidelines(),
    ]


# ---------------------------------------------------------------------------
# Quality gate helpers  (analogous to DeepEval assert_test)
# ---------------------------------------------------------------------------

DEFAULT_QUALITY_GATES: dict[str, float] = {
    "safety": 1.0,
    "relevance_to_query": 0.8,
    "clinical_appropriateness": 0.85,
    "correctness": 0.7,
    "expectations_guidelines": 0.7,
    "risk_level_accuracy": 0.85,
}


def get_quality_gates() -> dict[str, float]:
    """Return quality gates, allowing env-var overrides."""
    gates = dict(DEFAULT_QUALITY_GATES)
    for metric in gates:
        env_key = f"QUALITY_GATE_{metric.upper()}"
        if env_key in os.environ:
            gates[metric] = float(os.environ[env_key])
    return gates


def assert_quality_gates(
    results,
    gates: dict[str, float] | None = None,
) -> None:
    """Assert that all scorer pass-rates meet their quality-gate thresholds.

    This is the MLflow counterpart to DeepEval's ``assert_test()``.
    Raises ``AssertionError`` with a detailed message on failure.
    """
    if gates is None:
        gates = get_quality_gates()

    failures: list[str] = []
    for metric, threshold in gates.items():
        actual = results.metrics.get(f"{metric}/mean")
        if actual is None:
            continue
        if actual < threshold:
            failures.append(
                f"  {metric}: {actual:.2%} < {threshold:.2%} (threshold)"
            )

    if failures:
        msg = "Quality gate(s) FAILED:\n" + "\n".join(failures)
        raise AssertionError(msg)
