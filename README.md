# Medicine Side-Effect Risk Classification — LLM Eval & CI/CD with MLflow 3.0+

This repo demonstrates how to replicate [DeepEval's CI/CD unit-testing workflow](https://deepeval.com/docs/evaluation-unit-testing-in-ci-cd) using **MLflow 3.0+ on Databricks**, applied to a pharmacovigilance use case: **classifying patient-reported medicine side effects into high, medium, and low risk.**

## The Use Case

Patients report side effects after taking medications. An LLM-based classifier triages each report:

| Risk Level | Description | Example |
|-----------|-------------|---------|
| **High** | Life-threatening or potentially disabling | Anaphylaxis, seizures, jaundice, serotonin syndrome |
| **Medium** | Concerning, needs medical attention, not immediately life-threatening | Widespread rash (no blisters), persistent vomiting, vision changes |
| **Low** | Mild, expected, typically self-resolving | Headache, drowsiness, dry mouth, injection-site soreness |

The most critical quality gate: **high-risk cases must never be under-triaged.**

## Quick Start

```bash
# Install
pip install ".[dev]"

# Set environment
export OPENAI_API_KEY="sk-..."
export MLFLOW_TRACKING_URI="databricks"
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="dapi..."

# Run all evals
pytest tests/ -m llm_eval -v

# Run only quality-gate tests
pytest tests/ -m quality_gate -v

# Run component-level evals only
pytest tests/ -m component -v
```

## Project Structure

```
.
├── src/my_agent/
│   ├── __init__.py
│   └── agent.py                  # Side-effect classifier, RAG classifier, triage component
├── tests/
│   ├── conftest.py               # Shared fixtures, quality gate helpers
│   ├── eval_data/
│   │   ├── golden_dataset.json   # 12 patient reports with expected risk levels
│   │   ├── multi_turn_dataset.json  # Symptom-escalation conversations
│   │   └── component_dataset.json   # Reports for triage component testing
│   ├── test_e2e_single_turn.py   # End-to-end classification evals
│   ├── test_e2e_multi_turn.py    # Multi-turn escalation evals
│   └── test_component_level.py   # Component-level (triage + retriever) evals
├── notebooks/
│   └── run_all_evals.py          # Databricks notebook orchestrator (imports from project)
├── .github/workflows/
│   └── llm_eval.yml              # GitHub Actions CI/CD pipeline
├── pyproject.toml
└── README.md
```

---

## How the Unit Testing Workflow Works

The codebase is organized into four layers: **agents under test**, **scorers and quality gates**, **test suites**, and **CI/CD automation**. Each layer maps directly to a DeepEval concept, implemented natively in MLflow 3.0+.

### Layer 1: Agents Under Test

Five agent functions are defined in `src/my_agent/agent.py`, all decorated with `@mlflow.trace` so every invocation produces a full execution trace:

| Agent | Purpose | Trace Shape |
|-------|---------|-------------|
| `side_effect_classifier` | Direct LLM classification (end-to-end) | Single LLM span |
| `rag_side_effect_agent` | Retrieves from drug safety KB, then classifies | RETRIEVER span → LLM span |
| `triage_risk` | Lightweight component returning only the risk level | Single LLM span |
| `triage_then_explain` | Two-stage pipeline: `triage_risk` → `rag_side_effect_agent` | Multiple nested spans |
| `multi_turn_triage_agent` | Multi-turn conversation agent for escalation scenarios | Single LLM span per turn |

The `configure(client, model)` function makes the module portable — in CI it defaults to OpenAI; on Databricks it points at a Foundation Model API endpoint.

```
Patient Report
     │
     ├──→ [triage_risk]              Quick risk level (high/medium/low)
     │         │                     Component-level eval target
     │         ▼
     └──→ [rag_side_effect_agent]    Full classification with RAG
              │
              ├── [retrieve_drug_safety_docs]   RETRIEVER span
              │         │
              │         ▼
              └── [LLM classify]     Risk level + rationale + recommended action
                       │
                       ▼
              { risk_level, rationale, recommended_action }
```

The `triage_then_explain` pipeline is the key composition that enables component-level testing — each stage (`triage_risk`, `retrieve_drug_safety_docs`, the final LLM call) gets its own trace span, and scorers can inspect any of them independently.

### Layer 2: Scorers and Quality Gates

Scorers are the MLflow equivalent of DeepEval's metrics. The codebase uses two categories:

#### Built-in MLflow Scorers

| MLflow Scorer | DeepEval Equivalent | What It Checks |
|---------------|--------------------|----|
| `Safety()` | `ToxicityMetric` | Response is free of harmful content |
| `RelevanceToQuery()` | `AnswerRelevancyMetric` | Response addresses the patient's report |
| `Correctness()` | `CorrectnessMetric` | Rationale includes `expected_facts` from ground truth |
| `ExpectationsGuidelines()` | Per-row `GEval` | Response follows per-row `guidelines` (e.g. "must recommend ER") |
| `Guidelines(name, guidelines)` | `GEval` with custom criteria | Global guidelines like `clinical_appropriateness` |

#### Custom Domain-Specific Scorers

Custom scorers are defined with the `@scorer` decorator and return a `Feedback` object (`"yes"` / `"no"` with a rationale). These are where domain-specific evaluation logic lives:

**`risk_level_accuracy`** (in `test_e2e_single_turn.py`) — a deterministic scorer that compares the agent's structured `risk_level` output against the expected ground truth. No LLM judge needed:

```python
@scorer
def risk_level_accuracy(inputs: dict, outputs: dict, expectations: dict) -> Feedback:
    expected = expectations.get("expected_risk_level")
    actual = outputs.get("risk_level", "").strip().lower()
    is_correct = actual == expected.strip().lower()
    return Feedback(
        name="risk_level_accuracy",
        value="yes" if is_correct else "no",
        rationale=f"Expected '{expected}', got '{actual}'",
    )
```

**Component-level scorers** (in `test_component_level.py`) are the most powerful pattern — they accept a `trace: Trace` parameter and reach *inside* the pipeline execution to evaluate individual stages. This is the MLflow equivalent of DeepEval's `@observe` decorator:

| Scorer | What It Inspects |
|--------|-----------------|
| `triage_accuracy` | Finds the `triage_risk` span by name and checks its output against ground truth |
| `high_risk_not_undertriaged` | Same span inspection, but only for high-risk cases — fails if a high-risk case was classified as anything other than "high" |
| `retriever_called` | Checks for the presence of a `RETRIEVER`-typed span |
| `latency_budget` | Reads the root span's timing to enforce a 10-second budget |
| `response_not_empty` | Sanity check on the final output |

For example, `triage_accuracy` reaches into the trace to check what the triage *component* did, even though the overall pipeline returned a different composite output:

```python
@scorer
def triage_accuracy(inputs, outputs, expectations, trace: Trace) -> Feedback:
    triage_spans = [s for s in trace.search_spans() if "triage" in s.name.lower()]
    span_outputs = triage_spans[0].outputs or {}
    actual_level = span_outputs.get("risk_level", "").strip().lower()
    # ... compare against expectations["expected_risk_level"]
```

#### Quality Gates (`assert_quality_gates`)

The `assert_quality_gates` function in `conftest.py` is the direct counterpart to DeepEval's `assert_test()`. It takes the results from `mlflow.genai.evaluate()`, checks the mean pass-rate of each scorer against a threshold, and raises `AssertionError` if any threshold is breached — which means **pytest treats it as a test failure**:

```python
def assert_quality_gates(results, gates=None):
    for metric, threshold in gates.items():
        actual = results.metrics.get(f"{metric}/mean")
        if actual is not None and actual < threshold:
            failures.append(f"  {metric}: {actual:.2%} < {threshold:.2%}")
    if failures:
        raise AssertionError("Quality gate(s) FAILED:\n" + "\n".join(failures))
```

This is the bridge between LLM evaluation and unit testing — every `mlflow.genai.evaluate()` call produces metrics, and `assert_quality_gates` turns them into pass/fail assertions.

### Layer 3: The Three Test Suites

Every test function follows the same three-step pattern:

1. **Open an MLflow run** — `with mlflow.start_run(run_name="ci_...")`
2. **Batch-evaluate** — `mlflow.genai.evaluate(data=dataset, predict_fn=agent, scorers=[...])`
3. **Assert quality gates** — `assert_quality_gates(results, gates={...})`

#### End-to-End Single-Turn (`test_e2e_single_turn.py`) — 7 tests

These test the agent as a black box with increasing sophistication:

| Test | What It Proves |
|------|---------------|
| `test_classifier_golden_dataset` | Comprehensive: safety + relevance + clinical guidelines + risk accuracy across all 12 cases |
| `test_clinical_correctness` | The rationale mentions the expected clinical facts (uses `Correctness()`) |
| `test_rag_classifier` | Same battery but against the RAG variant — does retrieval help? |
| `test_per_row_guidelines` | Per-case guidelines like "must recommend ER" (uses `ExpectationsGuidelines()`) |
| `test_safety_gate` | Zero-tolerance: 100% safety or the build breaks |
| `test_high_risk_recall` | Filters to only high-risk ground-truth cases and demands 100% accuracy — the domain-critical test |
| `test_ab_comparison` | Runs both agents on the same data and prints accuracy side-by-side — an A/B test |

#### End-to-End Multi-Turn (`test_e2e_multi_turn.py`) — 1 test

Simulates patients whose symptoms worsen across conversation turns. The `escalation_awareness` guideline scorer checks that the agent escalates the risk level rather than repeating the initial low-risk assessment. The `clinical_action` scorer verifies the response includes a specific recommended action.

#### Component-Level (`test_component_level.py`) — 5 tests

This is where the pattern goes beyond what most LLM testing frameworks offer. The tests run the full `triage_then_explain` pipeline, but the scorers inspect *individual spans* within the trace:

| Test | What Span It Inspects |
|------|-----------------------|
| `test_triage_accuracy` | The `triage_risk` span's output — did the triage stage get it right? |
| `test_high_risk_never_undertriaged` | The triage span, but only for high-risk cases, at a 100% threshold |
| `test_retriever_is_called` | Checks for the presence of a `RETRIEVER`-typed span |
| `test_latency_budget` | Root span's timing — was the whole pipeline under 10 seconds? |
| `test_full_component_pipeline` | All of the above plus `Safety()` and `RelevanceToQuery()` in one run |

### Layer 4: CI/CD Automation

The `.github/workflows/llm_eval.yml` wires everything into GitHub Actions with three parallel jobs:

1. **End-to-End Evals** — runs `test_e2e_single_turn.py` and `test_e2e_multi_turn.py`
2. **Component-Level Evals** — runs `test_component_level.py`
3. **Quality Gate Check** — blocks the merge if either eval job failed

The e2e and component jobs run in parallel. If any `assert_quality_gates` raises `AssertionError`, pytest fails, the job fails, and the summary gate blocks the PR.

### How a Single Test Flows End-to-End

Taking `test_high_risk_recall` as a concrete example:

1. **pytest** collects the test; the `configure_mlflow` autouse fixture sets the tracking URI and experiment
2. The test filters `golden_dataset.json` to only high-risk cases (4 out of 12 rows)
3. `mlflow.genai.evaluate()` iterates over those 4 rows, calling `side_effect_classifier(patient_report=...)` for each
4. Each call is traced via `@mlflow.trace`, producing a `Trace` object logged to MLflow
5. The `risk_level_accuracy` scorer runs on each result, comparing `outputs["risk_level"]` to `expectations["expected_risk_level"]`
6. MLflow aggregates the scores into `results.metrics["risk_level_accuracy/mean"]`
7. `assert_quality_gates(results, gates={"risk_level_accuracy": 1.0})` checks that 100% of high-risk cases were correctly classified
8. If even one was under-triaged, the assertion fails, pytest reports a failure, and in CI the build is blocked

This is the same workflow as DeepEval's `assert_test(test_case, metrics=[...])`, but using MLflow's native evaluation engine with all traces and metrics logged to a centralized experiment for observability.

---

## Concept Mapping: DeepEval to MLflow 3.0+

| DeepEval | MLflow 3.0+ | This Project |
|----------|-------------|--------------|
| `EvaluationDataset` / `Golden` | `list[dict]` with `inputs`/`expectations` | `golden_dataset.json` with `patient_report` + `expected_risk_level` |
| `LLMTestCase` | Data record `{"inputs": {...}, "expectations": {...}}` | Auto-built from JSON records |
| `assert_test(test_case, metrics)` | `mlflow.genai.evaluate()` + `assert_quality_gates()` | Batch eval with domain-specific thresholds |
| `AnswerRelevancyMetric` | `RelevanceToQuery()` | Used in e2e tests |
| `CorrectnessMetric` | `Correctness()` | Validates clinical facts in rationale |
| `ToxicityMetric` | `Safety()` | Zero-tolerance gate |
| `GEval` (custom criteria) | `Guidelines(name, guidelines)` | `clinical_appropriateness`, `escalation_awareness` |
| `@observe` span metrics | `@scorer` + `Trace` inspection | `triage_accuracy`, `high_risk_not_undertriaged` |
| `deepeval test run` | `pytest -m llm_eval` | Standard pytest |
| Confident AI dashboard | MLflow experiment UI | All results in Databricks |

---

## Quality Gates

### Default Thresholds

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| `safety` | 100% | Classifier must never produce harmful content |
| `risk_level_accuracy` | 85% | Overall classification accuracy |
| `clinical_appropriateness` | 85% | Rationale and action must be clinically sound |
| `high_risk_not_undertriaged` | 100% | **Critical**: never miss a high-risk case |
| `relevance_to_query` | 80% | Response must address the reported symptoms |
| `escalation_awareness` | 80% | Multi-turn: must escalate when symptoms worsen |

### Customization

Override per test:
```python
assert_quality_gates(results, gates={"risk_level_accuracy": 0.95})
```

Override via environment:
```bash
QUALITY_GATE_RISK_LEVEL_ACCURACY=0.95 pytest tests/ -m quality_gate
```

---

## Databricks Deployment

The `notebooks/run_all_evals.py` notebook is a thin orchestrator designed to run on Databricks. It does **not** duplicate any agent or test code — instead it:

1. Installs dependencies (`mlflow>=3.8.0`, `openai`, `pytest`)
2. Adds the deployed `src/` and `tests/` directories to `sys.path`
3. Configures the agent module with `configure(client, model)` pointing at a Foundation Model API endpoint
4. Imports agents, scorers, and the quality-gate helper from the project modules
5. Runs all 9 evaluation scenarios under a single MLflow experiment

The full project (source, tests, eval data) is uploaded as workspace files and the notebook imports from them.

---

## CI/CD Pipeline

The `.github/workflows/llm_eval.yml` runs three jobs:

1. **End-to-End Evals** — single-turn and multi-turn classification tests (parallel)
2. **Component-Level Evals** — triage accuracy, retriever, latency (parallel)
3. **Quality Gate Check** — fails the build if any job failed

### Required Secrets

| Secret | Purpose |
|--------|---------|
| `OPENAI_API_KEY` | LLM calls (or replace with Databricks Foundation Model API) |
| `DATABRICKS_HOST` | MLflow tracking server |
| `DATABRICKS_TOKEN` | Databricks authentication |

---

## Extending

### Add a new side-effect test case

Add to `tests/eval_data/golden_dataset.json`:
```json
{
  "inputs": {"patient_report": "Patient report text..."},
  "expectations": {
    "expected_risk_level": "high",
    "expected_facts": ["Clinical fact 1"],
    "guidelines": ["Must recommend emergency care"]
  }
}
```

### Add a domain-specific scorer

```python
@scorer
def mentions_drug_interaction(inputs: dict, outputs: dict) -> Feedback:
    report = inputs.get("patient_report", "").lower()
    response = outputs.get("response", "").lower()
    if "interaction" in report and "interaction" not in response:
        return Feedback(value="no", rationale="Report mentions interaction but response doesn't address it")
    return Feedback(value="yes", rationale="OK")
```

### Persist datasets in Unity Catalog

```python
import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.create_dataset(
    uc_table_name="pharma.safety.side_effect_eval_v1"
)
eval_dataset.merge_records(golden_data)
```
