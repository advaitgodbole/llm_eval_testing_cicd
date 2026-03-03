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
├── .github/workflows/
│   └── llm_eval.yml              # GitHub Actions CI/CD pipeline
├── pyproject.toml
└── README.md
```

## Agent Architecture

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

Three agents are provided:
- **`side_effect_classifier`** — direct LLM classification (end-to-end)
- **`rag_side_effect_agent`** — retrieves from drug safety KB first, then classifies
- **`triage_risk`** — lightweight component that outputs only the risk level

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

## Test Suite Overview

### End-to-End Single Turn (`test_e2e_single_turn.py`)

| Test | What it verifies |
|------|-----------------|
| `test_classifier_golden_dataset` | Full dataset: safety + relevance + clinical quality + risk accuracy |
| `test_clinical_correctness` | Rationale includes expected clinical facts |
| `test_rag_classifier` | RAG-based classifier accuracy and quality |
| `test_per_row_guidelines` | Per-report clinical guidelines (e.g. "must recommend ER") |
| `test_safety_gate` | Zero-tolerance safety check |
| `test_high_risk_recall` | 100% recall on high-risk cases — most critical gate |
| `test_ab_comparison` | Plain vs RAG classifier accuracy comparison |

### End-to-End Multi Turn (`test_e2e_multi_turn.py`)

| Test | What it verifies |
|------|-----------------|
| `test_multi_turn_escalation` | Agent correctly escalates risk when symptoms worsen across turns |

### Component Level (`test_component_level.py`)

| Test | What it verifies |
|------|-----------------|
| `test_triage_accuracy` | Triage component assigns correct risk level |
| `test_high_risk_never_undertriaged` | High-risk cases never classified lower (100% gate) |
| `test_retriever_is_called` | Drug safety retriever span fires every time |
| `test_latency_budget` | Each classification completes within 10 seconds |
| `test_full_component_pipeline` | All component scorers together in one run |

---

## CI/CD Pipeline

The `.github/workflows/llm_eval.yml` runs three parallel jobs:

1. **End-to-End Evals** — single-turn and multi-turn classification tests
2. **Component-Level Evals** — triage accuracy, retriever, latency
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
