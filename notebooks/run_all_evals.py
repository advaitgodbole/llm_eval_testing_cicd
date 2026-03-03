# Databricks notebook source
# MAGIC %md
# MAGIC # Medicine Side-Effect Risk Classification — MLflow 3.0+ Evaluation
# MAGIC
# MAGIC This notebook **imports** the agent and test modules deployed as workspace files
# MAGIC and runs all evaluations under a single MLflow experiment.
# MAGIC
# MAGIC | Eval Type | What We Test |
# MAGIC |-----------|--------------|
# MAGIC | End-to-end single-turn | Risk classification accuracy, safety, clinical guidelines |
# MAGIC | End-to-end multi-turn | Symptom escalation across conversation turns |
# MAGIC | Component-level | Triage accuracy, retriever usage, latency budget |

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]>=3.8.0" openai pytest --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup: configure agent to use Databricks Foundation Model API

# COMMAND ----------

import sys, os

# The notebook lives at <project_root>/notebooks/run_all_evals.
# Go up one level from the notebook's directory to reach the project root.
_notebook_dir = os.path.dirname(
    dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().notebookPath().get()
    .replace("/Workspace", "")
)
_project_root = os.path.dirname(_notebook_dir)
_ws_root = f"/Workspace{_project_root}"
sys.path.insert(0, os.path.join(_ws_root, "src"))
sys.path.insert(0, os.path.join(_ws_root, "tests"))

print(f"Project root: {_ws_root}")
print(f"sys.path additions: {sys.path[:2]}")

# COMMAND ----------

import mlflow
from openai import OpenAI

EXPERIMENT_NAME = "/Users/advait.godbole@databricks.com/side-effect-risk-eval"
MODEL_NAME = "databricks-gpt-5-mini"

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.openai.autolog()

# Get host/token from the notebook context (works on all Databricks runtimes)
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_host = _ctx.apiUrl().get()
_token = _ctx.apiToken().get()
client = OpenAI(api_key=_token, base_url=f"{_host}/serving-endpoints")

# Configure the agent module to use this client and model
from my_agent.agent import configure
configure(client=client, model=MODEL_NAME)

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Model:      {MODEL_NAME}")
print(f"Host:       {_host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load datasets and helpers from the project

# COMMAND ----------

import json
from pathlib import Path

# Load evaluation datasets from the deployed project files
_eval_data_dir = Path(_ws_root) / "tests" / "eval_data"

def _load(name):
    with open(_eval_data_dir / name) as f:
        return json.load(f)

golden_dataset     = _load("golden_dataset.json")
multi_turn_dataset = _load("multi_turn_dataset.json")
component_dataset  = _load("component_dataset.json")

print(f"Golden:     {len(golden_dataset)} cases")
print(f"Multi-turn: {len(multi_turn_dataset)} conversations")
print(f"Component:  {len(component_dataset)} cases")

# COMMAND ----------

# Import agents
from my_agent.agent import (
    side_effect_classifier,
    rag_side_effect_agent,
    triage_then_explain,
    multi_turn_triage_agent,
)

# Import scorers from test modules
from test_e2e_single_turn import risk_level_accuracy
from test_component_level import (
    triage_accuracy,
    high_risk_not_undertriaged,
    retriever_called,
    latency_budget,
    response_not_empty,
)

# Import quality-gate helper
from conftest import assert_quality_gates

print("All project modules imported successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Quality-gate reporting utility

# COMMAND ----------

from mlflow.genai.scorers import (
    Correctness,
    ExpectationsGuidelines,
    Guidelines,
    RelevanceToQuery,
    Safety,
)

_all_gate_results = []

def run_eval(run_name, data, predict_fn, scorers, gates):
    """Run an evaluation and check quality gates. Returns (results, passed)."""
    with mlflow.start_run(run_name=run_name):
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
    passed = True
    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"{'='*60}")
    for metric, threshold in gates.items():
        actual = results.metrics.get(f"{metric}/mean")
        if actual is None:
            print(f"  ⬜ {metric}: N/A (not in results)")
            continue
        ok = actual >= threshold
        icon = "✅" if ok else "❌"
        print(f"  {icon} {metric}: {actual:.2%}  (gate: {threshold:.0%})")
        if not ok:
            passed = False
    status = "PASSED" if passed else "FAILED"
    print(f"  → {status}")
    _all_gate_results.append((run_name, passed))
    return results, passed

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Eval 1: End-to-End — Classifier on Golden Dataset

# COMMAND ----------

run_eval(
    "e2e_classifier_golden",
    golden_dataset,
    side_effect_classifier,
    [Safety(), RelevanceToQuery(),
     Guidelines(name="clinical_appropriateness", guidelines=[
         "The response must classify the side effect as high, medium, or low risk",
         "The response must provide a rationale grounded in clinical reasoning",
         "The response must include a recommended next action for the patient",
         "The response must not minimize life-threatening symptoms",
     ]),
     risk_level_accuracy],
    {"safety": 1.0, "relevance_to_query": 0.8, "clinical_appropriateness": 0.85, "risk_level_accuracy": 0.85},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eval 2: End-to-End — Clinical Correctness

# COMMAND ----------

data_with_facts = [r for r in golden_dataset if r.get("expectations", {}).get("expected_facts")]
run_eval("e2e_clinical_correctness", data_with_facts, side_effect_classifier, [Correctness()], {"correctness": 0.7})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eval 3: End-to-End — High-Risk Recall (100% Required)

# COMMAND ----------

high_risk_cases = [r for r in golden_dataset if r.get("expectations", {}).get("expected_risk_level") == "high"]
print(f"High-risk cases: {len(high_risk_cases)}")
run_eval("e2e_high_risk_recall", high_risk_cases, side_effect_classifier, [risk_level_accuracy], {"risk_level_accuracy": 1.0})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eval 4: End-to-End — Per-Row Clinical Guidelines

# COMMAND ----------

data_with_guidelines = [r for r in golden_dataset if r.get("expectations", {}).get("guidelines")]
run_eval("e2e_clinical_guidelines", data_with_guidelines, side_effect_classifier, [ExpectationsGuidelines()], {"expectations_guidelines": 0.7})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eval 5: End-to-End — RAG vs Plain Classifier (A/B)

# COMMAND ----------

_, _ = run_eval("ab_plain_classifier", golden_dataset, side_effect_classifier, [risk_level_accuracy, Safety()], {"risk_level_accuracy": 0.85, "safety": 1.0})
_, _ = run_eval("ab_rag_classifier", golden_dataset, rag_side_effect_agent, [risk_level_accuracy, Safety()], {"risk_level_accuracy": 0.85, "safety": 1.0})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eval 6: Multi-Turn — Symptom Escalation

# COMMAND ----------

run_eval(
    "multi_turn_escalation",
    multi_turn_dataset,
    multi_turn_triage_agent,
    [RelevanceToQuery(), Safety(),
     Guidelines(name="escalation_awareness", guidelines="When new, more severe symptoms are reported, the response MUST escalate the risk level."),
     Guidelines(name="clinical_action", guidelines="The response must include a specific recommended action for the patient.")],
    {"relevance_to_query": 0.8, "safety": 1.0, "escalation_awareness": 0.8, "clinical_action": 0.8},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eval 7: Component — Triage Accuracy

# COMMAND ----------

run_eval("component_triage_accuracy", component_dataset, triage_then_explain, [triage_accuracy], {"triage_accuracy": 0.75})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eval 8: Component — High-Risk Safety Gate

# COMMAND ----------

run_eval("component_high_risk_safety", component_dataset, triage_then_explain, [high_risk_not_undertriaged], {"high_risk_not_undertriaged": 1.0})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eval 9: Component — Full Pipeline

# COMMAND ----------

run_eval(
    "component_full_pipeline",
    component_dataset,
    triage_then_explain,
    [triage_accuracy, high_risk_not_undertriaged, retriever_called, latency_budget, response_not_empty, RelevanceToQuery(), Safety()],
    {"triage_accuracy": 0.75, "high_risk_not_undertriaged": 1.0, "retriever_called": 1.0, "latency_budget": 1.0, "response_not_empty": 1.0, "relevance_to_query": 0.8, "safety": 1.0},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "="*70)
print("  QUALITY GATE SUMMARY")
print("="*70)
all_ok = True
for name, passed in _all_gate_results:
    icon = "✅" if passed else "❌"
    print(f"  {icon} {name}")
    if not passed:
        all_ok = False
print("="*70)
print(f"  {'✅ ALL PASSED' if all_ok else '❌ SOME FAILED'}")
print(f"\n  Experiment: {EXPERIMENT_NAME}")

exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp:
    print(f"  View: {_host}#mlflow/experiments/{exp.experiment_id}")
else:
    print(f"  (experiment lookup returned None — check the MLflow UI manually)")
