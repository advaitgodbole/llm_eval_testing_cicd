"""
Medicine side-effect risk classification agent.

Components:
  - side_effect_classifier:     end-to-end risk classifier (high/medium/low)
  - rag_side_effect_agent:      RAG classifier with drug safety knowledge base
  - triage_risk:                standalone triage component (risk level only)
  - triage_then_explain:        two-stage pipeline (triage → RAG explain)
  - multi_turn_triage_agent:    multi-turn conversation agent

All functions use @mlflow.trace for full execution tracing.

Configuration:
  Call ``configure(client, model)`` before using any agent function.
  On Databricks this points at a Foundation Model API endpoint;
  in CI/local it points at OpenAI.
"""

import json

import mlflow
from openai import OpenAI

_client: OpenAI | None = None
_model: str = "gpt-4o-mini"


def configure(client: OpenAI, model: str = "gpt-4o-mini") -> None:
    """Set the OpenAI-compatible client and model used by all agent functions."""
    global _client, _model
    _client = client
    _model = model


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


CLASSIFICATION_SYSTEM_PROMPT = """\
You are a pharmacovigilance assistant that classifies medicine side effects
reported by patients into risk levels.

Given a patient report describing symptoms or side effects after taking a
medication, classify the report into exactly one risk level:

  - high: Life-threatening or potentially disabling symptoms (e.g. chest pain,
    difficulty breathing, seizures, anaphylaxis, severe bleeding, organ failure
    symptoms, suicidal ideation).
  - medium: Symptoms that are concerning and may need medical attention but are
    not immediately life-threatening (e.g. persistent vomiting, rash spreading
    over large areas, significant swelling, high fever, vision changes,
    irregular heartbeat).
  - low: Mild or expected side effects that typically resolve on their own
    (e.g. slight headache, mild nausea, drowsiness, dry mouth, minor stomach
    discomfort, temporary dizziness).

Your response MUST be valid JSON with the following structure:
{
  "risk_level": "high" | "medium" | "low",
  "rationale": "Brief clinical reasoning for the classification",
  "recommended_action": "What the patient should do next"
}
"""


def _parse_classification(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"risk_level": "unknown", "rationale": raw, "recommended_action": ""}
    return {
        "response": raw,
        "risk_level": parsed.get("risk_level", "unknown"),
        "rationale": parsed.get("rationale", ""),
        "recommended_action": parsed.get("recommended_action", ""),
    }


# ---------------------------------------------------------------------------
# End-to-end: side-effect risk classifier
# ---------------------------------------------------------------------------

@mlflow.trace
def side_effect_classifier(patient_report: str) -> dict:
    """Classify a patient-reported side effect into high/medium/low risk."""
    completion = _get_client().chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": patient_report},
        ],
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    return _parse_classification(completion.choices[0].message.content)


# ---------------------------------------------------------------------------
# RAG-based side-effect classifier with drug safety knowledge base
# ---------------------------------------------------------------------------

DRUG_SAFETY_KB = [
    {
        "id": "kb_anaphylaxis",
        "content": (
            "Anaphylaxis is a severe, potentially life-threatening allergic "
            "reaction that can occur within seconds or minutes of exposure. "
            "Symptoms include difficulty breathing, swelling of the throat, "
            "rapid pulse, dizziness, and drop in blood pressure. Drugs commonly "
            "associated: penicillin, NSAIDs, sulfa drugs. Risk level: HIGH. "
            "Action: Call emergency services immediately."
        ),
    },
    {
        "id": "kb_serotonin_syndrome",
        "content": (
            "Serotonin syndrome results from excess serotonergic activity, "
            "often caused by combining SSRIs with MAOIs or triptans. Symptoms "
            "include agitation, rapid heartbeat, high blood pressure, dilated "
            "pupils, muscle twitching, heavy sweating, and hyperthermia. "
            "Risk level: HIGH. Action: Seek emergency care; discontinue "
            "offending agents."
        ),
    },
    {
        "id": "kb_hepatotoxicity",
        "content": (
            "Drug-induced hepatotoxicity can manifest as jaundice, dark urine, "
            "abdominal pain, nausea, and fatigue. Common offenders include "
            "acetaminophen (overdose), statins, anti-TB drugs (isoniazid, "
            "rifampin), and methotrexate. Risk level: HIGH if jaundice is "
            "present; MEDIUM for mild enzyme elevation. Action: Stop medication "
            "and consult physician urgently."
        ),
    },
    {
        "id": "kb_gi_upset",
        "content": (
            "Gastrointestinal side effects like mild nausea, stomach discomfort, "
            "and diarrhea are among the most common drug reactions. Frequently "
            "seen with NSAIDs, antibiotics (especially erythromycin), and "
            "metformin. Risk level: LOW if mild and self-limiting. Action: Take "
            "with food; consult doctor if symptoms persist beyond 48 hours."
        ),
    },
    {
        "id": "kb_skin_rash",
        "content": (
            "Drug-induced skin rashes range from mild urticaria (hives) to "
            "severe Stevens-Johnson syndrome (SJS). Mild rash with no mucosal "
            "involvement: MEDIUM risk. Rash with blistering, mucosal "
            "involvement, or fever: HIGH risk (possible SJS/TEN). Common "
            "culprits: sulfonamides, allopurinol, phenytoin, carbamazepine."
        ),
    },
    {
        "id": "kb_drowsiness",
        "content": (
            "Drowsiness and sedation are expected side effects of many "
            "medications including antihistamines, benzodiazepines, opioids, "
            "and some antidepressants. Risk level: LOW if the patient is not "
            "operating machinery. Action: Avoid driving; adjust dose timing."
        ),
    },
]


@mlflow.trace(span_type="RETRIEVER")
def retrieve_drug_safety_docs(patient_report: str) -> list[dict]:
    """Keyword-based retriever over the drug safety knowledge base."""
    query_terms = set(patient_report.lower().split())
    scored = []
    for doc in DRUG_SAFETY_KB:
        doc_terms = set(doc["content"].lower().split())
        overlap = len(query_terms & doc_terms)
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:3]]


@mlflow.trace
def rag_side_effect_agent(patient_report: str) -> dict:
    """RAG pipeline: retrieve drug safety docs then classify risk."""
    docs = retrieve_drug_safety_docs(patient_report)
    context = "\n\n".join(d["content"] for d in docs)

    rag_system_prompt = (
        CLASSIFICATION_SYSTEM_PROMPT
        + "\n\nUse the following drug safety reference material to support "
        "your classification:\n\n"
        + context
    )

    completion = _get_client().chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": patient_report},
        ],
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    return _parse_classification(completion.choices[0].message.content)


# ---------------------------------------------------------------------------
# Component: standalone risk triage
# ---------------------------------------------------------------------------

@mlflow.trace
def triage_risk(patient_report: str) -> dict:
    """Triage a side-effect report into a risk level only.

    Returns {"risk_level": "high" | "medium" | "low"}.
    """
    completion = _get_client().chat.completions.create(
        model=_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical triage assistant. Classify the patient's "
                    "reported side effect into exactly one risk level: high, "
                    "medium, or low. Respond with ONLY the risk level, nothing else."
                ),
            },
            {"role": "user", "content": patient_report},
        ],
        max_tokens=10,
    )
    return {"risk_level": completion.choices[0].message.content.strip().lower()}


# ---------------------------------------------------------------------------
# Two-stage pipeline: triage → RAG explain
# ---------------------------------------------------------------------------

@mlflow.trace
def triage_then_explain(patient_report: str) -> dict:
    """Two-stage pipeline: quick triage then full RAG classification."""
    triage_result = triage_risk(patient_report)
    full_result = rag_side_effect_agent(patient_report)
    full_result["triage_risk_level"] = triage_result["risk_level"]
    return full_result


# ---------------------------------------------------------------------------
# Multi-turn triage agent
# ---------------------------------------------------------------------------

MULTI_TURN_SYSTEM_PROMPT = (
    "You are a pharmacovigilance triage assistant helping patients "
    "assess the risk of their reported medicine side effects.\n\n"
    "For each message, consider the FULL conversation history to "
    "determine the current risk level (high, medium, or low).\n\n"
    "If new symptoms are reported that are more severe than earlier "
    "ones, you MUST escalate the risk classification — never ignore "
    "worsening symptoms.\n\n"
    "Always respond with:\n"
    "1. The current risk level\n"
    "2. Clinical reasoning\n"
    "3. Recommended next action for the patient"
)


@mlflow.trace
def multi_turn_triage_agent(messages: list[dict]) -> dict:
    """Continue a multi-turn patient triage conversation."""
    system = {"role": "system", "content": MULTI_TURN_SYSTEM_PROMPT}
    completion = _get_client().chat.completions.create(
        model=_model,
        messages=[system] + messages,
        max_tokens=512,
    )
    return {"response": completion.choices[0].message.content}
