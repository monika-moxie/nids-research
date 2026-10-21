"""Bounded analyst decision workflow for the Phase 8 demonstrator."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def _humanize_feature_name(feature_name: str) -> str:
    if feature_name.startswith("num__"):
        return feature_name.removeprefix("num__")
    if feature_name.startswith("cat__"):
        return feature_name.removeprefix("cat__").replace("_", " = ", 1)
    return feature_name


def decide_escalation(prediction: dict[str, Any], explanation: dict[str, Any]) -> dict[str, str]:
    """Apply an auditable two-outcome policy using model and explainer tools."""

    probability = prediction["attack_probability"]
    positive_evidence = [item for item in explanation["top_features"] if item["direction"] == "toward_attack"]
    if probability >= 0.5 and positive_evidence:
        severity = "critical" if probability >= 0.9 else "high" if probability >= 0.7 else "medium"
        return {
            "action": "escalate",
            "severity": severity,
            "reason": "The model flags an attack and local explanation features support that prediction.",
        }
    if probability >= 0.5:
        return {
            "action": "escalate",
            "severity": "medium",
            "reason": "The model flags an attack, although local explanation evidence is mixed.",
        }
    return {
        "action": "suppress",
        "severity": "informational",
        "reason": "The model predicts normal traffic below the 0.50 attack threshold.",
    }


def build_plain_english_explanation(
    prediction: dict[str, Any], certification: dict[str, Any], explanation: dict[str, Any], decision: dict[str, str]
) -> str:
    likelihood = round(prediction["attack_probability"] * 100)
    feature_text = ", ".join(
        f"{_humanize_feature_name(item['feature'])} ({item['direction'].replace('_', ' ')})"
        for item in explanation["top_features"][:2]
    )
    certificate_text = (
        "The smoothing check abstained, so no local stability radius was certified."
        if certification["abstained"]
        else f"The smoothing check certified radius {certification['certified_radius']:.3f} in preprocessed feature space."
    )
    return f"{likelihood}% attack likelihood: {decision['action']}. Evidence: {feature_text}. {certificate_text}"


def build_incident_report(
    prediction: dict[str, Any], certification: dict[str, Any], explanation: dict[str, Any], decision: dict[str, str]
) -> dict[str, Any]:
    return {
        "report_id": f"NIDS-{uuid4().hex[:8].upper()}",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "draft",
        "recommended_action": decision["action"],
        "severity": decision["severity"],
        "summary": build_plain_english_explanation(prediction, certification, explanation, decision),
        "model_evidence": {
            "attack_probability": prediction["attack_probability"],
            "decision_threshold": prediction["decision_threshold"],
            "top_lime_features": explanation["top_features"],
            "certificate": certification,
        },
        "recommended_next_step": (
            "Review corresponding raw telemetry and correlate it with local security logs."
            if decision["action"] == "escalate"
            else "Retain the event for audit and reconsider it if correlated alerts appear."
        ),
        "limitation": "This is an analyst draft. LIME and certificate results are in preprocessed feature space.",
    }


class IntelligenceOrchestrator:
    """Calls named analysis tools and produces an auditable analyst recommendation."""

    def __init__(self, nids_service) -> None:
        self.nids_service = nids_service

    def analyze_flagged_flow(self, raw_features: dict[str, Any]) -> dict[str, Any]:
        analysis = self.nids_service.analyze(raw_features)
        explanation = self.nids_service.explain(raw_features)
        decision = decide_escalation(analysis, explanation)
        return {
            **analysis,
            "explanation": explanation,
            "decision": decision,
            "plain_english_explanation": build_plain_english_explanation(
                analysis, analysis["certification"], explanation, decision
            ),
            "incident_report": build_incident_report(analysis, analysis["certification"], explanation, decision),
        }
