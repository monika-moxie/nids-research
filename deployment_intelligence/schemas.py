"""FastAPI request and response schemas for analyst-facing NIDS inference."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FlowRequest(BaseModel):
    """One raw UNSW-NB15 flow keyed by the training-time field names."""

    features: dict[str, Any] = Field(
        description="All 42 raw model-input features; do not include id, label, or attack_cat."
    )


class PredictionResponse(BaseModel):
    """Clean-model prediction returned by the deployed baseline."""

    predicted_label: str
    attack_probability: float
    decision_threshold: float


class CertificateResponse(BaseModel):
    """Local randomized-smoothing result for one preprocessed flow."""

    smoothed_prediction: str | None
    top_class_probability: float | None
    lower_confidence_bound: float | None
    certified_radius: float
    abstained: bool
    sigma: float
    num_noise_samples: int
    certificate_space: str


class AnalysisResponse(PredictionResponse):
    """Prediction plus a flow-specific, explicitly scoped certificate."""

    certification: CertificateResponse


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    direction: str


class ExplanationResponse(BaseModel):
    method: str
    explanation_space: str
    top_features: list[FeatureContribution]


class DecisionResponse(BaseModel):
    action: str
    severity: str
    reason: str


class IncidentReportResponse(BaseModel):
    report_id: str
    created_at_utc: str
    status: str
    recommended_action: str
    severity: str
    summary: str
    model_evidence: dict[str, Any]
    recommended_next_step: str
    limitation: str


class IntelligenceResponse(AnalysisResponse):
    explanation: ExplanationResponse
    decision: DecisionResponse
    plain_english_explanation: str
    incident_report: IncidentReportResponse


class MetadataResponse(BaseModel):
    """Schema information the frontend needs to build a valid request."""

    model_input_fields: list[str]
    excluded_fields: list[str]
    preprocessed_feature_count: int
    certificate_space: str


class BatchAlert(BaseModel):
    """One model-flagged CSV row available for deeper analyst review."""

    row_number: int
    predicted_label: str
    attack_probability: float
    features: dict[str, Any]


class BatchScoreResponse(BaseModel):
    """Fast clean-model screening result for a bounded CSV batch."""

    total_rows: int
    flagged_count: int
    decision_threshold: float
    alerts: list[BatchAlert]
