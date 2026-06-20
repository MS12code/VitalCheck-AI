"""
Pydantic schemas for VitalCheck AI.

These models serve two purposes:
1. Force the Gemini LLM to return valid, predictable JSON (via LangChain's
   structured output / PydanticOutputParser).
2. Give the rest of the app (DB layer, Streamlit UI) typed, validated data
   to work with instead of raw dicts.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ExtractedParameter(BaseModel):
    """A single lab parameter pulled out of the raw report text."""

    name: str = Field(description="Name of the lab parameter, e.g. 'Hemoglobin'")
    value: Optional[float] = Field(default=None, description="Numeric value reported for this parameter")
    unit: str = Field(default="", description="Unit of measurement, e.g. 'g/dL'")
    normal_range: str = Field(
        default="", description="Normal range as stated on the report, e.g. '12-15'"
    )
    is_abnormal: Optional[bool] = Field(
        default=False, description="Whether the value falls outside the normal range"
    )


class ExtractionResult(BaseModel):
    """Output of the extraction chain: all parameters found in a report."""

    patient_name: Optional[str] = Field(
        default=None, description="Patient name if present on the report"
    )
    parameters: List[ExtractedParameter] = Field(default_factory=list)

    @model_validator(mode="after")
    def drop_unreadable_parameters(self) -> "ExtractionResult":
        """Silently drop any parameter rows where the numeric value could not
        be extracted (null/None), rather than crashing the whole extraction."""
        self.parameters = [
            p for p in self.parameters if p.value is not None
        ]
        return self


class HealthAnalysis(BaseModel):
    """
    Final combined output shown on the dashboard.
    This is assembled from the four parallel chains (summary, risk,
    recommendations, doctor questions) plus the health score.
    """

    health_score: int = Field(ge=0, le=100, description="Overall health score, 0-100")
    summary: str = Field(description="Plain-language executive summary of the report")
    abnormal_parameters: List[str] = Field(
        default_factory=list,
        description="Human-readable list of abnormal findings, e.g. 'Low Hemoglobin: 10.5 g/dL'",
    )
    risk_level: str = Field(description="One of: Low, Moderate, High")
    risk_factors: List[str] = Field(
        default_factory=list, description="Specific potential health risks identified"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Actionable lifestyle recommendations"
    )
    doctor_questions: List[str] = Field(
        default_factory=list, description="Suggested questions to ask a doctor"
    )

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, v: str) -> str:
        normalized = v.strip().title()
        if normalized not in {"Low", "Moderate", "High"}:
            return "Moderate"
        return normalized


class FamilyHistory(BaseModel):
    """Self-reported family risk factors, used as extra prompt context."""

    diabetes: bool = False
    hypertension: bool = False
    thyroid: bool = False

    def as_prompt_text(self) -> str:
        """Render as a short string to inject into prompts. Empty if nothing reported."""
        flags = []
        if self.diabetes:
            flags.append("family history of diabetes")
        if self.hypertension:
            flags.append("family history of hypertension")
        if self.thyroid:
            flags.append("family history of thyroid disorders")
        if not flags:
            return "No family history reported."
        return "Patient has reported: " + ", ".join(flags) + "."


class TrendPoint(BaseModel):
    """A single (date, value) observation for one parameter, used for trend charts."""

    report_date: str  # ISO format string, e.g. "2026-06-20"
    parameter: str
    value: float
    unit: str = ""


class TrendResult(BaseModel):
    """Computed trend for a single parameter across multiple reports."""

    parameter: str
    values: List[float]
    dates: List[str]
    direction: str  # "Increasing", "Decreasing", "Stable", "Insufficient data"
    note: str = ""