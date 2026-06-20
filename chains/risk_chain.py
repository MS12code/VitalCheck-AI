"""
Risk analysis chain.

Identifies an overall risk level (Low/Moderate/High) and specific
potential risk factors implied by abnormal parameters and family history.
Explicitly framed as "potential concerns to discuss", not a diagnosis.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from typing import List

from chains._shared import invoke_structured

_PROMPT = ChatPromptTemplate.from_template(
    """You are a cautious medical report risk-flagging assistant. You
identify POTENTIAL concerns implied by lab data and family history — you
never diagnose a condition or claim certainty.

Patient context (self-reported family history):
{family_history_context}

Extracted lab parameters (name, value, unit, normal range, abnormal flag):
{parameters_text}

Trend information from past reports, if available:
{trend_context}

Task:
1. Assign an overall risk_level: exactly one of "Low", "Moderate", or "High",
   based on the number and severity of abnormal parameters, worsening
   trends, and relevant family history.
2. List specific potential risk factors as short phrases (e.g. "Possible
   iron deficiency", "Prediabetes risk given elevated fasting glucose and
   family history of diabetes"). Frame these as possibilities to discuss
   with a doctor, not conclusions.

Do not include any medical advice disclaimer text yourself — that is
handled separately by the application.

{format_instructions}"""
)


class RiskOutput(BaseModel):
    risk_level: str = Field(description="Low, Moderate, or High")
    risk_factors: List[str] = Field(default_factory=list)


def build_risk_chain(llm: BaseChatModel):
    """Returns a callable that takes the standard chain inputs dict and
    returns a validated RiskOutput."""

    def run(inputs: dict) -> RiskOutput:
        return invoke_structured(
            llm=llm,
            prompt=_PROMPT,
            output_model=RiskOutput,
            inputs={
                "family_history_context": inputs.get(
                    "family_history_context", "No family history reported."
                ),
                "parameters_text": inputs["parameters_text"],
                "trend_context": inputs.get(
                    "trend_context", "No prior reports available for trend analysis."
                ),
                "format_instructions": "Respond with valid JSON only, matching the required schema.",
            },
        )

    return run