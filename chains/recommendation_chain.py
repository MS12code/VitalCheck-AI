"""
Recommendation chain.

Generates actionable lifestyle recommendations (diet, exercise, habits)
tied to the specific abnormal parameters found — not generic wellness
advice unrelated to the report.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from typing import List

from chains._shared import invoke_structured

_PROMPT = ChatPromptTemplate.from_template(
    """You are a lifestyle guidance assistant. Based on the abnormal lab
parameters below, suggest practical, general lifestyle recommendations
(diet, exercise, habits). Do NOT recommend specific drug dosages,
supplement dosages, or prescription changes — those require a doctor.

Extracted lab parameters (name, value, unit, normal range, abnormal flag):
{parameters_text}

Patient context:
{family_history_context}

Task:
Provide 4-6 specific, actionable recommendations directly tied to the
abnormal findings above (e.g. if iron/hemoglobin is low, suggest
iron-rich foods; if blood sugar is high, suggest reducing refined sugar
and increasing activity). If no parameters are abnormal, give general
maintenance recommendations instead.

Do not include any medical advice disclaimer text yourself — that is
handled separately by the application.

{format_instructions}"""
)


class RecommendationOutput(BaseModel):
    recommendations: List[str] = Field(default_factory=list)


def build_recommendation_chain(llm: BaseChatModel):
    """Returns a callable that takes the standard chain inputs dict and
    returns a validated RecommendationOutput."""

    def run(inputs: dict) -> RecommendationOutput:
        return invoke_structured(
            llm=llm,
            prompt=_PROMPT,
            output_model=RecommendationOutput,
            inputs={
                "parameters_text": inputs["parameters_text"],
                "family_history_context": inputs.get(
                    "family_history_context", "No family history reported."
                ),
                "format_instructions": "Respond with valid JSON only, matching the required schema.",
            },
        )

    return run