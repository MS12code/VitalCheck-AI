"""
Summary chain.

Generates the plain-language executive summary and the overall health
score. These two are bundled into one chain (rather than split further)
because the health score is fundamentally a holistic judgment over the
same information the summary describes — splitting them would mean
re-deriving the same context twice for no benefit.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from chains._shared import invoke_structured

_PROMPT = ChatPromptTemplate.from_template(
    """You are a medical report explainer assisting a non-medical person in
understanding their lab results. You do NOT diagnose conditions.

Patient context:
{family_history_context}

Extracted lab parameters (name, value, unit, normal range, abnormal flag):
{parameters_text}

Task:
1. Write a short executive summary (3-5 sentences) in plain, simple
   language explaining what the results show. Mention abnormal values
   specifically; do not dwell on normal ones beyond a brief confirmation.
2. Give an overall health score from 0 to 100, where 100 represents all
   parameters well within normal range and lower scores reflect more or
   more severe abnormalities. Base this only on the data given.

Do not state or imply a diagnosis. Do not include any medical advice
disclaimer text yourself — that is handled separately by the application.

{format_instructions}"""
)


class SummaryOutput(BaseModel):
    health_score: int = Field(ge=0, le=100)
    summary: str


def build_summary_chain(llm: BaseChatModel):
    """Returns a callable that takes the standard chain inputs dict and
    returns a validated SummaryOutput."""

    def run(inputs: dict) -> SummaryOutput:
        return invoke_structured(
            llm=llm,
            prompt=_PROMPT,
            output_model=SummaryOutput,
            inputs={
                "family_history_context": inputs.get(
                    "family_history_context", "No family history reported."
                ),
                "parameters_text": inputs["parameters_text"],
                "format_instructions": "Respond with valid JSON only, matching the required schema.",
            },
        )

    return run