"""
Doctor questions chain.

Generates specific, useful questions the patient can bring to a doctor's
appointment, tied to the abnormal findings — designed to make the most
of limited consultation time rather than being generic.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from typing import List

from chains._shared import invoke_structured

_PROMPT = ChatPromptTemplate.from_template(
    """You help patients prepare for doctor appointments by turning lab
report findings into specific, useful questions to ask.

Extracted lab parameters (name, value, unit, normal range, abnormal flag):
{parameters_text}

Patient context:
{family_history_context}

Task:
Generate 3-5 specific questions the patient should ask their doctor,
directly tied to the abnormal findings (e.g. "Should I get an HbA1c test
given my elevated fasting glucose?"). Avoid generic questions like "Am I
healthy?" — make each one specific and actionable.

Do not include any medical advice disclaimer text yourself — that is
handled separately by the application.

{format_instructions}"""
)


class DoctorQuestionsOutput(BaseModel):
    doctor_questions: List[str] = Field(default_factory=list)


def build_doctor_chain(llm: BaseChatModel):
    """Returns a callable that takes the standard chain inputs dict and
    returns a validated DoctorQuestionsOutput."""

    def run(inputs: dict) -> DoctorQuestionsOutput:
        return invoke_structured(
            llm=llm,
            prompt=_PROMPT,
            output_model=DoctorQuestionsOutput,
            inputs={
                "parameters_text": inputs["parameters_text"],
                "family_history_context": inputs.get(
                    "family_history_context", "No family history reported."
                ),
                "format_instructions": "Respond with valid JSON only, matching the required schema.",
            },
        )

    return run