"""
Main analysis pipeline.

This is the file that actually demonstrates the LCEL architecture:

    Cleaned PDF text
      -> extraction (regex/LLM hybrid)
      -> RunnableParallel(summary, risk, recommendations, doctor_questions)
      -> combined into one HealthAnalysis

The four analysis chains are run with RunnableParallel specifically
because they are independent of one another — none of them depends on
another's output, they all read from the same extracted-parameters
context. That independence is what makes parallelizing them valid (it's
not just "faster for the sake of it"; running them sequentially would
produce identical results, just slower).
"""

from typing import List, Optional

from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from chains._shared import get_llm, invoke_structured
from chains.extraction_chain import extract_parameters
from chains.summary_chain import build_summary_chain, SummaryOutput
from chains.risk_chain import build_risk_chain, RiskOutput
from chains.recommendation_chain import build_recommendation_chain, RecommendationOutput
from chains.doctor_chain import build_doctor_chain, DoctorQuestionsOutput
from models.schemas import ExtractedParameter, FamilyHistory, HealthAnalysis


class CombinedAnalysisOutput(BaseModel):
    """Unified schema combining the outputs of all analysis chains."""
    health_score: int = Field(ge=0, le=100)
    summary: str
    risk_level: str = Field(description="Low, Moderate, or High")
    risk_factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    doctor_questions: List[str] = Field(default_factory=list)


_COMBINED_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert AI Health Assistant. Your task is to analyze the patient's lab report findings and family history to produce a comprehensive health analysis.

Patient family history context:
{family_history_context}

Extracted lab parameters (name, value, unit, normal range, abnormal flag):
{parameters_text}

Your analysis must cover the following aspects:
1. Health Score: Provide an overall health score from 0 to 100, where 100 represents all parameters well within normal range and lower scores reflect more or more severe abnormalities.
2. Executive Summary: Write a short executive summary (3-5 sentences) in plain, simple language explaining what the results show. Mention each abnormal value by name and what direction it is off. Do not state or imply a diagnosis.
3. Risk Assessment:
   - Assign an overall risk level: exactly one of "Low", "Moderate", or "High" based on the abnormal parameters and family history.
   - List specific potential risk factors as short phrases (e.g., "Possible iron deficiency", "Prediabetes risk given elevated fasting glucose"). Frame these as possibilities/concerns to discuss with a doctor, not conclusions.
4. Lifestyle Recommendations: Suggest 4-6 specific, actionable lifestyle recommendations (diet, exercise, habits) directly tied to the abnormal findings. Do NOT recommend specific drug or supplement dosages.
5. Questions for Doctor: Generate 3-5 specific, useful questions the patient can bring to a doctor's appointment, directly tied to the abnormal findings.

Do not include any medical advice disclaimer text yourself — that is handled separately by the application.

{format_instructions}"""
)


def _format_parameters_for_prompt(parameters: List[ExtractedParameter]) -> str:
    """Render extracted parameters as a compact text block for prompts."""
    if not parameters:
        return "No structured parameters could be extracted from this report."

    lines = []
    for p in parameters:
        flag = "ABNORMAL" if p.is_abnormal else "normal"
        range_part = f", normal range: {p.normal_range}" if p.normal_range else ""
        lines.append(f"- {p.name}: {p.value} {p.unit} [{flag}{range_part}]")
    return "\n".join(lines)


def build_analysis_pipeline(llm=None):
    """
    Build the pipeline over the analysis chains.
    To avoid 429 rate limit issues on the Gemini free tier, we combine the
    four separate analysis calls into a single structured LLM call.
    
    Returns a Runnable whose .invoke(inputs) returns a dict with keys
    'summary', 'risk', 'recommendations', 'doctor_questions' matching
    the legacy parallel signature.
    """
    llm = llm or get_llm()

    def run_combined(inputs: dict) -> dict:
        combined_result = invoke_structured(
            llm=llm,
            prompt=_COMBINED_PROMPT,
            output_model=CombinedAnalysisOutput,
            inputs={
                "family_history_context": inputs.get(
                    "family_history_context", "No family history reported."
                ),
                "parameters_text": inputs["parameters_text"],
            },
        )
        return {
            "summary": SummaryOutput(
                health_score=combined_result.health_score,
                summary=combined_result.summary,
            ),
            "risk": RiskOutput(
                risk_level=combined_result.risk_level,
                risk_factors=combined_result.risk_factors,
            ),
            "recommendations": RecommendationOutput(
                recommendations=combined_result.recommendations,
            ),
            "doctor_questions": DoctorQuestionsOutput(
                doctor_questions=combined_result.doctor_questions,
            ),
        }

    return RunnableLambda(run_combined)


def run_full_analysis(
    report_text: str,
    family_history: Optional[FamilyHistory] = None,
    llm=None,
) -> tuple[HealthAnalysis, List[ExtractedParameter]]:
    """
    Run the complete pipeline: extraction then the combined analysis chain.

    Args:
        report_text: Cleaned text from the uploaded PDF.
        family_history: Optional self-reported family risk factors.
        llm: Optional pre-configured LLM (mainly for testing).

    Returns:
        Tuple of (HealthAnalysis, the list of ExtractedParameter used).
    """
    llm = llm or get_llm()

    extraction_result = extract_parameters(report_text, llm=llm)
    parameters = extraction_result.parameters

    family_history = family_history or FamilyHistory()
    inputs = {
        "parameters_text": _format_parameters_for_prompt(parameters),
        "family_history_context": family_history.as_prompt_text(),
    }

    pipeline = build_analysis_pipeline(llm=llm)
    # This single .invoke() call is what actually triggers the parallel
    # execution of all four chains against the same `inputs`.
    results = pipeline.invoke(inputs)

    abnormal_list = [
        f"{p.name}: {p.value} {p.unit} (normal: {p.normal_range})"
        for p in parameters
        if p.is_abnormal
    ]

    analysis = HealthAnalysis(
        health_score=results["summary"].health_score,
        summary=results["summary"].summary,
        abnormal_parameters=abnormal_list,
        risk_level=results["risk"].risk_level,
        risk_factors=results["risk"].risk_factors,
        recommendations=results["recommendations"].recommendations,
        doctor_questions=results["doctor_questions"].doctor_questions,
    )

    return analysis, parameters