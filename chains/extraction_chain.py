from langchain_core.prompts import ChatPromptTemplate
from chains._shared import invoke_structured
from models.schemas import ExtractionResult

_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert medical data extraction assistant. Your job is to extract structured lab parameters from raw medical test report text.

Extract every single lab parameter/test mentioned. For each parameter, extract:
- name: The name of the test/parameter (e.g., Hemoglobin, WBC, Cholesterol, TSH, Fasting Blood Sugar).
- value: The numerical value. Make sure to parse it as a number.
- unit: The unit of measurement (e.g., g/dL, mg/dL, uIU/mL).
- normal_range: The reference normal range as given in the report (e.g., 12.0 - 15.0, <200, 0.4-4.5).
- is_abnormal: A boolean (true/false) indicating if the value is outside the reference range.

Also extract the patient's name if it is mentioned in the text.

Raw report text:
{report_text}

{format_instructions}"""
)

def extract_parameters(report_text: str, llm) -> ExtractionResult:
    """
    Extract health parameters from raw text using LangChain and ChatGoogleGenerativeAI.
    
    Args:
        report_text: Cleaned text from the PDF.
        llm: ChatGoogleGenerativeAI chat model instance.
        
    Returns:
        ExtractionResult: Parsed and validated extraction results.
    """
    return invoke_structured(
        llm=llm,
        prompt=_PROMPT,
        output_model=ExtractionResult,
        inputs={
            "report_text": report_text,
            "format_instructions": "Respond with valid JSON only, matching the required schema.",
        }
    )
