"""
Shared building blocks for the four parallel analysis chains
(summary, risk, recommendations, doctor questions).

Centralizing the LLM instantiation and the structured-output retry logic
here avoids repeating it four times and keeps the behavior consistent
across every chain.
"""

import os
import re
import time
from typing import Type, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq  # type: ignore[import]
from pydantic import BaseModel
import streamlit as st

T = TypeVar("T", bound=BaseModel)

# Fixed, non-LLM-generated disclaimer. This is rendered directly by the UI
# (see app.py) rather than relied upon as model output, so it can never be
# dropped if the model decides to omit it for a given response.
MEDICAL_DISCLAIMER = (
    "This is not medical advice. Consult a qualified healthcare professional "
    "for diagnosis and treatment decisions."
)


class LLMConfigError(Exception):
    """Raised when the Groq API key is missing or invalid."""


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    """
    Construct the shared Groq chat model used by every chain.

    Args:
        temperature: Lower values (default 0.2) keep medical-adjacent
            output more consistent and less speculative across runs.

    Raises:
        LLMConfigError: If GROQ_API_KEY is not set in the environment.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMConfigError(
            "GROQ_API_KEY is not set. Add it to your .env file before running the app."
        )

    model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )


def invoke_structured(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    output_model: Type[T],
    inputs: dict,
    max_retries: int = 2,  # kept for API compatibility; retries managed internally
) -> T:
    """
    Run a prompt through the LLM using its native structured output
    (function/tool calling) and return a validated Pydantic model instance.

    Using with_structured_output() is far more reliable than prompt-based
    JSON parsing because the model is constrained by the schema at the API
    level via Groq's tool-calling interface, rather than just being asked
    to produce a specific format in text.

    Args:
        llm: Configured chat model.
        prompt: Prompt template; {format_instructions} placeholder is
            accepted but passed as empty string — not needed with
            structured output.
        output_model: Pydantic model class the response is parsed into.
        inputs: Variables to fill into the prompt template.
        max_retries: Kept for backward-compatibility.

    Returns:
        A validated instance of output_model.

    Raises:
        Exception: If all rate-limit retries are exhausted or a non-
            rate-limit error occurs.
    """
    # Bind the output schema to the LLM using function calling.
    # {format_instructions} is set to "" for prompt template compatibility —
    # the placeholder still exists in some prompts but is not needed here.
    structured_llm = llm.with_structured_output(output_model)
    chain = prompt | structured_llm

    attempt_inputs = dict(inputs)
    attempt_inputs.setdefault("format_instructions", "")

    api_retries = 4
    last_error: Exception = Exception("Unknown error.")

    for api_attempt in range(api_retries):
        try:
            result = chain.invoke(attempt_inputs)
            if result is None:
                raise ValueError("Structured output returned None unexpectedly.")
            return result  # type: ignore[return-value]
        except Exception as exc:
            exc_str = str(exc)
            # Detect 429, quota limits, or rate limit errors from Groq
            is_rate_limited = (
                "429" in exc_str
                or "quota" in exc_str.lower()
                or "resourceexhausted" in exc_str.lower()
                or "resource exhausted" in exc_str.lower()
                or "rate limit" in exc_str.lower()
            )
            if is_rate_limited and api_attempt < api_retries - 1:
                # Try to parse delay, e.g. "Please retry in 32.063s" or "try again in 5.5s"
                delay = 5.0
                match = re.search(r"(?:retry|try again) in ([\d\.]+)\s*s", exc_str, re.IGNORECASE)
                if match:
                    try:
                        delay = float(match.group(1)) + 1.5  # small safety buffer
                    except ValueError:
                        pass
                else:
                    delay = 5.0 * (2 ** api_attempt)  # exponential backoff: 5s, 10s, 20s

                try:
                    st.warning(
                        f"⚠️ Groq API rate limit hit. Retrying in {delay:.1f} seconds... "
                        f"(Attempt {api_attempt + 1}/{api_retries})"
                    )
                except Exception:
                    print(f"Warning: Groq API rate limit hit. Retrying in {delay:.1f}s...")

                time.sleep(delay)
                last_error = exc
                continue

            # Non-rate-limit error — fail immediately
            last_error = exc
            break

    raise last_error