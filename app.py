"""
VitalCheck AI - Streamlit dashboard.
This file is intentionally thin: it handles UI state, file upload, and
rendering. All LLM/LangChain logic lives in chains/, all PDF/text
processing lives in utils/, and all persistence lives in database.py.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from chains._shared import MEDICAL_DISCLAIMER, LLMConfigError
from chains.analysis_chain import run_full_analysis
from database import (
    get_family_history,
    init_db,
    save_family_history,
    save_report,
)
from models.schemas import FamilyHistory, HealthAnalysis
from utils.pdf_parser import CorruptedPDFError, EmptyPDFError, extract_text_from_pdf
from utils.report_generator import generate_health_report_pdf

load_dotenv()
init_db()

st.set_page_config(
    page_title="VitalCheck AI",
    page_icon="🩺",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_patient_name(raw_name: str) -> str:
    """Normalize whitespace/casing so 'sunita sharma' and 'Sunita Sharma'
    resolve to the same patient record."""
    return " ".join(raw_name.strip().split()).title()


def _render_disclaimer():
    """Fixed, non-LLM-generated disclaimer. Always rendered regardless of
    what the model output contains, so it can never be silently dropped."""
    st.warning(f"⚠️ {MEDICAL_DISCLAIMER}")


def _render_score_badge(score: int):
    if score >= 80:
        color = "🟢"
    elif score >= 50:
        color = "🟡"
    else:
        color = "🔴"
    st.metric(label="Health Score", value=f"{score} / 100", delta=None)
    st.markdown(f"### {color} {'Good' if score >= 80 else 'Needs Attention' if score >= 50 else 'Concerning'}")


def _render_analysis(analysis: HealthAnalysis):
    """Render the full dashboard for a completed HealthAnalysis."""

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### 🩺 Health Score")
        _render_score_badge(analysis.health_score)

        st.markdown("#### 🚨 Risk Assessment")
        risk_color = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(analysis.risk_level, "⚪")
        st.markdown(f"**Risk Level:** {risk_color} {analysis.risk_level}")
        if analysis.risk_factors:
            for factor in analysis.risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.markdown("_No specific risk factors identified._")

    with col2:
        st.markdown("#### 📋 Executive Summary")
        with st.container(border=True):
            st.write(analysis.summary)

        st.markdown("#### ⚠️ Abnormal Parameters")
        if analysis.abnormal_parameters:
            for param in analysis.abnormal_parameters:
                st.error(f"🔴 {param}")
        else:
            st.success("✅ No abnormal parameters detected.")

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🥗 Lifestyle Recommendations")
        with st.container(border=True):
            for i, rec in enumerate(analysis.recommendations, 1):
                st.markdown(f"**{i}.** {rec}")

    with col4:
        st.markdown("#### 👨‍⚕️ Questions for Your Doctor")
        with st.container(border=True):
            for i, q in enumerate(analysis.doctor_questions, 1):
                st.markdown(f"**{i}.** {q}")

    st.divider()
    _render_disclaimer()



# ---------------------------------------------------------------------------
# Sidebar: patient identity + family history
# ---------------------------------------------------------------------------

st.sidebar.title("🩺 VitalCheck AI")
st.sidebar.caption("Intelligent Medical Report Analyzer")

st.sidebar.divider()
st.sidebar.subheader("Patient")
patient_name_input = st.sidebar.text_input(
    "Patient name",
    placeholder="e.g. Sunita Sharma",
    help="Used to track your report history and trends across visits.",
)
patient_name = _normalize_patient_name(patient_name_input) if patient_name_input else ""

st.sidebar.subheader("Family Risk Factors")
st.sidebar.caption("Optional, but improves the personalization of insights.")

existing_history = get_family_history(patient_name) if patient_name else None

diabetes = st.sidebar.checkbox(
    "Family history of Diabetes", value=existing_history["diabetes"] if existing_history else False
)
hypertension = st.sidebar.checkbox(
    "Family history of Hypertension", value=existing_history["hypertension"] if existing_history else False
)
thyroid = st.sidebar.checkbox(
    "Family history of Thyroid disorder", value=existing_history["thyroid"] if existing_history else False
)

family_history = FamilyHistory(diabetes=diabetes, hypertension=hypertension, thyroid=thyroid)

if patient_name and st.sidebar.button("💾 Save family history", use_container_width=True):
    save_family_history(patient_name, diabetes, hypertension, thyroid)
    st.sidebar.success("Saved.")

st.sidebar.divider()
api_key_present = bool(os.environ.get("GROQ_API_KEY"))
if not api_key_present:
    st.sidebar.error("⚠️ GROQ_API_KEY not found in .env")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🩺 VitalCheck AI")
st.caption("Upload a blood test report (CBC, LFT, KFT, Lipid Profile, Diabetes panel, etc.) for an AI-powered breakdown.")

if not patient_name:
    st.info("👈 Enter a patient name in the sidebar to get started.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload report PDF",
    type=["pdf"],
    help="Multi-page PDFs are supported.",
)

if uploaded_file is not None:
    if not api_key_present:
        st.error(
            "⚠️ Cannot run analysis: GROQ_API_KEY is missing. "
            "Add it to your .env file and restart the app."
        )
        st.stop()

    with st.spinner("Reading PDF..."):
        try:
            report_text = extract_text_from_pdf(uploaded_file)
        except EmptyPDFError as e:
            st.error(f"📄 {e}")
            st.stop()
        except CorruptedPDFError as e:
            st.error(f"📄 {e}")
            st.stop()

    with st.spinner("Running AI analysis..."):
        try:
            analysis, parameters = run_full_analysis(
                report_text=report_text,
                family_history=family_history,
            )
        except LLMConfigError as e:
            st.error(f"⚠️ {e}")
            st.stop()
        except Exception as e:
            st.error(
                f"⚠️ The AI analysis failed. This can happen due to a temporary "
                f"API issue or an unusually formatted report. Details: {e}"
            )
            st.stop()

    if not parameters:
        st.warning(
            "⚠️ No lab parameters could be identified in this report. "
            "The analysis below may be limited. Make sure you uploaded a "
            "lab/blood test report PDF."
        )

    # Persist this report to the database.
    save_report(
        patient_name=patient_name,
        health_score=analysis.health_score,
        risk_level=analysis.risk_level,
        parameters=[p.model_dump() for p in parameters],
    )

    st.success(f"✅ Analysis complete for **{patient_name}**.")
    _render_analysis(analysis)

    st.divider()
    pdf_bytes = generate_health_report_pdf(analysis, patient_name=patient_name)
    st.download_button(
        label="📄 Download Health Report (PDF)",
        data=pdf_bytes,
        file_name=f"{patient_name.replace(' ', '_')}_health_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

else:
    st.info("👈 Upload a report PDF above to get started.")