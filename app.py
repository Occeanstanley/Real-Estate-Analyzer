import streamlit as st
import pdfplumber
from fpdf import FPDF
from openai import OpenAI
import io
import re

client = OpenAI()

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def sanitize_for_pdf(text: str) -> str:
    """Make text safe for FPDF (latin-1)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("latin-1", "replace").decode("latin-1")


def pretty_value(v):
    """Format extracted dicts and lists into clean human text."""
    if v is None:
        return "None"

    if isinstance(v, dict):
        parts = []
        for k, val in v.items():
            label = k.replace("_", " ").capitalize()
            parts.append(f"{label}: {val}")
        return "; ".join(parts)

    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(x) for x in v)

    return str(v)


def extract_text(file):
    with pdfplumber.open(file) as pdf:
        full = ""
        for page in pdf.pages:
            full += page.extract_text() + "\n"
        return full.strip()


# -------------------------------------------------------
# AI Extraction
# -------------------------------------------------------

def extract_structured_data(raw_text):
    prompt = f"""
Extract key information from the following real estate document.
Return a JSON object with these fields:

- property_address
- landlord
- tenant
- lease_start
- lease_end
- monthly_rent
- security_deposit
- late_fee
- utilities
- pet_policy
- termination_clause
- other_fees
- notes

If a field is not present, return null.
For 'other_fees', return a dictionary of fee_name: fee_value.

Document:
\"\"\"{raw_text}\"\"\"
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return resp.choices[0].message.parsed


def real_estate_agent_answer(question, extracted):
    prompt = f"""
You are a helpful real estate agent.
Answer the question using ONLY this document data:

{extracted}

User question: {question}
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content


# -------------------------------------------------------
# Property Value Estimator
# -------------------------------------------------------

def estimate_property_value(extracted):
    address = extracted.get("property_address", "Unknown")
    rent = extracted.get("monthly_rent")
    size_guess = "Not provided"

    prompt = f"""
Estimate a rough property value for the property located at:
{address}

Based on:
- Monthly Rent: {rent}
- Size: {size_guess}
- Market conditions typical of Long Island.

Return a SHORT readable explanation in paragraphs, no JSON.
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content


# -------------------------------------------------------
# PDF Generator
# -------------------------------------------------------

def build_summary_pdf(structured, value_estimate):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, sanitize_for_pdf("Lease Summary Report"), ln=1)

    pdf.set_font("Helvetica", "", 11)

    pdf.ln(5)
    pdf.cell(0, 8, sanitize_for_pdf("Extracted Key Information:"), ln=1)

    for key, value in structured.items():
        line = f"{key.replace('_',' ').title()}: {pretty_value(value)}"
        pdf.multi_cell(0, 6, sanitize_for_pdf(line))

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, sanitize_for_pdf("Property Value Estimate"), ln=1)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, sanitize_for_pdf(value_estimate))

    return pdf.output(dest="S").encode("latin-1", "replace")


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------

st.set_page_config(page_title="Real Estate Document Analyzer", layout="wide")

st.markdown("<h1>🏡 Real Estate Document Analyzer</h1>", unsafe_allow_html=True)
st.write("Upload a lease, contract, or purchase agreement to extract AI insights.")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    raw_text = extract_text(uploaded)
    st.subheader("📌 Extracted Text Preview")
    st.text_area("", raw_text, height=220)

    if st.button("Analyze document with AI"):
        extracted = extract_structured_data(raw_text)
        st.success("Analysis complete!")

        # -------------------------
        # Key Info Display
        # -------------------------
        st.subheader("📄 Key Information")

        col1, col2 = st.columns(2)
        keys_left = [
            "property_address", "landlord", "tenant",
            "lease_start", "lease_end", "monthly_rent",
            "security_deposit", "notes"
        ]
        keys_right = [
            "late_fee", "utilities", "pet_policy",
            "termination_clause", "other_fees"
        ]

        with col1:
            for key in keys_left:
                st.markdown(f"**{key.replace('_',' ').title()}**")
                st.write(pretty_value(extracted.get(key)))

        with col2:
            for key in keys_right:
                st.markdown(f"**{key.replace('_',' ').title()}**")
                st.write(pretty_value(extracted.get(key)))

        # -------------------------
        # Property Value Estimator
        # -------------------------
        st.subheader("📊 Property Value Estimator")
        value_estimate = estimate_property_value(extracted)
        st.write(value_estimate)

        # -------------------------
        # PDF Download
        # -------------------------
        st.subheader("📄 Download Lease Summary PDF")
        pdf_bytes = build_summary_pdf(extracted, value_estimate)
        st.download_button(
            "Download Summary PDF",
            data=pdf_bytes,
            file_name="lease_summary.pdf",
            mime="application/pdf"
        )

        # -------------------------
        # Q&A — Standard + Agent Persona
        # -------------------------
        st.subheader("💬 Ask Questions About the Document")

        tab1, tab2 = st.tabs(["Standard Q&A", "Real Estate Agent Persona"])

        with tab1:
            q = st.text_input("Ask a question:")
            if st.button("Answer question"):
                resp = real_estate_agent_answer(q, extracted)
                st.write(resp)

        with tab2:
            q2 = st.text_input("Ask the agent:")
            if st.button("Agent answer"):
                resp = real_estate_agent_answer(q2, extracted)
                st.write(resp)

        # -------------------------
        # Table Extraction (future)
        # -------------------------
        st.subheader("📑 Extracted Tables (from PDF)")
        st.write("No clear tables were detected, but this feature can be expanded.")
