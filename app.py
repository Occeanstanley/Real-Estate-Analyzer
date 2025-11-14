import streamlit as st
import pdfplumber
from fpdf import FPDF
from openai import OpenAI
from docx import Document
import pandas as pd
import io

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
    """Extract raw text from a PDF using pdfplumber."""
    with pdfplumber.open(file) as pdf:
        full = ""
        for page in pdf.pages:
            t = page.extract_text() or ""
            full += t + "\n"
        return full.strip()


# -------------------------------------------------------
# AI Extraction & Reasoning
# -------------------------------------------------------

def extract_structured_data(raw_text: str) -> dict:
    """
    Extract structured lease/contract info as JSON using gpt-4o-mini.
    Schema is lease-oriented but works for most real estate docs.
    """
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
For 'other_fees', you may return a dictionary of fee_name: fee_value or a short text.

Document:
\"\"\"{raw_text}\"\"\"
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return resp.choices[0].message.parsed


def real_estate_agent_answer(question: str, extracted: dict, raw_text: str) -> str:
    """
    Persona Q&A: friendly real estate agent answering based on extracted data + raw text.
    """
    prompt = f"""
You are a friendly, experienced real estate agent.
Use ONLY the info from the extracted fields and the document text.

Extracted structured data:
{extracted}

Document text:
\"\"\"{raw_text[:8000]}\"\"\"

User question: {question}
Answer clearly in plain language. If something is not in the document, say you are not sure.
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content


def estimate_property_value_text(extracted: dict, raw_text: str) -> str:
    """
    High-level natural language value estimate (explanation).
    """
    address = extracted.get("property_address", "Unknown")
    rent = extracted.get("monthly_rent")
    prompt = f"""
Estimate a rough property value and rent positioning for:

- Property address: {address}
- Monthly rent (if any): {rent}

Use typical US market logic (e.g. Long Island / suburban NYC style if unclear).

Return a short explanation in 1–3 paragraphs.
Clearly say this is NOT an appraisal, just an AI estimate.
Document context:
\"\"\"{raw_text[:4000]}\"\"\"    
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content


def estimate_property_value_range(extracted: dict, raw_text: str) -> dict:
    """
    Ask the model for a numeric low/mid/high estimated value so we can chart it.
    Returns dict: { "low": float, "mid": float, "high": float }
    """
    address = extracted.get("property_address", "Unknown")
    rent = extracted.get("monthly_rent")

    prompt = f"""
Based on the following info, estimate a rough property value range in USD.

Property address: {address}
Monthly rent (if present): {rent}
Document snippet:
\"\"\"{raw_text[:2000]}\"\"\"

Return ONLY JSON with this shape:

{{
  "low": 450000,
  "mid": 500000,
  "high": 550000
}}

Values should be numbers, no strings, no currency symbols.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    data = resp.choices[0].message.parsed
    # basic safety
    try:
        low = float(data.get("low", 0))
        mid = float(data.get("mid", 0))
        high = float(data.get("high", 0))
    except Exception:
        low, mid, high = 0.0, 0.0, 0.0

    return {"low": low, "mid": mid, "high": high}


# -------------------------------------------------------
# PDF & DOCX Generators
# -------------------------------------------------------

def build_summary_pdf(structured: dict, value_estimate_text: str) -> bytes:
    """
    Create a PDF summary using FPDF, with latin-1 safe text.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, sanitize_for_pdf("Lease / Real Estate Summary"), ln=1)

    pdf.set_font("Helvetica", "", 11)
    pdf.ln(4)
    pdf.cell(0, 8, sanitize_for_pdf("Extracted Key Information:"), ln=1)

    for key, value in structured.items():
        label = key.replace("_", " ").title()
        line = f"{label}: {pretty_value(value)}"
        pdf.multi_cell(0, 6, sanitize_for_pdf(line))

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, sanitize_for_pdf("Property Value Estimate (AI)"), ln=1)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, sanitize_for_pdf(value_estimate_text))

    return pdf.output(dest="S").encode("latin-1", "replace")


def build_summary_docx(structured: dict, value_estimate_text: str) -> bytes:
    """
    Create a DOCX summary using python-docx and return bytes.
    """
    doc = Document()
    doc.add_heading("Lease / Real Estate Summary", level=1)

    doc.add_heading("Extracted Key Information", level=2)
    for key, value in structured.items():
        label = key.replace("_", " ").title()
        doc.add_paragraph(f"{label}: {pretty_value(value)}")

    doc.add_heading("Property Value Estimate (AI)", level=2)
    doc.add_paragraph(value_estimate_text)

    # Save to bytes
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------

st.set_page_config(page_title="Real Estate Document Analyzer", layout="wide")

st.title("🏡 Real Estate Document Analyzer")
st.caption("Upload a lease, commercial lease, or purchase agreement to extract structured info, estimate value, and get Q&A.")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    raw_text = extract_text(uploaded)

    # MLS-style layout instruction
    st.subheader("📄 Extracted Text Preview")
    st.text_area("", raw_text, height=220)

    if st.button("Analyze document with AI"):
        with st.spinner("Analyzing document and extracting key fields..."):
            extracted = extract_structured_data(raw_text)
            value_estimate_text = estimate_property_value_text(extracted, raw_text)
            value_range = estimate_property_value_range(extracted, raw_text)

        st.success("Analysis complete!")

        # -------------------------
        # MLS-STYLE SUMMARY SECTIONS
        # -------------------------
        st.subheader("🧾 MLS-Style Summary")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### 🏠 Property Info")
            st.write("**Address:**", pretty_value(extracted.get("property_address")))
            st.write("**Landlord:**", pretty_value(extracted.get("landlord")))
            st.write("**Tenant:**", pretty_value(extracted.get("tenant")))

            st.markdown("### 📅 Lease / Term Info")
            st.write("**Lease Start:**", pretty_value(extracted.get("lease_start")))
            st.write("**Lease End:**", pretty_value(extracted.get("lease_end")))

        with colB:
            st.markdown("### 💵 Financials")
            st.write("**Monthly Rent:**", pretty_value(extracted.get("monthly_rent")))
            st.write("**Security Deposit:**", pretty_value(extracted.get("security_deposit")))
            st.write("**Other Fees:**", pretty_value(extracted.get("other_fees")))

            st.markdown("### 📜 Rules & Clauses")
            st.write("**Late Fee:**", pretty_value(extracted.get("late_fee")))
            st.write("**Utilities:**", pretty_value(extracted.get("utilities")))
            st.write("**Pet Policy:**", pretty_value(extracted.get("pet_policy")))
            st.write("**Termination Clause:**", pretty_value(extracted.get("termination_clause")))

        if extracted.get("notes"):
            st.markdown("### 📝 Notes")
            st.write(pretty_value(extracted.get("notes")))

        with st.expander("View raw JSON"):
            st.json(extracted)

        # -------------------------
        # PROPERTY VALUE ESTIMATOR
        # -------------------------
        st.subheader("📊 Property Value Estimator")

        st.write(value_estimate_text)

        # Chart for Low/Mid/High values
        if value_range["low"] > 0 and value_range["high"] > 0:
            chart_df = pd.DataFrame(
                {
                    "Estimate": ["Low", "Mid", "High"],
                    "Value": [
                        value_range["low"],
                        value_range["mid"],
                        value_range["high"],
                    ],
                }
            )
            st.bar_chart(chart_df.set_index("Estimate"))
        else:
            st.info("Value range not available for chart (model returned zeros).")

        # -------------------------
        # DOWNLOADS: PDF + DOCX
        # -------------------------
        st.subheader("📂 Export Summary")

        pdf_bytes = build_summary_pdf(extracted, value_estimate_text)
        st.download_button(
            "⬇️ Download Summary as PDF",
            data=pdf_bytes,
            file_name="lease_summary.pdf",
            mime="application/pdf",
        )

        docx_bytes = build_summary_docx(extracted, value_estimate_text)
        st.download_button(
            "⬇️ Download Summary as DOCX",
            data=docx_bytes,
            file_name="lease_summary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        # -------------------------
        # Q&A – STANDARD + AGENT PERSONA
        # -------------------------
        st.subheader("💬 Ask Questions About the Document")

        tab1, tab2 = st.tabs(["Standard Q&A", "Real Estate Agent Persona"])

        with tab1:
            q = st.text_input("Ask any question about this document (standard assistant):")
            if st.button("Answer question", key="qa_standard"):
                if q.strip():
                    with st.spinner("Thinking..."):
                        resp = real_estate_agent_answer(q, extracted, raw_text)
                    st.write(resp)
                else:
                    st.warning("Please enter a question.")

        with tab2:
            q2 = st.text_input("Ask Alex (real estate agent persona):")
            if st.button("Ask Alex", key="qa_agent"):
                if q2.strip():
                    with st.spinner("Alex is reviewing your document..."):
                        resp2 = real_estate_agent_answer(q2, extracted, raw_text)
                    st.write(resp2)
                else:
                    st.warning("Please enter a question for Alex.")

else:
    st.info("Upload a PDF lease, commercial lease, or purchase agreement to begin.")
