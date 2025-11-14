import streamlit as st
import pdfplumber
import re
import json
from openai import OpenAI
from fpdf import FPDF

client = OpenAI()

# ------------------------------
# Helper: Safe Field Finder
# ------------------------------
def find_field(text, keywords):
    for k in keywords:
        pattern = rf"{k}[:\-]?\s*(.*)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


# ------------------------------
# Robust JSON Extractor
# ------------------------------
def extract_structured_data(raw_text):

    prompt = f"""
Extract structured real estate information from this document.
Return ONLY valid JSON with these fields:

- property_address
- landlord
- tenant
- lease_start
- lease_end
- monthly_rent
- security_deposit
- other_fees
- termination_clause
- notes

Document:
{raw_text}
"""

    # Try JSON mode
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            response_format={"type": "json_object"}
        )
        return resp.output[0].content[0].parsed

    except Exception:
        # Try to read raw text and decode JSON manually
        try:
            raw = resp.output_text
            return json.loads(raw)
        except:
            pass

        # Final fallback (never crash)
        return {
            "property_address": find_field(raw_text, ["address", "property"]),
            "landlord": find_field(raw_text, ["landlord", "seller", "owner"]),
            "tenant": find_field(raw_text, ["tenant", "buyer"]),
            "lease_start": find_field(raw_text, ["start", "commence"]),
            "lease_end": find_field(raw_text, ["end", "terminate"]),
            "monthly_rent": find_field(raw_text, ["rent"]),
            "security_deposit": find_field(raw_text, ["deposit"]),
            "other_fees": find_field(raw_text, ["fee", "charge", "earnest", "closing"]),
            "termination_clause": find_field(raw_text, ["termination", "cancel"]),
            "notes": ""
        }


# ------------------------------
# Property Value Estimator
# ------------------------------
def estimate_property_value(structured):
    prompt = f"""
Estimate the property value (NOT an appraisal). Use rent/terms/context.

Return a friendly paragraph.
Property info: {json.dumps(structured)}
"""

    resp = client.responses.create(
        model="gpt-4o",
        input=prompt
    )
    return resp.output_text


# ------------------------------
# Real Estate Agent Persona Q&A
# ------------------------------
def agent_persona_answer(question, structured):
    prompt = f"""
You are a friendly Long Island real estate agent.
Use the extracted lease/purchase agreement info:

{json.dumps(structured, indent=2)}

Answer the user's question conversationally.
Question: {question}
"""

    resp = client.responses.create(model="gpt-4o", input=prompt)
    return resp.output_text


# ------------------------------
# PDF Summary Generator (FIXED)
# ------------------------------
def build_summary_pdf(structured, estimate):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Lease / Agreement Summary", ln=True)

    for k, v in structured.items():
        pdf.multi_cell(0, 10, f"{k.replace('_',' ').title()}: {v}")

    pdf.ln(5)
    pdf.multi_cell(0, 10, "Property Value Estimate:")
    pdf.multi_cell(0, 10, estimate)

    return pdf.output(dest="S").encode("latin-1", "replace")


# ------------------------------
# PDF Text Extraction + Tables
# ------------------------------
def extract_pdf_text_and_tables(file):
    text = ""
    tables = []

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            page_tables = page.extract_tables()
            if page_tables:
                tables.extend(page_tables)

    return text, tables


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Real Estate Analyzer", layout="wide")

st.markdown("""
<style>
body { background-color: #0d1117; color: white; }
.stMarkdown, .stText, .stHeader, .stDataFrame { color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏡 Real Estate Document Analyzer")
st.write("Upload a lease, commercial lease, or purchase agreement.")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    st.success(f"File uploaded: {uploaded.name}")

    raw_text, tables = extract_pdf_text_and_tables(uploaded)

    st.subheader("📄 Extracted Text Preview")
    st.code(raw_text[:2000] + "...")  # Show first 2000 chars only

    if st.button("Analyze document with AI"):
        with st.spinner("Analyzing with AI..."):
            structured = extract_structured_data(raw_text)

        st.subheader("📌 Key Information")

        # Clean printing (remove dict artifacts)
        for k, v in structured.items():
            st.write(f"**{k.replace('_', ' ').title()}:** {v}")

        # Property value estimate
        st.subheader("📈 Property Value Estimator")
        estimate = estimate_property_value(structured)
        st.write(estimate)

        # Tables
        st.subheader("📊 Extracted Tables")
        if tables:
            for t in tables:
                st.table(t)
        else:
            st.write("No tables detected.")

        # Q&A
        st.subheader("❓ Ask Questions About the Document")
        col1, col2 = st.columns(2)
        with col1:
            mode = st.radio("Mode", ["Standard Q&A", "Real Estate Agent Persona"])

        question = st.text_input("Enter your question:")
        if st.button("Answer question"):
            with st.spinner("Thinking..."):
                if mode == "Standard Q&A":
                    resp = client.responses.create(
                        model="gpt-4o",
                        input=f"Answer based on document info: {structured}. Question: {question}"
                    )
                    st.write(resp.output_text)
                else:
                    st.write(agent_persona_answer(question, structured))

        # PDF download
        st.subheader("📥 Download Lease Summary PDF")
        pdf_bytes = build_summary_pdf(structured, estimate)
        st.download_button("Download Summary PDF",
                           pdf_bytes,
                           file_name="lease_summary.pdf",
                           mime="application/pdf")
