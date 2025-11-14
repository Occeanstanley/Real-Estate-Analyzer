import streamlit as st
import base64
import pdfplumber
from fpdf import FPDF
from openai import OpenAI
import json
import re

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

def clean_unicode(text: str) -> str:
    """Convert Unicode → ASCII-safe for PDF"""
    return (
        text.encode("ascii", "replace")
            .decode()
            .replace("?", "")  # remove replacement marks
    )

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        all_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return all_text


# -----------------------------
# 1. STRUCTURED DATA EXTRACTION
# -----------------------------
def extract_structured_data(raw_text: str) -> dict:
    system_prompt = (
        "Extract structured information from the real estate document. "
        "Return ONLY valid JSON with the following keys:\n"
        "property_address, landlord, tenant, buyer, seller, lease_start, lease_end, "
        "monthly_rent, security_deposit, purchase_price, earnest_money, utilities, "
        "pet_policy, other_fees, termination_clause, notes"
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text}
        ]
    )

    try:
        raw_json = resp.output_text
        cleaned = raw_json.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
    except:
        data = {}

    # Always return a valid dictionary with all fields
    default_fields = {
        "property_address": "",
        "landlord": "",
        "tenant": "",
        "buyer": "",
        "seller": "",
        "lease_start": "",
        "lease_end": "",
        "monthly_rent": "",
        "security_deposit": "",
        "purchase_price": "",
        "earnest_money": "",
        "utilities": "",
        "pet_policy": "",
        "other_fees": "",
        "termination_clause": "",
        "notes": "",
    }

    default_fields.update(data)
    return default_fields


# -----------------------------
# 2. PROPERTY VALUE ESTIMATOR
# -----------------------------
def estimate_property_value(data: dict, raw_text: str) -> str:
    prompt = (
        "You are a real estate valuation assistant. "
        "Give a high-level estimated property value range (not an appraisal). "
        "Use the lease/purchase data, address, and rent.\n\n"
        f"Here is the extracted data:\n{json.dumps(data, indent=2)}\n\n"
        f"Here is the document text:\n{raw_text}"
    )

    resp = client.responses.create(
        model="gpt-4o",
        input=prompt
    )

    return resp.output_text


# -----------------------------
# 3. PDF SUMMARY BUILDER
# -----------------------------
def build_summary_pdf(data: dict, estimate_text: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Real Estate Document Summary", ln=True)

    # Content
    pdf.set_font("Arial", "", 12)

    for key, value in data.items():
        safe_key = clean_unicode(key.replace("_", " ").title())
        safe_val = clean_unicode(str(value))
        pdf.multi_cell(0, 8, f"{safe_key}: {safe_val}")

    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Estimated Property Value", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, clean_unicode(estimate_text))

    # Output PDF safely
    return pdf.output(dest="S").encode("latin-1", "replace")


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Real Estate Analyzer", layout="wide")

st.title("🏡 Real Estate Document Analyzer")
st.write("Upload a lease, purchase agreement, or real estate document to extract key data.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)

    st.subheader("📄 Extracted Text Preview")
    st.text_area("Document Text:", raw_text, height=300)

    if st.button("Analyze document with AI"):
        st.success("Analyzing...")

        structured = extract_structured_data(raw_text)

        st.subheader("📌 Key Information")
        for k, v in structured.items():
            st.write(f"**{k.replace('_',' ').title()}**: {v}")

        # Estimate
        st.subheader("📈 Property Value Estimator")
        estimate = estimate_property_value(structured, raw_text)
        st.write(estimate)

        # PDF Download
        st.subheader("📄 Download Lease Summary PDF")
        pdf_bytes = build_summary_pdf(structured, estimate)

        b64_pdf = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="summary.pdf">⬇️ Download Summary PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

