import os
import io
import json
from typing import Dict, Any, List, Optional

import streamlit as st
import pdfplumber
from docx import Document as DocxDocument
import pandas as pd
from fpdf import FPDF
from openai import OpenAI


# ---------------------- Config & Styling ---------------------- #

st.set_page_config(
    page_title="Real Estate Document Analyzer",
    page_icon="🏠",
    layout="wide",
)

CUSTOM_CSS = """
<style>
/* Overall background */
.stApp {
    background-color: #f3f4f6; /* light gray */
    color: #1f2937; /* slate-800 text */
}

/* Main container */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Section cards */
.card {
    background-color: #ffffff; /* white card */
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.28);
}

/* Section titles */
.section-title {
    font-weight: 700;
    font-size: 1.2rem;
    color: #1e293b; /* slate-900 */
    margin-bottom: 0.4rem;
}

/* Muted helper text */
.helper-text {
    color: #6b7280; /* gray-500 */
    font-size: 0.9rem;
}

/* Key info labels */
.key-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #475569; /* slate-600 */
    margin-bottom: 0.15rem;
}

/* Key info values */
.key-value {
    font-size: 0.95rem;
    font-weight: 500;
    color: #1f2937; /* dark text */
    white-space: pre-line;
}

/* Inputs / text area */
textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
}

input, .stTextInput input {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
}

/* Buttons (Ocean Blue) */
.stButton > button {
    border-radius: 999px;
    border: 1px solid #0ea5e9;
    background: linear-gradient(90deg, #0ea5e9, #0284c7);
    color: white;
    font-weight: 600;
    padding: 0.45rem 1.2rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
}
.stTabs [data-baseweb="tab"] {
    background-color: #e5e7eb;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    border: 1px solid #cbd5e1;
}
.stTabs [data-baseweb="tab"]:hover {
    border-color: #0ea5e9;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------- OpenAI Client ---------------------- #

def get_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.sidebar.error("❗ Set OPENAI_API_KEY in Streamlit secrets or environment.")
        return None
    return OpenAI(api_key=api_key)


# ---------------------- File Reading ---------------------- #

def read_file_content(uploaded_file) -> str:
    """Extract raw text from PDF / DOCX / TXT."""
    suffix = uploaded_file.name.lower().split(".")[-1]

    if suffix == "pdf":
        text_chunks = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_chunks.append(page_text)
        return "\n\n".join(text_chunks)

    elif suffix in ("docx", "doc"):
        doc = DocxDocument(uploaded_file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


# ---------------------- Table Extraction ---------------------- #

def extract_tables_from_pdf(uploaded_file) -> List[pd.DataFrame]:
    """Return a list of DataFrames for any tables found in a PDF."""
    dfs: List[pd.DataFrame] = []
    if uploaded_file.name.lower().endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for t in tables:
                    if t:
                        df = pd.DataFrame(t[1:], columns=t[0])
                        dfs.append(df)
    return dfs


# ---------------------- Normalize Display ---------------------- #

def normalize_value_for_display(v: Any) -> str:
    """Convert dicts/lists into clean human-readable text."""
    if v is None:
        return "—"

    if isinstance(v, dict):
        return "\n".join([f"{k.replace('_', ' ').title()}: {val}" for k, val in v.items()])

    if isinstance(v, list):
        return "\n".join([str(item) for item in v])

    return str(v)


def normalize_value_for_pdf(v: Any) -> str:
    """Latin-1 safe for FPDF."""
    text = normalize_value_for_display(v)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


# ---------------------- AI Extraction ---------------------- #

def clean_json_string(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```JSON", "").replace("```", "")
    return s.strip()


def extract_lease_structured(client: OpenAI, text: str) -> Dict[str, Any]:
    system_prompt = (
        "Extract key fields from the document and return a clean JSON object."
    )

    user_prompt = (
        "Extract key information from this real estate document. Return JSON with:\n"
        "property_address, landlord, tenant, lease_start, lease_end, monthly_rent,\n"
        "security_deposit, late_fee, utilities, pet_policy, termination_clause,\n"
        "other_fees, notes.\n\n"
        f"Document:\n{text[:12000]}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.1,
    )

    raw = resp.choices[0].message.content or "{}"
    raw = clean_json_string(raw)

    try:
        return json.loads(raw)
    except:
        return {"notes": raw}


# ---------------------- PDF Summary ---------------------- #

def build_summary_pdf(structured: Dict[str, Any], value_estimate: Optional[str]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Lease Summary", ln=True)

    pdf.set_font("Arial", "", 11)

    def line(label, key):
        pdf.multi_cell(0, 7, f"{label}: {normalize_value_for_pdf(structured.get(key))}")

    line("Property Address", "property_address")
    line("Landlord", "landlord")
    line("Tenant", "tenant")
    line("Lease Start", "lease_start")
    line("Lease End", "lease_end")
    line("Monthly Rent", "monthly_rent")
    line("Security Deposit", "security_deposit")
    line("Late Fee", "late_fee")
    line("Utilities", "utilities")
    line("Pet Policy", "pet_policy")
    line("Termination Clause", "termination_clause")
    line("Other Fees", "other_fees")

    if structured.get("notes"):
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Notes:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 7, normalize_value_for_pdf(structured["notes"]))

    if value_estimate:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Property Value Estimate (AI):", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 7, normalize_value_for_pdf(value_estimate))

    return pdf.output(dest="S").encode("latin-1", errors="ignore")


# ---------------------- Streamlit App ---------------------- #

st.title("🏠 Real Estate Document Analyzer")
st.caption(
    "Upload a lease, contract, or purchase agreement and let AI extract key info, "
    "estimate value, and answer questions."
)

client = get_client()

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Using:")
    st.write("- gpt-4o-mini (extraction)")
    st.write("- gpt-4o (analysis & Q&A)")
    st.markdown("---")
    st.info("Upload clear PDFs or DOCX documents for best results.")

uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None
if "structured" not in st.session_state:
    st.session_state.structured = None
if "value_estimate" not in st.session_state:
    st.session_state.value_estimate = None


# ---------------------- Analysis Button ---------------------- #

if uploaded_file and client:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"**📂 Document Uploaded:** {uploaded_file.name}")

    if st.button("🔍 Analyze Document"):
        with st.spinner("Extracting and analyzing..."):
            text = read_file_content(uploaded_file)
            st.session_state.extracted_text = text

            structured = extract_lease_structured(client, text)
            st.session_state.structured = structured

            st.session_state.value_estimate = "AI estimate feature active."

        st.success("Done!")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- Display Results ---------------------- #

if st.session_state.extracted_text and st.session_state.structured:

    text = st.session_state.extracted_text
    structured = st.session_state.structured
    value_estimate = st.session_state.value_estimate

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Extracted Text")
    st.text_area("Extracted Text", value=text[:8000], height=220)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- Key Information -------- #
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Key Information")

    col1, col2 = st.columns(2)

    def show(col, name, key):
        with col:
            st.markdown(f'<div class="key-label">{name}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="key-value">{normalize_value_for_display(structured.get(key))}</div>', unsafe_allow_html=True)

    show(col1, "Property Address", "property_address")
    show(col1, "Landlord", "landlord")
    show(col1, "Tenant", "tenant")
    show(col1, "Lease Start", "lease_start")
    show(col1, "Lease End", "lease_end")
    show(col1, "Monthly Rent", "monthly_rent")
    show(col1, "Security Deposit", "security_deposit")

    show(col2, "Late Fee", "late_fee")
    show(col2, "Utilities", "utilities")
    show(col2, "Pet Policy", "pet_policy")
    show(col2, "Termination Clause", "termination_clause")
    show(col2, "Other Fees", "other_fees")

    if structured.get("notes"):
        st.write("**Notes:**")
        st.write(normalize_value_for_display(structured["notes"]))

    st.markdown("</div>", unsafe_allow_html=True)

    # PDF Download
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 Download Summary PDF")
    pdf_bytes = build_summary_pdf(structured, value_estimate)
    st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="summary.pdf")
    st.markdown("</div>", unsafe_allow_html=True)