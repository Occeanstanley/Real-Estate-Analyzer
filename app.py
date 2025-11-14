# app.py — Real Estate Document Analyzer
import os
import io
import json
from typing import Dict, Any, List, Optional

import streamlit as st
import pdfplumber
import pandas as pd
from fpdf import FPDF
from openai import OpenAI

# ────────────────────────────
# Setup
# ────────────────────────────
st.set_page_config(
    page_title="Real Estate Document Analyzer",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

client = OpenAI()  # relies on OPENAI_API_KEY env/secret


# ────────────────────────────
# Styling
# ────────────────────────────
APP_CSS = """
<style>
body, .stApp {
    background: #0f172a;
    color: #e5e7eb;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Main container */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Card */
.card {
    background: #020617;
    border-radius: 18px;
    padding: 18px 20px;
    border: 1px solid #1f2937;
    box-shadow: 0 18px 40px rgba(15,23,42,0.55);
}

/* Subcard */
.subcard {
    background: #020617;
    border-radius: 14px;
    padding: 14px 16px;
    border: 1px solid #111827;
}

/* Headings */
h1, h2, h3, h4 {
    color: #e5e7eb !important;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

.section-title span.icon {
    font-size: 1.2rem;
}

/* Key info labels/values */
.key-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #9ca3af;
    margin-bottom: 0.15rem;
}
.key-value {
    font-size: 0.98rem;
    font-weight: 500;
    color: #e5e7eb;
    margin-bottom: 0.4rem;
}

/* Text preview */
.text-preview {
    background: #020617;
    color: #e5e7eb;
    border-radius: 12px;
    padding: 10px 12px;
    border: 1px solid #1f2937;
    max-height: 260px;
    overflow-y: auto;
    font-size: 0.86rem;
    white-space: pre-wrap;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1f2937;
}

/* Buttons */
.stButton > button {
    border-radius: 999px;
    border: 1px solid #60a5fa;
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    font-weight: 600;
}
.stButton > button:hover {
    border-color: #93c5fd;
}

/* Radio pills for persona */
.stRadio > label {
    font-weight: 500;
    color: #e5e7eb;
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


# ────────────────────────────
# Helpers
# ────────────────────────────
def clean_json_string(s: str) -> str:
    """Remove ``` and ```json fences etc."""
    if not s:
        return "{}"
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        # remove possible leading json tag
        s = s.replace("json\n", "", 1).replace("json\r\n", "", 1)
    return s.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF using pdfplumber."""
    text_chunks: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            text_chunks.append(txt)
    return "\n\n".join(text_chunks)


def extract_tables_from_pdf(file_bytes: bytes) -> List[pd.DataFrame]:
    """Extract tables (if any) from PDF using pdfplumber."""
    tables: List[pd.DataFrame] = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for tbl in page_tables:
                    if not tbl:
                        continue
                    df = pd.DataFrame(tbl)
                    # use first row as header if it looks like one
                    if df.shape[0] > 1:
                        df.columns = df.iloc[0]
                        df = df[1:]
                    tables.append(df)
    except Exception as e:
        st.warning(f"Could not parse tables: {e}")
    return tables


def normalize_value(value: Any) -> str:
    """Turn nested dicts/lists into readable strings."""
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if value is None or value == "":
        return "—"
    return str(value)


# ────────────────────────────
# OpenAI helpers
# ────────────────────────────
def extract_structured_data(text: str) -> Dict[str, Any]:
    """
    Universal extractor for:
    - residential lease
    - commercial lease
    - purchase agreement
    """

    system_prompt = """
    You are a real estate document parser.
    You must return a SINGLE valid JSON object with this exact schema:

    {
      "document_type": "",
      "property_address": "",
      "landlord": "",
      "tenant": "",
      "buyer": "",
      "seller": "",
      "purchase_price": "",
      "earnest_money": "",
      "lease_start": "",
      "lease_end": "",
      "monthly_rent": "",
      "security_deposit": "",
      "late_fee": "",
      "utilities": "",
      "other_fees": "",
      "pet_policy": "",
      "termination_clause": "",
      "notes": ""
    }

    Rules:
    - Works for leases, commercial leases, and purchase agreements.
    - If a field is not present, return empty string "".
    - Do NOT nest JSON inside fields. Flatten everything.
      Example: instead of {"CAM_charges": 450} use "CAM charges: 450".
    - Dates can be written naturally, e.g. "January 1, 2025" or "2025-01-01".
    - Do not add additional top-level keys.
    """

    user_prompt = f"""
    Extract data from this real estate document. It may be a lease or purchase agreement.

    DOCUMENT START:
    {text[:15000]}
    DOCUMENT END.
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    raw = resp.choices[0].message.content
    raw = clean_json_string(raw)

    try:
        data = json.loads(raw)
    except Exception:
        data = {"notes": raw}

    return data


def estimate_property_value(structured: Dict[str, Any], text: str) -> str:
    """LLM-based rough property value estimate / rent sanity check."""
    system_prompt = """
    You are an experienced New York real estate analyst.
    Based on the lease/purchase details, provide a SHORT narrative estimate.

    Output format (plain text, NO JSON):
    - One sentence with a rough value RANGE or rent sanity check.
    - Then 2–4 bullet points explaining the reasoning.
    - End with a short disclaimer that this is NOT an appraisal.
    """

    context = json.dumps(structured, indent=2)

    user_prompt = f"""
    Use the following extracted data and text to estimate a rough property value or
    check whether the rent seems above/below market. Be concise.

    EXTRACTED DATA:
    {context}

    RAW TEXT SNIPPET:
    {text[:6000]}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.35,
    )

    return resp.choices[0].message.content.strip()


def answer_question_standard(question: str, text: str, structured: Dict[str, Any]) -> str:
    system_prompt = """
    You are a precise real estate document assistant.
    Answer questions ONLY using the document text and structured fields.
    If you are not sure, say you are not sure.
    """

    context = json.dumps(structured, indent=2)
    user_prompt = f"""
    QUESTION: {question}

    STRUCTURED FIELDS:
    {context}

    DOCUMENT EXCERPT:
    {text[:8000]}
    """

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    return resp.choices[0].message.content.strip()


def answer_question_persona(question: str, text: str, structured: Dict[str, Any]) -> str:
    system_prompt = """
    You are a friendly, practical New York real estate agent.
    You explain things clearly to tenants, buyers, and landlords.
    Use plain, informal language, but stay accurate to the document.
    Always reference the key numbers if relevant (rent, deposit, dates, fees).
    Start with a one-sentence answer, then add 2–3 bullet tips.
    """

    context = json.dumps(structured, indent=2)
    user_prompt = f"""
    QUESTION: {question}

    STRUCTURED FIELDS:
    {context}

    DOCUMENT EXCERPT:
    {text[:8000]}
    """

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
    )

    return resp.choices[0].message.content.strip()


def build_summary_pdf(structured: Dict[str, Any], value_estimate: Optional[str]) -> bytes:
    """Generate a simple summary PDF, unicode-safe."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Real Estate Document Summary", ln=True)

    pdf.set_font("Arial", "", 12)
    fields = [
        ("Document Type", "document_type"),
        ("Property Address", "property_address"),
        ("Landlord", "landlord"),
        ("Tenant", "tenant"),
        ("Buyer", "buyer"),
        ("Seller", "seller"),
        ("Purchase Price", "purchase_price"),
        ("Earnest Money", "earnest_money"),
        ("Lease Start", "lease_start"),
        ("Lease End", "lease_end"),
        ("Monthly Rent", "monthly_rent"),
        ("Security Deposit", "security_deposit"),
        ("Late Fee", "late_fee"),
        ("Utilities", "utilities"),
        ("Other Fees", "other_fees"),
        ("Pet Policy", "pet_policy"),
        ("Termination Clause", "termination_clause"),
    ]

    for label, key in fields:
        val = normalize_value(structured.get(key))
        pdf.multi_cell(0, 7, f"{label}: {val}")

    notes = normalize_value(structured.get("notes"))
    if notes and notes != "—":
        pdf.ln(4)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Notes", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, notes)

    if value_estimate:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "AI Property Value Estimate", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, value_estimate)

    # latin-1 with "replace" avoids UnicodeEncodeError
    return pdf.output(dest="S").encode("latin-1", "replace")


# ────────────────────────────
# Session state
# ────────────────────────────
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "file_bytes" not in st.session_state:
    st.session_state.file_bytes = None
if "structured" not in st.session_state:
    st.session_state.structured = {}
if "value_estimate" not in st.session_state:
    st.session_state.value_estimate = ""
if "tables" not in st.session_state:
    st.session_state.tables = []


# ────────────────────────────
# Sidebar
# ────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(
        """
**Models used**

- `gpt-4o-mini` → extraction & value estimate  
- `gpt-4o` → Q&A / persona  

Tip: Use clear, text-based PDFs (not scans) for best results.
        """
    )
    st.markdown("---")
    st.markdown(
        "Built by **Stanley Occean** — AI Product / Data Science student. 🌊"
    )


# ────────────────────────────
# Main Layout
# ────────────────────────────
st.markdown(
    "<h1>🏡 Real Estate Document Analyzer</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='color:#9ca3af;'>Upload a lease, commercial lease, or purchase agreement and let AI extract key information, estimate value, and answer questions.</p>",
    unsafe_allow_html=True,
)

with st.container():
    col_u, col_btn = st.columns([4, 1])
    with col_u:
        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Max ~200MB. Use text-based PDFs for best results.",
        )
    with col_btn:
        st.write("")
        st.write("")
        analyze_clicked = st.button("Analyze document with AI", use_container_width=True)

# Handle upload
if uploaded is not None:
    file_bytes = uploaded.read()
    st.session_state.file_bytes = file_bytes
    st.session_state.raw_text = extract_text_from_pdf(file_bytes)

    st.markdown("#### 📎 Extracted Text Preview")
    st.markdown(
        f"<div class='text-preview'>{st.session_state.raw_text[:3000] or 'No text detected.'}</div>",
        unsafe_allow_html=True,
    )

else:
    st.info("Upload a real estate PDF to get started.")

# Run analysis
if analyze_clicked:
    if not st.session_state.file_bytes:
        st.error("Please upload a PDF first.")
    else:
        with st.spinner("Running AI extraction..."):
            try:
                structured = extract_structured_data(st.session_state.raw_text)
                st.session_state.structured = structured
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                structured = {}

        with st.spinner("Estimating property value..."):
            try:
                estimate = estimate_property_value(structured, st.session_state.raw_text)
                st.session_state.value_estimate = estimate
            except Exception as e:
                st.warning(f"Value estimate failed: {e}")
                st.session_state.value_estimate = ""

        with st.spinner("Looking for tables (fee schedules, rent rolls)..."):
            st.session_state.tables = extract_tables_from_pdf(
                st.session_state.file_bytes
            )

        st.success("Analysis complete ✅")


structured = st.session_state.structured
raw_text = st.session_state.raw_text

if structured:
    # ───────── Key Information & Value Estimator ─────────
    st.markdown("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'><span class='icon'>📌</span> <span>Key Information</span></div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    fields_left = [
        ("Document Type", "document_type"),
        ("Property Address", "property_address"),
        ("Landlord", "landlord"),
        ("Tenant", "tenant"),
        ("Buyer", "buyer"),
        ("Seller", "seller"),
    ]
    fields_right = [
        ("Purchase Price", "purchase_price"),
        ("Earnest Money", "earnest_money"),
        ("Monthly Rent", "monthly_rent"),
        ("Security Deposit", "security_deposit"),
        ("Late Fee", "late_fee"),
        ("Utilities", "utilities"),
        ("Other Fees", "other_fees"),
        ("Pet Policy", "pet_policy"),
        ("Termination Clause", "termination_clause"),
    ]

    with c1:
        for label, key in fields_left:
            st.markdown("<div class='key-label'>" + label + "</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='key-value'>"
                + normalize_value(structured.get(key))
                + "</div>",
                unsafe_allow_html=True,
            )

    with c2:
        for label, key in fields_right:
            st.markdown("<div class='key-label'>" + label + "</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='key-value'>"
                + normalize_value(structured.get(key))
                + "</div>",
                unsafe_allow_html=True,
            )

    notes = normalize_value(structured.get("notes"))
    if notes and notes != "—":
        st.markdown("##### Notes")
        st.markdown(notes)

    st.markdown("</div>", unsafe_allow_html=True)

    # ───────── Property Value Estimator ─────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'><span class='icon'>📈</span> <span>Property Value Estimator</span></div>",
        unsafe_allow_html=True,
    )
    if st.session_state.value_estimate:
        st.markdown(
            f"<div style='font-size:0.95rem; color:#d1d5db;'>{st.session_state.value_estimate}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.write("No estimate available for this document.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ───────── Extracted Tables ─────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'><span class='icon'>📊</span> <span>Extracted Tables (from PDF)</span></div>",
        unsafe_allow_html=True,
    )
    if st.session_state.tables:
        for i, df in enumerate(st.session_state.tables, start=1):
            st.markdown(f"**Table {i}**")
            st.dataframe(df, use_container_width=True)
    else:
        st.write("No clear tables were detected in this PDF.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ───────── Q&A + Persona ─────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'><span class='icon'>❓</span> <span>Ask Questions About the Document</span></div>",
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Mode",
        options=["Standard Q&A", "Real Estate Agent Persona"],
        horizontal=True,
    )

    question = st.text_input("Enter your question:", "")
    ask_clicked = st.button("Answer question")

    if ask_clicked:
        if not question.strip():
            st.error("Please type a question first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    if mode == "Standard Q&A":
                        answer = answer_question_standard(question, raw_text, structured)
                    else:
                        answer = answer_question_persona(question, raw_text, structured)
                    st.markdown("##### Answer")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Q&A failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ───────── Download Summary PDF ─────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'><span class='icon'>📥</span> <span>Download Lease / Deal Summary PDF</span></div>",
        unsafe_allow_html=True,
    )
    try:
        pdf_bytes = build_summary_pdf(structured, st.session_state.value_estimate)
        st.download_button(
            "Download Summary PDF",
            data=pdf_bytes,
            file_name="real_estate_summary.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"Could not build PDF: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        "<p style='color:#9ca3af; margin-top:1.5rem;'>Once you run the analysis, key information, value estimate, tables, persona Q&A, and the PDF export will appear here.</p>",
        unsafe_allow_html=True,
    )
