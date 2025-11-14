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
    background-color: #f5f5f7;
}

/* Main container card look */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Section cards */
.card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
}

/* Section titles */
.section-title {
    font-weight: 700;
    font-size: 1.1rem;
    color: #111827;
    margin-bottom: 0.3rem;
}

/* Muted helper text */
.helper-text {
    color: #6b7280;
    font-size: 0.9rem;
}

/* Key info labels */
.key-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #4b5563;
}

/* Key info values */
.key-value {
    font-size: 0.95rem;
    font-weight: 500;
    color: #111827;
}

/* Make text areas more readable */
textarea[aria-label="Extracted Text Preview"] {
    font-size: 0.85rem !important;
    line-height: 1.4 !important;
}

/* Buttons */
.stButton > button {
    border-radius: 999px;
    border: 1px solid #e5e7eb;
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    font-weight: 600;
}

/* Secondary buttons (download, etc.) */
button[kind="secondary"] {
    border-radius: 999px !important;
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

    else:  # txt and others
        return uploaded_file.read().decode("utf-8", errors="ignore")


# ---------------------- Table Extraction (PDF) ---------------------- #

def extract_tables_from_pdf(uploaded_file) -> List[pd.DataFrame]:
    """Return a list of DataFrames for any tables found in a PDF."""
    dfs: List[pd.DataFrame] = []
    suffix = uploaded_file.name.lower().split(".")[-1]
    if suffix != "pdf":
        return dfs

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for t in tables:
                if not t:
                    continue
                # First row is often header
                df = pd.DataFrame(t[1:], columns=t[0])
                dfs.append(df)
    return dfs


# ---------------------- LLM Helpers ---------------------- #

def clean_json_string(s: str) -> str:
    """Remove ```json fences if the model added them."""
    s = s.strip()
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```JSON", "")
        s = s.replace("```", "")
    return s.strip()


def extract_lease_structured(client: OpenAI, text: str) -> Dict[str, Any]:
    """Use GPT-4o-mini to extract key lease fields as JSON."""
    system_prompt = (
        "You are an assistant that extracts **structured data** from residential "
        "and commercial real estate documents such as leases or contracts. "
        "Always return a single valid JSON object."
    )
    user_prompt = (
        "Extract the key information from the following lease / real estate document.\n"
        "Return a JSON object with (if available):\n"
        "property_address, landlord, tenant, lease_start, lease_end, monthly_rent,\n"
        "security_deposit, late_fee, utilities, pet_policy, termination_clause,\n"
        "other_fees, notes.\n\n"
        f"Document text:\n{text[:12000]}"  # safety limit
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    raw = resp.choices[0].message.content or "{}"
    raw = clean_json_string(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # last-resort fallback: wrap as notes
        data = {"notes": raw}
    return data


def answer_question_about_doc(
    client: OpenAI,
    question: str,
    document_text: str,
    structured: Dict[str, Any],
    persona_mode: str = "neutral",
) -> str:
    """Use GPT-4o to answer questions about the document."""
    if persona_mode == "agent":
        system = (
            "You are Alex Morgan, an experienced New York real estate agent. "
            "Explain things clearly, practically, and avoid giving formal legal advice. "
            "Base all answers ONLY on the provided lease text and structured fields."
        )
    else:
        system = (
            "You are a helpful assistant answering questions about a real estate document. "
            "Base all answers ONLY on the provided lease text and structured fields."
        )

    structured_snippet = json.dumps(structured, ensure_ascii=False, indent=2)

    user = (
        f"Here is the structured lease info:\n{structured_snippet}\n\n"
        f"Here is the full document text:\n{document_text[:12000]}\n\n"
        f"Question: {question}\n"
        "If information is missing, say you cannot be certain instead of guessing."
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def estimate_property_value(
    client: OpenAI,
    structured: Dict[str, Any],
    document_text: str,
) -> str:
    """Rough value/rent estimate based on address + rent in the doc."""
    address = structured.get("property_address") or "Unknown"
    rent = structured.get("monthly_rent") or "Unknown"

    system = (
        "You are a real estate pricing assistant. "
        "Give a rough, **high-level** estimate of property value and market rent.\n"
        "Assume this is in the United States and clearly communicate that this is "
        "NOT an official appraisal, just an AI estimate."
    )

    user = (
        f"Property address: {address}\n"
        f"Monthly rent (from document, may be approximate or text): {rent}\n\n"
        "Document text snippet:\n"
        f"{document_text[:4000]}\n\n"
        "1. Give an estimated value range for the property (low–high) and explain assumptions.\n"
        "2. Comment on whether the rent seems under/over/around market.\n"
        "3. Mention 2–3 factors that could change this estimate (location details, condition, etc.)."
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


# ---------------------- PDF Summary Generation ---------------------- #

def build_summary_pdf(structured: Dict[str, Any], value_estimate: Optional[str]) -> bytes:
    """Create a simple summary PDF and return its bytes."""

    # Helper to ensure all text is Latin-1 safe for FPDF's core fonts
    def to_latin1(text: Any) -> str:
        s = str(text)
        return s.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, to_latin1("Lease Summary"), ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "", 11)

    def line(label: str, key: str):
        value = structured.get(key, "-")  # use ASCII hyphen instead of em dash
        pdf.multi_cell(0, 7, f"{to_latin1(label)}: {to_latin1(value)}")

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

    notes = structured.get("notes")
    if notes:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, to_latin1("Notes"), ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 7, to_latin1(notes))

    if value_estimate:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, to_latin1("Property Value Estimate (AI)"), ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 7, to_latin1(value_estimate))

    # Export as bytes
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes


# ---------------------- Streamlit App ---------------------- #

st.title("🏠 Real Estate Document Analyzer")
st.caption(
    "Upload a lease, contract, or other real-estate document and let AI extract key info, "
    "estimate value, and answer questions."
)

client = get_client()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.write("Models used:")
    st.write("- `gpt-4o-mini` for fast extraction")
    st.write("- `gpt-4o` for reasoning & Q&A")
    st.markdown("---")
    st.info(
        "Tip: For best results, upload clear PDFs or DOCX leases. "
        "You can also try rent roll PDFs, addenda, or purchase contracts."
    )

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, TXT (up to ~200 MB on Streamlit).",
)

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None
if "structured" not in st.session_state:
    st.session_state.structured = None
if "value_estimate" not in st.session_state:
    st.session_state.value_estimate = None

if uploaded_file and client:
    # --------- Extraction & Analysis Button --------- #
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📂 Document Uploaded</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="helper-text">File: <b>{uploaded_file.name}</b></div>',
        unsafe_allow_html=True,
    )

    if st.button("🔍 Analyze document with AI", type="primary"):
        with st.spinner("Reading document and extracting information..."):
            text = read_file_content(uploaded_file)
            st.session_state.extracted_text = text

            structured = extract_lease_structured(client, text)
            st.session_state.structured = structured

            value_estimate = estimate_property_value(client, structured, text)
            st.session_state.value_estimate = value_estimate

        st.success("Analysis complete.")
    st.markdown("</div>", unsafe_allow_html=True)

# Only show results if we have analysis
if st.session_state.extracted_text and st.session_state.structured:
    text = st.session_state.extracted_text
    structured = st.session_state.structured
    value_estimate = st.session_state.value_estimate

    # --------- Extracted Text Preview --------- #
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Extracted Text Preview</div>', unsafe_allow_html=True)
    st.text_area(
        "Extracted Text Preview",
        value=text[:8000],
        height=220,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --------- Key Information (Clean JSON, no ```json) --------- #
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧾 Key Information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="helper-text">AI-extracted fields from the document.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    def show_field(col, label, key):
        with col:
            st.markdown(f'<div class="key-label">{label}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="key-value">{structured.get(key, "-")}</div>',
                unsafe_allow_html=True,
            )

    show_field(col1, "Property Address", "property_address")
    show_field(col1, "Landlord", "landlord")
    show_field(col1, "Tenant", "tenant")
    show_field(col1, "Lease Start", "lease_start")
    show_field(col1, "Lease End", "lease_end")
    show_field(col1, "Monthly Rent", "monthly_rent")
    show_field(col1, "Security Deposit", "security_deposit")

    show_field(col2, "Late Fee", "late_fee")
    show_field(col2, "Utilities", "utilities")
    show_field(col2, "Pet Policy", "pet_policy")
    show_field(col2, "Termination Clause", "termination_clause")
    show_field(col2, "Other Fees", "other_fees")

    if structured.get("notes"):
        st.markdown("**Notes:**")
        st.write(structured["notes"])

    with st.expander("View raw JSON"):
        st.json(structured)

    st.markdown("</div>", unsafe_allow_html=True)

    # --------- Property Value Estimator --------- #
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏡 Property Value Estimator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="helper-text">Rough AI estimate based on rent, address, and lease context (not an appraisal).</div>',
        unsafe_allow_html=True,
    )

    st.write(value_estimate)
    st.markdown("</div>", unsafe_allow_html=True)

    # --------- Lease Summary PDF Download --------- #
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Download Lease Summary PDF</div>', unsafe_allow_html=True)
    pdf_bytes = build_summary_pdf(structured, value_estimate)
    st.download_button(
        label="⬇️ Download Summary PDF",
        data=pdf_bytes,
        file_name="lease_summary.pdf",
        mime="application/pdf",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --------- Q&A (Neutral + Agent Persona) --------- #
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 Ask Questions About the Document</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Standard Q&A", "Real Estate Agent Persona"])

    with tab1:
        q = st.text_input("Enter your question about this document:")
        if st.button("Answer question", key="qa_standard") and q.strip():
            with st.spinner("Thinking..."):
                answer = answer_question_about_doc(
                    client,
                    q.strip(),
                    text,
                    structured,
                    persona_mode="neutral",
                )
            st.write(answer)

    with tab2:
        q2 = st.text_input(
            "Ask Alex (NY real estate agent) a question about this document:",
            key="qa_agent_input",
        )
        if st.button("Ask Alex", key="qa_agent_button") and q2.strip():
            with st.spinner("Alex is reviewing your lease..."):
                answer2 = answer_question_about_doc(
                    client,
                    q2.strip(),
                    text,
                    structured,
                    persona_mode="agent",
                )
            st.write(answer2)

    st.markdown("</div>", unsafe_allow_html=True)

    # --------- Table Extraction (for PDFs) --------- #
    if uploaded_file and uploaded_file.name.lower().endswith(".pdf"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📝 Extracted Tables (from PDF)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="helper-text">AI-ready tables like fee schedules or rent rolls will appear here if detected.</div>',
            unsafe_allow_html=True,
        )

        tables = extract_tables_from_pdf(uploaded_file)
        if tables:
            for idx, df in enumerate(tables, start=1):
                st.markdown(f"**Table {idx}**")
                st.dataframe(df, use_container_width=True)
        else:
            st.write("No clear tables were detected in this PDF.")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    if uploaded_file is None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📥 Start by uploading a document</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="helper-text">Upload a PDF, DOCX, or TXT lease to begin the analysis.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
