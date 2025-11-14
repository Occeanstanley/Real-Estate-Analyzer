
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
.stApp { background-color: #f5f5f7; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
}
.section-title { font-weight: 700; font-size: 1.1rem; color: #111827; margin-bottom: 0.3rem; }
.helper-text { color: #6b7280; font-size: 0.9rem; }
.key-label { font-size: 0.9rem; font-weight: 600; color: #4b5563; }
.key-value { font-size: 0.95rem; font-weight: 500; color: #111827; }
textarea[aria-label="Extracted Text Preview"] {
    font-size: 0.85rem !important;
    line-height: 1.4 !important;
}
.stButton > button {
    border-radius: 999px;
    border: 1px solid #e5e7eb;
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    font-weight: 600;
}
button[kind="secondary"] { border-radius: 999px !important; }
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

# ---------------------- Table Extraction (PDF) ---------------------- #
def extract_tables_from_pdf(uploaded_file) -> List[pd.DataFrame]:
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
                df = pd.DataFrame(t[1:], columns=t[0])
                dfs.append(df)
    return dfs

# ---------------------- LLM Helpers ---------------------- #
def clean_json_string(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```JSON", "").replace("```", "")
    return s.strip()

def extract_lease_structured(client: OpenAI, text: str) -> Dict[str, Any]:
    system_prompt = (
        "You are an assistant that extracts structured data from real estate leases or contracts."
        " Always return a single valid JSON object."
    )
    user_prompt = (
        "Extract key information from this document and return a JSON object with:"
        " property_address, landlord, tenant, lease_start, lease_end, monthly_rent,"
        " security_deposit, late_fee, utilities, pet_policy, termination_clause, other_fees, notes.\n\n"
        f"Document text:\n{text[:12000]}"
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
        data = {"notes": raw}
    return data

def answer_question_about_doc(client: OpenAI, question: str, document_text: str, structured: Dict[str, Any], persona_mode: str = "neutral") -> str:
    if persona_mode == "agent":
        system = "You are Alex Morgan, a practical New York real estate agent. Be clear and concise."
    else:
        system = "You are a helpful assistant explaining real estate documents."
    structured_snippet = json.dumps(structured, ensure_ascii=False, indent=2)
    user = (
        f"Here is the structured lease info:\n{structured_snippet}\n\n"
        f"Document text:\n{document_text[:12000]}\n\n"
        f"Question: {question}\nIf unsure, say you cannot be certain."
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def estimate_property_value(client: OpenAI, structured: Dict[str, Any], document_text: str) -> str:
    address = structured.get("property_address") or "Unknown"
    rent = structured.get("monthly_rent") or "Unknown"
    system = "You are a real estate pricing assistant giving rough value/rent estimates (not appraisals)."
    user = (
        f"Property address: {address}\nMonthly rent: {rent}\n\n"
        f"Snippet:\n{document_text[:4000]}\n\n"
        "1. Estimate value range. 2. Is rent above/below market? 3. Note 2–3 influencing factors."
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()

# ---------------------- PDF Summary ---------------------- #
def build_summary_pdf(structured: Dict[str, Any], value_estimate: Optional[str]) -> bytes:
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
        value = structured.get(key, "-")
        pdf.multi_cell(0, 7, f"{to_latin1(label)}: {to_latin1(value)}")

    for label, key in [
        ("Property Address", "property_address"), ("Landlord", "landlord"),
        ("Tenant", "tenant"), ("Lease Start", "lease_start"), ("Lease End", "lease_end"),
        ("Monthly Rent", "monthly_rent"), ("Security Deposit", "security_deposit"),
        ("Late Fee", "late_fee"), ("Utilities", "utilities"),
        ("Pet Policy", "pet_policy"), ("Termination Clause", "termination_clause"),
        ("Other Fees", "other_fees")
    ]:
        line(label, key)

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

    return pdf.output(dest="S").encode("latin-1")

# ---------------------- Streamlit App ---------------------- #
st.title("🏠 Real Estate Document Analyzer")
st.caption("Upload a lease, contract, or real-estate document to extract key info, estimate value, and ask questions.")
client = get_client()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.write("Models used:")
    st.write("- `gpt-4o-mini` for fast extraction")
    st.write("- `gpt-4o` for reasoning & Q&A")
    st.markdown("---")
    st.info("Tip: Upload clear PDFs or DOCX leases for best results.")

uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"], help="Supported formats: PDF, DOCX, TXT (up to 200 MB).")

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None
if "structured" not in st.session_state:
    st.session_state.structured = None
if "value_estimate" not in st.session_state:
    st.session_state.value_estimate = None

if uploaded_file and client:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📂 Document Uploaded</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="helper-text">File: <b>{uploaded_file.name}</b></div>', unsafe_allow_html=True)

    if st.button("🔍 Analyze document with AI", type="primary"):
        with st.spinner("Analyzing document..."):
            text = read_file_content(uploaded_file)
            st.session_state.extracted_text = text
            structured = extract_lease_structured(client, text)
            st.session_state.structured = structured
            value_estimate = estimate_property_value(client, structured, text)
            st.session_state.value_estimate = value_estimate
        st.success("Analysis complete.")
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.extracted_text and st.session_state.structured:
    text = st.session_state.extracted_text
    structured = st.session_state.structured
    value_estimate = st.session_state.value_estimate

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧾 Key Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="helper-text">AI-extracted fields from the document.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    def format_display_value(val):
        if not val or val == "None":
            return None
        if isinstance(val, dict):
            lines = []
            for k, v in val.items():
                if isinstance(v, list):
                    v = ", ".join(str(x) for x in v)
                lines.append(f"**{k.replace('_', ' ').title()}**: {v}")
            return "<br>".join(lines)
        if isinstance(val, list):
            return "<br>".join(f"• {str(x)}" for x in val)
        return str(val)

    def show_field(col, label, key):
        value = structured.get(key)
        display = format_display_value(value)
        if display:
            with col:
                st.markdown(f'<div class="key-label">{label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="key-value">{display}</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏡 Property Value Estimator</div>', unsafe_allow_html=True)
    st.write(value_estimate)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Download Lease Summary PDF</div>', unsafe_allow_html=True)
    pdf_bytes = build_summary_pdf(structured, value_estimate)
    st.download_button("⬇️ Download Summary PDF", data=pdf_bytes, file_name="lease_summary.pdf", mime="application/pdf")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 Ask Questions About the Document</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Standard Q&A", "Real Estate Agent Persona"])

    with tab1:
        q = st.text_input("Enter your question about this document:")
        if st.button("Answer question", key="qa_standard") and q.strip():
            with st.spinner("Thinking..."):
                answer = answer_question_about_doc(client, q.strip(), text, structured, "neutral")
            st.write(answer)

    with tab2:
        q2 = st.text_input("Ask Alex (NY real estate agent):", key="qa_agent_input")
        if st.button("Ask Alex", key="qa_agent_button") and q2.strip():
            with st.spinner("Alex is reviewing your lease..."):
                answer2 = answer_question_about_doc(client, q2.strip(), text, structured, "agent")
            st.write(answer2)
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file and uploaded_file.name.lower().endswith(".pdf"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📝 Extracted Tables (from PDF)</div>', unsafe_allow_html=True)
        tables = extract_tables_from_pdf(uploaded_file)
        if tables:
            for idx, df in enumerate(tables, start=1):
                st.markdown(f"**Table {idx}**")
                st.dataframe(df, use_container_width=True)
        else:
            st.write("No clear tables detected in this PDF.")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    if uploaded_file is None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📥 Start by uploading a document</div>', unsafe_allow_html=True)
        st.markdown('<div class="helper-text">Upload a PDF, DOCX, or TXT lease to begin the analysis.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
