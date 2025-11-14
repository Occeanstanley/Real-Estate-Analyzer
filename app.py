import streamlit as st
import pdfplumber
import docx
import tempfile
import json
from openai import OpenAI

client = OpenAI()

st.set_page_config(
    page_title="Real Estate Document Analyzer",
    layout="wide"
)

st.title("🏡 Real Estate Document Analyzer")
st.write("Upload a lease, contract, or real estate document and let AI extract key information.")

# Function to extract text from PDFs
def extract_pdf_text(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# Function to extract text from Word (docx)
def extract_docx_text(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


# AI extraction
def extract_key_fields(text):
    prompt = f"""
    Extract key real estate lease or contract fields from this text.
    Return ONLY valid JSON with these fields:

    {{
      "property_address": "",
      "landlord": "",
      "tenant": "",
      "lease_start": "",
      "lease_end": "",
      "monthly_rent": "",
      "security_deposit": "",
      "late_fee": "",
      "utilities": "",
      "pet_policy": "",
      "termination_clause": ""
    }}

    Document text:
    {text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# Q&A about the document
def ask_question(text, question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in real estate documents."},
            {"role": "user", "content": f"Document: {text}\n\nAnswer this question: {question}"}
        ]
    )
    return response.choices[0].message.content


uploaded_file = st.file_uploader("📄 Upload Document", type=["pdf", "docx", "txt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Extract text
    if uploaded_file.type == "application/pdf":
        extracted_text = extract_pdf_text(temp_path)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        extracted_text = extract_docx_text(temp_path)
    else:
        extracted_text = open(temp_path).read()

    st.subheader("📌 Extracted Text Preview")
    st.text_area("", extracted_text[:5000], height=300)

    # Extract key fields using AI
    st.subheader("🔍 Key Information")
    extracted_json = extract_key_fields(extracted_text)

    try:
        parsed_data = json.loads(extracted_json)
        st.json(parsed_data)
    except:
        st.warning("AI returned invalid JSON. Showing raw output:")
        st.text(extracted_json)

    st.subheader("💬 Ask Questions About the Document")
    user_q = st.text_input("Enter your question here:")

    if user_q:
        answer = ask_question(extracted_text, user_q)
        st.write("### Answer:")
        st.write(answer)
