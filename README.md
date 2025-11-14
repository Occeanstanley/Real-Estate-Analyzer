# 🏡 Real Estate Document Analyzer

A smart AI-powered tool that extracts key information from **leases, rental agreements, commercial leases, and purchase agreements**.  
Built with **Streamlit + GPT-4o**, it turns messy PDFs into clean structured data.

## ✨ Features

### 🔍 Document Extraction
- Upload **PDF leases / contracts / purchase agreements**
- AI extracts:
  - Property address  
  - Landlord / Tenant  
  - Lease dates  
  - Rent + Deposits  
  - Fees (CAM charges, earnest deposits, etc.)  
  - Utilities, pet policy, late fees  
  - Termination clauses  
  - Notes + addenda  

### 🤖 Smart Real Estate AI
- **Standard Q&A** about the document  
- **Real Estate Agent Persona** for friendly explanations  
- Clean formatting (no JSON chunks shown to users)

### 📊 Property Value Estimator
- High-level AI estimate based on:
  - Rent  
  - Area (inferred from text)  
  - Long Island / NYC surrounding market logic  

### 📄 Downloadable PDF Summary
- Auto-generated **Lease Summary PDF**
- Fully Unicode-safe (no more errors from special characters)

### 📑 Table Extraction (beta)
- Detects tables like:
  - Rent roll  
  - Fee schedules  
  - Closing cost lines

---

## 🚀 Technology Used

- **Streamlit**
- **OpenAI GPT-4o & GPT-4o-mini**
- **PDFPlumber** (PDF text extraction)
- **FPDF** (PDF builder)
- **Python 3.9+**

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
streamlit run app.py
