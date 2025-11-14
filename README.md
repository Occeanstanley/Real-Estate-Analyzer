# 🏠 Real Estate Document Analyzer

**Real Estate Document Analyzer** is an AI-powered Streamlit web app that helps users automatically extract, summarize, and understand key information from real estate documents — such as leases, purchase agreements, and contracts.

Built with **OpenAI GPT-4o models**, it can analyze text, estimate property values, and even answer natural-language questions about uploaded documents.

---

## 🚀 Features

### 🔍 Document Analysis
- Upload and analyze **PDF**, **DOCX**, or **TXT** files (up to 200 MB)
- Automatically extract structured key fields:
  - Property address
  - Landlord / tenant names
  - Lease start & end dates
  - Rent, deposits, late fees
  - Utilities, pet policies, termination clauses, etc.
- Works with both residential and commercial real-estate documents

### 🧾 Clean Key Information Display
- Displays fields in a clear, readable layout  
- Hides empty or “None” values  
- Formats nested data like utilities or fees into labeled text

### 🏡 Property Value Estimator
- Uses AI to generate a **rough market value estimate**
- Comments on rent competitiveness and influencing factors

### 💬 Smart Q&A
- Ask natural-language questions about the uploaded document  
- Two modes:
  - **Standard Q&A:** factual and concise
  - **Agent Persona:** “Alex Morgan,” a friendly New York real-estate agent

### 📄 PDF Summary Export
- Download a **Lease Summary PDF** with all extracted fields and AI notes  
- Fully Unicode-safe output for international text

### 🧮 Table Extraction
- Detects and extracts tabular data (e.g., rent rolls, fee schedules) from PDFs  
- Displays them directly as editable dataframes

### 🧠 Built-in Help & Tips
- Quick sidebar guide for users on how to upload, analyze, and export results

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/real-estate-analyzer.git
cd real-estate-analyzer
