# Retrieval QA App using LangChain, Streamlit, and FastAPI

This project is a Retrieval-based QA (Question Answering) system using:

- 🧠 **LangChain + OpenAI Embeddings**
- 📦 **FAISS** or **ChromaDB** Vector Store
- 📄 PDF File Parsing with **PyMuPDF** and **pdfplumber**
- 🖥️ **Streamlit** UI for file upload and QA interface
- 🚀 **FastAPI** for potential frontend/backend separation and future extension

## 🔧 Features

- Upload a PDF and extract text using PyMuPDF/pdfplumber
- Embed the PDF text using OpenAI Embeddings
- Store embeddings in FAISS or ChromaDB
- Ask natural language questions about the PDF content
- Use LangChain RetrievalQA to return the answer

## 📂 Structure

```bash
retrieval_qa_app/
│
├── app.py                # Streamlit UI
├── backend.py            # Embedding, vector store, and QA logic
├── requirements.txt
└── README.md
