import streamlit as st
import tempfile
from qa_pipeline import load_pdf, split_docs, create_vectorstore, get_qa_chain

st.title("📄 PDF Q&A using LangChain + OpenAI")

# Step 1: User apni OpenAI API Key input kare
user_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

if user_api_key:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name
        
        with st.spinner("Loading and processing..."):
            docs = load_pdf(file_path)
            chunks = split_docs(docs)
            vectorstore = create_vectorstore(chunks, api_key=user_api_key)
            qa_chain = get_qa_chain(vectorstore, api_key=user_api_key)
        
        st.success("✅ PDF processed! You can now ask questions.")

        user_query = st.text_input("Ask a question from the PDF:")

        if user_query:
            with st.spinner("Thinking..."):
                response = qa_chain.run(user_query)
                st.write("**Answer:**", response)
