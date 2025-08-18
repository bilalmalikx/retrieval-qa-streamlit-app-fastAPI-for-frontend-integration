import streamlit as st
import tempfile
from qa_pipeline import load_pdf, split_docs, create_vectorstore, get_qa_chain

st.title("📄 PDF Q&A using LangChain + OpenAI")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
    
    with st.spinner("Loading and processing..."):
        docs = load_pdf(file_path)
        chunks = split_docs(docs)
        vectorstore = create_vectorstore(chunks)
        qa_chain = get_qa_chain(vectorstore)
    
    st.success("PDF processed! You can now ask questions.")

    user_query = st.text_input("Ask a question from the PDF:")

    if user_query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_query)
            st.write("**Answer:**", response)