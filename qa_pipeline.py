import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader, PDFPlumberLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def load_pdf(file_path):
    # loader = PyMuPDFLoader(file_path)
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def create_vectorstore(chunks, store_type="faiss"):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    if store_type == "faiss":
        return FAISS.from_documents(chunks, embeddings)
    else:
        return Chroma.from_documents(chunks, embeddings)

def get_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
