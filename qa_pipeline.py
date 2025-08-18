from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain.chains import RetrievalQA


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def create_vectorstore(chunks, api_key: str, store_type="faiss"):
    embeddings = OpenAIEmbeddings(api_key=api_key)   # 👈 User API key use ho rahi hai
    
    if store_type == "faiss":
        return FAISS.from_documents(chunks, embeddings)
    else:
        return Chroma.from_documents(chunks, embeddings)

def get_qa_chain(vectorstore, api_key: str):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0, api_key=api_key)  # 👈 User API key use ho rahi hai
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
