from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from qa_pipeline import load_pdf, split_docs, create_vectorstore, get_qa_chain
import tempfile

app = FastAPI()

# ✅ Enable CORS for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with Angular origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_chain = None

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global qa_chain
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    docs = load_pdf(tmp_path)
    chunks = split_docs(docs)
    vectorstore = create_vectorstore(chunks)
    qa_chain = get_qa_chain(vectorstore)
    return {"message": "PDF processed successfully"}

@app.post("/ask_question")
async def ask_question(query: str = Form(...)):
    global qa_chain
    if not qa_chain:
        return {"error": "PDF not uploaded/processed yet"}
    answer = qa_chain.run(query)
    return {"answer": answer}