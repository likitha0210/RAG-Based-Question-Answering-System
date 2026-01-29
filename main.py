from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
from rag_core import ingest_document, retrieve_chunks, generate_answer

app = FastAPI(title="RAG-Based Question Answering System")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    latency_ms: float

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")
    
    file_path = f"{UPLOAD_DIR}/current_document.pdf"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # ✅ SYNCHRONOUS processing - WAITS until complete
    ingest_document(file_path, "current")
    return {"message": "✅ PDF uploaded and processed successfully"}

@app.post("/ask/", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    chunks = retrieve_chunks("current", req.question)
    if not chunks:
        return AnswerResponse(answer="Please upload a PDF first", latency_ms=0)
    
    # ✅ Fix Document error
    context = "\n".join([str(chunk) for chunk in chunks])
    answer, latency = generate_answer(context, req.question)
    return AnswerResponse(answer=answer, latency_ms=latency)
